"""Capability-aware step budgeting, surplus drift control, and bounded aggregation."""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class CapDriftConfig:
    # Stage 1: Capability settings
    beta_ema: float = 0.80          # Smoothing factor for throughput EMA
    T_target_sec: float = 30.0      # Target local training interval (sec)
    S_max: int = 50                 # Maximum local step ceiling
    S_min: int = 5                  # Required minimum local workload (steps)
    S_0: int = 5                    # Initial local budget before throughput is observed
    
    # Stage 2: Drift gating settings
    K_block: int = 5                # Block size for local SGD steps
    tau_drift: float = 0.50         # Drift threshold (cosine similarity >= 0.50)
    epsilon: float = 1e-8           # Numerical stability safeguard
    
    # Stage 3: Contribution clipping settings
    kappa_c: float = 1.5            # Multiplier for median contribution norm calibration
    server_lr: float = 1.0          # Server update step size eta_s
    
    # Mode flags for ablations
    enable_capability: bool = True  # If False, S_cap = S_min (or fixed S_max for drift-only)
    enable_drift_gating: bool = True # Enables the selected drift control
    enable_clipping: bool = True    # If False, G_bar_i = G_i (no contribution bounding)
    dynamic_clipping: bool = True   # Dynamically re-calibrate median clipping cap each round

    # Defaults retain the legacy controller for existing ablation scripts.
    drift_mode: str = "gate"        # "gate" (legacy) or "projection"
    drift_parameter_prefix: Optional[str] = None  # e.g., "classifier.3"; None selects all
    scale_local_lr: bool = False    # eta_i = eta_0 * sqrt(S_min / S_cap)
    server_momentum: float = 0.0    # Polyak beta; 0 disables velocity accumulation
    anneal_start_round: Optional[int] = None  # apply taper strictly after this round
    anneal_power: float = 1.0
    budget_lr_power: float = 0.5
    client_update_normalization_power: float = 0.0
    server_lr_decay_start_round: Optional[int] = None
    server_lr_final_ratio: float = 1.0
    server_average_decay: Optional[float] = None  # evaluation-only exponential weight average
    server_average_start_round: int = 10
    local_prior_strength: float = 0.0  # local CE uses a smoothed label-prior offset
    gradient_mass_blend: float = 0.0   # bounded blend with solver-mass normalization

    def __post_init__(self):
        if self.drift_mode not in ("gate", "projection"):
            raise ValueError("drift_mode must be 'gate' or 'projection'")
        if not 0 <= self.server_momentum < 1:
            raise ValueError("server_momentum must be in [0, 1)")
        if not 1 <= self.S_min <= self.S_max or self.S_0 < 1 or self.K_block < 1:
            raise ValueError("step budgets and block size must be positive, with S_min <= S_max")
        if not math.isfinite(self.anneal_power) or self.anneal_power <= 0:
            raise ValueError("anneal_power must be finite and positive")
        if self.anneal_start_round is not None and self.anneal_start_round < 0:
            raise ValueError("anneal_start_round must be nonnegative")
        if not 0 <= self.budget_lr_power <= 1:
            raise ValueError("budget_lr_power must be in [0, 1]")
        if not 0 <= self.client_update_normalization_power <= 1:
            raise ValueError("client_update_normalization_power must be in [0, 1]")
        if not math.isfinite(self.server_lr) or self.server_lr <= 0:
            raise ValueError("server_lr must be finite and positive")
        if self.server_lr_decay_start_round is not None and self.server_lr_decay_start_round < 0:
            raise ValueError("server_lr_decay_start_round must be nonnegative")
        if not 0 <= self.server_lr_final_ratio <= 1:
            raise ValueError("server_lr_final_ratio must be in [0, 1]")
        if self.server_average_decay is not None and not 0 <= self.server_average_decay < 1:
            raise ValueError("server_average_decay must be in [0, 1)")
        if self.server_average_start_round < 1:
            raise ValueError("server_average_start_round must be positive")
        if not math.isfinite(self.local_prior_strength) or self.local_prior_strength < 0:
            raise ValueError("local_prior_strength must be finite and nonnegative")
        if not 0 <= self.gradient_mass_blend <= 1:
            raise ValueError("gradient_mass_blend must be in [0, 1]")

    def scheduled_server_lr(self, server_round: int, total_rounds: Optional[int] = None) -> float:
        """Cosine-taper the server gain without reducing the amount of client work."""
        start = self.server_lr_decay_start_round
        if start is None or server_round <= start:
            return self.server_lr
        if total_rounds is None or not start < server_round <= total_rounds:
            raise ValueError("server LR decay requires decay_start < server_round <= total_rounds")
        phase = (server_round - start) / (total_rounds - start)
        progress = 0.5 * (1 + math.cos(math.pi * phase))
        return self.server_lr * (self.server_lr_final_ratio + (1 - self.server_lr_final_ratio) * progress)


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    """Flattens a list of parameter tensors into a single 1D tensor."""
    return torch.cat([p.detach().flatten() for p in params])


def local_logit_offset(train_class_counts: List[int], strength: float) -> torch.Tensor:
    """Centered log of training-only class counts with four pseudocounts per class."""
    counts = torch.as_tensor(train_class_counts, dtype=torch.float64, device="cpu")
    if (counts.ndim != 1 or counts.numel() < 2 or not torch.isfinite(counts).all()
            or (counts < 0).any() or counts.sum() <= 0):
        raise ValueError("train_class_counts must give nonnegative counts for at least two classes")
    logs = (counts + 4).log()
    return strength * (logs - logs.mean())


def compute_directional_drift(
    delta_i_k: torch.Tensor,
    delta_G_prev: torch.Tensor,
    epsilon: float = 1e-8
) -> Optional[float]:
    """
    Computes directional drift d_i(t, k) according to Equation (5):
    d_i(t, k) = 1 - ( (Delta_i^(t,k))^T Delta_G^(t-1) ) / ( ||Delta_i^(t,k)||_2 * ||Delta_G^(t-1)||_2 )
    Returns value in [0, 2], with smaller values indicating stronger alignment.
    Returns None if either displacement norm is <= epsilon.
    """
    norm_local = torch.norm(delta_i_k)
    norm_global = torch.norm(delta_G_prev)
    
    if norm_local <= epsilon or norm_global <= epsilon:
        return None
    
    cosine_sim = torch.dot(delta_i_k, delta_G_prev) / (norm_local * norm_global)
    # Clamp cosine similarity to [-1.0, 1.0] to guard against precision float errors
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
    return float(1.0 - cosine_sim.item())


def attenuate_orthogonal_drift(
    local_delta: torch.Tensor,
    consensus: torch.Tensor,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[float]]:
    """Keep the signed parallel component, scaling the remainder by max(0, cos)."""
    # One scalar transfer supplies diagnostics and handles zero directions.
    dot = torch.dot(local_delta, consensus)
    local_norm = torch.linalg.vector_norm(local_delta)
    consensus_norm = torch.linalg.vector_norm(consensus)
    cosine_t = (dot / (local_norm * consensus_norm)).clamp(-1.0, 1.0)
    valid = (local_norm > epsilon) & (consensus_norm > epsilon)
    cosine = float(torch.where(valid, cosine_t, float("nan")).item())
    if not math.isfinite(cosine):
        return local_delta, None
    parallel = (dot / consensus_norm.square()) * consensus
    return parallel + max(0.0, cosine) * (local_delta - parallel), cosine


def _drift_parameter_indices(model: nn.Module, prefix: Optional[str]) -> List[int]:
    indices = [
        i for i, (name, _) in enumerate(model.named_parameters())
        if prefix is None or name == prefix or name.startswith(prefix + ".")
    ]
    if not indices:
        raise ValueError(f"no model parameters match drift prefix {prefix!r}")
    return indices


class CapDriftClientTrainer:
    """
    Executes Algorithm 1: Capability- and Drift-Aware Local Training on a client node.
    """
    def __init__(self, client_id: str, config: CapDriftConfig):
        self.client_id = client_id
        self.cfg = config
        
        # Capability state
        self.q_hat: Optional[float] = None  # Smoothed throughput (steps/sec)
        self.last_throughput: float = 0.0
        
    def determine_step_budget(
        self,
        server_round: int = 1,
        total_rounds: Optional[int] = None,
        simulated_step_duration_sec: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Intersect the throughput-estimated budget, round ceiling, and modeled deadline."""
        ceiling = self.cfg.S_max
        if self.cfg.anneal_start_round is not None and server_round > self.cfg.anneal_start_round:
            if total_rounds is None or not 1 <= server_round <= total_rounds:
                raise ValueError("annealing requires 1 <= server_round <= total_rounds")
            remaining_rounds = total_rounds - server_round
            if self.cfg.anneal_power == 1:
                annealed_steps = ceiling * remaining_rounds // total_rounds
            else:
                annealed_steps = math.floor(ceiling * (remaining_rounds / total_rounds)**self.cfg.anneal_power)
            ceiling = max(self.cfg.S_min, annealed_steps)

        if not self.cfg.enable_capability:
            # Baseline or drift-only: if drift-only, allow S_max work; if FedAvg, S_min
            s_cap = ceiling if self.cfg.enable_drift_gating else self.cfg.S_min
        elif self.q_hat is None or self.q_hat <= 0:
            s_cap = self.cfg.S_0
        else:
            s_cap = max(1, math.floor(self.q_hat * self.cfg.T_target_sec))

        s_cap = min(s_cap, ceiling)
        if simulated_step_duration_sec is not None:
            if not math.isfinite(simulated_step_duration_sec) or simulated_step_duration_sec <= 0:
                raise ValueError("simulated step duration must be finite and positive")
            s_cap = min(s_cap, math.floor(self.cfg.T_target_sec / simulated_step_duration_sec))
        return s_cap, min(self.cfg.S_min, s_cap)

    def train_round(
        self,
        local_model: nn.Module,
        global_model_weights: List[torch.Tensor],
        delta_G_prev: Optional[torch.Tensor],
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        simulated_step_duration_sec: Optional[float] = None,
        server_round: int = 1,
        total_rounds: Optional[int] = None,
        train_class_counts: Optional[List[int]] = None,
    ) -> Dict:
        """
        Runs Algorithm 1.
        Returns:
            delta_i: List of parameter displacement tensors (w_i^(t) - w^(t-1))
            q_i: measured throughput (steps/sec)
            steps_completed: S_i(t)
            drift_events: list of drift evaluations
            gated_early: bool, whether surplus training was stopped by drift gating
        """
        s_cap, s_req = self.determine_step_budget(
            server_round, total_rounds, simulated_step_duration_sec
        )
        lr_scale = self._lr_scale(s_cap)
        prior_offset = None
        if self.cfg.local_prior_strength:
            if train_class_counts is None:
                raise ValueError("local prior adjustment requires train_class_counts")
            prior_offset = local_logit_offset(train_class_counts, self.cfg.local_prior_strength).to(device)
        original_lrs = [g["lr"] for g in optimizer.param_groups]
        for group, lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = lr * lr_scale
        try:
            return self._train_blocks(
                local_model, global_model_weights, delta_G_prev, train_loader,
                criterion, optimizer, device, simulated_step_duration_sec,
                s_cap, s_req, lr_scale, prior_offset,
            )
        finally:
            for group, lr in zip(optimizer.param_groups, original_lrs):
                group["lr"] = lr
        
    def _lr_scale(self, budget: int) -> float:
        if not self.cfg.scale_local_lr:
            return 1.0
        ratio = self.cfg.S_min / max(self.cfg.S_min, budget)
        if self.cfg.budget_lr_power == 0.5:
            return math.sqrt(ratio)
        return ratio ** self.cfg.budget_lr_power

    def _train_blocks(
        self, local_model, global_model_weights, delta_G_prev, train_loader,
        criterion, optimizer, device, simulated_step_duration_sec, s_cap, s_req, lr_scale, prior_offset,
    ) -> Dict:
        S_i = 0
        k = 0
        drift_evaluations = []
        gated_early = False
        
        # Track training timer
        t_start = time.perf_counter()
        
        params = list(local_model.parameters())
        drift_active = self.cfg.enable_drift_gating and delta_G_prev is not None and s_cap > s_req
        projected = drift_active and self.cfg.drift_mode == "projection"
        drift_params = []
        if drift_active:
            drift_indices = _drift_parameter_indices(local_model, self.cfg.drift_parameter_prefix)
            drift_params = [params[i] for i in drift_indices]
            if delta_G_prev.numel() != sum(p.numel() for p in drift_params):
                raise ValueError("consensus displacement does not match the selected drift parameters")
            if not projected:
                w_global_flat = flatten_params([global_model_weights[i] for i in drift_indices])

        data_iter = iter(train_loader) if s_cap else None
        local_model.train()
        while S_i < s_cap:
            surplus = S_i >= s_req
            if surplus:
                if drift_active and not projected:
                    delta_i_k = flatten_params(drift_params) - w_global_flat
                    d_score = compute_directional_drift(delta_i_k, delta_G_prev, self.cfg.epsilon)
                    if d_score is not None:
                        drift_evaluations.append({
                            "block": k,
                            "step": S_i,
                            "drift_score": d_score,
                            "cosine_similarity": 1.0 - d_score,
                            "threshold": self.cfg.tau_drift,
                            "passed": d_score <= self.cfg.tau_drift
                        })
                        if d_score > self.cfg.tau_drift:
                            gated_early = True
                            break
                if projected:
                    before_block = flatten_params(drift_params)
                B = min(self.cfg.K_block, s_cap - S_i)
            else:
                # Required phase: train without drift restrictions
                B = min(self.cfg.K_block, s_req - S_i)
                
            # Execute B SGD steps
            for _ in range(B):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    images, labels = next(data_iter)
                    
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = local_model(images)
                if prior_offset is not None:
                    if outputs.shape[-1] != prior_offset.numel():
                        raise ValueError("train_class_counts does not match the model's class count")
                    outputs = outputs + prior_offset.to(dtype=outputs.dtype)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            S_i += B
            if surplus and projected:
                with torch.no_grad():
                    block_delta = flatten_params(drift_params) - before_block
                    adjusted, cosine = attenuate_orthogonal_drift(
                        block_delta, delta_G_prev, self.cfg.epsilon
                    )
                    if cosine is not None:
                        # Apply the correction in place; the next block starts at the adjusted weights.
                        correction = (adjusted - block_delta).split([p.numel() for p in drift_params])
                        for p, diff in zip(drift_params, correction):
                            p.add_(diff.reshape_as(p))
                        drift_evaluations.append({
                            "block": k,
                            "step": S_i,
                            "drift_score": 1.0 - cosine,
                            "cosine_similarity": cosine,
                            "orthogonal_scale": max(0.0, cosine),
                        })
            k += 1
            
        t_elapsed = time.perf_counter() - t_start
        
        # If simulated duration is provided (for heterogeneous hardware emulation), use it
        if simulated_step_duration_sec is not None:
            tau_tr = simulated_step_duration_sec * S_i
        else:
            tau_tr = max(1e-4, t_elapsed)
            
        # Measured throughput q_i(t) = S_i(t) / tau_tr_i(t)
        q_i = float(S_i / tau_tr) if tau_tr > 0 else 0.0
        self.last_throughput = q_i
        
        # Update local EMA throughput estimate
        if S_i:  # no throughput observation when the deadline allows zero work
            self.q_hat = (
                q_i if self.q_hat is None
                else self.cfg.beta_ema * self.q_hat + (1.0 - self.cfg.beta_ema) * q_i
            )
            
        # Compute returned displacement Delta_i(t) = w_i(t) - w^(t-1)
        delta_i = [p.detach() - w_t for p, w_t in zip(params, global_model_weights)]
        
        return {
            "client_id": self.client_id,
            "delta_i": delta_i,
            "q_i": q_i,
            "q_hat": self.q_hat,
            "S_cap": s_cap,
            "S_req": s_req,
            "lr_scale": lr_scale,
            "steps_completed": S_i,
            "tau_tr": tau_tr,
            "gated_early": gated_early,
            "drift_evaluations": drift_evaluations
        }


class CapDriftCoordinator:
    """
    Executes Algorithm 2: CapDrift-FL Server Coordination.
    Manages client capability EMA smoothing, contribution bounding (clipping),
    global aggregation, and global displacement tracking.
    """
    def __init__(self, config: CapDriftConfig):
        self.cfg = config
        self.client_q_hat: Dict[str, float] = {}
        self.delta_G_prev: Optional[torch.Tensor] = None
        self.server_velocity: Optional[List[torch.Tensor]] = None
        self.averaged_parameters: Optional[List[torch.Tensor]] = None
        self.clipping_cap_c: Optional[float] = None
        self.round_history: List[Dict] = []
        
    def calibrate_clipping_cap(self, sample_weighted_norms: List[float]):
        """
        Calibrates c = kappa_c * m_G, where m_G is the median norm of sample-weighted contributions.
        """
        if sample_weighted_norms:
            m_G = float(np.median(sample_weighted_norms))
            self.clipping_cap_c = float(self.cfg.kappa_c * m_G)
        else:
            self.clipping_cap_c = 1.0

    def aggregate_round(
        self,
        server_round: int,
        global_model: nn.Module,
        client_results: List[Dict],
        sample_counts: Dict[str, int],
        total_rounds: Optional[int] = None,
        local_gradient_masses: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Runs Algorithm 2.
        Takes client updates, applies sample weights p_i, norm bounds G_i with cap c,
        and aggregates into global model. For normalized averaging, an optional
        per-client local-solver coefficient supplies the FedNova normalizer.
        Apply it before CapDrift clipping so the bound applies to the actual
        contribution; keep the FedNova weight/update convention unchanged when
        both clipping and drift projection are disabled.
        """
        if not client_results:
            return {}
            
        server_lr = self.cfg.scheduled_server_lr(server_round, total_rounds)
        client_masses = [sample_counts[res["client_id"]] for res in client_results]
        weighting_power = self.cfg.client_update_normalization_power
        blend = self.cfg.gradient_mass_blend
        if blend and local_gradient_masses is None:
            raise ValueError("gradient_mass_blend requires local_gradient_masses")
        normalizers = None
        effective_mass = None
        if local_gradient_masses is not None:
            if weighting_power:
                raise ValueError("step weighting cannot be combined with gradient-mass normalization")
            normalizers = [local_gradient_masses[res["client_id"]] for res in client_results]
            if any(not math.isfinite(v) or v < 0 for v in normalizers):
                raise ValueError("local gradient mass must be finite and nonnegative")
            client_masses = [n if a > 0 else 0 for n, a in zip(client_masses, normalizers)]
            active_samples = sum(client_masses)
            if active_samples == 0:
                return {}
            effective_mass = sum(n / active_samples * a for n, a in zip(client_masses, normalizers))
        if weighting_power:
            # Temper the displacement advantage of high-step clients without removing their work.
            client_masses = [
                samples * (self.cfg.S_min / res["steps_completed"]) ** weighting_power
                if res["steps_completed"] > 0 else 0.0
                for samples, res in zip(client_masses, client_results)
            ]
        mass_total = sum(client_masses)
        participating_cids = [res["client_id"] for res in client_results]
        
        # 1. Update capability estimates on server (Eq. 4)
        for res in client_results:
            cid = res["client_id"]
            q_i = res["q_i"]
            if q_i <= 0:
                continue  # a zero-work client has no new throughput observation
            if cid not in self.client_q_hat:
                self.client_q_hat[cid] = q_i
            else:
                self.client_q_hat[cid] = self.cfg.beta_ema * self.client_q_hat[cid] + (1.0 - self.cfg.beta_ema) * q_i
                
        # 2. Form sample-weighted contributions G_i(t) = p_i(t) * Delta_i(t)
        weighted_contributions = []
        raw_norms = []
        
        for i, (res, mass) in enumerate(zip(client_results, client_masses)):
            if normalizers is not None and (normalizers[i] == 0 or mass == 0):
                continue
            cid = res["client_id"]
            if mass_total > 0:
                p_i = mass / float(mass_total)
            else:
                p_i = 0.0 if weighting_power else 1.0 / len(client_results)
            
            # G_i(t)
            mass_adjustment = 1.0
            if normalizers is None:
                G_i_tensors = [p_i * d.detach() for d in res["delta_i"]]
            elif blend:
                ratio = min(2.0, max(0.5, effective_mass / normalizers[i]))
                mass_adjustment = (1.0 - blend) + blend * ratio
                G_i_tensors = [p_i * mass_adjustment * d.detach() for d in res["delta_i"]]
            else:
                a_i = normalizers[i]
                G_i_tensors = [p_i * (d.detach() / a_i) * effective_mass for d in res["delta_i"]]
            norm_G_i = float(torch.norm(flatten_params(G_i_tensors)).item())
            
            weighted_contributions.append({
                "cid": cid,
                "weight": p_i,
                "mass_adjustment": mass_adjustment,
                "G_i": G_i_tensors,
                "norm_G_i": norm_G_i
            })
            if not weighting_power or p_i > 0:
                raw_norms.append(norm_G_i)
            
        # Dynamically calibrate clipping cap c each round (or first round if dynamic_clipping is False)
        if self.cfg.dynamic_clipping or self.clipping_cap_c is None:
            self.calibrate_clipping_cap(raw_norms)
            
        # 3. Apply norm bounding (clipping) (Eq. 7)
        c = self.clipping_cap_c if self.clipping_cap_c is not None else 1.0
        bounded_contributions = []
        clipping_events = []
        
        for wc in weighted_contributions:
            norm_val = wc["norm_G_i"]
            is_clipped = False
            scale = 1.0
            
            if self.cfg.enable_clipping:
                denom = max(norm_val, self.cfg.epsilon)
                if norm_val > c:
                    scale = c / denom
                    is_clipped = True
                    
            bounded_tensors = [t * scale for t in wc["G_i"]] if scale != 1.0 else wc["G_i"]
            bounded_contributions.append(bounded_tensors)
            
            clipping_events.append({
                "cid": wc["cid"],
                "aggregation_weight": wc["weight"],
                "mass_adjustment": wc["mass_adjustment"],
                "raw_norm": norm_val,
                "bounded_norm": norm_val * scale,
                "cap_c": c,
                "is_clipped": is_clipped,
                "scale_factor": scale
            })
            
        # 4. Aggregate clipped client updates, then apply optional Polyak momentum.
        deltas = []
        with torch.no_grad():
            params = list(global_model.parameters())
            drift_indices = _drift_parameter_indices(global_model, self.cfg.drift_parameter_prefix)
            if self.cfg.server_momentum and self.server_velocity is None:
                self.server_velocity = [torch.zeros_like(p) for p in params]
            for p_idx, p in enumerate(params):
                if self.cfg.server_momentum:
                    raw_delta = sum(bc[p_idx] for bc in bounded_contributions)
                    velocity = self.server_velocity[p_idx]
                    velocity.mul_(self.cfg.server_momentum).add_(raw_delta)
                    applied = velocity * server_lr
                else:
                    applied = sum(bc[p_idx] * server_lr for bc in bounded_contributions)
                p.add_(applied)
                deltas.append(applied)

            decay = self.cfg.server_average_decay
            if decay is not None and server_round >= self.cfg.server_average_start_round:
                if self.averaged_parameters is None:
                    self.averaged_parameters = [p.detach().clone() for p in params]
                else:
                    for averaged, p in zip(self.averaged_parameters, params):
                        averaged.lerp_(p, 1 - decay)

        # Momentum changes the trajectory: use the applied step, not the raw mean.
        flat_delta_G = flatten_params(deltas)
        self.delta_G_prev = (
            flat_delta_G if self.cfg.drift_parameter_prefix is None
            else flatten_params([deltas[i] for i in drift_indices])
        )
        
        telemetry = {
            "round": server_round,
            "server_lr": server_lr,
            "uses_averaged_parameters": self.averaged_parameters is not None,
            "effective_gradient_mass": effective_mass,
            "participating_clients": participating_cids,
            "clipping_cap_c": c,
            "clipping_events": clipping_events,
            "global_displacement_norm": float(torch.norm(flat_delta_G).item()),
            "drift_displacement_norm": float(torch.norm(self.delta_G_prev).item()),
            "client_capabilities": self.client_q_hat.copy(),
        }
        self.round_history.append(telemetry)
        return telemetry
