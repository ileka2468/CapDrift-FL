"""
CapDrift-FL Controller: Capability- and Drift-Aware Federated Learning
Implements the exact 3-stage controller specified in Professor Siasi's IEEE template:
1. Stage 1: Capability-Aware Step Budgeting (Eqs. 3, 4, 6)
2. Stage 2: Block-Wise Directional Drift Gating on Surplus Training (Eqs. 2, 5, Algorithm 1)
3. Stage 3: Coordinator-Side Contribution Bounding / Norm Clipping (Eqs. 7, 8, Algorithm 2)
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

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
    enable_drift_gating: bool = True # If False, surplus training never early-stops
    enable_clipping: bool = True    # If False, G_bar_i = G_i (no contribution bounding)


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    """Flattens a list of parameter tensors into a single 1D tensor."""
    return torch.cat([p.detach().flatten() for p in params])


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
        
    def determine_step_budget(self) -> Tuple[int, int]:
        """
        Computes S_cap_i(t) and S_req_i(t) using Equation (6).
        """
        if not self.cfg.enable_capability:
            # Baseline or drift-only: if drift-only, allow S_max work; if FedAvg, S_min
            if self.cfg.enable_drift_gating:
                s_cap = self.cfg.S_max
            else:
                s_cap = self.cfg.S_min
            s_req = min(self.cfg.S_min, s_cap)
            return s_cap, s_req
            
        if self.q_hat is None or self.q_hat <= 0:
            s_cap = self.cfg.S_0
        else:
            estimated_steps = int(np.floor(self.q_hat * self.cfg.T_target_sec))
            s_cap = min(self.cfg.S_max, max(1, estimated_steps))
            
        s_req = min(self.cfg.S_min, s_cap)
        return s_cap, s_req

    def train_round(
        self,
        local_model: nn.Module,
        global_model_weights: List[torch.Tensor],
        delta_G_prev: Optional[torch.Tensor],
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        simulated_step_duration_sec: Optional[float] = None
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
        s_cap, s_req = self.determine_step_budget()
        
        # Initialize local state
        S_i = 0
        k = 0
        drift_evaluations = []
        gated_early = False
        
        # Track training timer
        t_start = time.perf_counter()
        
        w_global_flat = flatten_params(global_model_weights)
        
        # Iterator for data
        data_iter = iter(train_loader)
        
        while S_i < s_cap:
            # Check phase
            if S_i >= s_req:
                # Surplus phase: check drift gating before running surplus block
                w_current_flat = flatten_params([p.data for p in local_model.parameters()])
                delta_i_k = w_current_flat - w_global_flat
                
                if (
                    self.cfg.enable_drift_gating
                    and delta_G_prev is not None
                    and torch.norm(delta_i_k) > self.cfg.epsilon
                    and torch.norm(delta_G_prev) > self.cfg.epsilon
                ):
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
                            # DRIFT GATING TRIGGERED! Stop surplus training immediately
                            gated_early = True
                            break
                            
                B = min(self.cfg.K_block, s_cap - S_i)
            else:
                # Required phase: train without drift restrictions
                B = min(self.cfg.K_block, s_req - S_i)
                
            # Execute B SGD steps
            local_model.train()
            for _ in range(B):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    images, labels = next(data_iter)
                    
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            S_i += B
            k += 1
            
        t_elapsed = time.perf_counter() - t_start
        
        # If simulated duration is provided (for heterogeneous hardware emulation), use it
        if simulated_step_duration_sec is not None and S_i > 0:
            tau_tr = simulated_step_duration_sec * S_i
        else:
            tau_tr = max(1e-4, t_elapsed)
            
        # Measured throughput q_i(t) = S_i(t) / tau_tr_i(t)
        q_i = float(S_i / tau_tr) if tau_tr > 0 else 1.0
        self.last_throughput = q_i
        
        # Update local EMA throughput estimate
        if self.q_hat is None:
            self.q_hat = q_i
        else:
            self.q_hat = self.cfg.beta_ema * self.q_hat + (1.0 - self.cfg.beta_ema) * q_i
            
        # Compute returned displacement Delta_i(t) = w_i(t) - w^(t-1)
        delta_i = [p.data.clone() - w_t for p, w_t in zip(local_model.parameters(), global_model_weights)]
        
        return {
            "client_id": self.client_id,
            "delta_i": delta_i,
            "q_i": q_i,
            "q_hat": self.q_hat,
            "S_cap": s_cap,
            "S_req": s_req,
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
        sample_counts: Dict[str, int]
    ) -> Dict:
        """
        Runs Algorithm 2.
        Takes client updates, applies sample weights p_i, norm bounds G_i with cap c,
        and aggregates into global model.
        """
        if not client_results:
            return {}
            
        total_samples = sum(sample_counts[res["client_id"]] for res in client_results)
        participating_cids = [res["client_id"] for res in client_results]
        
        # 1. Update capability estimates on server (Eq. 4)
        for res in client_results:
            cid = res["client_id"]
            q_i = res["q_i"]
            if cid not in self.client_q_hat:
                self.client_q_hat[cid] = q_i
            else:
                self.client_q_hat[cid] = self.cfg.beta_ema * self.client_q_hat[cid] + (1.0 - self.cfg.beta_ema) * q_i
                
        # 2. Form sample-weighted contributions G_i(t) = p_i(t) * Delta_i(t)
        weighted_contributions = []
        raw_norms = []
        
        for res in client_results:
            cid = res["client_id"]
            p_i = sample_counts[cid] / float(total_samples) if total_samples > 0 else (1.0 / len(client_results))
            
            # G_i(t)
            G_i_tensors = [p_i * d.clone() for d in res["delta_i"]]
            flat_G_i = flatten_params(G_i_tensors)
            norm_G_i = float(torch.norm(flat_G_i).item())
            
            weighted_contributions.append({
                "cid": cid,
                "p_i": p_i,
                "G_i": G_i_tensors,
                "flat_G_i": flat_G_i,
                "norm_G_i": norm_G_i
            })
            raw_norms.append(norm_G_i)
            
        # Calibrate clipping cap c on first round if not yet calibrated
        if self.clipping_cap_c is None:
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
                    
            bounded_tensors = [t * scale for t in wc["G_i"]]
            bounded_contributions.append(bounded_tensors)
            
            clipping_events.append({
                "cid": wc["cid"],
                "raw_norm": norm_val,
                "bounded_norm": norm_val * scale,
                "cap_c": c,
                "is_clipped": is_clipped,
                "scale_factor": scale
            })
            
        # 4. Aggregate: w^(t) = w^(t-1) + eta_s * sum_i G_bar_i(t) (Eq. 8)
        with torch.no_grad():
            for p_idx, p in enumerate(global_model.parameters()):
                agg_delta = sum(
                    bc[p_idx] * self.cfg.server_lr
                    for bc in bounded_contributions
                )
                p.data.add_(agg_delta)
                
        # 5. Compute global displacement Delta_G^(t) = w^(t) - w^(t-1)
        # sum of aggregated updates
        flat_delta_G = torch.zeros_like(weighted_contributions[0]["flat_G_i"])
        for bc in bounded_contributions:
            flat_delta_G.add_(flatten_params(bc) * self.cfg.server_lr)
            
        self.delta_G_prev = flat_delta_G.detach().clone()
        
        telemetry = {
            "round": server_round,
            "participating_clients": participating_cids,
            "clipping_cap_c": c,
            "clipping_events": clipping_events,
            "global_displacement_norm": float(torch.norm(flat_delta_G).item()),
            "client_capabilities": copy.deepcopy(self.client_q_hat)
        }
        self.round_history.append(telemetry)
        return telemetry

# Directional drift formulation: d_i = 1 - cos(Delta_i, Delta_G_prev)

# Mini-block surplus execution: evaluates drift at K-step boundaries

# Numeric stability: clamp cosine similarity to [-1.0, 1.0]
