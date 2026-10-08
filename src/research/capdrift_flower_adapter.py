"""
CapDrift-FL Flower Adapter for Physical Proxmox Heterogeneous Cluster.
Integrates the three-stage controller into real gRPC-based Flower clients and server:
1. Client-Side: Throughput estimation, step budgeting, and block-wise cosine drift gating.
2. Server-Side: Median contribution norm calculation, adaptive clipping, and weighted aggregation.
3. System Telemetry: Tracks real CPU contention (psutil), round duration, idle straggler waste, and gating.
"""

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
try:
    from src.model import get_model, get_parameters, set_parameters
except ImportError:
    from model import get_model, get_parameters, set_parameters


@dataclass
class ClusterControllerConfig:
    min_steps: int = 5
    max_steps: int = 50
    target_round_time: float = 12.0
    drift_block_size: int = 5
    drift_threshold_tau: float = 0.50
    clipping_multiplier_kappa: float = 1.5
    server_lr: float = 1.0
    local_lr: float = 0.01
    local_momentum: float = 0.9
    local_weight_decay: float = 1e-4
    use_drift_gating: bool = True
    use_capability_budgeting: bool = True
    use_clipping: bool = True


class CapDriftFlowerClient(fl.client.NumPyClient):
    """
    Physical edge client running inside a Proxmox LXC container (or local GPU workstation).
    Executes base steps followed by drift-gated surplus blocks if capability permits.
    """
    def __init__(
        self,
        node_id: str,
        data_loader,
        model: nn.Module,
        device: torch.device,
        config: ClusterControllerConfig,
        criterion: Optional[nn.Module] = None
    ):
        self.node_id = node_id
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.config = config
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        self.smoothed_throughput: Optional[float] = None
        self.ema_beta: float = 0.70
        self.total_gating_events: int = 0

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def _calculate_cosine_drift(
        self,
        base_delta: List[torch.Tensor],
        current_delta: List[torch.Tensor]
    ) -> float:
        dot_product = 0.0
        norm_b_sq = 0.0
        norm_c_sq = 0.0
        for b, c in zip(base_delta, current_delta):
            b_f = b.view(-1).float()
            c_f = c.view(-1).float()
            dot_product += torch.sum(b_f * c_f).item()
            norm_b_sq += torch.sum(b_f * b_f).item()
            norm_c_sq += torch.sum(c_f * c_f).item()
            
        norm_b = np.sqrt(norm_b_sq)
        norm_c = np.sqrt(norm_c_sq)
        if norm_b <= 1e-8 or norm_c <= 1e-8:
            return 0.0
        cosine_sim = dot_product / (norm_b * norm_c)
        return float(np.clip(1.0 - cosine_sim, 0.0, 2.0))

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # Synchronize global weights from coordinator
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        self.model.train()

        round_idx = int(config.get("round", 1))
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        optimizer = SGD(
            self.model.parameters(),
            lr=self.config.local_lr,
            momentum=self.config.local_momentum,
            weight_decay=self.config.local_weight_decay
        )

        data_iter = iter(self.data_loader)
        def next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                return next(data_iter)

        # 1. Step Budgeting: Eq. 6
        if not self.config.use_capability_budgeting:
            step_budget = self.config.min_steps
        elif self.smoothed_throughput is None:
            step_budget = self.config.min_steps
        else:
            raw_budget = int(np.floor(self.smoothed_throughput * self.config.target_round_time))
            step_budget = max(self.config.min_steps, min(self.config.max_steps, raw_budget))

        # 2. Phase 1: Base Local Training (S_min steps)
        t_start = time.perf_counter()
        cpu_start = psutil.cpu_percent(interval=None)
        
        base_steps = self.config.min_steps
        base_loss = 0.0
        num_samples = 0

        for _ in range(base_steps):
            x, y = next_batch()
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            optimizer.step()
            base_loss += loss.item() * len(y)
            num_samples += len(y)

        delta_base = [p.detach() - init_p for p, init_p in zip(self.model.parameters(), initial_params)]
        last_aligned_checkpoint = [p.clone().detach() for p in self.model.parameters()]
        steps_executed = base_steps
        drift_detected = False
        final_drift = 0.0

        # 3. Phase 2: Surplus Training with Cosine Drift Gating
        surplus_steps_target = step_budget - base_steps
        if self.config.use_drift_gating and surplus_steps_target > 0:
            block_size = self.config.drift_block_size
            num_blocks = int(np.ceil(surplus_steps_target / block_size))
            
            for b in range(num_blocks):
                current_block_steps = min(block_size, surplus_steps_target - (b * block_size))
                for _ in range(current_block_steps):
                    x, y = next_batch()
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(x)
                    loss = self.criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    num_samples += len(y)
                
                steps_executed += current_block_steps
                delta_current = [p.detach() - init_p for p, init_p in zip(self.model.parameters(), initial_params)]
                drift_k = self._calculate_cosine_drift(delta_base, delta_current)
                final_drift = drift_k

                if drift_k > self.config.drift_threshold_tau:
                    # Gating condition met: roll back to last aligned checkpoint
                    drift_detected = True
                    self.total_gating_events += 1
                    with torch.no_grad():
                        for p, aligned_p in zip(self.model.parameters(), last_aligned_checkpoint):
                            p.copy_(aligned_p)
                    steps_executed -= current_block_steps
                    break
                else:
                    last_aligned_checkpoint = [p.clone().detach() for p in self.model.parameters()]
        elif not self.config.use_drift_gating and surplus_steps_target > 0:
            # Unconstrained surplus
            for _ in range(surplus_steps_target):
                x, y = next_batch()
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()
                num_samples += len(y)
            steps_executed += surplus_steps_target

        train_duration = max(time.perf_counter() - t_start, 1e-4)
        cpu_end = psutil.cpu_percent(interval=None)
        ram_mb = psutil.virtual_memory().used / (1024 * 1024)

        # Update EMA throughput
        current_v = steps_executed / train_duration
        if self.smoothed_throughput is None:
            self.smoothed_throughput = current_v
        else:
            self.smoothed_throughput = (self.ema_beta * self.smoothed_throughput) + ((1.0 - self.ema_beta) * current_v)

        telemetry: Dict[str, Scalar] = {
            "node_id": self.node_id,
            "steps_executed": steps_executed,
            "step_budget": step_budget,
            "train_duration_sec": round(train_duration, 3),
            "throughput": round(current_v, 2),
            "smoothed_throughput": round(self.smoothed_throughput, 2),
            "drift_detected": int(drift_detected),
            "final_drift": round(final_drift, 4),
            "cpu_percent": round(max(cpu_start, cpu_end), 1),
            "ram_mb": round(ram_mb, 1)
        }

        return get_parameters(self.model), num_samples, telemetry


class CapDriftFlowerStrategy(FedAvg):
    """
    Flower Coordinator Strategy implementing Algorithm 2:
    - Median Contribution Norm calculation: m_G(t)
    - Contribution Norm Bounding: c(t) = kappa_c * m_G(t)
    - Global model update & centralized evaluation on holdout test set using CUDA.
    """
    def __init__(
        self,
        evaluator,
        base_model: nn.Module,
        config: ClusterControllerConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.base_model = base_model
        self.config = config
        self.current_weights = get_parameters(self.base_model)
        self.history: List[Dict[str, Any]] = []
        self.best_accuracy = 0.0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # 1. Parse client contributions
        client_deltas = []
        client_sample_counts = []
        client_telemetries = []

        raw_client_weights = []
        for _, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            raw_client_weights.append(client_weights)
            
            # Compute float delta for float tensors, zero for integer buffers
            delta_i = []
            for w, gw in zip(client_weights, self.current_weights):
                if np.issubdtype(gw.dtype, np.floating):
                    delta_i.append((w.astype(np.float32) - gw.astype(np.float32)))
                else:
                    delta_i.append(np.zeros_like(gw, dtype=np.float32))
            client_deltas.append(delta_i)
            client_sample_counts.append(fit_res.num_examples)
            client_telemetries.append(fit_res.metrics)

        total_samples = sum(client_sample_counts)
        num_clients = len(results)

        # 2. Compute Contribution Norms: Eq. 7 (on floating parameter deltas)
        client_norms = []
        for delta in client_deltas:
            norm_val = np.sqrt(sum(np.sum(p ** 2) for p in delta))
            client_norms.append(float(norm_val))

        median_norm = float(np.median(client_norms))
        clipping_cap = self.config.clipping_multiplier_kappa * median_norm
        num_clipped = 0

        # 3. Apply Norm Bounding
        bounded_deltas = []
        for delta, norm_val in zip(client_deltas, client_norms):
            if self.config.use_clipping and norm_val > clipping_cap and norm_val > 1e-8:
                scale = clipping_cap / norm_val
                b_delta = [p * scale for p in delta]
                num_clipped += 1
            else:
                b_delta = delta
            bounded_deltas.append(b_delta)

        # 4. Global Model Aggregation: Eq. 8
        new_weights = []
        for param_idx in range(len(self.current_weights)):
            orig_p = self.current_weights[param_idx]
            if not np.issubdtype(orig_p.dtype, np.floating):
                # Non-floating buffer (e.g. num_batches_tracked int64), adopt from primary client
                new_weights.append(raw_client_weights[0][param_idx])
            else:
                weighted_delta = np.zeros_like(orig_p, dtype=np.float32)
                for i in range(num_clients):
                    w_i = client_sample_counts[i] / total_samples
                    weighted_delta += float(w_i) * bounded_deltas[i][param_idx]
                
                updated_p = orig_p.astype(np.float32) + (self.config.server_lr * weighted_delta)
                new_weights.append(updated_p.astype(orig_p.dtype))

        self.current_weights = new_weights
        set_parameters(self.base_model, new_weights)

        # 5. Timing and Cluster Utilization Analysis
        durations = [float(m.get("train_duration_sec", 0.0)) for m in client_telemetries]
        round_duration = max(durations) if durations else 0.0
        straggler_waste = sum(round_duration - d for d in durations)
        utilization = (sum(durations) / (num_clients * round_duration) * 100.0) if round_duration > 0 else 0.0

        # 6. Global Evaluation on RTX 5070 Ti
        loss, acc, prec, rec, f1 = self.evaluator.evaluate(self.base_model)
        
        print("\n" + "=" * 75)
        print(f" [ROUND {server_round:02d} PHYSICAL CLUSTER AGGREGATION & EVALUATION]")
        print(f"  Holdout Accuracy:  {acc * 100:5.2f}% | Loss: {loss:.4f} | F1: {f1:.4f}")
        print(f"  Round Duration:    {round_duration:5.2f}s | Straggler Waste: {straggler_waste:5.2f}s")
        print(f"  Cluster Util:      {utilization:5.1f}% | Median Norm: {median_norm:.4f} | Cap c: {clipping_cap:.4f}")
        print(f"  Gating Events:     {sum(int(m.get('drift_detected', 0)) for m in client_telemetries)} | Clipped: {num_clipped}/{num_clients}")
        for m in client_telemetries:
            print(f"    - {m.get('node_id')}: Steps={m.get('steps_executed')}/{m.get('step_budget')}, "
                  f"Time={m.get('train_duration_sec')}s, Rate={m.get('throughput')} st/s, "
                  f"CPU={m.get('cpu_percent')}%, Drift={m.get('final_drift')}")
        print("=" * 75 + "\n")

        round_record = {
            "round": server_round,
            "accuracy": round(acc * 100, 2),
            "loss": round(loss, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "round_duration_sec": round(round_duration, 3),
            "straggler_waste_sec": round(straggler_waste, 3),
            "cluster_utilization_pct": round(utilization, 2),
            "median_norm": round(median_norm, 4),
            "clipping_cap": round(clipping_cap, 4),
            "num_clipped": num_clipped,
            "clients": client_telemetries
        }
        self.history.append(round_record)

        if acc >= self.best_accuracy:
            self.best_accuracy = acc
            os.makedirs("results/cluster_runs", exist_ok=True)
            torch.save(self.base_model.state_dict(), "results/cluster_runs/best_cluster_model.pth")

        return ndarrays_to_parameters(new_weights), {
            "accuracy": acc,
            "loss": loss,
            "utilization": utilization
        }

# Clipping cap: c_t = kappa_c * median_norm(G_i)
