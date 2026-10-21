"""
Heterogeneous Edge-Fog Scaling Harness & Benchmark Engine.
Executes horizontal scaling studies (N = 3, 5, 10, 20 clients) across heterogeneous compute tiers
with accurate physical hardware profiling and network latency emulation.
"""

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.model import get_model, get_model_size_mb
from src.research.non_iid import partition_dataset_dirichlet

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ComputeTier:
    name: str
    hardware_desc: str
    throughput_samples_per_sec: float
    network_latency_ms: float
    device_type: str  # "cuda" | "cpu"


TIER_PROFILES = {
    "GPU_WORKSTATION": ComputeTier("Tier 1", "NVIDIA RTX 5070 Ti Workstation", 1400.0, 0.5, "cuda"),
    "FOG_XEON_HIGH": ComputeTier("Tier 2", "Cisco UCS Dual-Xeon E5-2680 (pve01)", 160.0, 1.2, "cpu"),
    "FOG_XEON_MID": ComputeTier("Tier 3", "Cisco UCS Dual-Xeon E5-2650 (pve02/03)", 75.0, 1.8, "cpu"),
    "EDGE_STRAGGLER": ComputeTier("Tier 4", "Resource-Constrained Edge / Jitter Node", 22.0, 15.0, "cpu")
}


class HeterogeneousClusterHarness:
    """
    Simulates and executes multi-tier heterogeneous federated learning with precise
    physical latency models and real GPU/CPU compute execution.
    """

    def __init__(
        self,
        num_clients: int = 4,
        data_dir: Union[str, Path] = "data/global_test/train",
        val_dir: Union[str, Path] = "data/global_test/test",
        arch: str = "mobilenet_v3_small",
        num_classes: int = 2,
        alpha_dirichlet: float = 0.5,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        self.num_clients = num_clients
        self.data_dir = Path(data_dir)
        self.val_dir = Path(val_dir)
        if not self.val_dir.exists():
            self.val_dir = self.data_dir  # fallback

        self.arch = arch
        self.num_classes = num_classes
        self.alpha_dirichlet = alpha_dirichlet
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self._assign_compute_tiers()
        self._setup_datasets()

    def _assign_compute_tiers(self):
        """Assigns realistic heterogeneous compute tiers across the N clients."""
        self.client_tiers: List[ComputeTier] = []
        for i in range(self.num_clients):
            if i == 0 and torch.cuda.is_available():
                self.client_tiers.append(TIER_PROFILES["GPU_WORKSTATION"])
            elif i % 4 == 1:
                self.client_tiers.append(TIER_PROFILES["FOG_XEON_HIGH"])
            elif i % 4 == 2:
                self.client_tiers.append(TIER_PROFILES["FOG_XEON_MID"])
            else:
                self.client_tiers.append(TIER_PROFILES["EDGE_STRAGGLER"])

    def _setup_datasets(self):
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.full_train_ds = ImageFolder(self.data_dir, transform=train_tf)
        self.val_ds = ImageFolder(self.val_dir, transform=eval_tf)
        self.val_loader = DataLoader(self.val_ds, batch_size=32, shuffle=False)

        # Partition indices using Dirichlet label skew
        client_indices = partition_dataset_dirichlet(
            self.full_train_ds,
            num_clients=self.num_clients,
            alpha=self.alpha_dirichlet,
            seed=self.seed
        )
        self.client_loaders = [
            DataLoader(Subset(self.full_train_ds, idxs), batch_size=32, shuffle=True)
            for idxs in client_indices
        ]

    def run_simulation(
        self,
        strategy_mode: str = "CARC",  # "CARC" | "FEDAVG" | "FEDPROX" | "CARC_FEDPROX" | "CARC_V2" | "CARC_V3" | "CARC_NOVA" | "CARC_SCAFFOLD" | "CENTRALIZED"
        rounds: int = 5,
        target_round_time_sec: float = 6.0,
        mu_proximal: float = 0.01,
        target_accuracy: float = 90.0,
        beta_step_norm: float = 0.5,
        lambda_drift: float = 1.0,
        mu_min: float = 0.001,
        mu_max: float = 0.05,
        gamma_mu: float = 0.015,
        lr_local: float = 0.01,
        server_lr: float = 1.0,
        server_momentum: float = 0.0,
        lr_decay: float = 1.0,
        aggregate_bn: bool = True
    ) -> Dict:
        """
        Executes an end-to-end benchmark trial under the specified strategy.
        Tracks time-to-accuracy, round times, straggler idle penalties, and throughput.
        Supports CARC_NOVA (FedNova-style first-order step normalization) and CARC_SCAFFOLD (control variates).
        """
        start_wall_clock = time.time()
        global_model = get_model(self.arch, num_classes=self.num_classes, pretrained=True).to(self.device)
        model_size_mb = get_model_size_mb(global_model)
        criterion = nn.CrossEntropyLoss()

        history = []
        time_to_target_sec: Optional[float] = None
        cumulative_simulated_time = 0.0
        total_idle_time = 0.0
        total_network_mb = 0.0

        print(f"\n==========================================================================")
        print(f"  RUNNING BENCHMARK: {strategy_mode} | Clients: {self.num_clients} | Alpha: {self.alpha_dirichlet}")
        print(f"==========================================================================")

        if strategy_mode == "CENTRALIZED":
            # Centralized Upper Bound: 100% data processed on single GPU
            central_loader = DataLoader(self.full_train_ds, batch_size=32, shuffle=True)
            optimizer = torch.optim.AdamW(global_model.parameters(), lr=1e-3, weight_decay=1e-4)
            epoch_time_budget = 5

            for ep in range(rounds * 2):
                t0 = time.perf_counter()
                global_model.train()
                for x, y in central_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    out = global_model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                ep_duration = time.perf_counter() - t0
                cumulative_simulated_time += ep_duration

                val_acc, val_loss = self._evaluate_model(global_model, criterion)
                if val_acc >= target_accuracy and time_to_target_sec is None:
                    time_to_target_sec = cumulative_simulated_time

                history.append({
                    "step": ep + 1,
                    "simulated_time_sec": round(cumulative_simulated_time, 2),
                    "val_accuracy": round(val_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "round_time_sec": round(ep_duration, 2),
                    "idle_time_sec": 0.0
                })
                print(f"  Epoch {ep+1:02d} | Time: {cumulative_simulated_time:.2f}s | Acc: {val_acc:.2f}% | Loss: {val_loss:.4f}")

        else:
            # Federated Strategies (CARC, FEDAVG, FEDPROX, CARC_V2, CARC_V3, CARC_NOVA, CARC_SCAFFOLD)
            # Controller state for CARC_V2 / CARC_V3
            client_similarities = [1.0] * self.num_clients
            client_divergences = [0.0] * self.num_clients
            client_mu_values = [mu_proximal if mu_proximal > 0 else 0.01] * self.num_clients

            # Controller state for SCAFFOLD
            client_controls = None
            server_control = None
            if strategy_mode == "CARC_SCAFFOLD":
                client_controls = [[torch.zeros_like(p.data) for p in global_model.parameters()] for _ in range(self.num_clients)]
                server_control = [torch.zeros_like(p.data) for p in global_model.parameters()]

            server_velocity = [torch.zeros_like(p.data) for p in global_model.parameters()]

            for r in range(rounds):
                r_start = time.perf_counter()
                client_updates = []
                client_durations = []
                client_sample_weights = []
                client_steps_list = []
                client_bn_buffers = []

                # Distribute global model to clients
                global_weights = [p.data.clone() for p in global_model.parameters()]
                # Network cost: Downlink payload for N clients
                total_network_mb += (model_size_mb * self.num_clients)

                for c_id in range(self.num_clients):
                    loader = self.client_loaders[c_id]
                    tier = self.client_tiers[c_id]

                    local_model = copy.deepcopy(global_model)
                    local_model.train()

                    if strategy_mode in ("CARC_NOVA", "CARC_SCAFFOLD"):
                        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr_local, momentum=0.0)
                    else:
                        optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-3, weight_decay=1e-4)

                    # Workload assignment
                    if len(loader) == 0:
                        steps = 0
                    elif strategy_mode in ("CARC_NOVA", "CARC_SCAFFOLD"):
                        # Capability-proportional local step scaling
                        raw_steps = int((target_round_time_sec * tier.throughput_samples_per_sec) / 32)
                        steps = max(2, min(16, raw_steps))
                    elif strategy_mode in ("CARC_V2", "CARC_V3"):
                        # Heterogeneity-Aware Adaptive Step Allocation:
                        raw_steps = int((target_round_time_sec * tier.throughput_samples_per_sec) / 32)
                        k_cap = max(2, min(16, raw_steps))
                        sim = client_similarities[c_id]
                        div = client_divergences[c_id]
                        alignment = max(0.2, (1.0 + sim) / 2.0)
                        drift_factor = 1.0 / (1.0 + 0.5 * div)
                        steps = int(max(2, min(16, round(k_cap * alignment * drift_factor))))
                    elif strategy_mode in ("CARC", "CARC_FEDPROX"):
                        # Adaptive steps proportional to client capability kappa
                        raw_steps = int((target_round_time_sec * tier.throughput_samples_per_sec) / 32)
                        steps = max(2, min(16, raw_steps))
                    else:
                        # Vanilla FedAvg & FedProx: Uniform 1 epoch
                        steps = len(loader)

                    client_steps_list.append(steps)

                    # Determine active proximal penalty mu_i for this client
                    active_mu = 0.0
                    if strategy_mode == "CARC_V3":
                        # Dynamic proximal anchor: mu_i(t) = clip(mu_min + gamma_mu * D_i, mu_min, mu_max)
                        div = client_divergences[c_id]
                        active_mu = float(max(mu_min, min(mu_max, mu_min + gamma_mu * div)))
                    elif strategy_mode == "CARC_V2":
                        # Adaptive mu_i: higher for clients with step asymmetry or divergence
                        base_mu = client_mu_values[c_id]
                        step_ratio = steps / max(1.0, (target_round_time_sec * 50.0 / 32.0))
                        sim = client_similarities[c_id]
                        drift_penalty = max(1.0, 2.0 - max(0.0, sim))
                        active_mu = base_mu * (1.0 + 0.5 * step_ratio) * drift_penalty
                    elif (strategy_mode in ("FEDPROX", "CARC_FEDPROX") or (strategy_mode == "CARC" and mu_proximal > 0)) and mu_proximal > 0:
                        active_mu = mu_proximal

                    # Execute local training
                    t_comp_start = time.perf_counter()
                    step_count = 0
                    samples_processed = 0

                    if steps > 0 and len(loader) > 0:
                        done = False
                        while not done:
                            for x, y in loader:
                                if step_count >= steps:
                                    done = True
                                    break
                                x, y = x.to(self.device), y.to(self.device)
                                optimizer.zero_grad()
                                out = local_model(x)
                                loss = criterion(out, y)

                                # Proximal regularization
                                if active_mu > 0:
                                    prox_penalty = 0.0
                                    for w, w_t in zip(local_model.parameters(), global_weights):
                                        prox_penalty += ((w - w_t) ** 2).sum()
                                    loss += (active_mu / 2.0) * prox_penalty

                                loss.backward()

                                # Apply SCAFFOLD correction: g_i - c_i + c
                                if strategy_mode == "CARC_SCAFFOLD":
                                    with torch.no_grad():
                                        for p_idx, p in enumerate(local_model.parameters()):
                                            if p.grad is not None:
                                                p.grad.add_(server_control[p_idx] - client_controls[c_id][p_idx])

                                optimizer.step()
                                step_count += 1
                                samples_processed += len(y)

                    # Compute physical simulated duration based on hardware tier
                    ideal_compute_sec = samples_processed / max(1.0, tier.throughput_samples_per_sec)
                    network_delay_sec = (tier.network_latency_ms * 2.0) / 1000.0 + (model_size_mb * 8.0) / 100.0
                    client_duration = ideal_compute_sec + network_delay_sec

                    client_durations.append(client_duration)
                    # Sample weighting (p_i proportional to samples processed)
                    if strategy_mode in ("CARC_V2", "CARC_V3", "CARC_NOVA", "CARC_SCAFFOLD"):
                        weight = max(1, samples_processed)
                    elif strategy_mode in ("CARC", "CARC_FEDPROX"):
                        weight = max(1, samples_processed * steps)
                    else:
                        weight = max(1, samples_processed)
                    client_sample_weights.append(weight)

                    delta_weights = [p.data.clone() - w_t for p, w_t in zip(local_model.parameters(), global_weights)]
                    client_updates.append(delta_weights)

                    # Track BatchNorm buffers for synchronization
                    client_bn_buffers.append([b.data.clone() for b in local_model.buffers()])

                    # Update client control variate for SCAFFOLD:
                    # c_{i, t+1} = c_{i, t} - c_t - Delta_i / (eta_l * K_i)
                    if strategy_mode == "CARC_SCAFFOLD":
                        with torch.no_grad():
                            for p_idx in range(len(client_controls[c_id])):
                                step_grad_approx = delta_weights[p_idx] / (lr_local * max(1, steps))
                                client_controls[c_id][p_idx].add_(-server_control[p_idx] - step_grad_approx)

                # Uplink network cost
                total_network_mb += (model_size_mb * self.num_clients)

                # Synchronous round time is bounded by the slowest client (the straggler)
                round_time = max(client_durations)
                cumulative_simulated_time += round_time

                # Calculate idle penalty: time faster clients spent waiting
                round_idle = sum(round_time - d for d in client_durations)
                total_idle_time += round_idle

                # Update server control variate for SCAFFOLD: c_{t+1} = sum_i p_i c_{i, t+1}
                if strategy_mode == "CARC_SCAFFOLD":
                    total_w = sum(client_sample_weights)
                    with torch.no_grad():
                        for p_idx in range(len(server_control)):
                            server_control[p_idx].zero_()
                            for c_id in range(self.num_clients):
                                server_control[p_idx].add_(client_controls[c_id][p_idx] * (client_sample_weights[c_id] / total_w))

                # Aggregate model parameters
                if strategy_mode in ("CARC_NOVA", "CARC_SCAFFOLD"):
                    # Exact first-order step-normalized aggregation (FedNova-style):
                    # G_t = -sum_i p_i (Delta_i / (eta_l * K_i))
                    # w_{t+1} = w_t + s_t * mean_k * sum_i p_i (Delta_i / K_i)
                    total_w = sum(client_sample_weights)
                    mean_k = sum((client_sample_weights[c_id] / total_w) * client_steps_list[c_id] for c_id in range(self.num_clients))
                    with torch.no_grad():
                        for param_idx, p in enumerate(global_model.parameters()):
                            agg_delta = sum(
                                (client_updates[c_id][param_idx] / max(1, client_steps_list[c_id])) *
                                ((client_sample_weights[c_id] / total_w) * mean_k * server_lr)
                                for c_id in range(self.num_clients)
                            )
                            if server_momentum > 0.0:
                                server_velocity[param_idx].mul_(server_momentum).add_(agg_delta)
                                p.data.add_(server_velocity[param_idx])
                            else:
                                p.data.add_(agg_delta)

                elif strategy_mode == "CARC_V3":
                    # CARC-v3: Double Closed-Loop Controller
                    flat_deltas = []
                    for c_id in range(self.num_clients):
                        flat = torch.cat([d.flatten() for d in client_updates[c_id]])
                        flat_deltas.append(flat)

                    # Consensus direction vector
                    global_flat = torch.zeros_like(flat_deltas[0])
                    for c_id in range(self.num_clients):
                        global_flat += flat_deltas[c_id] * (1.0 / self.num_clients)
                    global_norm = torch.norm(global_flat) + 1e-8

                    for c_id in range(self.num_clients):
                        c_norm = torch.norm(flat_deltas[c_id]) + 1e-8
                        cos_sim = float(torch.dot(flat_deltas[c_id], global_flat) / (c_norm * global_norm))
                        div = float(torch.norm(flat_deltas[c_id] - global_flat) / global_norm)
                        client_similarities[c_id] = cos_sim
                        client_divergences[c_id] = div

                    mean_k = sum(client_steps_list) / max(1.0, len(client_steps_list))

                    trust_weights = []
                    for c_id in range(self.num_clients):
                        n_i = client_sample_weights[c_id]
                        div = client_divergences[c_id]
                        sim = client_similarities[c_id]
                        drift_trust = 1.0 / (1.0 + lambda_drift * div)
                        align_trust = max(0.05, (1.0 + sim) / 2.0)
                        trust_weights.append(n_i * drift_trust * align_trust)

                    total_trust = sum(trust_weights) + 1e-8
                    alpha_norm = [tw / total_trust for tw in trust_weights]

                    with torch.no_grad():
                        for param_idx, p in enumerate(global_model.parameters()):
                            avg_delta = sum(
                                (client_updates[c_id][param_idx] / ((max(1.0, client_steps_list[c_id]) / mean_k) ** beta_step_norm)) *
                                alpha_norm[c_id]
                                for c_id in range(self.num_clients)
                            )
                            p.data.add_(avg_delta)

                elif strategy_mode == "CARC_V2":
                    flat_deltas = []
                    for c_id in range(self.num_clients):
                        flat = torch.cat([d.flatten() for d in client_updates[c_id]])
                        flat_deltas.append(flat)

                    global_flat = torch.zeros_like(flat_deltas[0])
                    for c_id in range(self.num_clients):
                        global_flat += flat_deltas[c_id] * (1.0 / self.num_clients)
                    global_norm = torch.norm(global_flat) + 1e-8

                    for c_id in range(self.num_clients):
                        c_norm = torch.norm(flat_deltas[c_id]) + 1e-8
                        cos_sim = float(torch.dot(flat_deltas[c_id], global_flat) / (c_norm * global_norm))
                        div = float(torch.norm(flat_deltas[c_id] - global_flat) / global_norm)
                        client_similarities[c_id] = cos_sim
                        client_divergences[c_id] = div

                        if cos_sim < 0.2 or div > 1.5:
                            client_mu_values[c_id] = min(0.05, client_mu_values[c_id] * 1.5)
                        elif cos_sim > 0.8 and div < 0.8:
                            client_mu_values[c_id] = max(0.002, client_mu_values[c_id] * 0.85)

                    mean_k = sum(client_steps_list) / max(1.0, len(client_steps_list))
                    align_weights = [max(0.1, (1.0 + client_similarities[c_id]) / 2.0) for c_id in range(self.num_clients)]
                    total_w = sum(client_sample_weights[c_id] * align_weights[c_id] for c_id in range(self.num_clients))

                    with torch.no_grad():
                        for param_idx, p in enumerate(global_model.parameters()):
                            avg_delta = sum(
                                (client_updates[c_id][param_idx] / max(1.0, client_steps_list[c_id] / mean_k)) *
                                ((client_sample_weights[c_id] * align_weights[c_id]) / total_w)
                                for c_id in range(self.num_clients)
                            )
                            p.data.add_(avg_delta)
                else:
                    total_w = sum(client_sample_weights)
                    with torch.no_grad():
                        for param_idx, p in enumerate(global_model.parameters()):
                            avg_delta = sum(
                                client_updates[c_id][param_idx] * (client_sample_weights[c_id] / total_w)
                                for c_id in range(self.num_clients)
                            )
                            p.data.add_(avg_delta)

                # Aggregate BatchNorm running buffers if enabled
                if aggregate_bn or strategy_mode in ("CARC_NOVA", "CARC_SCAFFOLD"):
                    total_w = sum(client_sample_weights)
                    with torch.no_grad():
                        for b_idx, b in enumerate(global_model.buffers()):
                            if b.is_floating_point():
                                b.zero_()
                                for c_id in range(self.num_clients):
                                    b.add_(client_bn_buffers[c_id][b_idx] * (client_sample_weights[c_id] / total_w))

                val_acc, val_loss, rec_0, rec_1 = self._evaluate_model(global_model, criterion)
                if val_acc >= target_accuracy and time_to_target_sec is None:
                    time_to_target_sec = cumulative_simulated_time

                history.append({
                    "step": r + 1,
                    "simulated_time_sec": round(cumulative_simulated_time, 2),
                    "val_accuracy": round(val_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "recall_class_0": round(rec_0, 2),
                    "recall_class_1": round(rec_1, 2),
                    "round_time_sec": round(round_time, 2),
                    "idle_time_sec": round(round_idle, 2)
                })

                idle_pct = (round_idle / (sum(client_durations) + round_idle)) * 100.0
                print(
                    f"  Round {r+1:02d} | Duration: {round_time:.2f}s | "
                    f"Idle Waste: {round_idle:.2f}s ({idle_pct:.1f}%) | "
                    f"Acc: {val_acc:.2f}% | Loss: {val_loss:.4f} | "
                    f"Rec[0]: {rec_0:.1f}% | Rec[1]: {rec_1:.1f}%",
                    flush=True
                )

                if lr_decay < 1.0:
                    lr_local *= lr_decay

        final_acc = history[-1]["val_accuracy"] if history else 0.0
        final_loss = history[-1]["val_loss"] if history else 0.0
        total_samples = len(self.full_train_ds)
        throughput = (total_samples * len(history)) / max(1.0, cumulative_simulated_time)

        accs = [h["val_accuracy"] for h in history] if history else [0.0]
        times = [h["simulated_time_sec"] for h in history] if history else [0.0]
        auc_acc = float(sum(accs) / len(accs)) if accs else 0.0

        # Compute sustained accuracy targets (consecutive rounds >= threshold)
        consecutive_m = 2
        time_sustained_70 = None
        time_sustained_75 = None
        for i in range(len(accs) - consecutive_m + 1):
            if all(accs[i + k] >= 70.0 for k in range(consecutive_m)) and time_sustained_70 is None:
                time_sustained_70 = times[i]
            if all(accs[i + k] >= 75.0 for k in range(consecutive_m)) and time_sustained_75 is None:
                time_sustained_75 = times[i]

        return {
            "strategy": strategy_mode,
            "num_clients": self.num_clients,
            "alpha_dirichlet": self.alpha_dirichlet,
            "final_accuracy": final_acc,
            "peak_accuracy": max(accs) if accs else 0.0,
            "auc_accuracy": round(auc_acc, 2),
            "final_loss": final_loss,
            "time_to_target_sec": round(time_to_target_sec, 2) if time_to_target_sec else None,
            "time_sustained_70_sec": round(time_sustained_70, 2) if time_sustained_70 is not None else None,
            "time_sustained_75_sec": round(time_sustained_75, 2) if time_sustained_75 is not None else None,
            "total_simulated_time_sec": round(cumulative_simulated_time, 2),
            "total_idle_time_sec": round(total_idle_time, 2),
            "aggregate_throughput_samples_per_sec": round(throughput, 2),
            "total_network_mb": round(total_network_mb, 2),
            "history": history
        }

    def _evaluate_model(self, model: nn.Module, criterion: nn.Module) -> Tuple[float, float, float, float]:
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        c0_correct, c0_total = 0, 0
        c1_correct, c1_total = 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * len(y)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

                for p, label in zip(preds, y):
                    if label == 0:
                        c0_total += 1
                        if p == 0: c0_correct += 1
                    else:
                        c1_total += 1
                        if p == 1: c1_correct += 1

        acc = (correct / max(1, total)) * 100.0
        avg_loss = total_loss / max(1, total)
        rec_0 = (c0_correct / max(1, c0_total)) * 100.0 if c0_total > 0 else 0.0
        rec_1 = (c1_correct / max(1, c1_total)) * 100.0 if c1_total > 0 else 0.0
        return acc, avg_loss, rec_0, rec_1
