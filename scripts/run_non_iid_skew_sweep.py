"""
Non-IID Skew Sensitivity Benchmark for CapDrift-FL:
Evaluates FedAvg vs. CapDrift-FL across varying Dirichlet label skew:
- alpha = 0.1 (Extreme non-IID, near-orthogonal client distributions)
- alpha = 0.3 (Severe non-IID)
- alpha = 0.5 (Moderate non-IID)

Demonstrates how block-wise directional drift gating and contribution bounding
prevent catastrophic objective divergence in severe non-IID regimes.
Outputs:
- results/research/non_iid_skew_sweep_results.json
- results/research/non_iid_skew_sweep_summary.csv
"""

import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "research"))

from capdrift_controller import (
    CapDriftConfig,
    CapDriftClientTrainer,
    CapDriftCoordinator,
    flatten_params
)
from model import get_model, get_model_size_mb
from research.non_iid import partition_dataset_dirichlet
from research.heterogeneous_fleet import TIER_PROFILES, ClientTierConfig

OUTPUT_DIR = PROJECT_ROOT / "results" / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = (correct / total * 100.0) if total > 0 else 0.0
    avg_loss = (total_loss / total) if total > 0 else 0.0
    return acc, avg_loss


def run_trial(
    method_name: str,
    alpha: float,
    train_dataset: ImageFolder,
    val_loader: DataLoader,
    client_indices: List[List[int]],
    client_tiers: List[ClientTierConfig],
    num_rounds: int = 6,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_clients = len(client_indices)
    client_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=16, shuffle=True)
        for idxs in client_indices
    ]
    sample_counts = {f"client_{i}": len(idxs) for i, idxs in enumerate(client_indices)}

    global_model = get_model("mobilenet_v3_small", num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    model_size_mb = get_model_size_mb(global_model)

    if method_name == "FedAvg":
        cfg = CapDriftConfig(enable_capability=False, enable_drift_gating=False, enable_clipping=False, S_min=5, S_max=5)
    elif method_name == "CapDrift-FL":
        cfg = CapDriftConfig(enable_capability=True, enable_drift_gating=True, enable_clipping=True, T_target_sec=15.0, tau_drift=0.50, kappa_c=1.5, S_min=5, S_max=50)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    coordinator = CapDriftCoordinator(cfg)
    clients = {f"client_{i}": CapDriftClientTrainer(f"client_{i}", cfg) for i in range(num_clients)}

    for i, cid in enumerate(clients):
        tier = client_tiers[i]
        tier_steps_per_sec = tier.base_samples_per_sec / 16.0
        clients[cid].q_hat = tier_steps_per_sec
        coordinator.client_q_hat[cid] = tier_steps_per_sec

    round_telemetry = []
    cumulative_time = 0.0
    cumulative_idle = 0.0
    total_gating_events = 0
    total_clipping_events = 0

    print(f"\n---> Sweep Trial: {method_name} | alpha = {alpha} (Rounds={num_rounds})")

    for r in range(1, num_rounds + 1):
        global_weights = [p.data.clone() for p in global_model.parameters()]
        client_results = []
        client_durations = []

        for i in range(num_clients):
            cid = f"client_{i}"
            client = clients[cid]
            loader = client_loaders[i]
            tier = client_tiers[i]

            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            step_duration_sec = 16.0 / max(1.0, tier.base_samples_per_sec)

            res = client.train_round(
                local_model=local_model,
                global_model_weights=global_weights,
                delta_G_prev=coordinator.delta_G_prev,
                train_loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                simulated_step_duration_sec=step_duration_sec
            )

            compute_time = res["tau_tr"]
            network_time = (tier.network_latency_ms * 2.0 / 1000.0) + (model_size_mb * 8.0 / tier.network_bandwidth_mbps)
            tot_dur = compute_time + network_time
            res["total_client_duration"] = tot_dur
            client_durations.append(tot_dur)
            client_results.append(res)

            if res["gated_early"]:
                total_gating_events += 1

        round_duration = max(client_durations)
        round_idle = sum(round_duration - d for d in client_durations)
        cumulative_time += round_duration
        cumulative_idle += round_idle

        coord_telem = coordinator.aggregate_round(
            server_round=r,
            global_model=global_model,
            client_results=client_results,
            sample_counts=sample_counts
        )

        num_clipped = sum(1 for e in coord_telem.get("clipping_events", []) if e.get("is_clipped"))
        total_clipping_events += num_clipped

        val_acc, val_loss = evaluate_accuracy(global_model, val_loader, criterion, device)
        utilization = (sum(client_durations) / (num_clients * round_duration)) * 100.0 if round_duration > 0 else 0.0
        steps_per_client = {res["client_id"]: res["steps_completed"] for res in client_results}
        gated_per_client = {res["client_id"]: res["gated_early"] for res in client_results}

        round_info = {
            "round": r,
            "val_accuracy": round(val_acc, 2),
            "val_loss": round(val_loss, 4),
            "round_duration_sec": round(round_duration, 2),
            "round_idle_sec": round(round_idle, 2),
            "cluster_utilization_pct": round(utilization, 2),
            "steps_per_client": steps_per_client,
            "gated_per_client": gated_per_client,
            "num_clipped": num_clipped
        }
        round_telemetry.append(round_info)

        print(
            f"   R{r:02d} | Acc: {val_acc:5.2f}% | Loss: {val_loss:.4f} | "
            f"Dur: {round_duration:5.2f}s | Util: {utilization:4.1f}% | "
            f"Steps: {list(steps_per_client.values())} | Gated: {sum(gated_per_client.values())} | Clipped: {num_clipped}"
        )

    accuracies = [h["val_accuracy"] for h in round_telemetry]
    return {
        "method": method_name,
        "alpha": alpha,
        "peak_accuracy": max(accuracies),
        "final_accuracy": accuracies[-1],
        "final_loss": round_telemetry[-1]["val_loss"],
        "total_time_sec": round(cumulative_time, 2),
        "total_idle_sec": round(cumulative_idle, 2),
        "mean_utilization_pct": round(float(np.mean([h["cluster_utilization_pct"] for h in round_telemetry])), 2),
        "total_gating_events": total_gating_events,
        "total_clipping_events": total_clipping_events,
        "history": round_telemetry
    }


def main():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = ImageFolder(str(PROJECT_ROOT / "data" / "global_test" / "train"), transform=tf)

    n_total = len(full_dataset)
    indices = list(range(n_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    n_train = int(0.80 * n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    num_clients = 4
    client_tiers = [
        TIER_PROFILES[0], # GPU Workstation
        TIER_PROFILES[1], # Cisco UCS pve01
        TIER_PROFILES[2], # Cisco UCS pve02
        TIER_PROFILES[4]  # Constrained Edge
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alphas = [0.1, 0.3, 0.5]
    all_results = []

    for alpha in alphas:
        client_sub_indices = partition_dataset_dirichlet(train_subset, num_clients=num_clients, alpha=alpha, seed=42)
        client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]

        for method in ["FedAvg", "CapDrift-FL"]:
            res = run_trial(
                method_name=method,
                alpha=alpha,
                train_dataset=full_dataset,
                val_loader=val_loader,
                client_indices=client_indices,
                client_tiers=client_tiers,
                num_rounds=6,
                device=device,
                seed=42
            )
            all_results.append(res)

    json_path = OUTPUT_DIR / "non_iid_skew_sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved Non-IID Sweep Results] -> {json_path}")

    csv_path = OUTPUT_DIR / "non_iid_skew_sweep_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Alpha (Skew)", "Peak Acc (%)", "Final Acc (%)", "Final Loss", "Gated Events", "Clipped Events", "Mean Util (%)"])
        for r in all_results:
            writer.writerow([
                r["method"],
                r["alpha"],
                r["peak_accuracy"],
                r["final_accuracy"],
                r["final_loss"],
                r["total_gating_events"],
                r["total_clipping_events"],
                r["mean_utilization_pct"]
            ])
    print(f"[Saved Non-IID Sweep CSV] -> {csv_path}")


if __name__ == "__main__":
    main()
