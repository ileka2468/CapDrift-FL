"""
CapDrift-FL Drift Threshold & Minimum Workload Sensitivity Sweep.
Evaluates the impact of:
1. Directional drift threshold: tau_d in {0.2, 0.5, 0.8, 1.0}
2. Mandatory minimum workload: S_min in {5, 10, 15}
Across multiple independent random seeds (42, 101, 2024) on MobileNetV3-Small (Pneumonia Chest X-Ray).

Outputs:
- results/research/drift_threshold_sweep_results.json
- results/research/drift_threshold_sweep_summary.csv
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


class PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, images_tensor, labels_tensor):
        self.images = images_tensor
        self.labels = labels_tensor
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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


def run_threshold_trial(
    tau_drift: float,
    s_min: int,
    train_dataset: ImageFolder,
    val_loader: DataLoader,
    client_indices: List[List[int]],
    client_tiers: List[ClientTierConfig],
    num_rounds: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_clients = len(client_indices)
    client_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=16, shuffle=True)
        for idxs in client_indices
    ]
    sample_counts = {f"client_{i}": len(idxs) for i, idxs in enumerate(client_indices)}

    global_model = get_model("mobilenet_v3_small", num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    model_size_mb = get_model_size_mb(global_model)

    cfg = CapDriftConfig(
        enable_capability=True,
        enable_drift_gating=True,
        enable_clipping=True,
        T_target_sec=15.0,
        tau_drift=tau_drift,
        kappa_c=1.5,
        S_min=s_min,
        S_max=50,
        K_block=5
    )

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
    total_steps_executed_fleet = 0

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

            total_steps_executed_fleet += res["steps_completed"]
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
        total_clipping_events += len(coord_telem.get("clipping_events", []))

        val_acc, val_loss = evaluate_accuracy(global_model, val_loader, criterion, device)
        utilization = (sum(client_durations) / (num_clients * round_duration)) * 100.0 if round_duration > 0 else 0.0

        round_telemetry.append({
            "round": r,
            "val_accuracy": round(val_acc, 2),
            "val_loss": round(val_loss, 4),
            "round_duration_sec": round(round_duration, 2),
            "cumulative_time_sec": round(cumulative_time, 2),
            "round_idle_sec": round(round_idle, 2),
            "cumulative_idle_sec": round(cumulative_idle, 2),
            "cluster_utilization_pct": round(utilization, 2),
            "steps_per_client": {res["client_id"]: res["steps_completed"] for res in client_results},
            "gated_per_client": {res["client_id"]: res["gated_early"] for res in client_results}
        })

    accuracies = [h["val_accuracy"] for h in round_telemetry]
    return {
        "tau_drift": tau_drift,
        "s_min": s_min,
        "seed": seed,
        "peak_accuracy": max(accuracies),
        "final_accuracy": accuracies[-1],
        "final_loss": round_telemetry[-1]["val_loss"],
        "total_time_sec": round(cumulative_time, 2),
        "total_idle_sec": round(cumulative_idle, 2),
        "mean_utilization_pct": round(float(np.mean([h["cluster_utilization_pct"] for h in round_telemetry])), 2),
        "total_gating_events": total_gating_events,
        "total_clipping_events": total_clipping_events,
        "total_steps_executed_fleet": total_steps_executed_fleet,
        "history": round_telemetry
    }


def main():
    print("=" * 80)
    print("  CAPDRIFT-FL DRIFT THRESHOLD & S_MIN SENSITIVITY SWEEP")
    print("=" * 80)

    data_dir = PROJECT_ROOT / "data" / "global_test" / "train"
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Preloading dataset into high-speed memory...")
    raw_dataset = ImageFolder(str(data_dir), transform=tf)
    all_imgs = []
    all_lbls = []
    for img, lbl in raw_dataset:
        all_imgs.append(img)
        all_lbls.append(lbl)
    all_imgs = torch.stack(all_imgs)
    all_lbls = torch.tensor(all_lbls, dtype=torch.long)
    full_dataset = PreloadedDataset(all_imgs, all_lbls)
    print(f"Preloaded {len(full_dataset)} images into RAM.")

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
    alpha = 0.5
    client_sub_indices = partition_dataset_dirichlet(train_subset, num_clients=num_clients, alpha=alpha, seed=42)
    client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]

    client_tiers = [
        TIER_PROFILES[0],
        TIER_PROFILES[1],
        TIER_PROFILES[2],
        TIER_PROFILES[4]
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Experiments grid
    # 1. Sweep tau_d in {0.2, 0.5, 0.8, 1.0} with S_min = 5
    # 2. Sweep S_min in {10, 15} with tau_d = 0.8
    param_configs = [
        (0.20, 5),
        (0.50, 5),
        (0.80, 5),
        (1.00, 5),
        (0.50, 10),
        (0.80, 10),
        (0.80, 15),
    ]

    seeds = [42, 101, 2024]
    num_rounds = 20

    all_results = []
    summary_rows = []

    for tau_d, s_min in param_configs:
        config_name = f"tau={tau_d:.2f}_smin={s_min}"
        print(f"\n---> Evaluating Config: tau_d = {tau_d:.2f}, S_min = {s_min} across {len(seeds)} seeds...")
        seed_trials = []

        for seed in seeds:
            print(f"   [Seed {seed}] Running {num_rounds} rounds...")
            t0 = time.time()
            trial = run_threshold_trial(
                tau_drift=tau_d,
                s_min=s_min,
                train_dataset=full_dataset,
                val_loader=val_loader,
                client_indices=client_indices,
                client_tiers=client_tiers,
                num_rounds=num_rounds,
                device=device,
                seed=seed
            )
            elapsed = time.time() - t0
            print(f"   [Seed {seed}] Done in {elapsed:.1f}s | Final Acc: {trial['final_accuracy']:.2f}% | Gated: {trial['total_gating_events']} | Steps: {trial['total_steps_executed_fleet']}")
            seed_trials.append(trial)

        final_accs = [t["final_accuracy"] for t in seed_trials]
        peak_accs = [t["peak_accuracy"] for t in seed_trials]
        times = [t["total_time_sec"] for t in seed_trials]
        gatings = [t["total_gating_events"] for t in seed_trials]
        steps = [t["total_steps_executed_fleet"] for t in seed_trials]

        config_summary = {
            "config_name": config_name,
            "tau_drift": tau_d,
            "s_min": s_min,
            "mean_final_acc": round(float(np.mean(final_accs)), 2),
            "std_final_acc": round(float(np.std(final_accs)), 2),
            "mean_peak_acc": round(float(np.mean(peak_accs)), 2),
            "std_peak_acc": round(float(np.std(peak_accs)), 2),
            "mean_total_time_sec": round(float(np.mean(times)), 2),
            "std_total_time_sec": round(float(np.std(times)), 2),
            "mean_gating_events": round(float(np.mean(gatings)), 1),
            "mean_steps_executed": round(float(np.mean(steps)), 1),
            "trials": seed_trials
        }
        all_results.append(config_summary)

        summary_rows.append([
            tau_d,
            s_min,
            f"{np.mean(final_accs):.2f} ± {np.std(final_accs):.2f}",
            f"{np.mean(peak_accs):.2f} ± {np.std(peak_accs):.2f}",
            f"{np.mean(times):.2f} ± {np.std(times):.2f}",
            f"{np.mean(gatings):.1f}",
            f"{np.mean(steps):.1f}"
        ])

    # Save JSON
    json_path = OUTPUT_DIR / "drift_threshold_sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved Threshold Results] -> {json_path}")

    # Save CSV
    csv_path = OUTPUT_DIR / "drift_threshold_sweep_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau_drift", "S_min", "Final Acc (%) [Mean ± Std]", "Peak Acc (%) [Mean ± Std]", "Total Time (s) [Mean ± Std]", "Gating Events (Mean)", "Total Steps Executed (Mean)"])
        for row in summary_rows:
            writer.writerow(row)
    print(f"[Saved Threshold CSV] -> {csv_path}")


if __name__ == "__main__":
    main()
