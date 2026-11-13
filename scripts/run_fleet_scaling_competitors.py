"""
Multi-Scale Fleet Scalability Benchmark across Competitors:
Compares:
1. FedAvg (McMahan et al., AISTATS 2017)
2. FedProx (Li et al., MLSys 2020)
3. FedNova (Wang et al., NeurIPS 2020)
4. CapDrift-FL (Ours)

Evaluated across fleet scales N in {4, 10, 20} clients on MobileNetV3-Small (Pneumonia Chest X-Ray)
under Dirichlet alpha = 0.5 over 15 communication rounds across multiple seeds [42, 101, 2024].

Measures:
- Fleet Idle Waste (seconds) vs. N
- Cluster Execution Utilization (%) vs. N
- Final Validation Accuracy (%) vs. N
- Final Loss (Divergence Metric) vs. N

Outputs:
- results/research/fleet_scaling_competitors_results.json
- results/research/fleet_scaling_competitors_summary.csv
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


def build_fleet_tiers(num_clients: int) -> List[ClientTierConfig]:
    if num_clients == 4:
        return [
            TIER_PROFILES[0], # Fast GPU Workstation
            TIER_PROFILES[1], # Cisco UCS Fast Fog (Xeon E5-2680 v4)
            TIER_PROFILES[2], # Cisco UCS Normal Fog (Xeon E5-2620 v4)
            TIER_PROFILES[4]  # Constrained Edge Straggler
        ]
    elif num_clients == 10:
        return [
            TIER_PROFILES[0], TIER_PROFILES[0], # 2x GPU
            TIER_PROFILES[1], TIER_PROFILES[1], # 2x Fast Fog
            TIER_PROFILES[2], TIER_PROFILES[2], TIER_PROFILES[2], # 3x Normal Fog
            TIER_PROFILES[3], TIER_PROFILES[3], # 2x Slow Edge
            TIER_PROFILES[4]  # 1x Constrained Straggler
        ]
    elif num_clients == 20:
        return [
            TIER_PROFILES[0], TIER_PROFILES[0], # 2x GPU
            TIER_PROFILES[1], TIER_PROFILES[1], TIER_PROFILES[1], TIER_PROFILES[1], # 4x Fast Fog
            TIER_PROFILES[2], TIER_PROFILES[2], TIER_PROFILES[2], TIER_PROFILES[2],
            TIER_PROFILES[2], TIER_PROFILES[2], TIER_PROFILES[2], # 7x Normal Fog
            TIER_PROFILES[3], TIER_PROFILES[3], TIER_PROFILES[3], TIER_PROFILES[3], # 4x Slow Edge
            TIER_PROFILES[4], TIER_PROFILES[4], # 2x Constrained Edge
            TIER_PROFILES[5]  # 1x Network Straggler
        ]
    else:
        # Generic cyclic fallback
        tier_keys = [0, 1, 2, 3, 4]
        return [TIER_PROFILES[tier_keys[i % len(tier_keys)]] for i in range(num_clients)]


def run_scaling_trial(
    method_name: str,
    train_dataset: ImageFolder,
    val_loader: DataLoader,
    client_indices: List[List[int]],
    client_tiers: List[ClientTierConfig],
    num_rounds: int = 15,
    mu_prox: float = 0.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_clients = len(client_indices)
    batch_size = 16 if num_clients <= 10 else 8
    client_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=batch_size, shuffle=True)
        for idxs in client_indices
    ]
    sample_counts = {f"client_{i}": len(idxs) for i, idxs in enumerate(client_indices)}
    total_samples = sum(sample_counts.values())

    global_model = get_model("mobilenet_v3_small", num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    model_size_mb = get_model_size_mb(global_model)

    if method_name == "FedAvg":
        cfg = CapDriftConfig(enable_capability=False, enable_drift_gating=False, enable_clipping=False, S_min=5, S_max=5)
    elif method_name == "FedProx":
        cfg = CapDriftConfig(enable_capability=False, enable_drift_gating=False, enable_clipping=False, S_min=5, S_max=5)
    elif method_name == "FedNova":
        cfg = CapDriftConfig(enable_capability=True, enable_drift_gating=False, enable_clipping=False, T_target_sec=6.5, S_min=5, S_max=40)
    elif method_name == "CapDrift-FL":
        cfg = CapDriftConfig(enable_capability=True, enable_drift_gating=True, enable_clipping=True, dynamic_clipping=True, T_target_sec=6.5, tau_drift=0.85, kappa_c=1.5, S_min=5, S_max=40)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    coordinator = CapDriftCoordinator(cfg)
    clients = {f"client_{i}": CapDriftClientTrainer(f"client_{i}", cfg) for i in range(num_clients)}

    for i, cid in enumerate(clients):
        tier = client_tiers[i]
        tier_steps_per_sec = tier.base_samples_per_sec / float(batch_size)
        clients[cid].q_hat = tier_steps_per_sec
        coordinator.client_q_hat[cid] = tier_steps_per_sec

    round_telemetry = []
    cumulative_time = 0.0
    cumulative_idle = 0.0
    total_gating_events = 0
    total_clipping_events = 0

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
            step_duration_sec = float(batch_size) / max(1.0, tier.base_samples_per_sec)

            if method_name == "FedProx":
                s_cap, s_req = client.determine_step_budget()
                t_start = time.perf_counter()
                data_iter = iter(loader)
                local_model.train()
                for _ in range(s_cap):
                    try:
                        images, labels = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        images, labels = next(data_iter)
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    if mu_prox > 0:
                        prox = sum(torch.sum((p_loc - p_glob.to(device)) ** 2) for p_loc, p_glob in zip(local_model.parameters(), global_weights))
                        loss = loss + 0.5 * mu_prox * prox
                    loss.backward()
                    optimizer.step()

                t_comp = time.perf_counter() - t_start
                tau_tr = step_duration_sec * s_cap
                delta_i = [p.data.clone() - w_g for p, w_g in zip(local_model.parameters(), global_weights)]
                res = {
                    "client_id": cid,
                    "delta_i": delta_i,
                    "q_i": s_cap / max(1e-4, tau_tr),
                    "steps_completed": s_cap,
                    "drift_evaluations": [],
                    "gated_early": False,
                    "tau_tr": tau_tr
                }
            else:
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

        # Aggregation
        if method_name == "FedNova":
            tau_eff = sum((sample_counts[res["client_id"]] / total_samples) * res["steps_completed"] for res in client_results)
            with torch.no_grad():
                for param_idx, p in enumerate(global_model.parameters()):
                    agg_delta = sum(
                        (sample_counts[res["client_id"]] / total_samples) *
                        (res["delta_i"][param_idx] / max(1, res["steps_completed"])) *
                        tau_eff
                        for res in client_results
                    )
                    p.data.add_(agg_delta)
            coord_telem = {"clipping_events": [], "clipping_cap_c": 0.0}
        else:
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

        round_info = {
            "round": r,
            "val_accuracy": round(val_acc, 2),
            "val_loss": round(val_loss, 4),
            "round_duration_sec": round(round_duration, 2),
            "cumulative_time_sec": round(cumulative_time, 2),
            "round_idle_sec": round(round_idle, 2),
            "cluster_utilization_pct": round(utilization, 2),
            "num_clipped": num_clipped
        }
        round_telemetry.append(round_info)

    accuracies = [h["val_accuracy"] for h in round_telemetry]
    return {
        "method": method_name,
        "num_clients": num_clients,
        "seed": seed,
        "num_rounds": num_rounds,
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
    print("[Scaling Benchmark] Preloading dataset into high-speed memory...")
    raw_dataset = ImageFolder(str(PROJECT_ROOT / "data" / "global_test" / "train"), transform=tf)
    all_imgs = []
    all_lbls = []
    for img, lbl in raw_dataset:
        all_imgs.append(img)
        all_lbls.append(lbl)
    all_imgs = torch.stack(all_imgs)
    all_lbls = torch.tensor(all_lbls, dtype=torch.long)
    full_dataset = PreloadedDataset(all_imgs, all_lbls)
    print(f"[Scaling Benchmark] Preloaded {len(full_dataset)} images into RAM.")

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

    scales = [4, 10, 20]
    methods = [
        ("FedAvg", 0.0),
        ("FedProx", 0.01),
        ("FedNova", 0.0),
        ("CapDrift-FL", 0.0)
    ]
    seeds = [42, 101, 2024]
    num_rounds = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Scaling Benchmark] Device: {device} | Scales: {scales} | Seeds: {seeds}")

    results = {}
    summary_rows = []

    total_experiments = len(scales) * len(methods) * len(seeds)
    exp_counter = 0

    t_start_all = time.perf_counter()

    for N in scales:
        client_tiers = build_fleet_tiers(N)
        results[f"N_{N}"] = {}

        for method_name, mu in methods:
            results[f"N_{N}"][method_name] = {"trials": []}
            trials = []

            for s in seeds:
                exp_counter += 1
                # Partition data for this N and seed
                client_sub_indices = partition_dataset_dirichlet(train_subset, num_clients=N, alpha=0.5, seed=s)
                client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]

                print(f"\n>>> [{exp_counter}/{total_experiments}] Scale N={N} | Method={method_name} | Seed={s}")
                t0 = time.perf_counter()
                trial_res = run_scaling_trial(
                    method_name=method_name,
                    train_dataset=full_dataset,
                    val_loader=val_loader,
                    client_indices=client_indices,
                    client_tiers=client_tiers,
                    num_rounds=num_rounds,
                    mu_prox=mu,
                    device=device,
                    seed=s
                )
                t_elapsed = time.perf_counter() - t0
                print(f"    --> Done in {t_elapsed:.1f}s | Final Acc: {trial_res['final_accuracy']:.2f}% | Loss: {trial_res['final_loss']:.4f} | Idle: {trial_res['total_idle_sec']:.1f}s | Util: {trial_res['mean_utilization_pct']:.1f}%")
                trials.append(trial_res)

            # Aggregate over seeds
            final_accs = [t["final_accuracy"] for t in trials]
            peak_accs = [t["peak_accuracy"] for t in trials]
            losses = [t["final_loss"] for t in trials]
            times = [t["total_time_sec"] for t in trials]
            idles = [t["total_idle_sec"] for t in trials]
            utils = [t["mean_utilization_pct"] for t in trials]
            gatings = [t["total_gating_events"] for t in trials]
            clippings = [t["total_clipping_events"] for t in trials]

            agg_entry = {
                "num_clients": N,
                "method": method_name,
                "num_seeds": len(seeds),
                "mean_final_acc": round(float(np.mean(final_accs)), 2),
                "std_final_acc": round(float(np.std(final_accs)), 2),
                "mean_peak_acc": round(float(np.mean(peak_accs)), 2),
                "std_peak_acc": round(float(np.std(peak_accs)), 2),
                "mean_final_loss": round(float(np.mean(losses)), 4),
                "std_final_loss": round(float(np.std(losses)), 4),
                "mean_time_sec": round(float(np.mean(times)), 2),
                "std_time_sec": round(float(np.std(times)), 2),
                "mean_idle_sec": round(float(np.mean(idles)), 2),
                "std_idle_sec": round(float(np.std(idles)), 2),
                "mean_utilization_pct": round(float(np.mean(utils)), 2),
                "std_utilization_pct": round(float(np.std(utils)), 2),
                "mean_gating_events": round(float(np.mean(gatings)), 1),
                "mean_clipping_events": round(float(np.mean(clippings)), 1),
                "trials": trials
            }
            results[f"N_{N}"][method_name] = agg_entry

            summary_rows.append({
                "Scale (N)": N,
                "Method": method_name,
                "Final Acc (%)": f"{agg_entry['mean_final_acc']:.2f} ± {agg_entry['std_final_acc']:.2f}",
                "Peak Acc (%)": f"{agg_entry['mean_peak_acc']:.2f} ± {agg_entry['std_peak_acc']:.2f}",
                "Final Loss": f"{agg_entry['mean_final_loss']:.4f} ± {agg_entry['std_final_loss']:.4f}",
                "Total Time (s)": f"{agg_entry['mean_time_sec']:.1f}",
                "Idle Waste (s)": f"{agg_entry['mean_idle_sec']:.1f} ± {agg_entry['std_idle_sec']:.1f}",
                "Cluster Util (%)": f"{agg_entry['mean_utilization_pct']:.1f}%",
                "Mean Gated": agg_entry["mean_gating_events"],
                "Mean Clipped": agg_entry["mean_clipping_events"]
            })

    total_time = time.perf_counter() - t_start_all
    print(f"\n=======================================================")
    print(f"  All Scaling Experiments Completed in {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print(f"=======================================================")

    # Save JSON
    out_json = OUTPUT_DIR / "fleet_scaling_competitors_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] -> {out_json}")

    # Save CSV
    out_csv = OUTPUT_DIR / "fleet_scaling_competitors_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Saved] -> {out_csv}")


if __name__ == "__main__":
    main()
