"""
CapDrift-FL 5-Way Ablation Benchmark Runner.
Executes the matched 5-way ablation specified in Professor Siasi's IEEE template:
1. FedAvg: Uniform local work (S_min = 5), no drift gating, no clipping
2. Capability-Only: Capability step budgeting (S_cap), no drift gating, no clipping
3. Drift-Only: High work budget (S_max = 50), drift gating active on surplus > S_min, no capability adaptation, no clipping
4. Capability+Drift: Capability step budgeting + drift gating, no clipping
5. Full CapDrift-FL: Capability step budgeting + drift gating + coordinator contribution clipping (c = kappa_c * m_G)

Evaluated under Non-IID Dirichlet distribution (alpha = 0.5, 0.1) on MobileNetV3-Small with Pneumonia Chest X-ray.
Outputs:
- results/research/capdrift_ablation_results.json
- results/research/capdrift_ablation_telemetry.csv
- Detailed markdown summary report
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
    CapDriftCoordinator
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


def run_single_ablation_trial(
    strategy_name: str,
    cfg: CapDriftConfig,
    train_dataset: ImageFolder,
    val_loader: DataLoader,
    client_indices: List[List[int]],
    client_tiers: List[ClientTierConfig],
    num_rounds: int = 10,
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
    
    # Initialize global model
    global_model = get_model("mobilenet_v3_small", num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    model_size_mb = get_model_size_mb(global_model)
    
    # Coordinator
    coordinator = CapDriftCoordinator(cfg)
    
    # Clients
    clients = {
        f"client_{i}": CapDriftClientTrainer(f"client_{i}", cfg)
        for i in range(num_clients)
    }
    
    # Pre-seed initial throughput based on hardware tier
    for i, cid in enumerate(clients):
        tier = client_tiers[i]
        # steps/sec = samples_per_sec / batch_size
        batch_size = 16
        tier_steps_per_sec = tier.base_samples_per_sec / batch_size
        clients[cid].q_hat = tier_steps_per_sec
        coordinator.client_q_hat[cid] = tier_steps_per_sec
        
    round_telemetry = []
    cumulative_simulated_time = 0.0
    cumulative_idle_time = 0.0
    total_gating_events = 0
    total_clipping_events = 0
    
    print(f"\n---> Starting Trial: {strategy_name} (Rounds={num_rounds}, Seed={seed})")
    
    for r in range(1, num_rounds + 1):
        global_weights = [p.data.clone() for p in global_model.parameters()]
        client_results = []
        client_durations = []
        
        # Train clients
        for i in range(num_clients):
            cid = f"client_{i}"
            client = clients[cid]
            loader = client_loaders[i]
            tier = client_tiers[i]
            
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            
            # Step duration for hardware emulation: batch_size / samples_per_sec
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
            
            # Calculate physical time: compute time + network latency + transmission time
            compute_time = res["tau_tr"]
            network_time = (tier.network_latency_ms * 2.0 / 1000.0) + (model_size_mb * 8.0 / tier.network_bandwidth_mbps)
            total_client_duration = compute_time + network_time
            
            res["total_client_duration"] = total_client_duration
            client_durations.append(total_client_duration)
            client_results.append(res)
            
            if res["gated_early"]:
                total_gating_events += 1
                
        # Synchronous round time is bounded by the slowest client (straggler)
        round_duration = max(client_durations)
        round_idle = sum(round_duration - d for d in client_durations)
        cumulative_simulated_time += round_duration
        cumulative_idle_time += round_idle
        
        # Server Aggregation
        coord_telem = coordinator.aggregate_round(
            server_round=r,
            global_model=global_model,
            client_results=client_results,
            sample_counts=sample_counts
        )
        
        num_clipped = sum(1 for e in coord_telem.get("clipping_events", []) if e.get("is_clipped"))
        total_clipping_events += num_clipped
        
        # Evaluate global model
        val_acc, val_loss = evaluate_accuracy(global_model, val_loader, criterion, device)
        
        # Record steps taken per client
        steps_per_client = {res["client_id"]: res["steps_completed"] for res in client_results}
        gated_per_client = {res["client_id"]: res["gated_early"] for res in client_results}
        
        # Calculate cluster utilization: sum(d) / (N * max(d))
        cluster_utilization = (sum(client_durations) / (num_clients * round_duration)) * 100.0 if round_duration > 0 else 0.0
        
        round_info = {
            "round": r,
            "val_accuracy": round(val_acc, 2),
            "val_loss": round(val_loss, 4),
            "round_duration_sec": round(round_duration, 2),
            "cumulative_time_sec": round(cumulative_simulated_time, 2),
            "round_idle_sec": round(round_idle, 2),
            "cumulative_idle_sec": round(cumulative_idle_time, 2),
            "cluster_utilization_pct": round(cluster_utilization, 2),
            "steps_per_client": steps_per_client,
            "gated_per_client": gated_per_client,
            "num_clipped": num_clipped,
            "clipping_cap_c": round(coord_telem.get("clipping_cap_c", 0.0), 5)
        }
        round_telemetry.append(round_info)
        
        print(
            f"   Round {r:02d} | Acc: {val_acc:5.2f}% | Loss: {val_loss:.4f} | "
            f"Round Time: {round_duration:5.2f}s | Idle: {round_idle:5.2f}s | "
            f"Util: {cluster_utilization:4.1f}% | Steps: {list(steps_per_client.values())} | "
            f"Gated: {sum(gated_per_client.values())} | Clipped: {num_clipped}"
        )
        
    accuracies = [h["val_accuracy"] for h in round_telemetry]
    peak_acc = max(accuracies)
    final_acc = accuracies[-1]
    final_loss = round_telemetry[-1]["val_loss"]
    
    return {
        "strategy": strategy_name,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "peak_accuracy": peak_acc,
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "total_simulated_time_sec": round(cumulative_simulated_time, 2),
        "total_idle_time_sec": round(cumulative_idle_time, 2),
        "mean_cluster_utilization_pct": round(float(np.mean([h["cluster_utilization_pct"] for h in round_telemetry])), 2),
        "total_gating_events": total_gating_events,
        "total_clipping_events": total_clipping_events,
        "history": round_telemetry
    }


def execute_ablation_study(
    alpha: float = 0.5,
    num_rounds: int = 8,
    seeds: List[int] = [42],
    data_dir: str = "data/global_test/train"
):
    print("=" * 85)
    print("  CAPDRIFT-FL 5-WAY CONTROL ABLATION BENCHMARK")
    print(f"  Non-IID Dirichlet Skew: alpha = {alpha} | Rounds: {num_rounds} | Seeds: {seeds}")
    print("=" * 85)
    
    # Load and partition data
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = ImageFolder(data_dir, transform=tf)
    
    # Stratified 80/20 train/val split
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
    # Dirichlet partition of training subset
    client_sub_indices = partition_dataset_dirichlet(
        train_subset,
        num_clients=num_clients,
        alpha=alpha,
        seed=42
    )
    # Map back to train_dataset indices
    client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]
    
    # Hardware tiers (Tier 0: GPU 5070Ti, Tier 1: Xeon Fast, Tier 2: Xeon Mid, Tier 4: Edge Straggler)
    client_tiers = [
        TIER_PROFILES[0], # GPU Workstation
        TIER_PROFILES[1], # Cisco UCS pve01
        TIER_PROFILES[2], # Cisco UCS pve02
        TIER_PROFILES[4]  # Constrained Edge
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    for i, t in enumerate(client_tiers):
        print(f"  Client {i}: {t.name} - {t.description} ({len(client_indices[i])} samples)")
        
    # The 5 Ablation Strategy Configurations
    configurations = [
        (
            "1. FedAvg (Baseline)",
            CapDriftConfig(
                enable_capability=False,
                enable_drift_gating=False,
                enable_clipping=False,
                S_min=5,
                S_max=5
            )
        ),
        (
            "2. Capability-Only",
            CapDriftConfig(
                enable_capability=True,
                enable_drift_gating=False,
                enable_clipping=False,
                T_target_sec=15.0,
                S_min=5,
                S_max=50
            )
        ),
        (
            "3. Drift-Only",
            CapDriftConfig(
                enable_capability=False,
                enable_drift_gating=True,
                enable_clipping=False,
                tau_drift=0.50,
                S_min=5,
                S_max=25
            )
        ),
        (
            "4. Capability + Drift",
            CapDriftConfig(
                enable_capability=True,
                enable_drift_gating=True,
                enable_clipping=False,
                T_target_sec=15.0,
                tau_drift=0.50,
                S_min=5,
                S_max=50
            )
        ),
        (
            "5. Full CapDrift-FL",
            CapDriftConfig(
                enable_capability=True,
                enable_drift_gating=True,
                enable_clipping=True,
                T_target_sec=15.0,
                tau_drift=0.50,
                kappa_c=1.5,
                S_min=5,
                S_max=50
            )
        ),
    ]
    
    all_results = []
    telemetry_rows = []
    
    for strat_name, cfg in configurations:
        for seed in seeds:
            res = run_single_ablation_trial(
                strategy_name=strat_name,
                cfg=cfg,
                train_dataset=full_dataset,
                val_loader=val_loader,
                client_indices=client_indices,
                client_tiers=client_tiers,
                num_rounds=num_rounds,
                device=device,
                seed=seed
            )
            all_results.append(res)
            
            # Record flat telemetry rows for CSV
            for h in res["history"]:
                telemetry_rows.append({
                    "strategy": strat_name,
                    "seed": seed,
                    "alpha": alpha,
                    "round": h["round"],
                    "val_accuracy": h["val_accuracy"],
                    "val_loss": h["val_loss"],
                    "round_duration_sec": h["round_duration_sec"],
                    "cumulative_time_sec": h["cumulative_time_sec"],
                    "round_idle_sec": h["round_idle_sec"],
                    "cumulative_idle_sec": h["cumulative_idle_sec"],
                    "cluster_utilization_pct": h["cluster_utilization_pct"],
                    "steps_c0": h["steps_per_client"].get("client_0", 0),
                    "steps_c1": h["steps_per_client"].get("client_1", 0),
                    "steps_c2": h["steps_per_client"].get("client_2", 0),
                    "steps_c3": h["steps_per_client"].get("client_3", 0),
                    "gated_c0": int(h["gated_per_client"].get("client_0", False)),
                    "gated_c1": int(h["gated_per_client"].get("client_1", False)),
                    "gated_c2": int(h["gated_per_client"].get("client_2", False)),
                    "gated_c3": int(h["gated_per_client"].get("client_3", False)),
                    "num_clipped": h["num_clipped"],
                    "clipping_cap_c": h["clipping_cap_c"]
                })
                
    # Save JSON results
    json_path = OUTPUT_DIR / "capdrift_ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved JSON Artifact] -> {json_path}")
    
    # Save CSV telemetry
    csv_path = OUTPUT_DIR / "capdrift_ablation_telemetry.csv"
    if telemetry_rows:
        keys = list(telemetry_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(telemetry_rows)
        print(f"[Saved CSV Artifact] -> {csv_path}")
        
    # Render High-Level Markdown Comparison Table
    print("\n" + "=" * 95)
    print("  CAPDRIFT-FL 5-WAY ABLATION SUMMARY TABLE (Non-IID alpha=0.5)")
    print("=" * 95)
    header = "| Strategy | Peak Acc (%) | Final Acc (%) | Loss | Total Time (s) | Idle Waste (s) | Cluster Util (%) | Gating Events | Clipping Events |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for r in all_results:
        print(
            f"| **{r['strategy']}** | {r['peak_accuracy']:.2f}% | **{r['final_accuracy']:.2f}%** | "
            f"{r['final_loss']:.4f} | {r['total_simulated_time_sec']:.1f}s | {r['total_idle_time_sec']:.1f}s | "
            f"{r['mean_cluster_utilization_pct']:.1f}% | {r['total_gating_events']} | {r['total_clipping_events']} |"
        )
    print("=" * 95)
    return all_results


if __name__ == "__main__":
    execute_ablation_study(alpha=0.5, num_rounds=8, seeds=[42])
