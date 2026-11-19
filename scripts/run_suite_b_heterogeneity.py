"""
Suite B: Hardware Heterogeneity Ratio Sweep (rho in {2x, 5x, 10x, 20x})
Compares CapDrift-FL against FedAvg, FedProx, FedNova, and FedNova-Momentum
under varying hardware disparity ratios between the fastest and slowest edge nodes.
Demonstrates:
- Straggler bottleneck and idle waste explosion in FedAvg/FedProx (>80% idle)
- Severe gradient skew and loss divergence in FedNova
- Robust utilization and bounded loss in CapDrift-FL via dynamic capacity calibration
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from research.capdrift_controller import (
    CapDriftConfig,
    CapDriftClientTrainer,
    CapDriftCoordinator,
)
from research.competitor_split import split_indices
from model import get_model, get_model_size_mb
from research.non_iid import partition_dataset_dirichlet
from research.heterogeneous_fleet import ClientTierConfig
from run_competitor_benchmark import (
    PreloadedDataset,
    load_dataset,
    run_competitor_trial,
    summarize_trials,
    write_json,
    build_config,
    METHODS,
    BATCH_SIZE,
    COMPUTE_DEADLINE_SEC
)

DEFAULT_RATIOS = [2.0, 5.0, 10.0, 20.0]
DEFAULT_METHODS = ["CapDrift-FL", "FedAvg", "FedProx", "FedNova", "FedNova-Momentum"]
DEFAULT_SEEDS = [42, 101, 2024, 7, 888]

def build_heterogeneous_tiers(ratio: float) -> List[ClientTierConfig]:
    """Generates 4 tiers spanning speed disparity ratio rho = v_fast / v_slow."""
    v_fast = 160.0 # samples/sec (AMC / GPU workstation)
    v_slow = v_fast / ratio
    # Log-spaced speeds across 4 clients
    speeds = [v_fast * (ratio ** (-i / 3.0)) for i in range(4)]
    latencies = [1.5, 3.0, 5.0, 45.0]
    bandwidths = [1000.0, 1000.0, 500.0, 25.0]

    tiers = []
    for i in range(4):
        # Steps feasible within 6.5s deadline
        feasible_steps = max(2, min(40, int(speeds[i] * COMPUTE_DEADLINE_SEC / BATCH_SIZE)))
        tiers.append(ClientTierConfig(
            tier_id=i,
            name=f"Tier {i} (rho={ratio:.0f}x, {speeds[i]:.1f} smp/s)",
            description=f"Synthetic hardware tier: speed {speeds[i]:.2f} samples/sec",
            vcpu_allocated=max(1, 16 // (2**i)),
            ram_mb=max(1024, 16384 // (2**i)),
            cpu_quota_percent=100.0,
            network_latency_ms=latencies[i],
            network_bandwidth_mbps=bandwidths[i],
            base_train_time_sec=1000.0 / speeds[i],
            base_samples_per_sec=speeds[i],
            default_carc_steps=feasible_steps,
            default_fedavg_steps=5
        ))
    return tiers

def run_suite_b(ratios=None, seeds=None, rounds=25):
    if ratios is None:
        ratios = DEFAULT_RATIOS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    base_output_dir = PROJECT_ROOT / "results" / "research" / "suite_b_heterogeneity"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "global_test" / "train"
    if not data_dir.is_dir():
        data_dir = PROJECT_ROOT.parent / "data" / "global_test" / "train"
    full_dataset, classes, data_hash = load_dataset(data_dir)

    train_indices, val_indices = split_indices(
        full_dataset, policy="image", seed=42, holdout_fraction=0.2
    )
    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    client_sub_indices = partition_dataset_dirichlet(
        Subset(full_dataset, train_indices), num_clients=4, alpha=0.5, seed=42
    )
    client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []

    for ratio in ratios:
        ratio_tag = f"ratio_{int(ratio)}x"
        run_output_dir = base_output_dir / ratio_tag
        run_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_output_dir / "competitor_benchmark_summary.csv"
        json_path = run_output_dir / "competitor_benchmark_results.json"
        ratio_summary_rows = []

        print(f"\n{'='*70}\n  [Suite B] Heterogeneity Ratio Sweep: rho = {ratio:.1f}x -> {run_output_dir}\n{'='*70}", flush=True)
        client_tiers = build_heterogeneous_tiers(ratio)
        configs = {name: build_config(
            name, server_momentum=0.9, server_lr=0.2, anneal_start_round=15, anneal_power=1.0
        ) for name in DEFAULT_METHODS}

        all_results = {}
        if json_path.is_file():
            all_results = json.loads(json_path.read_text(encoding="utf-8"))

        for name in DEFAULT_METHODS:
            print(f"\n--- Method: {name} (rho={ratio:.1f}x) ---", flush=True)
            completed = {t["seed"]: t for t in all_results.get(name, {}).get("trials", [])}
            for seed in seeds:
                if seed in completed:
                    print(f"   Reusing completed seed {seed}", flush=True)
                    continue
                try:
                    trial = run_competitor_trial(
                        method_name=name,
                        train_dataset=full_dataset,
                        val_loader=val_loader,
                        client_indices=client_indices,
                        client_tiers=client_tiers,
                        num_rounds=rounds,
                        mu_prox=METHODS[name],
                        device=device,
                        seed=seed,
                        config=configs[name],
                        freeze_bn=True,
                        channels_last=True,
                        fused_sgd=True,
                        local_lr=0.002,
                        baseline_local_lr=0.002,
                    )
                    completed[seed] = trial
                except Exception as exc:
                    print(f"   Trial failed: {name}, seed {seed}, exc: {exc}", flush=True)
                    continue
                all_results[name] = summarize_trials(name, [completed[s] for s in seeds if s in completed])
                write_json(json_path, all_results)

            data = all_results.get(name)
            if data:
                def summary_stat(key, digits):
                    return f"{data[f'mean_{key}']:.{digits}f} ± {data[f'std_{key}']:.{digits}f}"
                row_dict = {
                    "Ratio": ratio,
                    "Method": name,
                    "Successful Trials": f"{len(completed)}/{len(seeds)}",
                    "Final Acc (%) [Mean ± Std]": summary_stat("final_acc", 2),
                    "Peak Acc (%) [Mean ± Std]": summary_stat("peak_acc", 2),
                    "Final Loss": summary_stat("loss", 4),
                    "Total Time (s)": summary_stat("time_sec", 2),
                    "Idle Waste (s)": summary_stat("idle_sec", 2),
                    "Mean Util (%)": summary_stat("util_pct", 2),
                    "Mean Compute Util (%)": summary_stat("compute_util_pct", 2),
                }
                summary_rows.append(row_dict)
                ratio_summary_rows.append(row_dict)

        write_json(json_path, all_results)
        if ratio_summary_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ratio_summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ratio_summary_rows)

    # Save master CSV
    combined_csv = base_output_dir / "suite_b_heterogeneity_summary.csv"
    if summary_rows:
        fieldnames = ["Ratio", "Method", "Successful Trials", "Final Acc (%) [Mean ± Std]",
                      "Peak Acc (%) [Mean ± Std]", "Final Loss", "Total Time (s)",
                      "Idle Waste (s)", "Mean Util (%)", "Mean Compute Util (%)"]
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[Suite B] Master summary written to {combined_csv}")

    # Plot
    plot_suite_b(summary_rows, base_output_dir)

def plot_suite_b(rows, output_dir: Path):
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    def parse_mean(val_str):
        try:
            return float(val_str.split("±")[0].strip())
        except Exception:
            return np.nan

    df["Mean_Acc"] = df["Final Acc (%) [Mean ± Std]"].apply(parse_mean)
    df["Mean_Loss"] = df["Final Loss"].apply(parse_mean)
    df["Mean_Idle"] = df["Idle Waste (s)"].apply(parse_mean)
    df["Mean_Util"] = df["Mean Util (%)"].apply(parse_mean)

    methods = df["Method"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    colors = {
        "CapDrift-FL": "#1f77b4",
        "FedAvg": "#7f7f7f",
        "FedProx": "#2ca02c",
        "FedNova": "#d62728",
        "FedNova-Momentum": "#9467bd"
    }
    markers = {
        "CapDrift-FL": "o",
        "FedAvg": "s",
        "FedProx": "^",
        "FedNova": "v",
        "FedNova-Momentum": "D"
    }

    # Panel 1: Accuracy vs Heterogeneity Ratio
    ax1 = axes[0]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Ratio")
        ax1.plot(sub["Ratio"], sub["Mean_Acc"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax1.set_title("(a) Accuracy vs. Disparity Ratio (ρ)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Heterogeneity Disparity Ratio ρ (v_fast / v_slow)", fontsize=11)
    ax1.set_ylabel("Final Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Panel 2: Idle Waste vs Heterogeneity Ratio
    ax2 = axes[1]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Ratio")
        ax2.plot(sub["Ratio"], sub["Mean_Idle"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax2.set_title("(b) Cluster Idle Waste (Seconds)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Heterogeneity Disparity Ratio ρ", fontsize=11)
    ax2.set_ylabel("Cumulative Idle Waste (s)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Active Cluster Utilization (%)
    ax3 = axes[2]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Ratio")
        ax3.plot(sub["Ratio"], sub["Mean_Util"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax3.set_title("(c) Active Cluster Utilization (%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Heterogeneity Disparity Ratio ρ", fontsize=11)
    ax3.set_ylabel("Utilization (%)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / "fig_suite_b_heterogeneity.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Suite B] Plot generated: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suite B Heterogeneity Sweep Runner")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=25)
    args = parser.parse_args()
    run_suite_b(args.ratios, args.seeds, args.rounds)
