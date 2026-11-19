"""
Suite D: Dataset Volume Scaling Benchmark (M in {763, 1500, 3000, 5216} images)
Evaluates CapDrift-FL against FedAvg, FedProx, FedNova, and FedNova-Momentum
as the dataset scales from small sample to the full chest X-ray cohort.
Demonstrates:
- How the performance disparity (Delta Acc) between CapDrift-FL and competitors expands as data scales
- Step starvation and underfitting in FedAvg/FedProx due to the straggler-constrained 5-step limit
- Gradient divergence and loss explosion in FedNova on large heterogeneous client partitions
- Bounded drift and rapid data assimilation in CapDrift-FL
"""

import argparse
import copy
import csv
import hashlib
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
from torchvision import transforms
from torchvision.datasets import ImageFolder

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
from research.heterogeneous_fleet import TIER_PROFILES, ClientTierConfig
from run_competitor_benchmark import (
    PreloadedDataset,
    run_competitor_trial,
    summarize_trials,
    write_json,
    build_config,
    METHODS,
    TIER_IDS,
    BATCH_SIZE,
    COMPUTE_DEADLINE_SEC
)

DEFAULT_SCALES = [763, 1500, 3000, 5216]
DEFAULT_METHODS = ["CapDrift-FL", "FedAvg", "FedProx", "FedNova", "FedNova-Momentum"]
DEFAULT_SEEDS = [42, 101, 2024, 7, 888]

def load_scaled_dataset(root: Path, target_samples: Optional[int] = None, seed: int = 42) -> Tuple[PreloadedDataset, List[str]]:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Preloading ImageFolder: {root}", flush=True)
    raw = ImageFolder(str(root), transform=tf)
    total_avail = len(raw)

    if target_samples is None or target_samples >= total_avail:
        selected_indices = list(range(total_avail))
    else:
        # Stratified sampling to preserve class prevalence
        targets = np.array(raw.targets)
        classes, counts = np.unique(targets, return_counts=True)
        rng = np.random.RandomState(seed)
        selected_indices = []
        for cls, count in zip(classes, counts):
            cls_idxs = np.where(targets == cls)[0]
            rng.shuffle(cls_idxs)
            n_take = int(round(target_samples * (count / total_avail)))
            selected_indices.extend(cls_idxs[:n_take])
        # Adjust if rounding gave +- a few samples
        if len(selected_indices) < target_samples:
            remaining = list(set(range(total_avail)) - set(selected_indices))
            rng.shuffle(remaining)
            selected_indices.extend(remaining[:target_samples - len(selected_indices)])
        elif len(selected_indices) > target_samples:
            selected_indices = selected_indices[:target_samples]
        rng.shuffle(selected_indices)

    N = len(selected_indices)
    print(f"Preloading {N} images into memory...", flush=True)
    images = torch.empty((N, 3, 224, 224))
    labels = torch.empty(N, dtype=torch.long)
    for out_i, in_i in enumerate(selected_indices):
        img, lbl = raw[in_i]
        images[out_i] = img
        labels[out_i] = lbl
    return PreloadedDataset(images, labels), raw.classes

def run_suite_d(scales=None, seeds=None, rounds=25):
    if scales is None:
        scales = DEFAULT_SCALES
    if seeds is None:
        seeds = DEFAULT_SEEDS

    base_output_dir = PROJECT_ROOT / "results" / "research" / "suite_d_dataset_scaling"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    data_dir_763 = PROJECT_ROOT / "data" / "global_test" / "train"
    if not data_dir_763.is_dir():
        data_dir_763 = PROJECT_ROOT.parent / "data" / "global_test" / "train"

    data_dir_full = PROJECT_ROOT / "data" / "raw" / "chest_xray" / "train"
    if not data_dir_full.is_dir():
        data_dir_full = PROJECT_ROOT.parent / "data" / "raw" / "chest_xray" / "train"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_tiers = [TIER_PROFILES[i] for i in TIER_IDS]
    configs = {name: build_config(
        name, server_momentum=0.9, server_lr=0.2, anneal_start_round=15, anneal_power=1.0
    ) for name in DEFAULT_METHODS}

    summary_rows = []

    for M in scales:
        scale_tag = f"scale_{M}"
        run_output_dir = base_output_dir / scale_tag
        run_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_output_dir / "competitor_benchmark_summary.csv"
        json_path = run_output_dir / "competitor_benchmark_results.json"
        scale_summary_rows = []

        print(f"\n{'='*70}\n  [Suite D] Dataset Scaling: M = {M} images -> {run_output_dir}\n{'='*70}", flush=True)

        if M == 763:
            dataset, classes = load_scaled_dataset(data_dir_763)
        else:
            dataset, classes = load_scaled_dataset(data_dir_full, target_samples=M)

        train_indices, val_indices = split_indices(
            dataset, policy="image", seed=42, holdout_fraction=0.2
        )
        val_subset = Subset(dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        client_sub_indices = partition_dataset_dirichlet(
            Subset(dataset, train_indices), num_clients=len(TIER_IDS), alpha=0.5, seed=42
        )
        client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]

        all_results = {}
        if json_path.is_file():
            all_results = json.loads(json_path.read_text(encoding="utf-8"))

        for name in DEFAULT_METHODS:
            print(f"\n--- Method: {name} (M={M}) ---", flush=True)
            completed = {t["seed"]: t for t in all_results.get(name, {}).get("trials", [])}
            for seed in seeds:
                if seed in completed:
                    print(f"   Reusing completed seed {seed}", flush=True)
                    continue
                try:
                    trial = run_competitor_trial(
                        method_name=name,
                        train_dataset=dataset,
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
                    "Dataset_Size": M,
                    "Train_Samples": len(train_indices),
                    "Val_Samples": len(val_indices),
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
                scale_summary_rows.append(row_dict)

        write_json(json_path, all_results)
        if scale_summary_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(scale_summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(scale_summary_rows)

    # Master CSV
    combined_csv = base_output_dir / "suite_d_dataset_scaling_summary.csv"
    if summary_rows:
        fieldnames = ["Dataset_Size", "Train_Samples", "Val_Samples", "Method", "Successful Trials",
                      "Final Acc (%) [Mean ± Std]", "Peak Acc (%) [Mean ± Std]", "Final Loss",
                      "Total Time (s)", "Idle Waste (s)", "Mean Util (%)", "Mean Compute Util (%)"]
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[Suite D] Master summary written to {combined_csv}")

    plot_suite_d(summary_rows, base_output_dir)

def plot_suite_d(rows, output_dir: Path):
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

    # Panel 1: Accuracy vs Dataset Size
    ax1 = axes[0]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Dataset_Size")
        ax1.plot(sub["Dataset_Size"], sub["Mean_Acc"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax1.set_title("(a) Accuracy vs. Cohort Scale (M)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Total Dataset Volume (Images)", fontsize=11)
    ax1.set_ylabel("Final Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Panel 2: Performance Disparity Delta (CapDrift - FedAvg & CapDrift - FedNova)
    ax2 = axes[1]
    capdrift_sub = df[df["Method"] == "CapDrift-FL"].set_index("Dataset_Size")["Mean_Acc"]
    fedavg_sub = df[df["Method"] == "FedAvg"].set_index("Dataset_Size")["Mean_Acc"]
    fednova_sub = df[df["Method"] == "FedNova"].set_index("Dataset_Size")["Mean_Acc"]

    common_sizes_fedavg = sorted(list(set(capdrift_sub.index) & set(fedavg_sub.index)))
    if common_sizes_fedavg:
        delta_fedavg = [capdrift_sub.loc[s] - fedavg_sub.loc[s] for s in common_sizes_fedavg]
        ax2.plot(common_sizes_fedavg, delta_fedavg, marker="s", color="#7f7f7f", label="Δ vs FedAvg", linewidth=2.5)

    common_sizes_fednova = sorted(list(set(capdrift_sub.index) & set(fednova_sub.index)))
    if common_sizes_fednova:
        delta_fednova = [capdrift_sub.loc[s] - fednova_sub.loc[s] for s in common_sizes_fednova]
        ax2.plot(common_sizes_fednova, delta_fednova, marker="v", color="#d62728", label="Δ vs FedNova", linewidth=2.5)

    ax2.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax2.set_title("(b) CapDrift-FL Disparity Margin (Δ%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Total Dataset Volume (Images)", fontsize=11)
    ax2.set_ylabel("Advantage Margin (Percentage Points)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    # Panel 3: Cross-Entropy Loss vs Dataset Size
    ax3 = axes[2]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Dataset_Size")
        ax3.plot(sub["Dataset_Size"], sub["Mean_Loss"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax3.set_title("(c) Loss vs. Cohort Scale", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Total Dataset Volume (Images)", fontsize=11)
    ax3.set_ylabel("Final Loss", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / "fig_suite_d_dataset_scaling.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Suite D] Plot generated: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suite D Dataset Scaling Runner")
    parser.add_argument("--scales", nargs="+", type=int, default=DEFAULT_SCALES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=25)
    args = parser.parse_args()
    run_suite_d(args.scales, args.seeds, args.rounds)
