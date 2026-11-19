"""
Suite A: Extreme Pathological Non-IID Dirichlet Label Skew Sweep
Evaluates CapDrift-FL against FedAvg, FedProx, FedNova, and FedNova-Momentum
under extreme Dirichlet label skew alpha in {0.05, 0.10, 0.20} across 5 seeds and 25 rounds.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALPHAS = [0.05, 0.10, 0.20]
DEFAULT_METHODS = ["CapDrift-FL", "FedAvg", "FedProx", "FedNova", "FedNova-Momentum"]
DEFAULT_SEEDS = [42, 101, 2024, 7, 888]

def run_suite_a(alphas=None, seeds=None, rounds=25):
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    base_output_dir = PROJECT_ROOT / "results" / "research" / "suite_a_extreme_skew"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for alpha in alphas:
        alpha_tag = f"alpha_{alpha:.2f}".replace(".", "_")
        run_output_dir = base_output_dir / alpha_tag
        print(f"\n{'='*70}\n  [Suite A] Running Skew Sweep: alpha = {alpha} -> {run_output_dir}\n{'='*70}", flush=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_competitor_benchmark.py"),
            "--output-dir", str(run_output_dir),
            "--methods", *DEFAULT_METHODS,
            "--seeds", *[str(s) for s in seeds],
            "--rounds", str(rounds),
            "--alpha", str(alpha),
            "--freeze-bn",
            "--channels-last",
            "--fused-sgd",
            "--server-momentum", "0.9",
            "--server-lr", "0.2",
            "--local-lr", "0.002",
            "--baseline-local-lr", "0.002",
            "--continue-on-nonfinite"
        ]

        csv_path = run_output_dir / "competitor_benchmark_summary.csv"
        if not csv_path.is_file():
            res = subprocess.run(cmd)
            if res.returncode not in (0, 2):
                print(f"[Suite A] Warning: Run returned exit code {res.returncode}")
        else:
            print(f"[Suite A] Found existing summary at {csv_path}, skipping run.")

        if csv_path.is_file():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["Alpha"] = alpha
                    summary_rows.append(row)

    # Write unified summary CSV
    combined_csv = base_output_dir / "suite_a_extreme_skew_summary.csv"
    if summary_rows:
        fieldnames = ["Alpha"] + [k for k in summary_rows[0].keys() if k != "Alpha"]
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[Suite A] Combined summary written to {combined_csv}")

    # Generate visualization
    plot_suite_a(summary_rows, base_output_dir)

def plot_suite_a(rows, output_dir: Path):
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    # Parse numerical mean values
    def parse_mean(val_str):
        try:
            return float(val_str.split("±")[0].strip())
        except Exception:
            return np.nan

    df["Mean_Acc"] = df["Final Acc (%) [Mean ± Std]"].apply(parse_mean)
    df["Mean_Loss"] = df["Final Loss"].apply(parse_mean)
    df["Mean_Util"] = df["Mean Util (%)"].apply(parse_mean)

    methods = df["Method"].unique()
    alphas = sorted(df["Alpha"].unique())

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

    # Panel 1: Final Accuracy vs Alpha
    ax1 = axes[0]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Alpha")
        ax1.plot(sub["Alpha"], sub["Mean_Acc"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax1.set_title("(a) Final Validation Accuracy vs. Skew", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Dirichlet Parameter α (Lower = More Severe Skew)", fontsize=11)
    ax1.set_ylabel("Final Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Panel 2: Final Loss vs Alpha
    ax2 = axes[1]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Alpha")
        ax2.plot(sub["Alpha"], sub["Mean_Loss"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax2.set_title("(b) Cross-Entropy Loss (Stability)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Dirichlet Parameter α", fontsize=11)
    ax2.set_ylabel("Final Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Active Cluster Utilization (%)
    ax3 = axes[2]
    for m in methods:
        sub = df[df["Method"] == m].sort_values("Alpha")
        ax3.plot(sub["Alpha"], sub["Mean_Util"], marker=markers.get(m, "o"),
                 color=colors.get(m, "#333333"), label=m, linewidth=2, markersize=7)
    ax3.set_title("(c) Active Compute Utilization (%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Dirichlet Parameter α", fontsize=11)
    ax3.set_ylabel("Modeled Utilization (%)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / "fig_suite_a_extreme_skew.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Suite A] Plot generated: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suite A Extreme Skew Runner")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=25)
    args = parser.parse_args()
    run_suite_a(args.alphas, args.seeds, args.rounds)
