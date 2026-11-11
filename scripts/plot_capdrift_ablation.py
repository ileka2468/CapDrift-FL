"""
Plotting Script for CapDrift-FL 5-Way Ablation Benchmark.
Generates publication-quality IEEE style figures:
1. fig_ablation_accuracy_convergence.png: Accuracy vs Round and Accuracy vs Simulated Wall-Clock Time
2. fig_ablation_idle_and_utilization.png: Total Training Duration, Straggler Idle Waste, and Cluster Utilization
3. fig_ablation_steps_and_gating.png: Per-Client Executed Steps and Gating Truncations across Tiers
"""

import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "research"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = PROJECT_ROOT / "docs"

# Set high-quality IEEE style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "lines.markersize": 5
})

COLORS = {
    "1. FedAvg (Baseline)": "#7f7f7f",       # Grey
    "2. Capability-Only": "#d62728",         # Red
    "3. Drift-Only": "#ff7f0e",              # Orange
    "4. Capability + Drift": "#1f77b4",       # Blue
    "5. Full CapDrift-FL": "#2ca02c"         # Green
}

MARKERS = {
    "1. FedAvg (Baseline)": "o",
    "2. Capability-Only": "s",
    "3. Drift-Only": "^",
    "4. Capability + Drift": "d",
    "5. Full CapDrift-FL": "*"
}


def generate_ablation_plots():
    json_path = RESULTS_DIR / "capdrift_ablation_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group runs by strategy
    strat_groups = {}
    for run in data:
        strat = run["strategy"]
        if strat not in strat_groups:
            strat_groups[strat] = []
        strat_groups[strat].append(run)

    # -------------------------------------------------------------
    # Figure 1: Accuracy Convergence vs Round and vs Wall-Clock Time
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for strat, runs in strat_groups.items():
        color = COLORS.get(strat, "#333333")
        marker = MARKERS.get(strat, "o")
        num_rounds = len(runs[0]["history"])
        rounds = np.arange(1, num_rounds + 1)

        # Matrix of accuracies: shape (num_runs, num_rounds)
        acc_mat = np.array([[h["val_accuracy"] for h in r["history"]] for r in runs])
        time_mat = np.array([[h["cumulative_time_sec"] for h in r["history"]] for r in runs])

        mean_acc = np.mean(acc_mat, axis=0)
        std_acc = np.std(acc_mat, axis=0)
        mean_time = np.mean(time_mat, axis=0)

        # Plot (a) Accuracy vs Communication Rounds
        label = f"{strat} (N={len(runs)})" if len(runs) > 1 else strat
        ax1.plot(rounds, mean_acc, label=label, color=color, marker=marker, markevery=max(1, num_rounds // 10))
        ax1.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)

        # Plot (b) Accuracy vs Cumulative Simulated Wall-Clock Time
        ax2.plot(mean_time, mean_acc, label=label, color=color, marker=marker, markevery=max(1, num_rounds // 10))
        ax2.fill_between(mean_time, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)

    ax1.set_xlabel("Communication Round ($t$)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("(a) Accuracy vs. Communication Rounds")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim([0, 100])
    ax1.legend(loc="lower right", framealpha=0.9)

    ax2.set_xlabel("Cumulative Simulated Execution Time (s)")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("(b) Accuracy vs. Simulated Execution Time")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim([0, 100])
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig1_path = PLOTS_DIR / "fig_ablation_accuracy_convergence.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Copy to docs/figures and docs
    shutil.copy2(fig1_path, DOCS_FIGURES_DIR / "fig_ablation_accuracy_convergence.png")
    shutil.copy2(fig1_path, DOCS_DIR / "fig_ablation_accuracy_convergence.png")
    print(f"[Generated Plot & Copied] -> {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Total Time, Straggler Idle Waste, and Utilization
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    strategies = [s.split(". ")[-1] for s in strat_groups.keys()]
    mean_times = [np.mean([r["total_simulated_time_sec"] for r in runs]) for runs in strat_groups.values()]
    mean_idles = [np.mean([r["total_idle_time_sec"] for r in runs]) for runs in strat_groups.values()]
    mean_utils = [np.mean([r["mean_cluster_utilization_pct"] for r in runs]) for runs in strat_groups.values()]

    x = np.arange(len(strategies))
    width = 0.35

    ax1.bar(x - width/2, mean_times, width, label="Active Compute Time (s)", color="#4a90e2")
    ax1.bar(x + width/2, mean_idles, width, label="Straggler Idle Waste (s)", color="#e94e77")
    ax1.set_ylabel("Simulated Time (seconds)")
    ax1.set_title("(a) Execution Time & Straggler Waste")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    bars = ax2.bar(x, mean_utils, width=0.5, color="#50e3c2", edgecolor="#00a884")
    ax2.set_ylabel("Execution-Time Utilization (%)")
    ax2.set_title("(b) Cluster Execution-Time Utilization")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha="right")
    ax2.set_ylim([0, 100])
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig2_path = PLOTS_DIR / "fig_ablation_idle_and_utilization.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[Generated Plot] -> {fig2_path}")


if __name__ == "__main__":
    generate_ablation_plots()
