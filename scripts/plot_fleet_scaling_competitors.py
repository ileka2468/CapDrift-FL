"""
Plotting Script for Multi-Scale Fleet Scalability Benchmark:
Compares:
1. FedAvg (McMahan et al., AISTATS 2017)
2. FedProx (Li et al., MLSys 2020)
3. FedNova (Wang et al., NeurIPS 2020)
4. CapDrift-FL (Ours)

Across fleet scales N in {4, 10, 20} clients.
Outputs:
- results/research/plots/fig_fleet_scaling_competitors.png
- docs/figures/fig_fleet_scaling_competitors.png
- docs/fig_fleet_scaling_competitors.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "research"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    "lines.linewidth": 2.2
})


def plot_fleet_scaling():
    json_path = RESULTS_DIR / "fleet_scaling_competitors_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract scales
    scale_keys = [k for k in data.keys() if k.startswith("N_")]
    scales = [int(k.replace("N_", "")) for k in scale_keys]
    scales.sort()

    methods = ["FedAvg", "FedProx", "FedNova", "CapDrift-FL"]
    colors = {
        "FedAvg": "#7f7f7f",
        "FedProx": "#1f77b4",
        "FedNova": "#ff7f0e",
        "CapDrift-FL": "#2ca02c"
    }
    markers = {
        "FedAvg": "o",
        "FedProx": "s",
        "FedNova": "^",
        "CapDrift-FL": "D"
    }

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Fleet Straggler Idle Waste vs N
    for m in methods:
        idles = [data[f"N_{n}"][m]["mean_idle_sec"] for n in scales]
        idle_errs = [data[f"N_{n}"][m]["std_idle_sec"] for n in scales]
        ax1.errorbar(
            scales, idles, yerr=idle_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax1.set_xlabel("Fleet Size ($N$ Clients)")
    ax1.set_ylabel("Fleet Idle Waste (seconds)")
    ax1.set_title("(a) Aggregate Straggler Idle Waste vs. Fleet Scale")
    ax1.set_xticks(scales)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # (b) Cluster Utilization vs N
    for m in methods:
        utils = [data[f"N_{n}"][m]["mean_utilization_pct"] for n in scales]
        util_errs = [data[f"N_{n}"][m]["std_utilization_pct"] for n in scales]
        ax2.errorbar(
            scales, utils, yerr=util_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax2.set_xlabel("Fleet Size ($N$ Clients)")
    ax2.set_ylabel("Mean Cluster Utilization (%)")
    ax2.set_title("(b) Cluster Execution Utilization vs. Fleet Scale")
    ax2.set_xticks(scales)
    ax2.set_ylim([0, 100])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    # (c) Final Validation Accuracy vs N
    for m in methods:
        accs = [data[f"N_{n}"][m]["mean_final_acc"] for n in scales]
        acc_errs = [data[f"N_{n}"][m]["std_final_acc"] for n in scales]
        ax3.errorbar(
            scales, accs, yerr=acc_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax3.set_xlabel("Fleet Size ($N$ Clients)")
    ax3.set_ylabel("Final Validation Accuracy (%)")
    ax3.set_title("(c) Validation Accuracy Retention vs. Scale")
    ax3.set_xticks(scales)
    ax3.set_ylim([0, 100])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower left")

    # (d) Final Loss vs N (Stability / Divergence Metric)
    for m in methods:
        losses = [data[f"N_{n}"][m]["mean_final_loss"] for n in scales]
        loss_errs = [data[f"N_{n}"][m]["std_final_loss"] for n in scales]
        ax4.errorbar(
            scales, losses, yerr=loss_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax4.set_xlabel("Fleet Size ($N$ Clients)")
    ax4.set_ylabel("Final Cross-Entropy Loss")
    ax4.set_title("(d) Objective Loss Stability under Scaling")
    ax4.set_xticks(scales)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    out_path = PLOTS_DIR / "fig_fleet_scaling_competitors.png"
    plt.savefig(out_path, dpi=300)
    docs_fig = PROJECT_ROOT / "docs" / "figures" / "fig_fleet_scaling_competitors.png"
    plt.savefig(docs_fig, dpi=300)
    docs_root = PROJECT_ROOT / "docs" / "fig_fleet_scaling_competitors.png"
    plt.savefig(docs_root, dpi=300)
    plt.close()
    print(f"[Generated Fleet Scaling Competitor Figure] -> {out_path}")


if __name__ == "__main__":
    plot_fleet_scaling()
