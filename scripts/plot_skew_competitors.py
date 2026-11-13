"""
Plotting Script for Dirichlet Non-IID Skew Sensitivity Benchmark:
Compares:
1. FedAvg (McMahan et al., AISTATS 2017)
2. FedProx (Li et al., MLSys 2020)
3. FedNova (Wang et al., NeurIPS 2020)
4. CapDrift-FL (Ours)

Across Dirichlet alpha in {0.1, 0.3, 0.5, 1.0}.
Outputs:
- results/research/plots/fig_skew_competitors.png
- docs/figures/fig_skew_competitors.png
- docs/fig_skew_competitors.png
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


def plot_skew_competitors():
    json_path = RESULTS_DIR / "skew_competitors_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    alpha_keys = [k for k in data.keys() if k.startswith("alpha_")]
    alphas = [float(k.replace("alpha_", "")) for k in alpha_keys]
    alphas.sort()

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

    # (a) Validation Accuracy vs alpha
    for m in methods:
        accs = [data[f"alpha_{a}"][m]["mean_final_acc"] for a in alphas]
        acc_errs = [data[f"alpha_{a}"][m]["std_final_acc"] for a in alphas]
        ax1.errorbar(
            alphas, accs, yerr=acc_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax1.set_xlabel(r"Dirichlet Label Skew Parameter ($\alpha$)")
    ax1.set_ylabel("Final Validation Accuracy (%)")
    ax1.set_title("(a) Accuracy across Heterogeneity Regimes")
    ax1.set_xticks(alphas)
    ax1.set_ylim([0, 100])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # (b) Final Loss vs alpha (Divergence Metric)
    for m in methods:
        losses = [data[f"alpha_{a}"][m]["mean_final_loss"] for a in alphas]
        loss_errs = [data[f"alpha_{a}"][m]["std_final_loss"] for a in alphas]
        ax2.errorbar(
            alphas, losses, yerr=loss_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax2.set_xlabel(r"Dirichlet Label Skew Parameter ($\alpha$)")
    ax2.set_ylabel("Final Cross-Entropy Loss")
    ax2.set_title("(b) Objective Divergence Resistance")
    ax2.set_xticks(alphas)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # (c) CapDrift-FL Adaptive Interventions vs alpha
    gated_events = [data[f"alpha_{a}"]["CapDrift-FL"]["mean_gating_events"] for a in alphas]
    clipped_events = [data[f"alpha_{a}"]["CapDrift-FL"]["mean_clipping_events"] for a in alphas]
    width = 0.035
    ax3.bar([a - width/2 for a in alphas], gated_events, width=width, label="Drift Gating Interventions", color="#2ca02c", alpha=0.85, edgecolor="#111")
    ax3.bar([a + width/2 for a in alphas], clipped_events, width=width, label="Dynamic Clipping Events", color="#d62728", alpha=0.85, edgecolor="#111")
    ax3.set_xlabel(r"Dirichlet Label Skew Parameter ($\alpha$)")
    ax3.set_ylabel("Mean Interventions per Trial")
    ax3.set_title("(c) Controller Interventions vs. Non-IID Skew")
    ax3.set_xticks(alphas)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    # (d) Straggler Idle Waste vs alpha
    for m in methods:
        idles = [data[f"alpha_{a}"][m]["mean_idle_sec"] for a in alphas]
        idle_errs = [data[f"alpha_{a}"][m]["std_idle_sec"] for a in alphas]
        ax4.errorbar(
            alphas, idles, yerr=idle_errs,
            label=m, color=colors[m], marker=markers[m],
            capsize=4, markersize=8
        )
    ax4.set_xlabel(r"Dirichlet Label Skew Parameter ($\alpha$)")
    ax4.set_ylabel("Fleet Idle Waste (seconds)")
    ax4.set_title("(d) Synchronization Overhead vs. Skew")
    ax4.set_xticks(alphas)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    out_path = PLOTS_DIR / "fig_skew_competitors.png"
    plt.savefig(out_path, dpi=300)
    docs_fig = PROJECT_ROOT / "docs" / "figures" / "fig_skew_competitors.png"
    plt.savefig(docs_fig, dpi=300)
    docs_root = PROJECT_ROOT / "docs" / "fig_skew_competitors.png"
    plt.savefig(docs_root, dpi=300)
    plt.close()
    print(f"[Generated Skew Competitor Figure] -> {out_path}")


if __name__ == "__main__":
    plot_skew_competitors()
