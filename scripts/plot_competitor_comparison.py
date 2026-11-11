"""
Plotting Script for Competitor Baseline Benchmark:
Compares:
1. FedAvg (McMahan et al., AISTATS 2017)
2. FedProx (Li et al., MLSys 2020)
3. FedNova (Wang et al., NeurIPS 2020)
4. CapDrift-FL (Ours)

Outputs:
- results/research/plots/fig_competitor_comparison.png
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


def plot_competitors():
    json_path = RESULTS_DIR / "competitor_benchmark_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

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

    # (a) Accuracy Convergence
    for method, info in data.items():
        color = colors.get(method, "#333333")
        marker = markers.get(method, "o")
        if "trials" in info:
            trials = info["trials"]
            rounds = [h["round"] for h in trials[0]["history"]]
            acc_matrix = np.array([[h["val_accuracy"] for h in t["history"]] for t in trials])
            mean_accs = np.mean(acc_matrix, axis=0)
            std_accs = np.std(acc_matrix, axis=0)
            ax1.plot(
                rounds,
                mean_accs,
                label=f"{method} ({info['mean_final_acc']:.1f} ± {info['std_final_acc']:.1f}%)",
                color=color,
                marker=marker,
                markersize=5,
                markevery=2
            )
            ax1.fill_between(rounds, mean_accs - std_accs, mean_accs + std_accs, color=color, alpha=0.15)
        else:
            rounds = [h["round"] for h in info["history"]]
            accs = [h["val_accuracy"] for h in info["history"]]
            ax1.plot(
                rounds,
                accs,
                label=f"{method} (Final: {info['final_accuracy']:.1f}%)",
                color=color,
                marker=marker,
                markersize=6
            )

    ax1.set_xlabel("Communication Round ($t$)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("(a) Convergence under Heterogeneity (alpha = 0.5)")
    ax1.set_ylim([0, 100])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # (b) Pareto Tradeoff: Final Accuracy vs Execution Time (Scatter Plot)
    for method, info in data.items():
        color = colors.get(method, "#333333")
        marker = markers.get(method, "o")
        if "trials" in info:
            m_acc = info["mean_final_acc"]
            s_acc = info["std_final_acc"]
            m_time = info["mean_time_sec"]
            s_time = info["std_time_sec"]
            ax2.errorbar(
                m_time, m_acc, xerr=s_time, yerr=s_acc, fmt=marker, color=color,
                ecolor=color, elinewidth=1.5, capsize=4, markersize=9, label=method
            )
            offset_y = 1.2 if "CapDrift" in method else -1.8
            ax2.annotate(
                f"{method}\n({m_acc:.1f}%, {m_time:.0f}s)",
                (m_time, m_acc), xytext=(8, offset_y), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color
            )
        else:
            m_acc = info["final_accuracy"]
            m_time = info["total_time_sec"]
            ax2.scatter(m_time, m_acc, color=color, marker=marker, s=90, label=method)
            ax2.annotate(
                f"{method} ({m_acc:.1f}%, {m_time:.0f}s)",
                (m_time, m_acc), xytext=(5, 5), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color
            )

    ax2.set_xlabel("Total Simulated Execution Time (seconds)")
    ax2.set_ylabel("Final Validation Accuracy (%)")
    ax2.set_title("(b) Accuracy vs. Execution Time (Tradeoff)")
    ax2.set_ylim([0, 100])
    ax2.grid(True, linestyle="--", alpha=0.5)

    # (c) Cumulative Straggler Idle Waste
    methods = list(data.keys())
    if "trials" in data[methods[0]]:
        wastes = [data[m]["mean_idle_sec"] for m in methods]
        waste_errs = [data[m]["std_idle_sec"] for m in methods]
        bars3 = ax3.bar(methods, wastes, yerr=waste_errs, capsize=4, color=[colors.get(m, "#555") for m in methods], edgecolor="#222", alpha=0.85, width=0.45)
    else:
        wastes = [data[m]["total_idle_sec"] for m in methods]
        bars3 = ax3.bar(methods, wastes, color=[colors.get(m, "#555") for m in methods], edgecolor="#222", alpha=0.85, width=0.45)

    ax3.set_xlabel("Federated Learning Baseline")
    ax3.set_ylabel("Straggler Idle Waste (seconds)")
    ax3.set_title("(c) Straggler Synchronization Waste")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.set_xticklabels(methods, rotation=15)

    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 2.0, f"{h:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "fig_competitor_comparison.png"
    plt.savefig(out_path, dpi=300)
    docs_fig = PROJECT_ROOT / "docs" / "figures" / "fig_competitor_comparison.png"
    plt.savefig(docs_fig, dpi=300)
    docs_root = PROJECT_ROOT / "docs" / "fig_competitor_comparison.png"
    plt.savefig(docs_root, dpi=300)
    plt.close()
    print(f"[Generated Competitor Benchmark Figure] -> {out_path}")


if __name__ == "__main__":
    plot_competitors()
