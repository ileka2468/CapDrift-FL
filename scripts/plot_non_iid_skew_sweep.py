"""
Plotting Script for Non-IID Skew Sensitivity Benchmark and Drift Threshold Sensitivity.
Generates publication-quality figure:
- results/research/plots/fig_non_iid_skew_robustness.png
- docs/figures/fig_non_iid_skew_robustness.png
- docs/fig_non_iid_skew_robustness.png

Panels:
(a) Final Validation Accuracy under Non-IID Dirichlet Skew (alpha = 0.1, 0.3, 0.5) with 0-100% y-axis and error bars
(b) Drift Threshold Sensitivity: Gating Frequency and Executed Local Steps across tau_d in {0.2, 0.5, 0.8, 1.0}
(c) Convergence Trajectory under Extreme Skew (alpha = 0.1) across rounds with shaded error bands
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
    "lines.linewidth": 2.2,
    "lines.markersize": 6
})


def plot_skew_sweep():
    skew_json_path = RESULTS_DIR / "non_iid_skew_sweep_results.json"
    drift_json_path = RESULTS_DIR / "drift_threshold_sweep_results.json"

    if not skew_json_path.exists():
        print(f"Results file not found: {skew_json_path}")
        return

    with open(skew_json_path, "r", encoding="utf-8") as f:
        skew_data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    # =========================================================================
    # Panel (a): Accuracy vs Non-IID Skew with 0-100% axis and error bars
    # =========================================================================
    alphas = [0.1, 0.3, 0.5]
    x = np.arange(len(alphas))
    width = 0.35

    fedavg_means = []
    fedavg_stds = []
    capdrift_means = []
    capdrift_stds = []

    for a in alphas:
        f_trials = [r["final_accuracy"] for r in skew_data if r["method"] == "FedAvg" and r["alpha"] == a]
        c_trials = [r["final_accuracy"] for r in skew_data if r["method"] == "CapDrift-FL" and r["alpha"] == a]
        fedavg_means.append(np.mean(f_trials) if f_trials else 0.0)
        fedavg_stds.append(np.std(f_trials) if f_trials else 0.0)
        capdrift_means.append(np.mean(c_trials) if c_trials else 0.0)
        capdrift_stds.append(np.std(c_trials) if c_trials else 0.0)

    bars1 = ax1.bar(x - width/2, fedavg_means, width, yerr=fedavg_stds, capsize=4,
                    label="FedAvg (Baseline)", color="#e94e77", edgecolor="#a8173d")
    bars2 = ax1.bar(x + width/2, capdrift_means, width, yerr=capdrift_stds, capsize=4,
                    label="CapDrift-FL (Ours)", color="#2ca02c", edgecolor="#145214")

    ax1.set_xlabel(r"Dirichlet Skew Parameter $\alpha$ (Lower = Extreme Skew)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title(r"(a) Accuracy Robustness vs. Non-IID Skew ($\alpha$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r"$\alpha = 0.1$" + "\n(Extreme)", r"$\alpha = 0.3$" + "\n(Severe)", r"$\alpha = 0.5$" + "\n(Moderate)"])
    ax1.set_ylim([0, 100])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # =========================================================================
    # Panel (b): Drift Threshold Sensitivity (tau_d in {0.2, 0.5, 0.8, 1.0})
    # =========================================================================
    if drift_json_path.exists():
        with open(drift_json_path, "r", encoding="utf-8") as f:
            drift_data = json.load(f)

        # Filter for S_min = 5 configs
        tau_vals = [0.20, 0.50, 0.80, 1.00]
        mean_steps = []
        std_steps = []
        mean_gatings = []
        std_gatings = []

        for tau in tau_vals:
            matches = [r for r in drift_data if abs(r.get("tau_drift", 0.0) - tau) < 1e-4 and r.get("s_min", 0) == 5]
            if matches:
                m = matches[0]
                mean_steps.append(m.get("mean_steps_executed", 0.0))
                std_steps.append(0.0)
                mean_gatings.append(m.get("mean_gating_events", 0.0))
                std_gatings.append(0.0)
            else:
                mean_steps.append(0.0)
                std_steps.append(0.0)
                mean_gatings.append(0.0)
                std_gatings.append(0.0)

        color1 = "#1f77b4"
        color2 = "#d62728"

        ax2.set_xlabel(r"Drift Tolerance Threshold $\tau_d$")
        ax2.set_ylabel("Total Executed Local Steps", color=color1)
        line1 = ax2.errorbar(tau_vals, mean_steps, yerr=std_steps, fmt="-o", color=color1,
                             linewidth=2.2, capsize=4, label="Executed Steps")
        ax2.tick_params(axis="y", labelcolor=color1)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2_twin = ax2.twinx()
        ax2_twin.set_ylabel("Drift Gating Interventions", color=color2)
        line2 = ax2_twin.errorbar(tau_vals, mean_gatings, yerr=std_gatings, fmt="--s", color=color2,
                                  linewidth=2.2, capsize=4, label="Gating Events")
        ax2_twin.tick_params(axis="y", labelcolor=color2)

        ax2.set_title(r"(b) Drift-Gating Sensitivity vs. $\tau_d$ ($S_{\min}=5$)")
        ax2.set_xticks(tau_vals)
        ax2.set_xticklabels([f"{t:.2f}" for t in tau_vals])

        # Combined legend
        lines = [line1, line2]
        labels = ["Executed Steps", "Gating Events"]
        ax2.legend(lines, labels, loc="center left", framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Drift sweep data pending...", ha="center", va="center")
        ax2.set_title(r"(b) Drift-Gating Sensitivity vs. $\tau_d$")

    # =========================================================================
    # Panel (c): Convergence Trajectory under Extreme Skew (alpha = 0.1)
    # =========================================================================
    fedavg_01_runs = [r for r in skew_data if r["method"] == "FedAvg" and r["alpha"] == 0.1]
    capdrift_01_runs = [r for r in skew_data if r["method"] == "CapDrift-FL" and r["alpha"] == 0.1]

    if fedavg_01_runs and capdrift_01_runs:
        num_rounds = len(fedavg_01_runs[0]["history"])
        rounds = np.arange(1, num_rounds + 1)

        f_mat = np.array([[h["val_accuracy"] for h in r["history"]] for r in fedavg_01_runs])
        c_mat = np.array([[h["val_accuracy"] for h in r["history"]] for r in capdrift_01_runs])

        f_mean, f_std = np.mean(f_mat, axis=0), np.std(f_mat, axis=0)
        c_mean, c_std = np.mean(c_mat, axis=0), np.std(c_mat, axis=0)

        ax3.plot(rounds, f_mean, label="FedAvg", color="#e94e77", marker="o", markevery=2, linewidth=2.0)
        ax3.fill_between(rounds, f_mean - f_std, f_mean + f_std, color="#e94e77", alpha=0.15)

        ax3.plot(rounds, c_mean, label="CapDrift-FL (Ours)", color="#2ca02c", marker="D", markevery=2, linewidth=2.0)
        ax3.fill_between(rounds, c_mean - c_std, c_mean + c_std, color="#2ca02c", alpha=0.15)

    ax3.set_xlabel("Communication Round ($t$)")
    ax3.set_ylabel("Validation Accuracy (%)")
    ax3.set_title(r"(c) Convergence Trajectory ($\alpha = 0.1$)")
    ax3.set_ylim([0, 100])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right")

    plt.tight_layout()
    out_path = PLOTS_DIR / "fig_non_iid_skew_robustness.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Copy to docs/figures and docs
    shutil.copy2(out_path, DOCS_FIGURES_DIR / "fig_non_iid_skew_robustness.png")
    shutil.copy2(out_path, DOCS_DIR / "fig_non_iid_skew_robustness.png")
    print(f"[Generated Plot & Copied] -> {out_path}")


if __name__ == "__main__":
    plot_skew_sweep()
