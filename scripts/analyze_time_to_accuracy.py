"""
Time-to-Accuracy and Communication Efficiency Analysis:
Evaluates FedAvg, FedProx, FedNova, and CapDrift-FL across 5 seeds:
- Rounds to reach target accuracy thresholds (55%, 60%, 65%, 70%)
- Cumulative simulated wall-clock seconds to target accuracy
- Cumulative network communication payload (MB) to target accuracy
- Early-round acceleration factor

Outputs:
- results/research/time_to_accuracy_summary.csv
- results/research/plots/fig_time_to_accuracy.png
- docs/figures/fig_time_to_accuracy.png
- docs/fig_time_to_accuracy.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

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


def analyze_time_to_accuracy():
    json_path = RESULTS_DIR / "competitor_benchmark_results.json"
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds = [55.0, 60.0, 65.0, 70.0]
    methods = ["FedAvg", "FedProx", "FedNova", "CapDrift-FL"]

    # Model size is ~7.4 MB for MobileNetV3-Small (weights upload + download = 14.8 MB per client per round)
    model_payload_per_round_mb = 7.42 * 2.0 * 4 # 4 clients = 59.36 MB/round

    summary_rows = []

    print("\n" + "=" * 80)
    print("  TIME-TO-ACCURACY & COMMUNICATION EFFICIENCY BENCHMARK (5 SEEDS)")
    print("=" * 80)

    for target in thresholds:
        print(f"\n--- Target Accuracy Threshold: {target:.0f}% ---")
        for m in methods:
            trials = data[m]["trials"]
            rounds_to_target = []
            times_to_target = []
            bytes_to_target = []

            for t in trials:
                history = t["history"]
                hit_round = None
                hit_time = None
                for h in history:
                    if h["val_accuracy"] >= target:
                        hit_round = h["round"]
                        hit_time = h["cumulative_time_sec"]
                        break
                if hit_round is not None:
                    rounds_to_target.append(hit_round)
                    times_to_target.append(hit_time)
                    bytes_to_target.append(hit_round * model_payload_per_round_mb)

            success_rate = len(rounds_to_target) / len(trials) * 100.0
            mean_r = np.mean(rounds_to_target) if rounds_to_target else float("nan")
            std_r = np.std(rounds_to_target) if rounds_to_target else float("nan")
            mean_t = np.mean(times_to_target) if times_to_target else float("nan")
            std_t = np.std(times_to_target) if times_to_target else float("nan")
            mean_b = np.mean(bytes_to_target) if bytes_to_target else float("nan")

            summary_rows.append({
                "Target Acc (%)": target,
                "Method": m,
                "Success Rate (%)": f"{success_rate:.0f}%",
                "Rounds to Target": f"{mean_r:.1f} ± {std_r:.1f}" if not np.isnan(mean_r) else "DNC",
                "Simulated Time (s)": f"{mean_t:.1f} ± {std_t:.1f}" if not np.isnan(mean_t) else "DNC",
                "Comm Payload (MB)": f"{mean_b:.1f}" if not np.isnan(mean_b) else "DNC"
            })

            print(f"  {m:12s} | Hit Rate: {success_rate:3.0f}% | Rounds: {mean_r:4.1f} ± {std_r:3.1f} | Time: {mean_t:5.1f}s | Comm: {mean_b:5.1f} MB")

    out_csv = RESULTS_DIR / "time_to_accuracy_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[Saved CSV] -> {out_csv}")

    # Plot Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {
        "FedAvg": "#7f7f7f",
        "FedProx": "#1f77b4",
        "FedNova": "#ff7f0e",
        "CapDrift-FL": "#2ca02c"
    }

    # (a) Mean Rounds across 25 rounds
    for m in methods:
        trials = data[m]["trials"]
        acc_mat = np.array([[h["val_accuracy"] for h in t["history"]] for t in trials])
        mean_accs = np.mean(acc_mat, axis=0)
        rounds = list(range(1, len(mean_accs) + 1))
        ax1.plot(rounds, mean_accs, label=m, color=colors[m], linewidth=2.2)

    ax1.axhline(60.0, color="#d62728", linestyle=":", alpha=0.7, label="60% Target")
    ax1.axhline(70.0, color="#9467bd", linestyle=":", alpha=0.7, label="70% Target")
    ax1.set_xlabel("Communication Round ($t$)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("(a) Convergence Rate across Rounds")
    ax1.set_xlim([1, 25])
    ax1.set_ylim([0, 100])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # (b) Time-to-Accuracy bar chart for 60% target
    target_60_rows = [r for r in summary_rows if r["Target Acc (%)"] == 60.0]
    m_names = [r["Method"] for r in target_60_rows]
    m_times = []
    for r in target_60_rows:
        val = r["Simulated Time (s)"]
        if "±" in val:
            m_times.append(float(val.split("±")[0].strip()))
        else:
            m_times.append(0.0)

    bars = ax2.bar(m_names, m_times, color=[colors[m] for m in m_names], edgecolor="#222", alpha=0.85, width=0.45)
    ax2.set_xlabel("Federated Learning Baseline")
    ax2.set_ylabel("Simulated Wall-Clock Seconds to 60% Acc")
    ax2.set_title("(b) Wall-Clock Time to 60% Accuracy Target")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., h + 1.0, f"{h:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_png = PLOTS_DIR / "fig_time_to_accuracy.png"
    plt.savefig(out_png, dpi=300)
    plt.savefig(PROJECT_ROOT / "docs" / "figures" / "fig_time_to_accuracy.png", dpi=300)
    plt.savefig(PROJECT_ROOT / "docs" / "fig_time_to_accuracy.png", dpi=300)
    plt.close()
    print(f"[Generated Time-to-Accuracy Figure] -> {out_png}")


if __name__ == "__main__":
    analyze_time_to_accuracy()
