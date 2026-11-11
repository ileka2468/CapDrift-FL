"""
Plotting Script for Multi-Scale Physical Proxmox Cluster Benchmark (N = 5, 10, 20).
Generates publication-quality IEEE figure with strictly separated measurement units:
fig_cluster_multinode_scaling.png:
- (a) Multi-Scale Execution-Time Utilization (%) across N = 5, 10, 20 real LXC containers.
- (b) Straggler Synchronization Idle Waste (seconds): Cumulative Idle Waste vs. Mean Idle Waste per Client.
- (c) Virtualization Contention: Per-Host Execution Throughput Degradation under Multi-Tenant CPU sharing (steps/s).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR = PROJECT_ROOT / "docs" / "experiment_logs"
PLOTS_DIR = PROJECT_ROOT / "results" / "research" / "plots"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "lines.linewidth": 1.8
})


def generate_scaling_plots():
    p5 = CLUSTER_DIR / "cluster_telemetry_capdrift_5clients.json"
    p10 = CLUSTER_DIR / "cluster_telemetry_capdrift_10clients.json"
    p20 = CLUSTER_DIR / "cluster_telemetry_capdrift_20clients.json"

    if not (p5.exists() and p10.exists() and p20.exists()):
        print("Missing cluster telemetry files.")
        return

    with open(p5, "r", encoding="utf-8") as f:
        d5 = json.load(f)
    with open(p10, "r", encoding="utf-8") as f:
        d10 = json.load(f)
    with open(p20, "r", encoding="utf-8") as f:
        d20 = json.load(f)

    # Use Round 2 (adapted capability stage)
    runs = [("N = 5 Clients", d5[1]), ("N = 10 Clients", d10[1]), ("N = 20 Clients", d20[1])]
    client_counts = [5, 10, 20]
    scales = ["5 Clients\n(2/2/1)", "10 Clients\n(4/3/3)", "20 Clients\n(8/6/6)"]

    round_times = [r[1]["round_duration_sec"] for r in runs]
    utilizations = [r[1]["cluster_utilization_pct"] for r in runs]
    cumulative_idle_wastes = [r[1]["straggler_waste_sec"] for r in runs]
    mean_idle_wastes = [w / c for w, c in zip(cumulative_idle_wastes, client_counts)]

    # 3 Separate Subplots to completely eliminate mixed units
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    x = np.arange(len(scales))
    width = 0.45

    # Subplot (a): Execution-Time Utilization (%)
    bars_u = ax1.bar(x, utilizations, width, color="#2ca02c", edgecolor="#145214", linewidth=1.2)
    ax1.set_xlabel("Physical Cluster Scale (LXC Containers)")
    ax1.set_ylabel("Execution-Time Utilization (%)")
    ax1.set_title("(a) Execution-Time Utilization (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.set_ylim([0, 115])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_u:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot (b): Straggler Idle Waiting - Mean Idle Waiting per Client (seconds)
    # Displaying mean idle waiting on primary axis so differences across scales are prominent
    bars_mean = ax2.bar(x, mean_idle_wastes, width, color="#ff9f43", edgecolor="#c25e00", linewidth=1.2)
    ax2.set_xlabel("Physical Cluster Scale (LXC Containers)")
    ax2.set_ylabel("Mean Idle Waiting per Client (seconds)")
    ax2.set_title("(b) Mean Straggler Idle Waiting per Client (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scales)
    ax2.set_ylim([0, 4.0])
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_mean:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#8a4300")

    # Subplot (c): Virtualization Contention: Mean Throughput per Host
    hosts = ["pve01\n(Dual E5-2680)", "pve02\n(Dual E5-2620)", "pve03\n(Dual E5-2620)"]
    host_rates_5 = []
    host_rates_10 = []
    host_rates_20 = []

    for host_tag in ["pve01", "pve02", "pve03"]:
        rates5 = [m["throughput"] for m in d5[1]["clients"] if host_tag in m["node_id"]]
        rates10 = [m["throughput"] for m in d10[1]["clients"] if host_tag in m["node_id"]]
        rates20 = [m["throughput"] for m in d20[1]["clients"] if host_tag in m["node_id"]]
        host_rates_5.append(np.mean(rates5) if rates5 else 0.0)
        host_rates_10.append(np.mean(rates10) if rates10 else 0.0)
        host_rates_20.append(np.mean(rates20) if rates20 else 0.0)

    xh = np.arange(len(hosts))
    wh = 0.25

    b1 = ax3.bar(xh - wh, host_rates_5, wh, label="N = 5 (Low Contention)", color="#4a90e2", edgecolor="#2161ad")
    b2 = ax3.bar(xh, host_rates_10, wh, label="N = 10 (Moderate Contention)", color="#54a0ff", edgecolor="#1a5db0")
    b3 = ax3.bar(xh + wh, host_rates_20, wh, label="N = 20 (High Contention)", color="#a29bfe", edgecolor="#5c53c4")

    ax3.set_xlabel("Physical Cisco UCS Fog Host")
    ax3.set_ylabel("Training Throughput (steps/s)")
    ax3.set_title("(c) Multi-Tenant Host Throughput Contention")
    ax3.set_xticks(xh)
    ax3.set_xticklabels(hosts)
    ax3.set_ylim([0, 2.6])
    ax3.legend(loc="upper right")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in b3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path1 = PLOTS_DIR / "fig_cluster_multinode_scaling.png"
    out_path2 = FIGURES_DIR / "fig_cluster_multinode_scaling.png"
    plt.savefig(out_path1, dpi=300, bbox_inches="tight")
    plt.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Generated Multi-Scale Cluster Plot] -> {out_path1}")
    print(f"[Generated Multi-Scale Cluster Plot] -> {out_path2}")


if __name__ == "__main__":
    generate_scaling_plots()
