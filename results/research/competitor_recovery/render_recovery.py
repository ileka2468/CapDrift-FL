"""Render the audited recovery reports without touching any training artifacts."""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
reports = {split: json.loads((root / f"report_{split}.json").read_text(encoding="utf-8"))
           for split in ("target", "source")}
assert reports["target"]["selected_config_sha256"] == reports["source"]["selected_config_sha256"]
rows = []
labels = {"v3-anchor": "Frozen CapDrift", "prior-progress": "Prior-progress",
          "FedNova-Momentum": "FedNova-Momentum", "FedNova": "FedNova"}
for split, report in reports.items():
    for method, summary in report["methods"].items():
        means = summary["all_seed_mean"] or {}
        rows.append({
            "holdout": split, "method": method, "completed": summary["completed"],
            "attempted": summary["attempted"], "mean_final_accuracy_pct": means.get("final_accuracy"),
            "mean_final_loss": means.get("final_loss"), "modeled_seconds_per_completed_run": report["modeled_seconds"],
        })
with (root / "recovery_accuracy_summary.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), layout="constrained", sharey=True)
references = ["v3-anchor", "FedNova", "FedNova-Momentum"]
colors = ["#13746c", "#3766a4", "#b65d23"]
for ax, (split, report) in zip(axes, reports.items()):
    for reference, color in zip(references, colors):
        pair = report["paired"][reference]
        if pair["seeds"]:
            ax.plot(range(len(pair["seeds"])), pair["accuracy_delta_pp"], marker="o", color=color,
                    linewidth=1.7, label=f"vs. {labels[reference]}")
            ax.set_xticks(range(len(pair["seeds"])), pair["seeds"])
    ax.axhline(0, color="#666666", lw=1, zorder=0)
    ax.set(xlabel="Paired seed", title="Original holdout (153 images)" if split == "target"
           else "Source holdout (423 images)")
    ax.grid(axis="y", alpha=0.23)
axes[0].set_ylabel("Challenger minus reference accuracy (pp)")
axes[1].legend(loc="best", frameon=True)
fig.suptitle("CapDrift recovery: paired round-25 accuracy differences", fontsize=14)
fig.savefig(root / "paired_accuracy.png", dpi=200)
plt.close(fig)
