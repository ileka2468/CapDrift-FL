"""Audit all attempted seeds without hiding or imputing a failed trial."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys

out = Path(sys.argv[1])
methods = ("CapDrift-Accuracy", "FedNova-Momentum", "FedNova")
accuracy_label = methods[0]
seed_ids = ["42", "101", "2024", "7", "888"]
read = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
hash_file = lambda name: hashlib.sha256((out / name).read_bytes()).hexdigest()
selected = read("selected_config.json")
assert hash_file("development.json") == selected["development_sha256"]
protocol = selected["protocol"]
anchor = protocol["round_duration_sec"]
assert math.isclose(anchor * 25, 205.6197213692418, rel_tol=0, abs_tol=1e-8)
n = protocol["holdout"]["evaluated"]
counts = list(protocol["holdout"]["class_counts"].values())
assert sum(counts) == n
manifest = hash_file("selected_config.json")

records, failures = {}, []
partition = None
audited_rounds = 0
max_compute = 0.0
for method in methods:
    result = read(f"holdout_{method}.json")
    identity = dict(result["protocol"])
    assert identity.pop("manifest_sha256") == manifest and result["method"] == method
    this_partition = identity.pop("partition_sha256")
    assert identity == protocol and partition in (None, this_partition)
    partition = this_partition
    assert list(result["trials"]) == seed_ids
    records[method] = result["trials"]
    for seed, record in records[method].items():
        assert record["seed"] == int(seed)
        if "failure" in record:
            assert "history" not in record and 1 <= record["failed_round"] <= 25
            failures.append({"method": method, "seed": int(seed), "failed_round": record["failed_round"],
                             "failure": record["failure"]})
            continue
        hist = record["history"]
        assert [r["round"] for r in hist] == list(range(1, 26))
        assert math.isclose(record["total_time_sec"], anchor * 25, abs_tol=1e-8)
        for i, vals in enumerate(hist, 1):
            assert list(vals["steps_per_client"].values()) == [40, 28, 16, 5]
            max_compute = max(max_compute, *vals["compute_sec_per_client"].values())
            assert max_compute <= 6.5
            assert math.isclose(max(vals["client_duration_sec"].values()), anchor, abs_tol=1e-8)
            assert math.isclose(vals["round_duration_sec"], anchor, abs_tol=1e-8)
            assert math.isclose(vals["cumulative_time_sec"], i * anchor, abs_tol=1e-8)
            assert math.isclose(vals["cluster_utilization_pct"], 25 * sum(vals["client_duration_sec"].values())
                                / anchor, abs_tol=1e-8)
            assert math.isfinite(vals["val_loss"]) and 0 <= vals["val_accuracy"] <= 100
        final = hist[-1]
        assert final["class_sample_counts"] == counts
        hits = final["class_correct_counts"]
        assert all(0 <= k <= total for k, total in zip(hits, counts))
        assert math.isclose(record["final_accuracy"], 100 * sum(hits) / n, abs_tol=1e-8)
        assert math.isclose(record["final_balanced_accuracy"],
                            50 * sum(k / total for k, total in zip(hits, counts)), abs_tol=1e-8)
        assert record["final_loss"] == final["val_loss"]
        audited_rounds += len(hist)
common = [seed for seed in seed_ids if all("history" in records[m][seed] for m in methods)]
mean = lambda trials, key: statistics.mean(t[key] for t in trials)
summaries = []
for method in methods:
    match = [records[method][seed] for seed in common]
    successful = [r for r in records[method].values() if "history" in r]
    summaries.append({"method": method, "completed": len(successful), "attempted": len(seed_ids),
        "five_seed_final_accuracy_pct": mean(successful, "final_accuracy") if len(successful) == 5 else None,
        "matched_seed_final_accuracy_pct": mean(match, "final_accuracy"),
        "matched_seed_final_balanced_accuracy_pct": mean(match, "final_balanced_accuracy"),
        "matched_seed_final_loss": mean(match, "final_loss"),
        "mean_modeled_time_sec_completed": mean(successful, "total_time_sec"),
        "mean_active_utilization_pct_completed": mean(successful, "mean_utilization_pct")})
pairs = {}
for m in methods[1:]:
    deltas = {seed: records[accuracy_label][seed]["final_accuracy"] - records[m][seed]["final_accuracy"]
              for seed in common}
    pairs[m] = {"paired_accuracy_delta_pp": deltas,
                "mean_delta_pp": statistics.mean(deltas.values()),
                "paired_seeds_higher": sum(x > 0 for x in deltas.values())}
outcome = {
    "qualified_accuracy_win": not failures and all(x["mean_delta_pp"] > 0 for x in pairs.values()),
    "selected_variant": selected["name"], "holdout": protocol["holdout"], "attempted_trials": 15,
    "completed_trials": sum(x["completed"] for x in summaries),
    "audited_completed_rounds": audited_rounds, "max_modeled_client_compute_sec": max_compute,
    "matched_seeds": list(map(int, common)), "failures": failures, "summary": summaries,
    "comparisons_over_completed_pairs_only": pairs, "frozen_manifest_sha256": manifest,
}
with (out / "attempts_audit.json").open("w", encoding="utf-8") as f:
    json.dump(outcome, f, indent=2, allow_nan=False)
    f.write("\n")
with (out / "attempts_summary.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summaries[0]))
    w.writeheader()
    w.writerows(summaries)
print(json.dumps(outcome, indent=2))

# Preserve the failed attempt as a visible gap, not as a zero or a deleted seed.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
for index, method in enumerate(methods):
    scores = [records[method][s].get("final_accuracy", float("nan")) for s in seed_ids]
    ax.plot(range(len(seed_ids)), scores, marker=("o", "s", "^")[index],
            color=("#007c69", "#d65b13", "#2570a4")[index], label=method)
for failure in failures:
    ax.annotate(f"{failure['method']}\nNonfinite loss, round {failure['failed_round']}",
                (seed_ids.index(str(failure["seed"])), 85.7),
                ha="center", va="bottom", color="#b22534", fontsize=9)
ax.set(title=f"Final accuracy on the reserved {n}-image source-test holdout",
       xlabel="Paired optimizer seed", ylabel="Round-25 accuracy (%)", ylim=(85, 93))
ax.set_xticks(range(len(seed_ids)), seed_ids)
ax.legend(loc="upper left", ncol=3, fontsize=8)
ax.grid(axis="y", alpha=0.25)
fig.savefig(out / "accuracy_by_seed.png", dpi=220)
plt.close(fig)
