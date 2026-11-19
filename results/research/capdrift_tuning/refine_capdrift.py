"""Paired development-only neighborhood search around a frozen first-stage winner."""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
import run_capdrift_tuning as tuning

bench = tuning.bench
torch = tuning.torch
d = root / "results/research/capdrift_tuning"
prior = json.loads((d / "dev_tuning_results.json").read_text(encoding="utf-8"))
manifest_path = d / "selected_tuning_config.json"
frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
protocol = frozen["provenance"]
assert frozen["name"] == "short-memory"
config = bench.CapDriftConfig(**frozen["controller_config"])
variants = {
    "short-memory-early-taper": replace(config, server_lr_decay_start_round=7,
        server_lr_final_ratio=0.05, server_average_decay=0.85, server_average_start_round=8),
    "short-memory-tempered": replace(config, client_update_normalization_power=0.35),
}
pairs = list(zip(protocol["finalists"]["folds"], protocol["finalists"]["seeds"]))
args = SimpleNamespace(dev_fraction=protocol["dev_fraction"], alpha=protocol["dirichlet_alpha"],
                       partition_seed=protocol["partition_seed"])
rounds = protocol["finalists"]["rounds"]
tail = protocol["ranking"]["tail_rounds"]
lr = frozen["local_lr"]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
dataset, _, dataset_hash = bench.load_dataset(tuning.resolve_data_dir(None))
assert dataset_hash == protocol["data_sha256"]
outer_train, holdout = bench.split_indices(dataset, policy="filename-group",
        seed=protocol["outer_split_seed"], holdout_fraction=protocol["outer_fraction"])
assert tuning.fingerprint([outer_train, holdout]) == protocol["outer_split_sha256"]
folds = {fold: tuning.split_for_fold(dataset, outer_train, fold, args) for fold, _ in pairs}
prior_folds = {f["fold_seed"]: f for f in protocol["dev_folds"]}
assert all(row[3] == prior_folds[f] for f, row in folds.items())
tiers = [bench.TIER_PROFILES[i] for i in bench.TIER_IDS]
reference_round = protocol["reference_round_sec"]

# Freeze thresholds before running the variants, and never read outer scores here.
rules = {"score_improvement_strictly_over": 0.003,
         "allowed_accuracy_regression_pp": 0.5, "max_per_fold_ce_regression": 0.02}
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out = d / "refinement_results.json"
winner_out = d / "selected_refinement_config.json"
refinement = {
    "protocol": protocol, "source_sha256": source_hash, "rules": rules,
    "configs": {name: asdict(item) for name, item in variants.items()}, "trials": {},
}
if out.exists():
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert all(saved[k] == refinement[k] for k in ("protocol", "source_sha256", "rules", "configs"))
    refinement = saved
else:
    bench.write_json(out, refinement)
reference = [prior["finalist_trials"][f"short-memory/{f}/{s}"] for f, s in pairs]
ref_row = tuning.scores(reference, tail)
rows = {frozen["name"]: ref_row}
eligible = []
for name, cfg in variants.items():
    runs = []
    for fold, seed in pairs:
        key = f"{name}/{fold}/{seed}"
        if key in refinement["trials"]:
            trial = refinement["trials"][key]
        else:
            data, validation, clients, _ = folds[fold]
            print(f"refinement | {name} | fold={fold} seed={seed} rounds={rounds}", flush=True)
            try:
                trial = bench.run_competitor_trial(
                    "CapDrift-FL", data, validation, clients, tiers, num_rounds=rounds,
                    config=cfg, seed=seed, device=device, rng_policy="paired",
                    freeze_bn=True, channels_last=True, fused_sgd=True, local_lr=lr,
                    baseline_local_lr=lr, schedule_total_rounds=rounds)
            except bench.NonFiniteMetricError as exc:
                trial = {"failure": str(exc), "seed": seed, "round": exc.round_idx}
            refinement["trials"][key] = trial
            bench.write_json(out, refinement)
        if "history" in trial:
            tuning.validate_trial(trial, rounds, tiers, True, reference_round)
        runs.append(trial)
    if not all("history" in t for t in runs):
        continue
    row = rows[name] = tuning.scores(runs, tail)
    row["score_delta"] = row["loss_score"] - ref_row["loss_score"]
    row["accuracy_delta_pp"] = row["mean_final_accuracy"] - ref_row["mean_final_accuracy"]
    row["per_fold_final_ce_delta"] = [a["final_loss"] - b["final_loss"] for a, b in zip(runs, reference)]
    if (row["score_delta"] < -rules["score_improvement_strictly_over"]
            and row["accuracy_delta_pp"] >= -rules["allowed_accuracy_regression_pp"]
            and max(row["per_fold_final_ce_delta"]) <= rules["max_per_fold_ce_regression"]):
        eligible.append(name)
refinement["scores"] = rows
selected = min(eligible, key=lambda name: tuning.rank_key(name, rows[name])) if eligible else frozen["name"]
refinement["selected"] = selected
bench.write_json(out, refinement)
if selected != frozen["name"]:
    bench.write_json(winner_out, {
        "name": selected, "controller_config": asdict(variants[selected]), "local_lr": lr,
        "provenance": {**protocol, "dev_scores": rows[selected],
                       "refinement": {"parent": frozen["name"], "rules": rules,
                                      "source_sha256": source_hash}},
    })
print(json.dumps({"selected": selected, "scores": rows}, indent=2), flush=True)
