"""Development-only challenger: blend FedNova step-mass normalization with CapDrift."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import os
import sys
from types import SimpleNamespace

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
import run_capdrift_tuning as tuning
bench, torch = tuning.bench, tuning.torch

name = "projected-momentum-mass"
d = root / "results/research/capdrift_tuning"
prior = json.loads((d / "refinement_results.json").read_text(encoding="utf-8"))
winner = json.loads((d / "selected_refinement_config.json").read_text(encoding="utf-8"))
base = bench.CapDriftConfig(**winner["controller_config"])
cfg = replace(base, budget_lr_power=0.25, server_lr=0.75, server_momentum=0.0, kappa_c=2.0)

class StepMassCoordinator(bench.CapDriftCoordinator):
    """Normalize effective SGDM coefficients, then clip and project as usual."""
    def aggregate_round(self, server_round, global_model, client_results, sample_counts, total_rounds=None):
        observed = [(res, bench.fednova_momentum_normalizer(res["steps_completed"], bench.LOCAL_MOMENTUM)
                     * res["lr_scale"]) for res in client_results]
        n = sum(sample_counts[res["client_id"]] for res, mass in observed if mass > 0)
        a_eff = sum(sample_counts[res["client_id"]] * mass for res, mass in observed) / n if n else 1
        normalized = [
            {**res, "delta_i": [delta * (a_eff / mass if mass > 0 else 0.0) for delta in res["delta_i"]]}
            for res, mass in observed
        ]
        return super().aggregate_round(server_round, global_model, normalized, sample_counts, total_rounds)

bench.CapDriftCoordinator = StepMassCoordinator
proto = winner["provenance"]
pairs = list(zip(proto["finalists"]["folds"], proto["finalists"]["seeds"]))
args = SimpleNamespace(dev_fraction=proto["dev_fraction"], alpha=proto["dirichlet_alpha"],
                       partition_seed=proto["partition_seed"])
rounds, tail, lr = proto["finalists"]["rounds"], proto["ranking"]["tail_rounds"], winner["local_lr"]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
dataset, _, fp = bench.load_dataset(tuning.resolve_data_dir(None))
assert fp == proto["data_sha256"]
training, held = bench.split_indices(dataset, policy="filename-group",
    seed=proto["outer_split_seed"], holdout_fraction=proto["outer_fraction"])
assert tuning.fingerprint([training, held]) == proto["outer_split_sha256"]
layers = [bench.TIER_PROFILES[i] for i in bench.TIER_IDS]
out = d / "momentum_mass_challenger.json"
record = {"name": name, "selected_parent": winner["name"], "controller_config": asdict(cfg),
          "outer_split_sha256": proto["outer_split_sha256"], "unpublished_prototype": True,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": [],
          "selection_rules": prior["rules"]}
bench.write_json(out, record)
for f, s in pairs:
    data, dev, clients, fold = tuning.split_for_fold(dataset, training, f, args)
    assert fold in proto["dev_folds"]
    print(f"Step-mass prototype, fold {f}, seed {s}", flush=True)
    try:
        test = bench.run_competitor_trial("CapDrift-FL", data, dev, clients, layers, num_rounds=rounds,
            config=cfg, seed=s, device=device, rng_policy="paired", freeze_bn=True,
            channels_last=True, fused_sgd=True, local_lr=lr, baseline_local_lr=lr, schedule_total_rounds=rounds)
        tuning.validate_trial(test, rounds, layers, True, proto["reference_round_sec"])
    except bench.NonFiniteMetricError as err:
        test = {"failure": str(err), "round": err.round_idx, "seed": s}
    record["trials"].append({"fold_seed": f, **test})
    bench.write_json(out, record)
if all("history" in t for t in record["trials"]):
    score = tuning.scores(record["trials"], tail)
    previous = prior["scores"][winner["name"]]
    score["score_delta"] = score["loss_score"] - previous["loss_score"]
    score["accuracy_delta_pp"] = score["mean_final_accuracy"] - previous["mean_final_accuracy"]
    score["per_fold_final_ce_delta"] = [trial["final_loss"] - prior["trials"][f"{winner['name']}/{f}/{s}"]["final_loss"]
        for trial, (f, s) in zip(record["trials"], pairs)]
    rules = record["selection_rules"]
    record["promote"] = (score["score_delta"] < -rules["score_improvement_strictly_over"]
        and score["accuracy_delta_pp"] >= -rules["allowed_accuracy_regression_pp"]
        and max(score["per_fold_final_ce_delta"]) <= rules["max_per_fold_ce_regression"])
    record["score"] = score
else:
    record["promote"] = False
bench.write_json(out, record)
print(json.dumps({key: value for key, value in record.items() if key not in ("controller_config", "trials")}, indent=2))
