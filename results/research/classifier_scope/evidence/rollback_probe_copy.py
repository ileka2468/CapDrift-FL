"""Accuracy-first ablations rooted in the frozen 94.38% CapDrift-Tuned run.

Only nested development results can select a challenger. The already-reported
153-image comparison is a fixed regression target, not a fresh test. The
423-image source split is replayed separately, without selecting on its scores.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import run_capdrift_accuracy as accuracy
import run_capdrift_tuning as tuning

bench = tuning.bench
torch = bench.torch
ROOT = bench.PROJECT_ROOT
RESULTS = ROOT / "results" / "research"
BASELINE_MANIFEST = RESULTS / "capdrift_tuning" / "selected_refinement_config.json"
TARGET_ARCHIVE = RESULTS / "competitor_tuning_final"
SOURCE_ARCHIVE = RESULTS / "competitor_accuracy"
BASELINE = "v3-anchor"
RIVALS = ("FedNova-Momentum", "FedNova")
SEEDS = (42, 101, 2024, 7, 888)
ROUNDS = 25
STEPS = (40, 28, 16, 5)
SPLIT_SEED = 42
PARTITION_SEED = 42
ALPHA = 0.5
DEV = ((2309, 42, 42), (5171, 2024, 314))  # group split, optimizer, client partition
TOP = 2


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_baseline():
    manifest = read_json(BASELINE_MANIFEST)
    target = read_json(TARGET_ARCHIVE / "competitor_benchmark_metadata.json")
    digest = sha256(BASELINE_MANIFEST)
    if (manifest["name"] != "short-memory-early-taper"
            or digest != target["capdrift_tuned_manifest_sha256"]
            or manifest["controller_config"] != target["controller_configs"]["CapDrift-Tuned"]
            or manifest["local_lr"] != target["local_optimizer"]["capdrift_tuned_lr"]):
        raise ValueError("baseline is not the frozen CapDrift-Tuned configuration")
    return bench.CapDriftConfig(**manifest["controller_config"]), manifest["local_lr"], digest, target


def candidates(base):
    """Keep all unrelated controller settings unchanged for each ablation."""
    return {
        BASELINE: base,
        "taper-floor": replace(base, server_lr_final_ratio=0.25),
        "later-taper": replace(base, server_lr_decay_start_round=13, server_lr_final_ratio=0.20),
        "prior-adjusted": replace(base, local_prior_strength=1.0),
        "tempered-mass": replace(base, gradient_mass_blend=0.35),
        "prior-progress": replace(
            base, local_prior_strength=0.75, budget_lr_power=0.25, gradient_mass_blend=0.25,
            server_lr_decay_start_round=10, server_lr_final_ratio=0.15,
        ),
    }


@dataclass
class Study:
    output: Path
    plan: dict
    dataset: object
    train: list
    held: list
    clients: list
    tiers: list
    device: object
    configs: dict
    source_loader: object = None


def target_identity(metadata, dataset_hash, train, held, clients, payload, anchor, environment):
    expected = {
        "dataset_content_sha256": dataset_hash, "split_policy": "filename-group",
        "split_seed": SPLIT_SEED, "holdout_fraction": 0.20, "partition_seed": PARTITION_SEED,
        "outer_split_sha256": bench.split_fingerprint(train, held),
        "split_partition_sha256": hashlib.sha256(
            json.dumps([train, held, clients]).encode("utf-8")).hexdigest(),
        "train_samples": len(train), "val_samples": len(held), "seeds": list(SEEDS),
        "num_rounds": ROUNDS, "alpha": ALPHA, "rng_policy": "paired",
        "model": "mobilenet_v3_small",
        "weights": bench.torchvision.models.MobileNet_V3_Small_Weights.DEFAULT.name,
        "model_payload_mib": payload, "training_batch_size": bench.BATCH_SIZE,
        "modeled_reference_round_duration_sec": anchor,
        "tier_ids": list(bench.TIER_IDS),
        "tiers": [asdict(bench.TIER_PROFILES[i]) for i in bench.TIER_IDS],
        "torch_version": environment["torch"], "torchvision_version": environment["torchvision"],
        "freeze_bn": True, "channels_last": True, "deterministic_algorithms": True,
    }
    mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
    opt = metadata.get("local_optimizer", {})
    if (opt.get("name"), opt.get("momentum"), opt.get("weight_decay"),
            opt.get("baseline_lr"), opt.get("capdrift_tuned_lr")) != (
            "SGD", bench.LOCAL_MOMENTUM, bench.WEIGHT_DECAY,
            environment["local_lr"], environment["local_lr"]):
        mismatch.append("local_optimizer")
    if mismatch:
        raise ValueError(f"153-image historical protocol mismatch: {', '.join(mismatch)}")
    # CPU smoke runs can validate mechanics, not historical replay or a qualification claim.
    return (metadata["device"] == environment["device"]
            and metadata.get("device_name") == environment["device_name"]
            and opt.get("fused") == environment["fused_sgd"])


def prepare(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data_dir = tuning.resolve_data_dir(args.data_dir)
    src = args.source_test.resolve(strict=True)
    decode = args.stage == "evaluate" and args.holdout == "source"
    source_meta, source_loader = accuracy.source_holdout(data_dir, src, decode=decode)
    dataset, classes, data_hash = bench.load_dataset(data_dir)
    train, held = bench.split_indices(dataset, policy="filename-group", seed=SPLIT_SEED)
    test_groups = {dataset.group_ids[i] for i in held}
    if set(train).intersection(held) or test_groups.intersection(dataset.group_ids[i] for i in train):
        raise ValueError("filename-group development/evaluation overlap")
    train_set = bench.Subset(dataset, train)
    train_set.targets = dataset.targets[train]
    local = bench.partition_dataset_dirichlet(
        train_set, num_clients=len(bench.TIER_IDS), alpha=ALPHA, seed=PARTITION_SEED)
    if any(not c for c in local) or sorted(i for c in local for i in c) != list(range(len(train))):
        raise ValueError("invalid outer-training partition")
    clients = [[train[i] for i in c] for c in local]
    tiers = [bench.TIER_PROFILES[i] for i in bench.TIER_IDS]
    probe = bench.get_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    payload = bench.get_model_size_mb(probe)
    anchor = bench.reference_round_duration(tiers, payload)
    config, local_lr, baseline_hash, metadata = frozen_baseline()
    env = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__), "torchvision": str(bench.torchvision.__version__),
        "fused_sgd": device.type == "cuda", "local_lr": local_lr,
        "freeze_bn": True, "channels_last": True, "rng_policy": "paired",
    }
    replay = target_identity(metadata, data_hash, train, held, clients, payload, anchor, env)
    code = [Path(__file__), Path(bench.__file__), Path(tuning.__file__), Path(accuracy.__file__)]
    code += [ROOT / "src" / "research" / f for f in
             ("capdrift_controller.py", "competitor_split.py", "non_iid.py", "heterogeneous_fleet.py")]
    plan = {
        "schema": 1, "baseline_manifest_sha256": baseline_hash, "dataset_sha256": data_hash,
        "training_samples": len(train), "target_samples": len(held),
        "target_classes": {label: int((dataset.targets[held] == i).sum()) for i, label in enumerate(classes)},
        "target_split_sha256": bench.split_fingerprint(train, held),
        "target_partition_sha256": metadata["split_partition_sha256"],
        "train_index_sha256": tuning.fingerprint(train),
        "local_partition_sha256": tuning.fingerprint([train, local]),
        "source": source_meta, "seeds": list(SEEDS), "rounds": ROUNDS, "steps": list(STEPS),
        "dev_scenarios": [list(item) for item in DEV], "top": TOP, "round_seconds": anchor,
        "environment": env, "historical_environment_match": replay,
        "configs": {k: asdict(v) for k, v in candidates(config).items()},
        "code_sha256": {f.name: sha256(f) for f in code},
        "archived_target_metadata_sha256": sha256(TARGET_ARCHIVE / "competitor_benchmark_metadata.json"),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.is_file():
        if read_json(plan_path) != plan:
            raise ValueError("study plan changed; use a fresh output directory")
    else:
        bench.write_json(plan_path, plan)
    return Study(output, plan, dataset, train, held, clients, tiers, device, candidates(config), source_loader)


def audit_trial(trial, method, seed, tiers, anchor):
    if trial.get("seed") != seed or trial.get("method") != method:
        raise ValueError("trial identity mismatch")
    if "failure" in trial:
        if "history" in trial or not 1 <= trial.get("failed_round", 0) <= ROUNDS:
            raise ValueError("invalid failed trial")
        return
    tuning.validate_trial(trial, ROUNDS, tiers, True, anchor)
    last = trial["history"][-1]
    if (trial["final_accuracy"] != last["val_accuracy"] or trial["final_loss"] != last["val_loss"]
            or not 0 <= trial["final_accuracy"] <= 100 or not math.isfinite(trial["final_loss"])):
        raise ValueError("trial endpoint does not match valid round-25 telemetry")


def run_trial(study, method, dataset, loader, clients, seed, config, *, class_metrics=False):
    backend_method = "CapDrift-Tuned" if config is not None else method
    lr = study.plan["environment"]["local_lr"]
    print(f"Recovery | {method} | seed={seed} | full {ROUNDS} rounds", flush=True)
    try:
        trial = bench.run_competitor_trial(
            backend_method, dataset, loader, clients, study.tiers, num_rounds=ROUNDS,
            config=config, seed=seed, device=study.device, local_lr=lr, baseline_local_lr=lr,
            rng_policy="paired", freeze_bn=True, channels_last=True,
            fused_sgd=study.plan["environment"]["fused_sgd"], measure_balanced_accuracy=class_metrics,
        )
    except bench.NonFiniteMetricError as exc:
        trial = {"seed": seed, "failed_round": exc.round_idx, "failure": str(exc)}
    trial["method"] = method
    audit_trial(trial, method, seed, study.tiers, study.plan["round_seconds"])
    return trial


def dev_path(output, fold, case):
    return output / f"dev_{fold}_{case}.json"


def dev_result(study, fold, case):
    data = read_json(dev_path(study.output, fold, case))
    if data["plan_sha256"] != sha256(study.output / "plan.json") or data["fold"] != fold:
        raise ValueError("development plan mismatch")
    audit_trial(data["trial"], case, DEV[fold][1], study.tiers, study.plan["round_seconds"])
    return data["trial"]


def metrics(trials):
    return {
        "final_accuracy": statistics.mean(t["final_accuracy"] for t in trials),
        "final_loss": statistics.mean(t["final_loss"] for t in trials),
    }


def rank_key(name, values):
    return (-values["final_accuracy"], values["final_loss"], name)


def promote(study):
    observed = {n: dev_result(study, 0, n) for n in study.configs}
    scores = {n: metrics([t]) for n, t in observed.items() if "history" in t}
    names = sorted((n for n in scores if n != BASELINE), key=lambda n: rank_key(n, scores[n]))[:TOP]
    if not names or BASELINE not in scores:
        raise RuntimeError("screen requires a finite frozen baseline and at least one finite challenger")
    record = {
        "plan_sha256": sha256(study.output / "plan.json"), "challengers": names, "scores": scores,
        "screen_inputs": {n: sha256(dev_path(study.output, 0, n)) for n in study.configs},
    }
    path = study.output / "finalists.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("screen changed after promotion")
    if not path.exists():
        bench.write_json(path, record)
    print(f"Promoted on round-25 development accuracy: {names}", flush=True)
    return names


def screen(study, fold, case):
    names = list(study.configs) if fold == 0 else [BASELINE, *promote(study)]
    if case is not None:
        if case not in names:
            raise ValueError(f"{case!r} is not eligible in development fold {fold}")
        names = [case]
    group_seed, opt_seed, partition_seed = DEV[fold]
    nested = SimpleNamespace(alpha=ALPHA, dev_fraction=0.20, partition_seed=partition_seed)
    data, loader, clients, identity = tuning.split_for_fold(
        study.dataset, study.train, group_seed, nested)
    for name in names:
        path = dev_path(study.output, fold, name)
        if path.exists():
            if read_json(path)["split"] != identity:
                raise ValueError("saved development split changed")
            dev_result(study, fold, name)
            print(f"Reused {path.name}", flush=True)
            continue
        trial = run_trial(study, name, data, loader, clients, opt_seed, study.configs[name])
        bench.write_json(path, {"trial": trial, "fold": fold, "split": identity,
                                "plan_sha256": sha256(study.output / "plan.json")})


def select(study):
    names = [BASELINE, *promote(study)]
    attempts = {n: [dev_result(study, i, n) for i in range(len(DEV))] for n in names}
    scores = {n: metrics(rows) for n, rows in attempts.items() if all("history" in t for t in rows)}
    challengers = [n for n in scores if n != BASELINE]
    if BASELINE not in scores or not challengers:
        raise RuntimeError("no stable two-fold challenger and baseline for selection")
    winner = min(challengers, key=lambda n: rank_key(n, scores[n]))
    delta = scores[winner]["final_accuracy"] - scores[BASELINE]["final_accuracy"]
    frozen = {
        "name": winner, "controller_config": asdict(study.configs[winner]), "development": scores,
        "development_accuracy_delta_pp": delta, "exploratory_only": delta <= 0,
        "plan_sha256": sha256(study.output / "plan.json"),
        "trial_sha256": {dev_path(study.output, i, n).name: sha256(dev_path(study.output, i, n))
                         for n in names for i in range(len(DEV))},
    }
    path = study.output / "selected_config.json"
    if path.exists() and read_json(path) != frozen:
        raise ValueError("development selection changed after freezing")
    if not path.exists():
        bench.write_json(path, frozen)
    print(f"Frozen {winner}; two-fold accuracy delta {delta:+.3f} pp vs tuned baseline", flush=True)


def frozen_selection(output):
    frozen = read_json(output / "selected_config.json")
    if (frozen["plan_sha256"] != sha256(output / "plan.json")
            or any(sha256(output / path) != digest for path, digest in frozen["trial_sha256"].items())):
        raise ValueError("frozen selection or development input changed")
    return frozen


def evaluate(study, holdout, method):
    if holdout == "target":
        loader = bench.DataLoader(bench.Subset(study.dataset, study.held), batch_size=32, shuffle=False)
    else:
        loader = study.source_loader
    role = [BASELINE, "challenger"] if method == "all" else [BASELINE if method == "tuned" else "challenger"]
    for item in role:
        if item == BASELINE:
            name, cfg, selection_sha = BASELINE, study.configs[BASELINE], None
        else:
            frozen = frozen_selection(study.output)
            name = frozen["name"]
            cfg = study.configs[name]
            if frozen["controller_config"] != asdict(cfg):
                raise ValueError("frozen challenger configuration changed")
            selection_sha = sha256(study.output / "selected_config.json")
        identity = {
            "holdout": holdout, "plan_sha256": sha256(study.output / "plan.json"),
            "selection_sha256": selection_sha, "method": name,
        }
        path = study.output / f"holdout_{holdout}_{item}.json"
        saved = read_json(path) if path.is_file() else {"identity": identity, "trials": {}}
        if saved["identity"] != identity or not set(saved["trials"]) <= {str(seed) for seed in SEEDS}:
            raise ValueError("saved evaluation protocol or seeds changed")
        for seed in SEEDS:
            if str(seed) in saved["trials"]:
                audit_trial(saved["trials"][str(seed)], name, seed, study.tiers, study.plan["round_seconds"])
                continue
            saved["trials"][str(seed)] = run_trial(
                study, name, study.dataset, loader, study.clients, seed, cfg, class_metrics=True)
            bench.write_json(path, saved)


def archive_target(plan):
    if not plan["historical_environment_match"]:
        raise ValueError("historical controls do not match execution environment")
    if sha256(TARGET_ARCHIVE / "competitor_benchmark_metadata.json") != plan["archived_target_metadata_sha256"]:
        raise ValueError("153-image archive metadata changed")
    path = TARGET_ARCHIVE / "competitor_benchmark_results.json"
    raw = read_json(path)
    if not math.isclose(metrics(raw["CapDrift-Tuned"]["trials"])["final_accuracy"], 94.3790849673,
                        rel_tol=0, abs_tol=1e-7):
        raise ValueError("153-image archive does not reproduce the stated tuned baseline")
    result = {name: {str(t["seed"]): t for t in raw[name]["trials"]} for name in RIVALS}
    result[BASELINE] = {str(t["seed"]): {**t, "method": BASELINE} for t in raw["CapDrift-Tuned"]["trials"]}
    return result, {str(path): sha256(path)}


def archive_source(plan):
    rows, refs = {}, {}
    env = plan["environment"]
    expected = {
        "old_cohort_sha256": plan["dataset_sha256"], "train_index_sha256": plan["train_index_sha256"],
        "holdout": plan["source"], "partition_sha256": plan["local_partition_sha256"],
        "holdout_seeds": list(SEEDS), "rounds": ROUNDS, "active_steps": list(STEPS),
        "device": env["device"], "torch": env["torch"], "torchvision": env["torchvision"],
        "freeze_bn": True, "channels_last": True, "paired_rng": True, "fused_sgd": env["fused_sgd"],
        "local_lr": env["local_lr"], "local_momentum": bench.LOCAL_MOMENTUM,
        "round_duration_sec": plan["round_seconds"], "alpha": ALPHA,
    }
    for name in RIVALS:
        path = SOURCE_ARCHIVE / f"holdout_{name}.json"
        raw = read_json(path)
        if raw["method"] != name or any(raw["protocol"].get(k) != v for k, v in expected.items()):
            raise ValueError(f"423-image {name} archive does not match the study")
        refs[str(path)], rows[name] = sha256(path), raw["trials"]
    return rows, refs


def paired_report(challenger, rows):
    successes = {name: {s: t for s, t in attempts.items() if "history" in t}
                 for name, attempts in rows.items()}
    output = {"methods": {}, "paired": {}}
    for method, attempts in rows.items():
        good = successes[method]
        output["methods"][method] = {
            "completed": len(good), "attempted": len(attempts),
            "all_seed_mean": metrics(list(good.values())) if len(good) == len(SEEDS) else None,
            "failures": [row for row in attempts.values() if "failure" in row],
        }
        if method == challenger:
            continue
        paired = [str(s) for s in SEEDS if str(s) in good and str(s) in successes[challenger]]
        diffs = [successes[challenger][s]["final_accuracy"] - good[s]["final_accuracy"] for s in paired]
        output["paired"][method] = {
            "seeds": list(map(int, paired)), "accuracy_delta_pp": diffs,
            "mean_delta_pp": statistics.mean(diffs) if diffs else None,
            "higher": sum(d > 0 for d in diffs), "tied": sum(d == 0 for d in diffs),
        }
    output["matched_accuracy_win"] = all(
        len(row["seeds"]) == len(SEEDS) and row["mean_delta_pp"] > 0
        for row in output["paired"].values()
    )
    return output


def report(output, holdout):
    plan = read_json(output / "plan.json")
    selection = frozen_selection(output)
    plan_hash = sha256(output / "plan.json")
    selector_hash = sha256(output / "selected_config.json")
    name = selection["name"]
    tiers = [bench.TIER_PROFILES[i] for i in bench.TIER_IDS]

    def evaluation(case, method, selector):
        path = output / f"holdout_{holdout}_{case}.json"
        record = read_json(path)
        if record["identity"] != {"holdout": holdout, "plan_sha256": plan_hash,
                                  "selection_sha256": selector, "method": method}:
            raise ValueError("stored holdout evaluation does not match the study")
        return record["trials"]

    if holdout == "target":
        rows, archive_hashes = archive_target(plan)
        # An already-run live anchor is stricter than the reusable archived scores.
        if (output / f"holdout_{holdout}_{BASELINE}.json").exists():
            live = evaluation(BASELINE, BASELINE, None)
            for seed, trial in live.items():
                if any(trial.get(k) != rows[BASELINE][seed][k]
                       for k in ("final_accuracy", "final_loss")):
                    raise ValueError("live 153-image anchor does not replay archived endpoints")
    else:
        rows, archive_hashes = archive_source(plan)
        rows[BASELINE] = evaluation(BASELINE, BASELINE, None)
    rows[name] = evaluation("challenger", name, selector_hash)
    for method, attempts in rows.items():
        if set(attempts) != {str(s) for s in SEEDS}:
            raise ValueError(f"{method} did not attempt all five seeds")
        for seed in SEEDS:
            audit_trial(attempts[str(seed)], method, seed, tiers, plan["round_seconds"])
    record = {
        **paired_report(name, rows), "selected": name, "holdout": holdout, "plan_sha256": plan_hash,
        "selected_config_sha256": selector_hash, "archived_references": archive_hashes,
        "modeled_seconds": plan["round_seconds"] * ROUNDS, "exploratory_selection": selection["exploratory_only"],
        "interpretation": ("Replay on previously scored 153 images; not independent replication."
                           if holdout == "target" else
                           "Separate 423-image source split; previously scored for other methods."),
    }
    bench.write_json(output / f"report_{holdout}.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("plan", "screen", "promote", "select", "evaluate", "report"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source-test", type=Path, default=ROOT.parent / accuracy.SOURCE_TEST_PATH)
    parser.add_argument("--fold", type=int, choices=tuple(range(len(DEV))), default=0)
    parser.add_argument("--case")
    parser.add_argument("--holdout", choices=("target", "source"), default="target")
    parser.add_argument("--method", choices=("champion", "tuned", "all"), default="champion")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "report":
        report(args.output_dir.resolve(), args.holdout)
        return
    study = prepare(args)
    if args.stage == "plan":
        print(json.dumps(study.plan, indent=2))
    elif args.stage == "screen":
        screen(study, args.fold, args.case)
    elif args.stage == "promote":
        promote(study)
    elif args.stage == "select":
        select(study)
    else:
        evaluate(study, args.holdout, args.method)


if __name__ == "__main__":
    main()
