"""Accuracy-first FedNova ablations with a reserved source-test holdout."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import run_capdrift_tuning as tuning

bench = tuning.bench
torch = bench.torch

ROUNDS = 25
BASE_LR = 0.002
ALPHA = 0.5
OUTER_SPLIT_SEED = 42
OUTER_PARTITION_SEED = 42
DEV = ((731, 42, 42), (1729, 2024, 314))  # (group split, optimizer seed, partition seed)
SEEDS = (42, 101, 2024, 7, 888)
REFERENCE = "FedNova-Momentum"
TEST_METHODS = ("CapDrift-Accuracy", REFERENCE, "FedNova")
SOURCE_TEST_PATH = Path("data/raw/chest_xray/test")


def candidate_configs():
    base = bench.build_config(REFERENCE, server_momentum=0, server_lr=1,
                              anneal_start_round=15, anneal_power=1)
    projection = {"enable_drift_gating": True, "drift_mode": "projection",
                  "drift_parameter_prefix": "classifier.3"}
    clipping = {"enable_clipping": True, "kappa_c": 1.5}
    taper = {"server_lr_decay_start_round": 10, "server_lr_final_ratio": 0.15}
    average = {"server_average_decay": 0.75, "server_average_start_round": 12}
    return {
        "projection-only": replace(base, **projection),
        "clipping-only": replace(base, **clipping),
        "taper-only": replace(base, **taper),
        "average-only": replace(base, **average),
        "projection-clipping-taper": replace(base, **projection, **clipping, **taper),
        "taper-and-average": replace(base, **taper, **average),
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_records(images):
    return [(Path(path), label, sha256(Path(path))) for path, label in images.samples]


def eligible_source_rows(previous, test):
    """Exclude any source-test image linked to any image in the old cohort."""
    joined = previous + test
    groups, recognized = bench.filename_group_ids(
        [path for path, _, _ in joined], [digest for _, _, digest in joined],
    )
    seen = set(groups[:len(previous)])
    holdout = []
    eligible_groups = set()
    unique_bytes = set()
    for record, group in zip(test, groups[len(previous):]):
        if group in seen or record[2] in unique_bytes:
            continue
        eligible_groups.add(group)
        unique_bytes.add(record[2])
        holdout.append(record)
    return holdout, len(eligible_groups), recognized


def source_holdout(data_dir, source_test, *, decode):
    previous = bench.ImageFolder(str(data_dir))
    transform = bench.transforms.Compose([
        bench.transforms.Resize((224, 224)), bench.transforms.ToTensor(),
        bench.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    raw = bench.ImageFolder(str(source_test), transform=transform)
    if previous.classes != raw.classes or len(raw.classes) != 2:
        raise ValueError("expected matching two-class ImageFolder data")
    previous_records, candidates = image_records(previous), image_records(raw)
    eligible, groups, recognized = eligible_source_rows(previous_records, candidates)
    counts = [sum(label == i for _, label, _ in eligible) for i in range(len(raw.classes))]
    if min(counts) < 30:
        raise ValueError("not enough source-test images to assess both classes")
    metadata = {
        "source_test_total": len(candidates),
        "discarded_overlaps_and_duplicates": len(candidates) - len(eligible),
        "evaluated": len(eligible),
        "class_counts": dict(zip(raw.classes, counts)),
        "groups": groups,
        "recognized_filenames_across_both_cohorts": recognized,
        "verified_patient_ids": False,
        "eligible_sha256": tuning.fingerprint([
            [path.relative_to(source_test).as_posix(), label, digest]
            for path, label, digest in eligible
        ]),
    }
    if not decode:
        return metadata, None
    images = torch.stack([transform(raw.loader(str(path))) for path, _, _ in eligible])
    labels = torch.tensor([label for _, label, _ in eligible], dtype=torch.long)
    loader = bench.DataLoader(bench.PreloadedDataset(images, labels), batch_size=32, shuffle=False)
    return metadata, loader


def score(trials):
    return {"final_accuracy": statistics.mean(t["final_accuracy"] for t in trials),
            "final_loss": statistics.mean(t["final_loss"] for t in trials)}


def rank_key(name, summary):
    return (-summary["final_accuracy"], summary["final_loss"], name)


def read_or_start(path, expected):
    if path.exists():
        saved = read_json(path)
        if saved.get("protocol") != expected:
            raise ValueError(f"saved protocol changed: {path}")
        return saved
    return {"protocol": expected, "trials": {}}


def protocol(data_dir, source_test, dataset_hash, holdout, device, anchor):
    paths = (Path(__file__), Path(bench.__file__), Path(tuning.__file__),
             bench.PROJECT_ROOT / "src/research/capdrift_controller.py",
             bench.PROJECT_ROOT / "src/research/competitor_split.py",
             bench.PROJECT_ROOT / "src/research/non_iid.py")
    return {
        "schema": 1, "old_cohort": str(data_dir), "old_cohort_sha256": dataset_hash,
        "source_test": str(source_test), "holdout": holdout,
        "outer_split_seed": OUTER_SPLIT_SEED, "outer_partition_seed": OUTER_PARTITION_SEED,
        "dev_scenarios": [list(t) for t in DEV], "alpha": ALPHA, "rounds": ROUNDS,
        "holdout_seeds": list(SEEDS), "local_lr": BASE_LR, "local_momentum": bench.LOCAL_MOMENTUM,
        "active_steps": [40, 28, 16, 5], "round_duration_sec": anchor,
        "device": str(device), "torch": torch.__version__, "torchvision": bench.torchvision.__version__,
        "freeze_bn": True, "channels_last": True, "paired_rng": True, "fused_sgd": device.type == "cuda",
        "candidate_configs": {name: asdict(config) for name, config in candidate_configs().items()},
        "selection": "Round-25 accuracy descending; final CE ascending for ties. Promote two "
                     "hybrids from the first development scenario, rank on both paired scenarios.",
        "code_sha256": {path.name: sha256(path) for path in paths},
    }


def run(name, data, loader, clients, device, seed, anchor, config=None, *, class_metrics=False):
    tiers = [bench.TIER_PROFILES[i] for i in bench.TIER_IDS]
    hybrid = config is not None
    method = "CapDrift-Tuned" if hybrid else name
    print(f"Trial: {name} | seed={seed}, rounds={ROUNDS}", flush=True)
    try:
        trial = bench.run_competitor_trial(
            method, data, loader, clients, tiers, num_rounds=ROUNDS, device=device, seed=seed,
            config=config, local_lr=BASE_LR, baseline_local_lr=BASE_LR, rng_policy="paired",
            freeze_bn=True, channels_last=True, fused_sgd=device.type == "cuda",
            normalize_momentum_updates=hybrid, measure_balanced_accuracy=class_metrics,
        )
    except bench.NonFiniteMetricError as err:
        return {"seed": seed, "failed_round": err.round_idx, "failure": str(err)}
    tuning.validate_trial(trial, ROUNDS, tiers, True, anchor)
    if class_metrics:
        last = trial["history"][-1]
        assert math.isclose(100 * sum(last["class_correct_counts"]) / sum(last["class_sample_counts"]),
                            trial["final_accuracy"], abs_tol=1e-8)
        trial["final_balanced_accuracy"] = last["balanced_accuracy"]
    trial["method"] = name
    return trial


def develop(output, dataset, train, device, anchor, fixed):
    configs = candidate_configs()
    checkpoint = output / "development.json"
    record = read_or_start(checkpoint, fixed)
    candidates = list(configs)
    for stage, (fold_seed, rng_seed, partition_seed) in enumerate(DEV):
        args = SimpleNamespace(dev_fraction=0.2, alpha=ALPHA, partition_seed=partition_seed)
        fold = tuning.split_for_fold(dataset, train, fold_seed, args)
        record.setdefault("folds", {})[str(stage)] = fold[3]
        for name in [REFERENCE, *candidates]:
            key = f"{stage}/{name}"
            if key in record["trials"]:
                prior = record["trials"][key]
                if prior["seed"] != rng_seed:
                    raise ValueError(f"saved development trial has the wrong seed: {key}")
                if "history" in prior:
                    if prior["method"] != name:
                        raise ValueError(f"saved development trial has the wrong method: {key}")
                    tuning.validate_trial(prior, ROUNDS, [bench.TIER_PROFILES[i] for i in bench.TIER_IDS],
                                          True, anchor)
                elif not (1 <= prior.get("failed_round", 0) <= ROUNDS and prior.get("failure")):
                    raise ValueError(f"saved development trial has no result: {key}")
                continue
            record["trials"][key] = run(name, *fold[:3], device, rng_seed, anchor, configs.get(name))
            bench.write_json(checkpoint, record)
        if stage == 0:
            screen = {n: score([record["trials"][f"0/{n}"]]) for n in candidates
                      if "history" in record["trials"][f"0/{n}"]}
            candidates = sorted(screen, key=lambda n: rank_key(n, screen[n]))[:2]
            if not candidates:
                raise RuntimeError("No candidate completed the screen")
            print(f"Accuracy-first finalists: {candidates}", flush=True)
    scores = {}
    for name in [REFERENCE, *candidates]:
        trials = [record["trials"][f"{i}/{name}"] for i in range(len(DEV))]
        if all("history" in t for t in trials):
            scores[name] = score(trials)
    viable = [name for name in candidates if name in scores]
    if not viable:
        raise RuntimeError("No finalist completed both development scenarios")
    ref_failures = [i for i in range(len(DEV)) if "history" not in record["trials"][f"{i}/{REFERENCE}"]]
    if ref_failures:
        print(f"Development reference failed in scenarios {ref_failures}; retained failures "
              "and excluded them from two-scenario means.", flush=True)
    best = min(viable, key=lambda n: rank_key(n, scores[n]))
    manifest = {"name": best, "controller_config": asdict(configs[best]),
                "scores": scores, "finalists": candidates, "protocol": fixed,
                "reference_failed_scenarios": ref_failures,
                "development_sha256": None}
    record["final_scores"] = scores
    record["finalists"] = candidates
    record["selected"] = best
    bench.write_json(checkpoint, record)
    manifest["development_sha256"] = sha256(checkpoint)
    bench.write_json(output / "selected_config.json", manifest)
    print(f"Frozen development winner: {best}; {scores}", flush=True)


def frozen_selection(output):
    path = output / "selected_config.json"
    frozen = read_json(path)
    if frozen["development_sha256"] != sha256(output / "development.json"):
        raise ValueError("development record does not match the frozen configuration")
    if asdict(candidate_configs()[frozen["name"]]) != frozen["controller_config"]:
        raise ValueError("selected configuration differs from the preregistered grid")
    return frozen, sha256(path)


def evaluate(output, name, dataset, train, loader, device, anchor, fixed, frozen, manifest_hash):
    train_data = bench.Subset(dataset, train)
    train_data.targets = dataset.targets[train]
    clients = bench.partition_dataset_dirichlet(
        train_data, num_clients=4, alpha=ALPHA, seed=OUTER_PARTITION_SEED)
    if sorted(i for local in clients for i in local) != list(range(len(train))):
        raise ValueError("held-out benchmark partition does not cover training data exactly once")
    path = output / f"holdout_{name}.json"
    identity = {**fixed, "manifest_sha256": manifest_hash,
                "partition_sha256": tuning.fingerprint([train, clients])}
    record = read_or_start(path, identity)
    record["method"] = name
    cfg = bench.CapDriftConfig(**frozen["controller_config"]) if name == "CapDrift-Accuracy" else None
    for seed in SEEDS:
        if str(seed) not in record["trials"]:
            record["trials"][str(seed)] = run(name, train_data, loader, clients, device, seed,
                                              anchor, cfg, class_metrics=True)
            bench.write_json(path, record)
    if any("history" not in t for t in record["trials"].values()):
        raise RuntimeError(f"incomplete benchmark: {name} produced a nonfinite trial")


def report(output):
    frozen, fingerprint = frozen_selection(output)
    results, summary = {}, []
    pt = None
    for name in TEST_METHODS:
        record = read_json(output / f"holdout_{name}.json")
        proto = record["protocol"]
        if (proto["manifest_sha256"] != fingerprint or
                {k: v for k, v in proto.items() if k not in {"manifest_sha256", "partition_sha256"}} != frozen["protocol"] or
                (pt is not None and proto["partition_sha256"] != pt)):
            raise ValueError("the evaluation protocol changed across methods")
        pt = proto["partition_sha256"]
        trials = [record["trials"][str(seed)] for seed in SEEDS]
        if [t["seed"] for t in trials] != list(SEEDS) or not all("history" in t for t in trials):
            raise ValueError(f"missing or invalid evaluation trials for {name}")
        for t in trials:
            tuning.validate_trial(t, ROUNDS, [bench.TIER_PROFILES[i] for i in bench.TIER_IDS],
                                  True, proto["round_duration_sec"])
            assert sum(t["history"][-1]["class_sample_counts"]) == proto["holdout"]["evaluated"]
        results[name] = bench.summarize_trials(name, trials)
        summary.append({
            "method": name, "n": len(trials), "mean_final_accuracy": score(trials)["final_accuracy"],
            "sd_final_accuracy": statistics.pstdev(t["final_accuracy"] for t in trials),
            "mean_final_balanced_accuracy": statistics.mean(t["final_balanced_accuracy"] for t in trials),
            "mean_final_loss": score(trials)["final_loss"],
            "mean_modeled_time_sec": statistics.mean(t["total_time_sec"] for t in trials),
            "mean_active_utilization_pct": statistics.mean(t["mean_utilization_pct"] for t in trials),
        })
    metadata = {**frozen["protocol"], "complete": True, "manifest_sha256": fingerprint,
                "selected": frozen["name"], "methods": list(TEST_METHODS), "seeds": list(SEEDS),
                "modeled_reference_total_time_sec": proto["round_duration_sec"] * ROUNDS,
                "partition_sha256": pt}
    comparisons = {}
    for name in TEST_METHODS[1:]:
        deltas = [a["final_accuracy"] - b["final_accuracy"]
                  for a, b in zip(results[TEST_METHODS[0]]["trials"], results[name]["trials"])]
        comparisons[name] = {"paired_accuracy_deltas_pp": deltas, "mean_delta_pp": statistics.mean(deltas),
                             "higher_seeds": sum(d > 0 for d in deltas),
                             "tied_seeds": sum(d == 0 for d in deltas)}
    bench.write_json(output / "competitor_benchmark_results.json", results)
    bench.write_json(output / "competitor_benchmark_metadata.json", metadata)
    bench.write_json(output / "paired_differences.json", comparisons)
    with (output / "competitor_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"verified_round_records": len(TEST_METHODS) * len(SEEDS) * ROUNDS,
                      "holdout": proto["holdout"], "summary": summary, "paired": comparisons}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source-test", type=Path,
                        default=bench.PROJECT_ROOT.parent / SOURCE_TEST_PATH)
    parser.add_argument("--stage", choices=["plan", "develop", "evaluate", "report"], required=True)
    parser.add_argument("--method", choices=TEST_METHODS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if (args.stage == "evaluate") != (args.method is not None):
        parser.error("--method is required exactly for the evaluate stage")
    output = args.output_dir.resolve()
    if args.stage == "report":
        report(output)
        return
    data_dir = tuning.resolve_data_dir(args.data_dir)
    source_test = args.source_test.resolve(strict=True)
    # Fingerprint the eligible test files; only decode them after the winner is frozen.
    frozen = frozen_selection(output) if args.stage == "evaluate" else None
    torch.set_num_threads(4)
    holdout, loader = source_holdout(data_dir, source_test, decode=frozen is not None)
    if args.stage == "plan":
        print(json.dumps(holdout, indent=2))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset, _, dataset_hash = bench.load_dataset(data_dir)
    train, _ = bench.split_indices(dataset, policy="filename-group", seed=OUTER_SPLIT_SEED)
    probe = bench.get_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    anchor = bench.reference_round_duration([bench.TIER_PROFILES[i] for i in bench.TIER_IDS],
                                            bench.get_model_size_mb(probe))
    fixed = protocol(data_dir, source_test, dataset_hash, holdout, device, anchor)
    fixed["train_index_sha256"] = tuning.fingerprint(train)
    fixed["training_images"] = len(train)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "develop":
        if (output / "selected_config.json").exists():
            raise FileExistsError("selection is already frozen; use a new path")
        develop(output, dataset, train, device, anchor, fixed)
    else:
        selected, manifest_hash = frozen
        if selected["protocol"] != fixed:
            raise ValueError("holdout run does not match the frozen selection protocol")
        evaluate(output, args.method, dataset, train, loader, device, anchor, fixed, selected, manifest_hash)


if __name__ == "__main__":
    main()
