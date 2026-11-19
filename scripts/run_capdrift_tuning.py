"""Select a deadline-filling CapDrift preset on nested filename-group dev data.

The outer holdout is reserved before any training and never used to score a
candidate. Screening uses a prefix of the full schedule, then the finalists
start over from the same pretrained initialization on paired dev trials.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from torch.utils.data import DataLoader, Subset

import run_competitor_benchmark as bench

torch = bench.torch


def variants() -> dict:
    """A small, prespecified search; no outer-holdout metrics enter ranking."""
    projection_v2 = bench.build_config(
        "CapDrift-FL", server_momentum=0.9, server_lr=0.2,
        anneal_start_round=15, anneal_power=1.0,
    )
    full_cosine = replace(
        projection_v2, anneal_start_round=None,
        server_lr_decay_start_round=10, server_lr_final_ratio=0.1,
    )
    short_memory = replace(
        full_cosine, server_momentum=0.55, server_lr=0.65,
        server_lr_final_ratio=0.15, server_average_decay=0.8,
    )
    return {
        # name -> (controller config, base local step size, selection eligible)
        "projection-v2": (projection_v2, 0.002, False),
        "full-cosine": (full_cosine, 0.002, True),
        "full-cos-ema": (replace(full_cosine, server_average_decay=0.8), 0.002, True),
        "short-memory": (short_memory, 0.002, True),
        "balanced": (
            replace(short_memory, budget_lr_power=0.65, client_update_normalization_power=0.25),
            0.002, True,
        ),
        "no-server-momentum": (
            replace(short_memory, server_momentum=0.0, server_lr=1.0, server_average_decay=0.75),
            0.002, True,
        ),
        "full-ema": (
            replace(short_memory, server_momentum=0.8, server_lr=0.3,
                    server_lr_decay_start_round=15, server_lr_final_ratio=0.2),
            0.002, True,
        ),
    }


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_for_fold(dataset, outer_train, fold, args):
    train_global, dev_global = bench.split_indices(
        dataset, policy="filename-group", indices=outer_train,
        seed=fold, holdout_fraction=args.dev_fraction,
    )
    assert set(train_global).isdisjoint(dev_global)
    assert set(train_global) | set(dev_global) == set(outer_train)
    assert {dataset.group_ids[i] for i in train_global}.isdisjoint(
        dataset.group_ids[i] for i in dev_global
    )
    reverse = {global_ix: local_ix for local_ix, global_ix in enumerate(outer_train)}
    train_rows = [reverse[i] for i in train_global]
    dev_rows = [reverse[i] for i in dev_global]
    # Keep the outer-holdout image records outside every training/dev loader.
    outer_data = Subset(dataset, outer_train)
    train_data = Subset(outer_data, train_rows)
    train_data.targets = dataset.targets[train_global]
    partition = bench.partition_dataset_dirichlet(
        train_data, num_clients=len(bench.TIER_IDS), alpha=args.alpha, seed=args.partition_seed,
    )
    clients = [[train_rows[i] for i in subset] for subset in partition]
    if any(not rows for rows in clients) or sorted(i for rows in clients for i in rows) != sorted(train_rows):
        raise RuntimeError("A dev fold's client partition omitted or repeated data")
    return (
        outer_data,
        DataLoader(Subset(outer_data, dev_rows), batch_size=32, shuffle=False),
        clients,
        {
            "fold_seed": fold,
            "training_samples": len(train_rows),
            "dev_samples": len(dev_rows),
            "client_samples": [len(p) for p in clients],
            "split_partition_sha256": fingerprint([train_global, dev_global, clients]),
        },
    )


def validate_trial(trial, rounds, tiers, full_utilization, reference_round):
    history = trial["history"]
    if [h["round"] for h in history] != list(range(1, rounds + 1)):
        raise RuntimeError("Dev trial history does not have all expected rounds")
    expected_steps = [min(bench.MAX_STEPS, math.floor(bench.COMPUTE_DEADLINE_SEC /
                      (bench.BATCH_SIZE / tier.base_samples_per_sec))) for tier in tiers]
    for rd in history:
        if not math.isclose(rd["round_duration_sec"], reference_round, rel_tol=0, abs_tol=1e-8):
            raise RuntimeError("Dev trial did not meet the common modeled round latency")
        if max(rd["compute_sec_per_client"].values()) > bench.COMPUTE_DEADLINE_SEC + 1e-8:
            raise RuntimeError("Dev trial exceeded modeled compute deadline")
        found_steps = [rd["steps_per_client"][f"client_{i}"] for i in range(len(tiers))]
        if full_utilization and found_steps != expected_steps:
            raise RuntimeError(f"Selectable candidate used {found_steps}, expected full capacity {expected_steps}")
    if not math.isclose(trial["total_time_sec"], rounds * reference_round, rel_tol=0, abs_tol=1e-8):
        raise RuntimeError("Dev trial total modeled time did not match the expected sum")


def scores(trials, tail):
    final_loss = statistics.mean(t["final_loss"] for t in trials)
    tail_loss = statistics.mean(
        statistics.mean(h["val_loss"] for h in t["history"][-tail:]) for t in trials
    )
    return {
        "n": len(trials),
        "loss_score": 0.7 * final_loss + 0.3 * tail_loss,
        "mean_final_loss": final_loss,
        "mean_tail_loss": tail_loss,
        "mean_final_accuracy": statistics.mean(t["final_accuracy"] for t in trials),
        "mean_utilization_pct": statistics.mean(t["mean_utilization_pct"] for t in trials),
    }


def rank_key(name, row):
    return (row["loss_score"], -row["mean_final_accuracy"], name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-split-seed", type=int, default=42)
    parser.add_argument("--outer-fraction", type=float, default=0.2)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--screen-fold", type=int, default=731)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--screen-rounds", type=int, default=12)
    parser.add_argument("--full-rounds", type=int, default=25)
    parser.add_argument("--finalist-folds", nargs="+", type=int, default=[731, 1729])
    parser.add_argument("--finalist-seeds", nargs="+", type=int, default=[101, 2024])
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--reference", choices=list(bench.METHODS), default=None,
                        help="Optionally score a non-selectable competitor in the screen")
    parser.add_argument("--variants", nargs="+", choices=list(variants()),
                        help="Run only these prespecified variants; default is entire grid")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.screen_rounds <= args.full_rounds or args.tail < 1 or
            args.keep < 1 or args.cpu_threads < 1):
        parser.error("Require 1 <= screen-rounds <= full-rounds, and positive tail, keep, cpu-threads")
    if not (0.0 < args.outer_fraction < 1.0 and 0.0 < args.dev_fraction < 1.0 and
            math.isfinite(args.alpha) and args.alpha > 0.0):
        parser.error("Require fractions in (0, 1) and positive finite Dirichlet alpha")
    if (len(args.finalist_folds) != len(args.finalist_seeds) or
            len(set(zip(args.finalist_folds, args.finalist_seeds))) != len(args.finalist_seeds)):
        parser.error("Provide one fold per finalist seed, without duplicate (fold, seed) pairs")
    if any(not 0 <= s < 2**32 for s in [args.outer_split_seed, args.partition_seed,
            args.screen_fold, args.screen_seed, *args.finalist_folds, *args.finalist_seeds]):
        parser.error("Seeds must be integers in [0, 2**32)")
    if args.variants and len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants")
    return args


def resolve_data_dir(path):
    choices = [path] if path is not None else [
        bench.PROJECT_ROOT / bench.DATA_RELATIVE_PATH,
        bench.PROJECT_ROOT.parent / bench.DATA_RELATIVE_PATH,
    ]
    for choice in choices:
        if choice.is_dir():
            return choice.resolve()
    raise FileNotFoundError("ImageFolder root not found; pass --data-dir")


def main(argv=None):
    args = parse_args(argv)
    grid = variants()
    if args.variants:
        grid = {name: grid[name] for name in args.variants}
    if not any(selectable for _, _, selectable in grid.values()):
        raise ValueError("At least one selectable candidate is required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if dev == "auto":
        dev = "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    device = torch.device(dev)
    data_root = resolve_data_dir(args.data_dir)
    dataset, _, dataset_hash = bench.load_dataset(data_root)
    outer_train, outer_holdout = bench.split_indices(
        dataset, policy="filename-group", seed=args.outer_split_seed,
        holdout_fraction=args.outer_fraction,
    )
    if not set(outer_train).isdisjoint(outer_holdout):
        raise RuntimeError("Outer holdout overlaps outer training data")
    if not {dataset.group_ids[i] for i in outer_train}.isdisjoint(
            dataset.group_ids[i] for i in outer_holdout):
        raise RuntimeError("Outer holdout filename groups overlap outer training data")
    folds = {
        fold: split_for_fold(dataset, outer_train, fold, args)
        for fold in sorted(set([args.screen_fold, *args.finalist_folds]))
    }
    tiers = [bench.TIER_PROFILES[i] for i in bench.TIER_IDS]
    time_probe = bench.get_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    reference_round = bench.reference_round_duration(tiers, bench.get_model_size_mb(time_probe))
    provenance = {
        "schema": 1,
        "data_sha256": dataset_hash,
        "outer_split_policy": "filename-group",
        "outer_split_seed": args.outer_split_seed,
        "outer_fraction": args.outer_fraction,
        "outer_split_sha256": fingerprint([outer_train, outer_holdout]),
        "outer_training_samples": len(outer_train),
        "reserved_holdout_samples": len(outer_holdout),
        "dev_fraction": args.dev_fraction,
        "dev_folds": [folds[fold][3] for fold in folds],
        "dirichlet_alpha": args.alpha,
        "partition_seed": args.partition_seed,
        "screen": {"seed": args.screen_seed, "fold": args.screen_fold, "rounds": args.screen_rounds},
        "finalists": {"seeds": args.finalist_seeds, "folds": args.finalist_folds, "rounds": args.full_rounds},
        "keep": args.keep,
        "ranking": {"tail_rounds": args.tail, "score": "0.7 * final CE + 0.3 * tail-mean CE; final accuracy tiebreak"},
        "reference": args.reference,
        "reference_round_sec": reference_round,
        "environment": {
            "device": dev, "device_name": torch.cuda.get_device_name(device) if dev == "cuda" else "CPU",
            "torch": torch.__version__, "torchvision": bench.torchvision.__version__,
            "paired_rng": True, "freeze_bn": True, "channels_last": True,
            "fused_sgd": dev == "cuda", "cpu_threads": args.cpu_threads,
        },
        "variants": {
            name: {"controller_config": asdict(config), "local_lr": lr, "eligible": eligible}
            for name, (config, lr, eligible) in grid.items()
        },
    }
    controller_path = bench.PROJECT_ROOT / "src" / "research" / "capdrift_controller.py"
    provenance["code_sha256"] = {
        Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in [__file__, bench.__file__, controller_path]
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trial_file = output / "dev_tuning_results.json"
    winner_file = output / "selected_tuning_config.json"
    if args.resume:
        record = json.loads(trial_file.read_text(encoding="utf-8"))
        if record["provenance"] != provenance:
            raise ValueError("Resume requested with different config, code, dataset, or split")
    else:
        if trial_file.exists() or winner_file.exists():
            raise FileExistsError("Use an empty output directory or --resume")
        record = {"provenance": provenance, "screen_trials": {}, "finalist_trials": {}}
        bench.write_json(trial_file, record)

    def run(stage, name, fold, seed, rounds, *, reference=False):
        trial_key = f"{name}/{fold}/{seed}"
        trial_records = record[stage]
        config, lr, eligible = grid[name] if not reference else (None, 0.002, False)
        if trial_key in trial_records:
            saved = trial_records[trial_key]
            if "history" in saved:
                validate_trial(saved, rounds, tiers, eligible, reference_round)
            print(f"Reusing {stage}: {trial_key}", flush=True)
            return trial_records[trial_key]
        data, dev_loader, clients, _ = folds[fold]
        method = name if reference else "CapDrift-FL"
        print(f"{stage} | {name} | fold={fold} seed={seed} rounds={rounds}", flush=True)
        try:
            trial = bench.run_competitor_trial(
                method, data, dev_loader, clients, tiers, num_rounds=rounds,
                config=config, seed=seed, device=device, mu_prox=bench.METHODS[method],
                rng_policy="paired", freeze_bn=True, channels_last=True, fused_sgd=dev == "cuda",
                local_lr=lr, baseline_local_lr=lr, schedule_total_rounds=args.full_rounds,
            )
            validate_trial(trial, rounds, tiers, eligible, reference_round)
        except bench.NonFiniteMetricError as exc:
            trial = {"failure": str(exc), "round": exc.round_idx, "seed": seed}
            print(f"Non-finite dev trial: {trial_key}: {exc}", flush=True)
        trial_records[trial_key] = trial
        bench.write_json(trial_file, record)
        return trial

    screen_rows = {}
    for name in grid:
        trial = run("screen_trials", name, args.screen_fold, args.screen_seed, args.screen_rounds)
        if "history" in trial:
            screen_rows[name] = scores([trial], args.tail)
    if args.reference:
        trial = run("screen_trials", args.reference, args.screen_fold, args.screen_seed,
                    args.screen_rounds, reference=True)
        if "history" in trial:
            screen_rows[args.reference] = scores([trial], args.tail)
    finalist_names = sorted(
        (n for n in grid if grid[n][2] and n in screen_rows),
        key=lambda n: rank_key(n, screen_rows[n]),
    )[:args.keep]
    record["screen_scores"] = screen_rows
    record["finalists"] = finalist_names
    bench.write_json(trial_file, record)
    print(f"Promoted: {finalist_names}", flush=True)

    finalist_rows = {}
    for name in finalist_names:
        trials = [
            run("finalist_trials", name, fold, seed, args.full_rounds)
            for fold, seed in zip(args.finalist_folds, args.finalist_seeds)
        ]
        if all("history" in t for t in trials):
            finalist_rows[name] = scores(trials, args.tail)
    record["finalist_scores"] = finalist_rows
    bench.write_json(trial_file, record)
    if not finalist_rows:
        print(f"No complete selectable finalist; history: {trial_file}", file=sys.stderr)
        return 2
    name = min(finalist_rows, key=lambda n: rank_key(n, finalist_rows[n]))
    config, lr, _ = grid[name]
    bench.write_json(winner_file, {
        "name": name, "controller_config": asdict(config), "local_lr": lr,
        "provenance": {"dev_scores": finalist_rows[name], **provenance},
    })
    record["selected"] = name
    bench.write_json(trial_file, record)
    print(f"Selected {name}: {finalist_rows[name]}\n{winner_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
