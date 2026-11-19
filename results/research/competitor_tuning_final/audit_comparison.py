"""Join separately run, matched CapDrift final holdout benchmarks after audit."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


RESULTS = "competitor_benchmark_results.json"
METADATA = "competitor_benchmark_metadata.json"
SUMMARY = "competitor_benchmark_summary.csv"
PAIR_CSV = "paired_differences.csv"
PAIR_JSON = "paired_differences_summary.json"
REFERENCES = {"FedAvg", "FedProx", "FedNova", "FedNova-Momentum",
              "CapDrift-Legacy", "CapDrift-FL"}
TUNED = "CapDrift-Tuned"
SEEDS = [42, 101, 2024, 7, 888]
ROUNDS = 25
DEADLINE = 6.5
EXPECTED_STEPS = [40, 28, 16, 5]
IDENTITY = ("schema_version", "complete", "seeds", "num_rounds", "num_clients",
            "alpha", "split_seed", "split_policy", "holdout_fraction", "outer_split_sha256",
            "partition_seed", "rng_policy", "data_dir", "dataset_content_sha256",
            "train_samples", "val_samples", "train_class_counts", "val_class_counts",
            "client_class_counts", "split_partition_sha256", "filename_groups",
            "model", "weights", "model_payload_mib", "training_batch_size",
            "freeze_bn", "batchnorm_protocol", "fednova_protocol", "device",
            "device_name", "torch_version", "torchvision_version", "deterministic_algorithms",
            "channels_last", "cpu_threads", "tier_ids", "tiers", "compute_deadline_sec",
            "modeled_reference_round_duration_sec", "modeled_reference_total_time_sec",
            "time_protocol", "cluster_utilization_protocol", "compute_utilization_protocol")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, tol=1e-8):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def read_json(path):
    with path.open(encoding="utf-8-sig") as source:
        return json.load(source)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(folder, expected):
    meta, methods = [read_json(folder / name) for name in (METADATA, RESULTS)]
    require(meta.get("complete") is True and not meta.get("failed_trials"),
            f"Incomplete or failed benchmark: {folder}")
    require(set(meta["methods"]) == set(methods) == expected and len(meta["methods"]) == len(expected),
            f"Method mismatch in {folder}: {meta['methods']}")
    require(set(meta["controller_configs"]) == expected,
            f"Missing or unexpected method configuration in {folder}")
    return meta, methods


def validate_metadata(ref, tuned, selection_path):
    mismatched = [key for key in IDENTITY if key not in ref or key not in tuned or ref[key] != tuned[key]]
    require(not mismatched, f"Comparison protocol mismatch: {', '.join(mismatched)}")
    require((ref["seeds"], ref["num_rounds"], ref["num_clients"], ref["tier_ids"]) ==
            (SEEDS, ROUNDS, 4, [0, 1, 2, 4]), "Unexpected seed list, round count or tier ordering")
    require(ref["split_policy"] == "filename-group" and ref["filename_groups"]["overlap_groups"] == 0,
            "Filename group overlap is not zero")
    require(ref["filename_groups"]["verified_patient_ids"] is False,
            "Filename groups should be reported as proxies, not verified patient IDs")
    require(ref["train_samples"] == 610 and ref["val_samples"] == 153 and ref["alpha"] == 0.5,
            "Unexpected split size or Dirichlet alpha")
    require(ref["rng_policy"] == "paired" and ref["deterministic_algorithms"] and ref["freeze_bn"],
            "Expected paired RNG, deterministic setup and shared frozen BN")
    require("fednova_momentum_protocol" in ref, "Missing FedNova-Momentum protocol description")
    expected_metric = "Per-trial final round (not best round); population SD across listed seeds"
    require(ref["metric_protocol"] == expected_metric and tuned["metric_protocol"] ==
            expected_metric + "; tuned uses EMA shadow if configured, with raw metrics recorded",
            "Unexpected checkpoint or metric protocol")
    require(set(ref["local_optimizer"]) == {"name", "baseline_lr", "capdrift_lr", "momentum",
            "weight_decay", "fused"}, "Unexpected reference optimizer fields")
    require({k: v for k, v in tuned["local_optimizer"].items() if k != "capdrift_tuned_lr"} ==
            ref["local_optimizer"], "Optimizer protocol mismatch")
    require(ref["local_optimizer"]["name"] == "SGD", "Expected common SGD local optimizer")

    manifest = read_json(selection_path)
    manifest_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    require(tuned.get("capdrift_tuned_manifest_sha256") == manifest_hash and
            tuned.get("capdrift_tuned_manifest") == manifest,
            "Frozen development manifest or its bytes do not match")
    require(tuned["controller_configs"][TUNED] == manifest["controller_config"],
            "Applied controller config differs from frozen selection")
    require(manifest["local_lr"] == tuned["local_optimizer"]["capdrift_tuned_lr"] ==
            ref["local_optimizer"]["baseline_lr"] == ref["local_optimizer"]["capdrift_lr"],
            "Effective base local learning rates are not matched")

    prov = manifest["provenance"]
    expected_prov = {
        "data_sha256": ref["dataset_content_sha256"], "outer_split_sha256": ref["outer_split_sha256"],
        "outer_split_policy": ref["split_policy"], "outer_split_seed": ref["split_seed"],
        "outer_fraction": ref["holdout_fraction"], "dirichlet_alpha": ref["alpha"],
        "partition_seed": ref["partition_seed"], "outer_training_samples": ref["train_samples"],
        "reserved_holdout_samples": ref["val_samples"],
    }
    require(all(prov.get(k) == v for k, v in expected_prov.items()), "Selection used a different dataset or split")
    require(prov["finalists"]["rounds"] == ROUNDS and prov["dev_folds"],
            "Missing full-horizon nested development provenance")
    if "refinement" not in prov:
        require(prov["variants"][manifest["name"]]["controller_config"] == manifest["controller_config"] and
                prov["variants"][manifest["name"]]["eligible"],
                "Selected config does not match the eligible sweep entry")
    else:
        method = manifest["name"]
        refinement_info = prov["refinement"]
        development = selection_path.with_name("refinement_results.json")
        record = read_json(development)
        parent = refinement_info["parent"]
        original = read_json(selection_path.with_name("selected_tuning_config.json"))
        require(original["name"] == parent and original["provenance"] == record["protocol"] and
                {k: v for k, v in prov.items() if k not in {"dev_scores", "refinement"}} ==
                {k: v for k, v in original["provenance"].items() if k != "dev_scores"},
                "Refinement did not reuse the original nested protocol")
        code = selection_path.with_name("refine_capdrift.py")
        require(refinement_info["source_sha256"] == record["source_sha256"] ==
                hashlib.sha256(code.read_bytes()).hexdigest(), "Refinement script hash does not match")
        require(record["selected"] == method and record["configs"][method] == manifest["controller_config"]
                and record["scores"][method] == prov["dev_scores"], "Refinement selection/config mismatch")
        require(record["rules"] == refinement_info["rules"], "Refinement rules mismatch")
        rules = record["rules"]
        def passes(row):
            return (row["score_delta"] < -rules["score_improvement_strictly_over"]
                    and row["accuracy_delta_pp"] >= -rules["allowed_accuracy_regression_pp"]
                    and max(row["per_fold_final_ce_delta"]) <= rules["max_per_fold_ce_regression"])
        admitted = [n for n, row in record["scores"].items() if n != parent and passes(row)]
        best = min(admitted, key=lambda n: (record["scores"][n]["loss_score"],
                    -record["scores"][n]["mean_final_accuracy"], n), default=parent)
        require(best == method, "Refined selection did not follow the recorded development rule")
    expected_env = {
        "device": ref["device"], "device_name": ref["device_name"], "torch": ref["torch_version"],
        "torchvision": ref["torchvision_version"], "paired_rng": True, "freeze_bn": ref["freeze_bn"],
        "channels_last": ref["channels_last"], "fused_sgd": ref["local_optimizer"]["fused"],
        "cpu_threads": ref["cpu_threads"],
    }
    require(all(prov["environment"].get(k) == v for k, v in expected_env.items()),
            "Selection/evaluation environments differ")
    require(ref["compute_deadline_sec"] == DEADLINE and
            close(ref["modeled_reference_total_time_sec"], 205.6197213692418) and
            close(ref["modeled_reference_round_duration_sec"] * ROUNDS, ref["modeled_reference_total_time_sec"]),
            "Modeled time or compute deadline does not match publication anchor")
    return manifest, manifest_hash


def validate_histories(methods, meta, manifest):
    rounds_seen = 0
    busiest = 0.0
    anchor = meta["modeled_reference_round_duration_sec"]
    tiers = meta["tiers"]
    clients = [f"client_{i}" for i in range(len(tiers))]
    batch_size = meta["training_batch_size"]
    payload = meta["model_payload_mib"]
    network = [2 * t["network_latency_ms"] / 1000 + 8 * payload / t["network_bandwidth_mbps"]
               for t in tiers]
    step = [batch_size / t["base_samples_per_sec"] for t in tiers]
    require(EXPECTED_STEPS == [min(40, math.floor(DEADLINE / s)) for s in step],
            "Physical deadline step maxima do not match tier profiles")
    require(close(anchor, max(5 * s + n for s, n in zip(step, network))),
            "Reported modeled anchor differs from physical compute and payload formula")
    for method, results in methods.items():
        trials = results["trials"]
        require(results["method"] == method and [t["seed"] for t in trials] == SEEDS,
                f"{method}: missing, duplicated, or unordered trial seeds")
        for trial in trials:
            pfx = f"{method}/seed {trial['seed']}"
            hist = trial["history"]
            require(trial["method"] == method and trial["num_rounds"] == ROUNDS and
                    [r["round"] for r in hist] == list(range(1, ROUNDS + 1)),
                    f"{pfx}: wrong round grid or trial identity")
            require(close(trial["total_time_sec"], anchor * ROUNDS),
                    f"{pfx}: total time does not match modeled anchor")
            for i, r in enumerate(hist, start=1):
                here = f"{pfx}/round {i}"
                for key in ("steps_per_client", "compute_sec_per_client", "client_duration_sec"):
                    require(set(r[key]) == set(clients), f"{here}: {key} has missing/unexpected clients")
                require(close(r["round_duration_sec"], anchor) and close(r["cumulative_time_sec"], i * anchor),
                        f"{here}: modeled round/cumulative time mismatch")
                counts = [r["steps_per_client"][cid] for cid in clients]
                comp = [r["compute_sec_per_client"][cid] for cid in clients]
                total = [r["client_duration_sec"][cid] for cid in clients]
                require(all(type(k) is int and 0 <= k <= cap for k, cap in zip(counts, EXPECTED_STEPS)),
                        f"{here}: invalid step count")
                if method in {TUNED, "FedNova", "FedNova-Momentum"}:
                    require(counts == EXPECTED_STEPS, f"{here}: not all physically feasible steps used")
                require(all(0 <= seconds <= DEADLINE + 1e-8 and close(seconds, n * s)
                            for seconds, n, s in zip(comp, counts, step)),
                        f"{here}: compute failed deadline or is not modeled from steps")
                require(all(close(actual, seconds + n) for actual, seconds, n in zip(total, comp, network))
                        and close(max(total), anchor), f"{here}: client latencies disagree with the model")
                require(close(r["round_idle_sec"], sum(anchor - t for t in total)) and
                        close(r["cluster_utilization_pct"], 100 * sum(total) / (len(clients) * anchor)) and
                        close(r["compute_utilization_pct"], 100 * sum(comp) / (len(clients) * anchor)),
                        f"{here}: utilization/idle accounting mismatch")
                require(0 <= r["val_accuracy"] <= 100 and math.isfinite(r["val_loss"]) and r["val_loss"] >= 0,
                        f"{here}: invalid evaluation values")
                if method == TUNED:
                    config = manifest["controller_config"]
                    average_start = config.get("server_average_start_round", ROUNDS + 1)
                    config_ema = config.get("server_average_decay") is not None and i >= average_start
                    require(("raw_val_accuracy" in r) == config_ema and ("raw_val_loss" in r) == config_ema,
                            f"{here}: raw EMA companion metrics missing or unexpectedly present")
                    if config_ema:
                        require(0 <= r["raw_val_accuracy"] <= 100 and
                                math.isfinite(r["raw_val_loss"]) and r["raw_val_loss"] >= 0,
                                f"{here}: invalid raw companion metrics")
                rounds_seen += 1
                busiest = max(busiest, *comp)
            require(close(trial["final_accuracy"], hist[-1]["val_accuracy"]) and
                    close(trial["final_loss"], hist[-1]["val_loss"]) and
                    close(trial["peak_accuracy"], max(r["val_accuracy"] for r in hist)) and
                    close(trial["total_idle_sec"], sum(r["round_idle_sec"] for r in hist)) and
                    close(trial["max_client_compute_time_sec"],
                          max(max(r["compute_sec_per_client"].values()) for r in hist)) and
                    close(trial["mean_utilization_pct"], statistics.mean(r["cluster_utilization_pct"] for r in hist)) and
                    close(trial["mean_compute_utilization_pct"], statistics.mean(r["compute_utilization_pct"] for r in hist)),
                    f"{pfx}: trial aggregates do not agree with round history")
    return rounds_seen, busiest


def fmt_stats(raw, key, digits):
    values = [t[key] for t in raw]
    return f"{statistics.mean(values):.{digits}f} ± {statistics.pstdev(values):.{digits}f}"


def summary_rows(methods):
    labels = [
        ("Final Acc (%) [Mean ± Std]", "final_accuracy", 2), ("Peak Acc (%) [Mean ± Std]", "peak_accuracy", 2),
        ("Final Loss", "final_loss", 4), ("Total Time (s)", "total_time_sec", 2),
        ("Idle Waste (s)", "total_idle_sec", 2), ("Mean Util (%)", "mean_utilization_pct", 2),
        ("Mean Compute Util (%)", "mean_compute_utilization_pct", 2), ("Mean Wall Time (s)", "wall_time_sec", 2),
        ("Gated Events", "total_gating_events", 1), ("Clipped Events", "total_clipping_events", 1),
        ("Projected Blocks", "total_projection_evaluations", 1),
    ]
    return [{"Method": name, "Successful Trials": f"{len(m['trials'])}/{len(SEEDS)}",
             **{label: fmt_stats(m["trials"], key, digits) for label, key, digits in labels}}
            for name, m in methods.items()]


def paired_rows(methods):
    tuned_trials = {t["seed"]: t for t in methods[TUNED]["trials"]}
    differences = []
    summaries = {}
    for name, method in methods.items():
        if name == TUNED:
            continue
        batch = []
        for ref in method["trials"]:
            t = tuned_trials[ref["seed"]]
            row = {"reference": name, "seed": ref["seed"],
                   "tuned_minus_reference_accuracy_pp": t["final_accuracy"] - ref["final_accuracy"],
                   "tuned_minus_reference_loss": t["final_loss"] - ref["final_loss"],
                   "tuned_minus_reference_utilization_pp": t["mean_utilization_pct"] - ref["mean_utilization_pct"],
                   "tuned_minus_reference_idle_sec": t["total_idle_sec"] - ref["total_idle_sec"]}
            differences.append(row)
            batch.append(row)
        summaries[name] = {
            field: {"mean": statistics.mean(r[field] for r in batch),
                    "sd": statistics.pstdev(r[field] for r in batch),
                    "positive": sum(r[field] > 1e-8 for r in batch),
                    "tied": sum(abs(r[field]) <= 1e-8 for r in batch),
                    "negative": sum(r[field] < -1e-8 for r in batch)}
            for field in batch[0] if field.startswith("tuned_minus_")
        }
    return differences, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ref_meta, refs = load_run(args.references, REFERENCES)
        tuned_meta, tuned = load_run(args.tuned, {TUNED})
        selection, selected_hash = validate_metadata(ref_meta, tuned_meta, args.selection)
        combined = {**tuned, **refs}
        round_records, busiest = validate_histories(combined, ref_meta, selection)
        rows = summary_rows(combined)
        diffs, summaries = paired_rows(combined)
        require(not any((args.output_dir / f).exists() for f in (RESULTS, METADATA, SUMMARY, PAIR_CSV, PAIR_JSON)),
                "Output files already exist; choose a new --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        combined_meta = {
            **ref_meta, "methods": list(combined),
            "controller_configs": {**tuned_meta["controller_configs"], **ref_meta["controller_configs"]},
            "capdrift_tuned_manifest": selection, "capdrift_tuned_manifest_sha256": selected_hash,
            "local_optimizer": tuned_meta["local_optimizer"],
            "metric_protocol": tuned_meta["metric_protocol"],
            "audit": {
                "round_records": round_records, "max_client_compute_time_sec": busiest,
                "tuned_steps_all_rounds": EXPECTED_STEPS,
                "sources": {"references": str(args.references.resolve()), "tuned": str(args.tuned.resolve()),
                            "selection": str(args.selection.resolve())},
            },
        }
        write_json(args.output_dir / RESULTS, combined)
        write_json(args.output_dir / METADATA, combined_meta)
        write_csv(args.output_dir / SUMMARY, rows)
        write_csv(args.output_dir / PAIR_CSV, diffs)
        write_json(args.output_dir / PAIR_JSON, summaries)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Verified {len(combined)} methods x {len(SEEDS)} seeds x {ROUNDS} rounds = {round_records} records; "
          f"zero overlapping inferred filename groups")
    print(f"Every trial: {ref_meta['modeled_reference_total_time_sec']:.12f} modeled s; "
          f"maximum modeled compute {busiest:.9f} s; tuned steps {EXPECTED_STEPS}")
    print(f"Selected {selection['name']}; manifest SHA-256 {selected_hash}")
    for row in rows:
        print(f"{row['Method']}: accuracy {row['Final Acc (%) [Mean ± Std]']}%, "
              f"cross-entropy {row['Final Loss']}, active utilization {row['Mean Util (%)']}%")
    for name, row in summaries.items():
        acc, loss, util = [row[k] for k in ("tuned_minus_reference_accuracy_pp",
                         "tuned_minus_reference_loss", "tuned_minus_reference_utilization_pp")]
        print(f"Paired tuned minus {name}: accuracy {acc['mean']:+.3f} pp ({acc['positive']}/{len(SEEDS)} higher), "
              f"loss {loss['mean']:+.4f} ({loss['negative']}/{len(SEEDS)} lower), utilization {util['mean']:+.3f} pp")
    print(f"Reproducible merged results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
