"""Replay only a promoted paired-error change on both previously scored cohorts."""

import argparse
import json
import math
import statistics
from pathlib import Path

import run_capdrift_paired_errors as study_code

p = study_code.protocol
b = p.bench
A = study_code.ANCHOR
F = study_code.FEATURE
read = p.read_json
digest = p.sha256
SEEDS = p.SEEDS


def freeze(output):
    assessment = read(output / "assessment.json")
    if not assessment["promoted"]:
        raise ValueError("development evaluation did not promote a challenger")
    plan_hash = digest(output / "paired_plan.json")
    if assessment["plan_sha256"] != plan_hash or any(
        digest(output / k) != h for k, h in assessment["input_sha256"].items()
    ):
        raise ValueError("development input changed")
    if assessment["route"] == F:
        thresholds = {F: 0.0}
    else:
        thresholds = {
            method: study_code.choose_threshold([
                row for fold in range(len(p.DEV))
                for row in study_code.load_dev(output, fold, method)["predictions"]
            ]) for method in study_code.CONTROL
            if all(study_code.dev_path(output, f, method).exists() for f in range(len(p.DEV)))
        }
    plan = read(output / "plan.json")
    plan_path = output / "replay_freeze.json"
    details = {"plan_sha256": plan_hash, "runner_sha256": digest(Path(__file__)),
               "assessment_sha256": digest(output / "assessment.json"), "thresholds": thresholds,
               "route": assessment["route"], "seeds": list(SEEDS)}
    if plan_path.exists() and read(plan_path) != details:
        raise ValueError("replay freeze changed")
    if not plan_path.exists():
        b.write_json(plan_path, details)
    return details


def measure(rows, threshold):
    labels = [[r for r in rows if r["label"] == k] for k in (0, 1)]
    base, adjusted = study_code.score(rows), study_code.score(rows, threshold)
    base_rates = [study_code.score(rs) / len(rs) for rs in labels]
    adjusted_rates = [study_code.score(rs, threshold) / len(rs) for rs in labels]
    def ce(r):
        x = (1 - 2*r["label"]) * r["margin"]
        return max(x, 0) + math.log1p(math.exp(-abs(x)))
    return {"samples": len(rows), "default_hits": base, "hits": adjusted,
            "default_accuracy": base * 100.0 / len(rows), "accuracy": adjusted * 100.0 / len(rows),
            "ce": statistics.mean(ce(r) for r in rows),
            "default_balanced_accuracy": 100 * statistics.mean(base_rates),
            "balanced_accuracy": 100 * statistics.mean(adjusted_rates)}


def replay(study, frozen, seeds, methods):
    metadata, source = p.accuracy.source_holdout(
        p.tuning.resolve_data_dir(None), p.ROOT.parent / p.accuracy.SOURCE_TEST_PATH, decode=True)
    if metadata != study.plan["source"]:
        raise ValueError("source holdout changed")
    target = b.DataLoader(b.Subset(study.dataset, study.held), batch_size=32, shuffle=False)
    for method in methods:
        cfg = study.configs[A] if method in (A, F) else None
        backend = "CapDrift-Tuned" if cfg is not None else method
        lr = study.plan["environment"]["local_lr"]
        t = frozen["thresholds"][method]
        for seed in seeds:
            path = study.output / f"replay_{method}_{seed}.json"
            identity = {"freeze_sha256": digest(study.output / "replay_freeze.json"),
                        "method": method, "seed": seed}
            if path.exists():
                if read(path)["identity"] != identity:
                    raise ValueError("saved replay identity changed")
                continue
            results = {}
            def observer(model):
                for name, loader, ids in (("target", target, study.held),
                                           ("source", source, range(len(source.dataset)))):
                    rows = study_code.observed_logits(model, loader, study.device, ids)
                    results[name] = measure(rows, t)
            run = b.run_competitor_trial(
                backend, study.dataset, target, study.clients, study.tiers,
                num_rounds=p.ROUNDS, config=cfg, seed=seed, device=study.device, local_lr=lr,
                baseline_local_lr=lr, rng_policy="paired", freeze_bn=True, channels_last=True, fused_sgd=True,
                final_model_observer=observer, feature_lr_multiplier=.5 if method == F else 1.0,
            )
            run["method"] = method
            p.audit_trial(run, method, seed, study.tiers, study.plan["round_seconds"])
            if not math.isclose(run["final_accuracy"], results["target"]["default_accuracy"], abs_tol=1e-10):
                raise ValueError("target scores from observer do not match archived telemetry semantics")
            b.write_json(path, {"identity": identity, "trial": run, "scores": results})


def report(output, frozen):
    plan = read(output / "plan.json")
    target, _ = p.archive_target(plan)
    source, _ = p.archive_source(plan)
    source[A] = read(p.RESULTS / "competitor_recovery" / f"holdout_source_{A}.json")["trials"]
    output_records = {}
    for cohort, controls in (("target", target), ("source", source)):
        per_seed = {
            method: [controls[method][str(s)]["final_accuracy"] for s in SEEDS]
            for method in study_code.CONTROL
        }
        for method in frozen["thresholds"]:
            scores = []
            for s in SEEDS:
                record = read(output / f"replay_{method}_{s}.json")
                if record["identity"] != {"method": method, "seed": s,
                        "freeze_sha256": digest(output / "replay_freeze.json")}:
                    raise ValueError("replay result does not match freeze")
                scores.append(record["scores"][cohort])
            if method in per_seed and any(not math.isclose(r["default_accuracy"], a, abs_tol=1e-10)
                    for r, a in zip(scores, per_seed[method], strict=True)):
                raise ValueError("fresh unadjusted replay does not match archive")
            per_seed[method + "-threshold" if frozen["route"] == "threshold" else method] = [
                r["accuracy"] for r in scores]
        selected = A + "-threshold" if frozen["route"] == "threshold" else F
        eligible_rivals = [
            n + "-threshold" if n in frozen["thresholds"] else n for n in p.RIVALS]
        mean = {k: statistics.mean(v) for k, v in per_seed.items()}
        output_records[cohort] = {
            "accuracy_by_seed": per_seed, "mean_accuracy": mean,
            "selected": selected, "wins_against": {
                n: sum(a > b for a, b in zip(per_seed[selected], per_seed[n], strict=True))
                for n in eligible_rivals}, "accuracy_delta_pp": {
                n: mean[selected] - mean[n] for n in (A, *p.RIVALS, *sorted(set(eligible_rivals) - set(p.RIVALS)))}
        }
    b.write_json(output / "replay_report.json", output_records)
    print(json.dumps(output_records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "replay", "report"))
    parser.add_argument("--output-dir", type=Path, default=p.RESULTS / "paired_errors")
    parser.add_argument("--seed", nargs="+", choices=SEEDS, type=int)
    parser.add_argument("--method", nargs="+", choices=(*study_code.CONTROL, F))
    args = parser.parse_args()
    frozen = freeze(args.output_dir)
    if args.stage == "report":
        report(args.output_dir, frozen)
    elif args.stage == "replay":
        names = args.method or list(frozen["thresholds"])
        if not set(names) <= frozen["thresholds"].keys():
            raise ValueError("requested method not in frozen qualification plan")
        replay(study_code.prepare(args), frozen, args.seed or SEEDS, names)
    else:
        print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
