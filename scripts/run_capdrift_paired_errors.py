"""Round-25 paired errors and a preregistered follow-up, using only outer training data."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import run_capdrift_recovery as protocol

bench = protocol.bench
torch = bench.torch
ANCHOR = protocol.BASELINE
NOVA, HISTORICAL = protocol.RIVALS
FEATURE = "half-feature-lr"
CONTROL = (ANCHOR, NOVA, HISTORICAL)
THRESHOLDS = (0.0, -0.1, 0.1, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75)
NEAR = math.log(0.65 / 0.35)  # probability in [.35, .65]
ROOT = protocol.ROOT
RESULTS = protocol.RESULTS


def score(rows, threshold=0.0):
    return sum(int(r["margin"] > threshold) == r["label"] for r in rows)


def choose_threshold(rows):
    # The fixed order favors the smallest offset on ties, including default.
    return max(THRESHOLDS, key=lambda t: score(rows, t))


def json_write(path, obj):
    bench.write_json(path, obj)


def dev_path(output, fold, method):
    return output / f"paired_dev_{fold}_{method}.json"


def load_dev(output, fold, method):
    row = protocol.read_json(dev_path(output, fold, method))
    if row["plan_sha256"] != protocol.sha256(output / "paired_plan.json"):
        raise ValueError("development record does not belong to this plan")
    if row["fold"] != fold or row["method"] != method:
        raise ValueError("development trial identity mismatch")
    return row


def observed_logits(model, loader, device, ids):
    rows = []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).detach().cpu().tolist()
        for label, (p0, p1) in zip(y.tolist(), logits, strict=True):
            margin = p1 - p0
            if label not in (0, 1) or not math.isfinite(margin):
                raise ValueError("invalid label or nonfinite logit margin")
            rows.append({"id": int(ids[len(rows)]), "label": label, "margin": margin})
    if len(rows) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("evaluation row IDs must be unique and complete")
    return rows


def dev_inputs(study, fold):
    group_seed, seed, partition_seed = protocol.DEV[fold]
    args = SimpleNamespace(alpha=protocol.ALPHA, dev_fraction=0.2, partition_seed=partition_seed)
    data, val, clients, identity = protocol.tuning.split_for_fold(study.dataset, study.train, group_seed, args)
    ids = [study.train[i] for i in val.dataset.indices]
    cal_a, cal_b = bench.split_indices(
        study.dataset, policy="filename-group", indices=ids, seed=9143 + fold, holdout_fraction=.50)
    groups = study.dataset.group_ids
    if {groups[i] for i in cal_a} & {groups[i] for i in cal_b}:
        raise ValueError("calibration and assessment share a filename group")
    if set(cal_a) & set(cal_b) or set(cal_a) | set(cal_b) != set(ids):
        raise ValueError("invalid calibration split")
    return data, val, clients, ids, seed, {**identity, "calibration_ids": [cal_a, cal_b]}


def prepare(args):
    study = protocol.prepare(SimpleNamespace(cpu=False, data_dir=None,
        source_test=ROOT.parent / protocol.accuracy.SOURCE_TEST_PATH, stage="plan", holdout=None,
        output_dir=args.output_dir))
    if study.device.type != "cuda" or not study.plan["historical_environment_match"]:
        raise RuntimeError("matched CUDA execution is required")
    plan = {
        "protocol_plan_sha256": protocol.sha256(study.output / "plan.json"),
        "runner_sha256": protocol.sha256(Path(__file__)),
        "development": [dev_inputs(study, fold)[-1] for fold in range(len(protocol.DEV))],
        "near_tie_probability": [.35, .65],
        "route": "Threshold if >50% of canonical FedNova-only correct rows are near ties; else feature LR.",
        "thresholds_logit_margin": list(THRESHOLDS),
        "threshold_eval": "Within each fold swap filename-group calibration and assessment halves.",
        "feature_lr_multiplier": .5,
        "promotion": "Positive total paired gain on both-fold assessment without any fold losing accuracy.",
        "rounds": protocol.ROUNDS,
    }
    path = study.output / "paired_plan.json"
    if path.exists() and protocol.read_json(path) != plan:
        raise ValueError("paired error plan changed")
    if not path.exists():
        json_write(path, plan)
    return study


def train(study, method, data, val, clients, ids, seed):
    config = study.configs[ANCHOR] if method in (ANCHOR, FEATURE) else None
    backend = "CapDrift-Tuned" if config is not None else method
    lr = study.plan["environment"]["local_lr"]
    rows = []

    def save(model):
        rows.extend(observed_logits(model, val, study.device, ids))

    trial = bench.run_competitor_trial(
        backend, data, val, clients, study.tiers, num_rounds=protocol.ROUNDS,
        config=config, seed=seed, device=study.device, local_lr=lr, baseline_local_lr=lr,
        rng_policy="paired", freeze_bn=True, channels_last=True, fused_sgd=True,
        final_model_observer=save, feature_lr_multiplier=.5 if method == FEATURE else 1.,
    )
    trial["method"] = method
    protocol.audit_trial(trial, method, seed, study.tiers, study.plan["round_seconds"])
    if not math.isclose(score(rows) * 100 / len(rows), trial["final_accuracy"], abs_tol=1e-10):
        raise ValueError("captured logits do not match benchmark endpoint")
    return trial, rows


def run_dev(study, fold, method):
    if method == FEATURE and protocol.read_json(study.output / "diagnostic.json")["route"] != FEATURE:
        raise ValueError("paired errors did not select the feature LR branch")
    path = dev_path(study.output, fold, method)
    if path.exists():
        load_dev(study.output, fold, method)
        print(f"Reused {path}")
        return
    data, val, clients, ids, seed, split = dev_inputs(study, fold)
    trial, rows = train(study, method, data, val, clients, ids, seed)
    if method == ANCHOR:
        reference_path = RESULTS / "competitor_recovery" / f"dev_{fold}_{ANCHOR}.json"
        reference = protocol.read_json(reference_path)
        if trial["history"] != reference["trial"]["history"]:
            raise ValueError("control did not reproduce archived development history")
    json_write(path, {"fold": fold, "method": method, "trial": trial, "predictions": rows,
                      "plan_sha256": protocol.sha256(study.output / "paired_plan.json")})


def aligned(left, right):
    ident = lambda rows: [(r["id"], r["label"]) for r in rows]
    if ident(left) != ident(right):
        raise ValueError("paired prediction IDs or labels disagree")
    return zip(left, right)


def hit(row, threshold=0.0):
    return int(row["margin"] > threshold) == row["label"]


def compare(cap, rival):
    pairs = list(aligned(cap, rival))
    misses = [a for a, b in pairs if not hit(a) and hit(b)]
    exclusive_wins = sum(hit(a) and not hit(b) for a, b in pairs)
    return {
        "samples": len(cap), "cap_correct": score(cap), "rival_correct": score(rival),
        "rival_only_correct": len(misses), "cap_only_correct": exclusive_wins,
        "rival_only_near": sum(abs(a["margin"]) <= NEAR for a in misses),
        "rival_only_confident_ge_80pct": sum(abs(a["margin"]) >= math.log(4) for a in misses),
        "rival_only_by_label": [sum(a["label"] == k for a in misses) for k in (0, 1)],
    }


def diagnose(output):
    folds = list(range(len(protocol.DEV)))
    rows = {
        name: [compare(load_dev(output, f, ANCHOR)["predictions"],
                       load_dev(output, f, name)["predictions"]) for f in folds]
        for name in protocol.RIVALS
    }
    near = sum(r["rival_only_near"] for r in rows[NOVA])
    count = sum(r["rival_only_correct"] for r in rows[NOVA])
    route = "threshold" if count and 2 * near > count else FEATURE
    plan_hash = protocol.sha256(output / "paired_plan.json")
    inputs = {dev_path(output, f, n).name: protocol.sha256(dev_path(output, f, n))
              for f in folds for n in CONTROL}
    result = {"route": route, "plan_sha256": plan_hash, "by_rival_and_fold": rows,
              "canonical_rival_only_near": near, "canonical_rival_only_cases": count,
              "input_sha256": inputs}
    json_write(output / "diagnostic.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def threshold_study(rows, calibration_ids):
    by_id = {r["id"]: r for r in rows}
    if set(calibration_ids[0]) & set(calibration_ids[1]) or (
        set(calibration_ids[0]) | set(calibration_ids[1]) != set(by_id)
    ):
        raise ValueError("invalid threshold evaluation partition")
    out = []
    for cal, assess in (calibration_ids, calibration_ids[::-1]):
        cal_rows = [by_id[i] for i in cal]
        eval_rows = [by_id[i] for i in assess]
        threshold = choose_threshold(cal_rows)
        out.append({"threshold": threshold, "assessment_samples": len(eval_rows),
                    "default_hits": score(eval_rows), "adjusted_hits": score(eval_rows, threshold)})
    return out


def assess(output):
    plan = protocol.read_json(output / "paired_plan.json")
    audit = diagnose(output)
    folds = list(range(len(protocol.DEV)))
    details = {}
    branch_hashes = {}
    if audit["route"] == FEATURE:
        gains = []
        for f in folds:
            control = load_dev(output, f, ANCHOR)
            altered = load_dev(output, f, FEATURE)
            c = compare(altered["predictions"], control["predictions"])
            c["feature_ce"] = altered["trial"]["final_loss"]
            c["anchor_ce"] = control["trial"]["final_loss"]
            c["feature_accuracy"] = altered["trial"]["final_accuracy"]
            c["anchor_accuracy"] = control["trial"]["final_accuracy"]
            gains.append(c["cap_correct"] - c["rival_correct"])
            details[str(f)] = c
            p = dev_path(output, f, FEATURE)
            branch_hashes[p.name] = protocol.sha256(p)
    else:
        for name in CONTROL:
            details[name] = [
                threshold_study(load_dev(output, f, name)["predictions"],
                                plan["development"][f]["calibration_ids"]) for f in folds
            ]
        gains = [sum(row["adjusted_hits"] - row["default_hits"] for row in fold)
                 for fold in details[ANCHOR]]
    result = {
        "route": audit["route"], "plan_sha256": audit["plan_sha256"], "details": details,
        "decision_signed_gain_per_fold": gains, "promoted": all(g >= 0 for g in gains) and sum(gains) > 0,
        "input_sha256": {**audit["input_sha256"], **branch_hashes},
    }
    json_write(output / "assessment.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "dev", "diagnose", "assess"))
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "paired_errors")
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--method", choices=(*CONTROL, FEATURE), default=ANCHOR)
    args = parser.parse_args()
    if args.stage == "diagnose":
        diagnose(args.output_dir)
    elif args.stage == "assess":
        assess(args.output_dir)
    else:
        study = prepare(args)
        if args.stage == "plan":
            print(study.plan["environment"], flush=True)
        else:
            run_dev(study, args.fold, args.method)


if __name__ == "__main__":
    main()
