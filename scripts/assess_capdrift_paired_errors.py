"""Prespecified-for-remaining-analysis fallback when canonical FedNova diverges.

Retain every attempted fold. If canonical lacks a complete pair of folds, use
the rival with the most completed matched folds (canonical breaks ties).
If only one survives, record the resulting direction as a one-fold diagnostic.
Do not replace a failed score or substitute one fold from a different rival.
"""

import argparse
import json
import re
from pathlib import Path

import run_capdrift_paired_errors as core

p = core.protocol


def amended_diagnosis(output):
    base = p.read_json(output / "paired_plan.json")
    regions = list(range(len(p.DEV)))
    rows, hashes, failures = {n: [] for n in p.RIVALS}, {}, {}
    for fold in regions:
        a = core.load_dev(output, fold, core.ANCHOR)
        name = core.dev_path(output, fold, core.ANCHOR)
        hashes[name.name] = p.sha256(name)
        for rival in p.RIVALS:
            path = core.dev_path(output, fold, rival)
            if not path.exists():
                log = output / "evidence" / f"dev-{fold}-{rival}.log"
                failure = re.search(r"NonFiniteMetricError: round (\d+): accuracy=[\d.]+, loss=(\S+)",
                                    log.read_text(encoding="utf-8-sig"))
                if failure is None:
                    raise ValueError(f"Not a verified nonfinite run: {rival} fold {fold}")
                failures[path.name] = {"failed_round": int(failure[1]), "loss": failure[2]}
                hashes[log.relative_to(output).as_posix()] = p.sha256(log)
                rows[rival].append({"failure": failures[path.name]})
                continue
            b = core.load_dev(output, fold, rival)
            rows[rival].append(core.compare(a["predictions"], b["predictions"]))
            hashes[path.name] = p.sha256(path)
    complete = {n: [f for f in rows[n] if "failure" not in f] for n in p.RIVALS}
    selecting = max(p.RIVALS, key=lambda n: len(complete[n]))
    if not complete[selecting]:
        raise ValueError("Neither rival completed a development fold; no paired diagnosis")
    near = sum(f["rival_only_near"] for f in complete[selecting])
    count = sum(f["rival_only_correct"] for f in complete[selecting])
    result = {
        "route": "threshold" if count and 2 * near > count else core.FEATURE,
        "selecting_rival": selecting, "complete_matched_folds": len(complete[selecting]), "near": near, "rival_only_cases": count,
        "by_rival_and_fold": rows, "failures": failures, "input_sha256": hashes,
        "plan_sha256": p.sha256(output / "paired_plan.json"),
        "amendment_sha256": p.sha256(Path(__file__)),
        "selection_policy": __doc__.strip(),
    }
    core.json_write(output / "diagnostic.json", result)
    print(json.dumps(result, indent=2))
    return result


def assess(output):
    core.diagnose = amended_diagnosis
    diag = amended_diagnosis(output)
    if diag["route"] != "threshold":
        core.assess(output)
        return
    plan = p.read_json(output / "paired_plan.json")
    details = {}
    for n in core.CONTROL:
        details[n] = []
        for f in range(len(p.DEV)):
            path = core.dev_path(output, f, n)
            details[n].append(core.threshold_study(core.load_dev(output, f, n)["predictions"],
                    plan["development"][f]["calibration_ids"]) if path.exists() else None)
    gains = [sum(row["adjusted_hits"] - row["default_hits"] for row in fold)
             for fold in details[core.ANCHOR]]
    result = {
        "route": diag["route"], "details": details, "plan_sha256": diag["plan_sha256"],
        "input_sha256": diag["input_sha256"], "decision_signed_gain_per_fold": gains,
        "promoted": all(g >= 0 for g in gains) and sum(gains) > 0,
    }
    core.json_write(output / "assessment.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("diagnose", "assess"))
    parser.add_argument("--output-dir", type=Path, default=p.RESULTS / "paired_errors")
    args = parser.parse_args()
    (amended_diagnosis if args.stage == "diagnose" else assess)(args.output_dir)


if __name__ == "__main__":
    main()
