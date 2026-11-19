# Accuracy-first CapDrift ablation

## Question and controls

Does contribution clipping, classifier projection, server gain taper, or
evaluation-only averaging improve final accuracy over momentum-aware FedNova
without changing its local optimizer or training budget? The historical
step-count-only FedNova variant is also reported in the final comparison.
All methods use ImageNet-initialized MobileNetV3-Small, SGD with momentum 0.9
and base LR 0.002, frozen BatchNorm running statistics, paired RNG, and the
same four-tier fleet and data. The modeled work budget remains [40, 28, 16, 5]
steps in every round. Timing is simulated, not measured on four devices.

The six ablations were screened at round 25 on one development split from
the existing 610 training images. The two highest-accuracy variants were
rerun on a second development split and different client partition. Selection
used mean round-25 accuracy across the development scenarios, then mean
cross-entropy only as a tiebreaker. Neither the model choices nor the
checkpoint were selected on the final holdout.

Clipping-only and average-only advanced. Clipping-only was the only finalist
to complete both development scenarios (95.90% and 90.98%); on the second
scenario, the momentum-aware reference and average-only produced nonfinite
loss in round 5. Projection-only and projection+clipping+taper produced
nonfinite loss in round 6 of screening. Those failures remain in
development.json. The winner adds just CapDrift's norm clipping (1.5 times
the median contribution norm per round) **after** momentum-aware
normalization. It does not use projection, server taper, or EMA.

## Reserved test set

Of 624 images in the source dataset's original test folder, 201 were excluded
because of filename-group overlap with the entire previous 763-image
benchmark cohort or duplicate bytes. The benchmark uses all 423 eligible
images (167 normal, 256 pneumonia) from 312 inferred groups. These are
filename/content proxies, not verified patient IDs or a different hospital
cohort. The new scores should not be compared numerically with the earlier
153-image validation results as if they were the same test.

## Reproduce

From OmniFed, use a new output directory. The default input locations are
within this checkout's parent; the runner also accepts --data-dir and
--source-test. Dev and evaluation checkpoint each completed trial.

    $env:PYTHONUTF8 = '1'
    $out = 'results/research/competitor_accuracy_repro'
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_accuracy.py --stage plan --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_accuracy.py --stage develop --output-dir $out
    foreach ($method in @('CapDrift-Accuracy','FedNova-Momentum','FedNova')) {
        & ..\.venv\Scripts\python.exe scripts/run_capdrift_accuracy.py --stage evaluate --method $method --output-dir $out
    }
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_accuracy.py --stage report --output-dir $out

The original development runner recorded all ten trials and then exited
because it required a complete reference. Its original JSON and source
are preserved in the verification bundle. The current runner retains
reference failures and replays the same selection rule; the original training
and development measurements were not changed or repeated.

## Final result: no qualified win

| Method | Completed | Final accuracy, matched 4 seeds | Balanced accuracy, matched 4 | Cross-entropy, matched 4 | Final accuracy, all 5 seeds |
|---|---:|---:|---:|---:|---:|
| CapDrift-Accuracy | 4/5 | 89.07% | 87.04% | 0.4400 | not available |
| FedNova-Momentum | 5/5 | 89.01% | 86.96% | 0.4324 | 88.61% |
| FedNova | 5/5 | 88.89% | 86.60% | 0.4752 | 89.03% |

CapDrift-Accuracy failed on seed 7 at round 3 with nonfinite loss. Both
FedNova references completed all five seeds. The four completed paired
comparisons (42, 101, 2024, 888) show +0.06 percentage points versus
FedNova-Momentum and +0.18 versus the historical FedNova implementation,
with two wins and two losses against each. These are descriptive survivor
comparisons, not five-seed evidence of superiority. No score was imputed.
Every completed trial took 205.62 modeled seconds and had 75.95% modeled
active utilization; the maximum audited client compute was 6.317 s.

The strict report command rejects this incomplete comparison (exit 1).
To reproduce the incomplete-attempt audit and the plot, preserving failure:

    & ..\.venv\Scripts\python.exe results/research/competitor_accuracy/audit_attempts.py results/research/competitor_accuracy

The full per-round histories, failed attempt, matched-seed means, and
configuration hashes are recorded in the JSON and CSV files in this folder.
Treat nonfinite updates as a reliability failure, not as a reason to change
the seed set or select another configuration on this holdout.
