# CapDrift-Tuned: full-budget comparison

On the same 153-image filename-group holdout, the refined preset improved the
previous projection controller by 1.18 percentage points in mean final accuracy,
0.0454 lower mean cross-entropy, and 13.16 percentage points in modeled active
utilization. It scored higher accuracy and lower loss on four of the five paired
seeds against that previous controller.

The selected preset stayed at [40, 28, 16, 5] steps in all rounds. It kept
the classifier projection and local step-budget LR scaling, used server
momentum 0.55 with a cosine gain taper from 0.65 after round 7 to 0.0325,
and evaluated an exponential weight average (decay 0.85, starting round 8).
Selection used the training-side development folds only. See the
[development report](../capdrift_tuning/README.md) and
[frozen manifest](../capdrift_tuning/selected_refinement_config.json).

## Held-out results

Five paired seeds (42, 101, 2024, 7, 888); 25 rounds, N=4, alpha=0.5.
Accuracy and cross-entropy are at round 25, not a selected checkpoint.
Values are mean +/- population SD across seeds.

| Method | Final accuracy (%) | Final cross-entropy | Active utilization (%) |
|---|---:|---:|---:|
| CapDrift-Tuned | 94.38 ± 0.67 | 0.1391 ± 0.0236 | 75.95 ± 0.00 |
| FedNova-Momentum | 95.69 ± 0.67 | 0.1278 ± 0.0220 | 75.95 ± 0.00 |
| FedNova | 95.03 ± 1.06 | 0.1449 ± 0.0296 | 75.95 ± 0.00 |
| FedAvg | 93.73 ± 2.13 | 0.1660 ± 0.0475 | 36.52 ± 0.00 |
| FedProx | 92.55 ± 2.13 | 0.1887 ± 0.0574 | 36.52 ± 0.00 |
| CapDrift-Legacy | 94.64 ± 0.87 | 0.1359 ± 0.0131 | 40.68 ± 1.29 |
| CapDrift-FL | 93.20 ± 1.97 | 0.1845 ± 0.0597 | 62.79 ± 0.00 |

All 35 trials completed, and the independent audit checked 875 per-round
records. Model-based completion time was 205.619721369 s for every trial,
and maximum client compute was 6.316976875 s under a 6.5 s deadline. The 75.95%
active utilization is the ceiling with these step limits and timing model,
not measured GPU occupancy.

FedNova-Momentum had the highest final accuracy and lowest final loss; the
tuned controller trailed it by 1.31 percentage points and 0.0113 loss, with
equal utilization. The tuned controller did have a lower mean loss over
rounds 21–25 (0.1381 vs. 0.1647). CapDrift-Legacy also had slightly better
endpoint accuracy and loss, but used 35.27 percentage points less capacity.
The earlier short-memory preset (before refinement) scored 94.12% and 0.1510
at the same utilization. Within this search, the earlier-taper preset is
the recommended full-budget CapDrift variant, not a global optimum.

The inference groups are filename/content proxies (zero overlap), not verified
patient identities. This one cohort and one outer client partition do not
establish generalization across sites or label-skew settings.

## Reproduce from OmniFed

Use a new path in place of the output directory. The same command can run all
seven methods together with the frozen manifest.

    $env:PYTHONUTF8 = '1'
    & ..\.venv\Scripts\python.exe scripts/run_competitor_benchmark.py --output-dir results/research/competitor_tuning_repro --methods CapDrift-Tuned FedNova-Momentum FedNova FedAvg FedProx CapDrift-Legacy CapDrift-FL --split-policy filename-group --freeze-bn --channels-last --fused-sgd --server-momentum 0.9 --server-lr 0.2 --local-lr 0.002 --baseline-local-lr 0.002 --tuned-config results/research/capdrift_tuning/selected_refinement_config.json
    & ..\.venv\Scripts\python.exe scripts/plot_competitor_comparison.py --input results/research/competitor_tuning_repro/competitor_benchmark_results.json --output results/research/competitor_tuning_repro/fig_competitor_comparison.png

FedNova-Momentum uses the momentum normalizer in the
[authors' implementation](https://github.com/JYWa/FedNova/blob/master/distoptim/FedNova.py).
FedNova is the historical step-count-only reference. All methods shared
pretrained MobileNetV3-Small, frozen BN running statistics, SGD with the same
base learning rate, and deterministic per-client/round RNG.
