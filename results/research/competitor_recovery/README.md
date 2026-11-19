# Recovering accuracy from the 94.38% CapDrift run

This experiment keeps the complete **short-memory-early-taper** preset as the
control. It does not start from the separate clipping-only variant, which had
a failed seed on the 423-image test. The archived control's manifest and
input/partition/environment fingerprints are verified before reuse.

## Competitor code audit

Momentum-aware FedNova divides the client update by the sum of its integrated
SGD-momentum coefficients and re-scales the aggregate by their sample-weighted
mean. The historical FedNova reference instead divides by the number of steps.
FedAvg and FedProx use five steps on every client and leave the faster tiers idle.
The code paths are in the [benchmark runner](../../../scripts/run_competitor_benchmark.py);
the [official FedNova code](https://github.com/JYWa/FedNova/blob/master/distoptim/FedNova.py#L138-L169)
uses the same momentum accumulation in its nonproximal case.

At client step counts [40, 28, 16, 5], the integrated coefficients are
approximately [311.33, 194.71, 86.68, 13.14]. Even after CapDrift's square-root
budget LR scaling, a *constant-gradient, pre-clipping proxy* gives the fastest
client 29.47% of the update mass versus 18.20% of training samples, and the
slowest 0.92% versus 4.75% of samples. This is not an estimate of learned
gradient importance; it indicates why raising fast-client LR by itself is a
poor default. The frozen controller also tapers server LR to 0.0325 and averaged
its model weights from round 8. It is less variable late in training than the
reference, but ends 1.31 accuracy points behind it on the 153-image target.
Ten net additional correct predictions across the five seed runs would tie;
eleven would strictly pass that archived FedNova-Momentum score.

## Ablations

- Taper floor: keep the original schedule and EMA, but raise the final LR
  ratio from .05 to .25.
- Later taper: delay onset to round 13 and retain a .20 final ratio.
- Prior adjustment: add a centered, smoothed log label-count offset to the
  **local training** logits only. This tests the class-skew mechanism of
  [Balanced Softmax](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html),
  not post-hoc test threshold tuning.
- Tempered mass: multiply each sample-weighted update by a bounded mixture
  of its original scaling and the ratio of the mean to local momentum
  coefficient. Use the LR-adjusted mass (integrated coefficient times actual
  client LR scale), bound the raw ratio to [.5, 2], and apply CapDrift clipping
  after that adjustment.
- Prior and progress: combine milder prior/mass correction with lower
  budget-LR exponent and a revised taper.

All ablations retain projection, clipping, the EMA evaluation rule,
[40, 28, 16, 5] steps, and the 205.62 s modeled runtime. The training plan uses
CUDA with fused SGD on the NVIDIA GeForce RTX 5070 Ti. CPU-only unit tests are
separate. Trials use training-only development folds and rank by round-25
accuracy with CE as a tiebreaker. Evaluation never changes the frozen winner.
The 153-image target and the 423-image source holdout were both scored in
earlier work; results here are fixed comparisons, not new independent
confirmation.

## Reproduce from OmniFed

    $env:PYTHONUTF8='1'
    $out = 'results/research/competitor_recovery_repro'
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage plan --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage screen --fold 0 --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage promote --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage screen --fold 1 --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage select --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage evaluate --holdout target --method champion --output-dir $out
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage evaluate --holdout source --method all --output-dir $out
    foreach ($split in @('target','source')) {
        & ..\.venv\Scripts\python.exe scripts/run_capdrift_recovery.py --stage report --holdout $split --output-dir $out
    }

Use a new output directory to start a new study. The stage commands reuse
completed records only when the plan fingerprints still agree.

## Verified result

The two-fold development accuracy moved from **92.62%** for the frozen anchor
to **93.44%** for prior-progress. Later taper reached 93.03%. The selected
configuration was frozen prior to either challenger evaluation.

| Method | Final accuracy, original 153 | Final accuracy, source 423 |
|---|---:|---:|
| Frozen CapDrift | 94.38% | **91.25%** |
| Prior-progress challenger | 94.64% | 91.11% |
| FedNova | 95.03% | 89.03% |
| FedNova-Momentum | **95.69%** | 88.61% |

Five seeds per row and split, 25 rounds per trial. Every completed trial
used [40,28,16,5] steps and 205.62 modeled seconds. The new GPU run of the
frozen controller beat both previously protocol-matched FedNova references
in all five paired seeds on the 423-image source set (+2.22 percentage points
over FedNova; +2.65 over FedNova-Momentum).

The challenger recovered **+0.26 percentage points** on the 153-image target
(two net correct classifications across all five seeds), with lower
cross-entropy (0.1169 vs 0.1391 for the frozen controller and 0.1278 for
FedNova-Momentum). It still missed both FedNova versions on endpoint accuracy.
It beat both FedNova means on the 423-image set, but was 0.14 points below
the frozen CapDrift and had higher cross-entropy (0.2888 vs 0.2544). Do not
promote it as the default on this evidence. Keep the unchanged model as
the primary CapDrift result; show both splits. The study does not identify
which member of the combined prior-progress change caused the small gain.

The original seed 42 was rerun on the RTX 5070 Ti with the frozen manifest.
All 25 round accuracies, losses, raw-model summaries if present, client
step counts and modeled durations matched the archived record exactly.

[Audited summary](recovery_accuracy_summary.csv) · [Paired plot](paired_accuracy.png)
· [Frozen challenger](selected_config.json) · [153-image audit](report_target.json)
· [423-image audit](report_source.json)
