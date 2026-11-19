# CapDrift-FL projection benchmark

## Design

The benchmark compares the projected controller, the original hard-gate
controller (CapDrift-Legacy), FedAvg, FedProx, and the repository's historical
step-normalized FedNova variant. All methods use the same MobileNetV3-Small
ImageNet initialization, shared base SGD learning rate (0.002), frozen
pretrained BatchNorm running statistics, paired seed list, and fixed
image-level split and client partition. The FedNova baseline uses SGD with
local momentum and step-count normalization, not the momentum-aware
normalizer from [the FedNova paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html).
Do not compare these scores directly with the old frozen-server/local-BN
protocol and attribute the difference to the controller.

The candidate attenuates each surplus block's classifier displacement
orthogonal to the previous applied server direction by max(0, cosine).
It preserves the signed parallel displacement and does not project the
required 5 steps or the backbone. With no usable previous direction it makes
an ordinary update. Each client scales its base step size by sqrt(5 / budget).
The server applies Polyak momentum after clipping (beta 0.9, server LR 0.2).
The ceiling is 40 through round 15 and then
max(5, floor(40 * (25 - round) / 25)). The qualification settings were chosen
from a nested development split, rather than selecting a run or checkpoint on
the 153-image validation split.

## Reproduce from OmniFed

Use a fresh output directory. The dataset is auto-detected in this checkout or
its parent; for a different layout use --data-dir with the two-class ImageFolder
train directory.

    $env:PYTHONUTF8 = '1'
    & ..\.venv\Scripts\python.exe scripts/run_competitor_benchmark.py --output-dir results/research/competitor_projection_repro --methods CapDrift-FL FedAvg FedProx FedNova CapDrift-Legacy --freeze-bn --channels-last --fused-sgd --server-momentum 0.9 --server-lr 0.2 --local-lr 0.002 --baseline-local-lr 0.002
    & ..\.venv\Scripts\python.exe scripts/plot_competitor_comparison.py --input results/research/competitor_projection_repro/competitor_benchmark_results.json --output results/research/competitor_projection_repro/fig_competitor_comparison.png

## Interpretation

This is 25 rounds at N=4 and alpha=0.5, with five seeds
(42, 101, 2024, 7, 888) and the same 610/153 split for each seed. The reported
variability is population standard deviation over training seeds, not a
confidence interval for unseen patients. No exact image bytes were shared
between splits. Eleven filename-based subject proxies appeared in both splits;
patient-disjoint evaluation with verified IDs is still needed.

The 205.62 s completion time is a model, not measured distributed latency.
It is the maximum modeled client compute-plus-network duration in each round.
Reported active utilization counts both compute and communication; the CSV
also reports compute-only utilization. With the same step ceiling and network
model, removing steps in the last ten rounds necessarily reduces active
utilization against a method that executes the largest feasible budget every
round. A useful next experiment is a patient-group-disjoint validation with a
training-only sweep of server gain/momentum and ablations of shared BatchNorm,
LR scaling, projection, and annealing.

## Verified final metrics

All 25 trials succeeded. Shown as mean +/- population SD across five seeds.

| Method | Successful Trials | Final Acc (%) | Peak Acc (%) | Final Loss | Modeled Time (s) | Idle Waste (s) | Active Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CapDrift-FL** | 5/5 | **93.59 ± 1.33** | 95.69 ± 0.67 | 0.2549 ± 0.0755 | 205.62 ± 0.00 | 306.06 ± 0.00 | 62.79 ± 0.00 |
| **FedAvg** | 5/5 | **92.29 ± 2.91** | 94.77 ± 0.58 | 0.2126 ± 0.0804 | 205.62 ± 0.00 | 522.15 ± 0.00 | 36.52 ± 0.00 |
| **FedProx** | 5/5 | **92.68 ± 1.72** | 94.77 ± 0.41 | 0.1941 ± 0.0287 | 205.62 ± 0.00 | 522.15 ± 0.00 | 36.52 ± 0.00 |
| **FedNova** | 5/5 | **94.64 ± 1.05** | 95.82 ± 0.67 | 0.2319 ± 0.0499 | 205.62 ± 0.00 | 197.84 ± 0.00 | 75.95 ± 0.00 |
| **CapDrift-Legacy** | 5/5 | **92.42 ± 1.88** | 94.38 ± 0.32 | 0.2288 ± 0.0566 | 205.62 ± 0.00 | 475.82 ± 5.30 | 42.15 ± 0.64 |

The projected controller exceeded the requested accuracy and loss thresholds
and improved mean accuracy and utilization against FedAvg and CapDrift-Legacy.
It did not dominate: FedNova led on accuracy and utilization, and every
comparison method had lower mean final loss. The projected controller's mean
validation loss reached 0.1591 at round 18 and rose to 0.2549 at round 25;
the table uses the predeclared final round, not the best checkpoint.

The independent audit verified the 6.5 s compute deadline across every client
and round (observed maximum 6.316976875 modeled seconds) and identical
205.619721369 modeled seconds per trial. The full JSON and CSV include the
mean compute-only utilization and the measured single-GPU trial wall times.
