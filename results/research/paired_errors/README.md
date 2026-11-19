# Paired CapDrift / FedNova error analysis

## Result

| Round-25 development accuracy | Fold 0 (seed 42) | Fold 1 (seed 2024) | Mean | Mean CE |
|---|---:|---:|---:|---:|
| Frozen CapDrift | 91.80% (112/122) | 93.44% (114/122) | 92.62% | 0.2180 |
| Half feature LR, unchanged classifier LR | 90.98% (111/122) | 92.62% (113/122) | 91.80% | 0.2548 |

The variant lost one prediction in each fold and had higher loss in both. It did
not qualify for either benchmark. Every completed run used an RTX 5070 Ti,
fused SGD, 25 rounds and [40,28,16,5] client steps (205.62 modeled seconds).
The frozen CapDrift control reproduced all 25 archived rounds in both folds.

## Which errors motivated the experiment?

In the only completed matched FedNova comparison, historical FedNova reached
114/122 vs CapDrift's 112/122. Both exclusive disagreements were label 1;
only one CapDrift error was near the threshold (model score between .35 and
.65) and neither assigned .80 probability to the wrong class. With only two
discordant cases, this does not establish a dominant mechanism. Threshold
adjustment was not selected or tested on the benchmark.

FedNova-Momentum produced nonfinite loss at rounds 10 and 2 in folds 0 and 1;
historical FedNova did so at round 15 in fold 1. No completion was imputed and
no replacement split was selected. The post-failure amendment used the one
available matched fold to choose the direction, then compared the LR variant
against frozen CapDrift on both development folds. Development scenarios are
not new independent test data. Previously reported five-seed FedNova results
on the 153 and 423 images are unchanged.

See [paired errors](diagnostic.json), [development decision](assessment.json),
and [verification and exact commands](evidence/VERIFICATION.txt).
