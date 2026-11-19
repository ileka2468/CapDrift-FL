# Training-only CapDrift tuning

A class-stratified filename/content-group holdout reserved 153 images before model selection. Each development fold used 488 training and 122 development images drawn exclusively from the 610 outer training images. Client allocations differ between the inner and outer splits. The filename groups are proxies, not verified patient IDs.

1. Screened six candidate presets on the first 12 rounds of a 25-round schedule, with the previous projection controller and momentum-aware FedNova as non-selectable references. The two promoted candidates restarted on two fold/seed pairs and finished all 25 rounds.
2. Compared the selected short-memory preset against earlier server taper / stronger averaging and a tempered step-weighting alternative on the same fold/seed pairs. Promoting a refinement required a composite loss gain over 0.003, accuracy within 0.5 percentage point, and no fold's final loss rising by more than 0.02.
3. Tested a projected, momentum-step-normalized prototype on the same development pairs. Its composite loss was worse, so it was not promoted to the main implementation.

The development score is 70% final cross-entropy and 30% mean cross-entropy of the last five rounds. Lower is better. Validation metrics from the outer benchmark did not enter these scores or the promotion rule. The outer holdout belongs to the previously available dataset and is not an independent clinical cohort.

| Full-horizon development candidate | Composite loss | Final loss | Final accuracy | Disposition |
|---|---:|---:|---:|---|
| Short memory, earlier taper | 0.20134 | 0.20170 | 92.21% | Selected |
| Short memory | 0.20713 | 0.20786 | 92.62% | First-stage winner |
| Short memory, tempered weighting | 0.21476 | 0.21585 | 91.80% | Not selected |
| Full-budget cosine | 0.23657 | 0.23475 | 92.21% | First-stage finalist |
| Projected momentum-step-mass prototype | 0.26002 | 0.26065 | 92.21% | Not promoted |

- `dev_tuning_results.json`: all screen and finalist histories, source hashes and selection.
- `refinement_results.json`: paired refinement trials and promotion rules.
- `momentum_mass_challenger.json`: rejected prototype and paired development comparison.
- `selected_tuning_config.json`: original short-memory manifest.
- `selected_refinement_config.json`: final frozen manifest for CapDrift-Tuned.

The selected version projects surplus classifier updates, uses budget-scaled local rates, and retains the fixed [40, 28, 16, 5] step budget. Its server uses momentum 0.55, a 0.65 gain that starts cosine decay after round 7 and ends at 0.0325, and evaluation-only weight EMA with decay 0.85 from round 8. No checkpoint or seed was chosen from final holdout performance.

To repeat the screen in a new output directory from OmniFed:

    $env:PYTHONUTF8 = '1'
    & ..\.venv\Scripts\python.exe scripts/run_capdrift_tuning.py --output-dir results/research/capdrift_tuning_repro --reference FedNova-Momentum

The archived refinement and prototype scripts expect the recorded working directory and protocol folder. Keep the original reports; use an isolated copy of the output folder to reproduce those trials.
