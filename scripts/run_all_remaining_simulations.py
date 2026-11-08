"""
Master Orchestration Script for CapDrift-FL Extended Multi-Seed Benchmarks.
Runs sequentially:
1. run_capdrift_ablation.py (25 rounds, seeds [42, 101, 2024])
2. run_competitor_benchmark.py (25 rounds, seeds [42, 101, 2024])
3. run_non_iid_skew_sweep.py (15 rounds, seeds [42, 101, 2024])
4. Re-plots all 4 IEEE figures:
   - plot_capdrift_ablation.py
   - plot_cluster_multinode_scaling.py
   - plot_competitor_comparison.py
   - plot_non_iid_skew_sweep.py
5. Generates the final 11-slide presentation PPTX:
   - generate_final_presentation_pptx.py
6. Packages docs/Overleaf_Figures.zip
"""

import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_capdrift_ablation import execute_ablation_study
from scripts.run_competitor_benchmark import main as run_competitors
from scripts.run_non_iid_skew_sweep import main as run_skew
from scripts.plot_capdrift_ablation import generate_ablation_plots
from scripts.plot_cluster_multinode_scaling import generate_scaling_plots
from scripts.plot_competitor_comparison import plot_competitors
from scripts.plot_non_iid_skew_sweep import plot_skew_sweep
from scripts.generate_final_presentation_pptx import build_presentation


def main():
    start_time = time.time()
    print("=" * 85)
    print("  CAPDRIFT-FL MASTER EXPERIMENT & FIGURE ORCHESTRATION PIPELINE")
    print("=" * 85)

    # 1. CapDrift Ablation Benchmark (25 rounds, 3 seeds)
    print("\n>>> [STEP 1/6] Running 5-Way Control Ablation (25 Rounds, Seeds [42, 101, 2024])...")
    execute_ablation_study(alpha=0.5, num_rounds=25, seeds=[42, 101, 2024])

    # 2. Competitor Benchmark (25 rounds, 3 seeds)
    print("\n>>> [STEP 2/6] Running Competitor Benchmark (FedAvg, FedProx, FedNova, CapDrift-FL)...")
    run_competitors()

    # 3. Non-IID Skew Sweep (15 rounds, 3 seeds, alphas [0.1, 0.3, 0.5])
    print("\n>>> [STEP 3/6] Running Non-IID Skew Sensitivity Sweep...")
    run_skew()

    # 4. Re-plot All 4 Figures
    print("\n>>> [STEP 4/6] Rendering All 4 Publication-Quality Figures...")
    generate_ablation_plots()
    generate_scaling_plots()
    plot_competitors()
    plot_skew_sweep()

    # 5. Build PowerPoint Presentation
    print("\n>>> [STEP 5/6] Building Final Presentation (.pptx)...")
    build_presentation()

    # 6. Package Overleaf Figures Zip
    print("\n>>> [STEP 6/6] Packaging docs/Overleaf_Figures.zip...")
    figures = [
        PROJECT_ROOT / "docs" / "figures" / "fig_ablation_accuracy_convergence.png",
        PROJECT_ROOT / "docs" / "figures" / "fig_cluster_multinode_scaling.png",
        PROJECT_ROOT / "docs" / "figures" / "fig_competitor_comparison.png",
        PROJECT_ROOT / "docs" / "figures" / "fig_non_iid_skew_robustness.png",
    ]
    zip_path = PROJECT_ROOT / "docs" / "Overleaf_Figures.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fig in figures:
            if fig.exists():
                zf.write(fig, arcname=fig.name)
                print(f"   Added {fig.name} to {zip_path.name}")
            else:
                print(f"   [WARNING] Missing {fig.name}!")

    elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"  ALL BENCHMARKS, FIGURES, AND SLIDES COMPLETED SUCCESSFULLY IN {elapsed:.1f}s ({elapsed/60:.1f}m)!")
    print("=" * 85)


if __name__ == "__main__":
    main()

