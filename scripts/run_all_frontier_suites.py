"""
Master Orchestrator for CapDrift-FL Frontier Validation Suites:
Executes:
1. Suite A: Extreme Pathological Dirichlet Skew Sweep (alpha in {0.05, 0.10, 0.20})
2. Suite B: Hardware Heterogeneity Disparity Ratio Sweep (rho in {2x, 5x, 10x, 20x})
3. Suite D: Dataset Volume Cohort Scaling (M in {763, 1500, 3000, 5216} images)

Saves all summary tables, JSON metadata, and publication-ready comparative multi-panel plots.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_suite_a_skew import run_suite_a
from run_suite_b_heterogeneity import run_suite_b
from run_suite_d_dataset_scaling import run_suite_d

def main():
    parser = argparse.ArgumentParser(description="Master Frontier Suites Orchestrator")
    parser.add_argument("--skip-a", action="store_true", help="Skip Suite A (Extreme Skew)")
    parser.add_argument("--skip-b", action="store_true", help="Skip Suite B (Heterogeneity Sweep)")
    parser.add_argument("--skip-d", action="store_true", help="Skip Suite D (Dataset Scaling)")
    parser.add_argument("--rounds", type=int, default=25, help="Rounds per trial (default: 25)")
    args = parser.parse_args()

    start_total = time.time()
    print("================================================================================")
    print("   CapDrift-FL Frontier Empirical Validation Battery Launched")
    print(f"   Target Compute Deadline: 6.5s per round (205.62s modeled runtime)")
    print("================================================================================", flush=True)

    if not args.skip_a:
        t0 = time.time()
        print("\n>>> STARTING SUITE A: Extreme Pathological Dirichlet Skew Sweep (alpha in {0.05, 0.10, 0.20})...", flush=True)
        run_suite_a(rounds=args.rounds)
        print(f">>> COMPLETED SUITE A in {time.time() - t0:.1f}s\n", flush=True)

    if not args.skip_b:
        t0 = time.time()
        print("\n>>> STARTING SUITE B: Hardware Heterogeneity Disparity Sweep (rho in {2x, 5x, 10x, 20x})...", flush=True)
        run_suite_b(rounds=args.rounds)
        print(f">>> COMPLETED SUITE B in {time.time() - t0:.1f}s\n", flush=True)

    if not args.skip_d:
        t0 = time.time()
        print("\n>>> STARTING SUITE D: Dataset Volume Cohort Scaling (M in {763, 1500, 3000, 5216})...", flush=True)
        run_suite_d(rounds=args.rounds)
        print(f">>> COMPLETED SUITE D in {time.time() - t0:.1f}s\n", flush=True)

    total_elapsed = time.time() - start_total
    print("\n================================================================================")
    print(f"   All Frontier Validation Suites Finished Successfully in {total_elapsed/60.0:.2f} minutes!")
    print("================================================================================", flush=True)

if __name__ == "__main__":
    main()
