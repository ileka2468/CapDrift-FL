# OmniFed: Capability- and Drift-Aware Federated Learning

OmniFed (CapDrift-FL) is a high-performance federated learning system designed for heterogeneous edge AI environments.
It dynamically budgets client local workloads based on measured processing capability while adaptively gating client drift on non-IID data distributions.

## Key Empirical Findings (25 Rounds, Multi-Seed Evaluation)
- **45.9% Faster than FedNova:** CapDrift-FL converges in 213.2 s vs. 394.0 s for FedNova while avoiding late-stage parameter divergence on non-IID data (59.7% vs. 55.5% final accuracy).
- **Directional Drift Gating:** Dynamically eliminates surplus training when cosine similarity to consensus displacement breaches $\tau_d \in [0.2, 1.0]$, preserving synchronization efficiency.
- **Physical Testbed Verification:** Evaluated on Cisco UCS C220 M4 enterprise servers with Proxmox VE LXC containers, reducing barrier idle waiting by $3.25\times$.

## Repository Structure
- `src/`: Core federated learning implementation, capability estimator, drift gating controller, and Flower strategy adapter.
- `scripts/`: Benchmark runners, parameter sweeps, plotting utilities, and numerical reproducibility verifiers.
- `results/`: Multi-seed experimental telemetry logs (JSON and CSV) for ablation, competitor, and sensitivity benchmarks.
- `docs/`: IEEE conference manuscript source (`main.tex`, `OVERLEAF_CapDrift_FL.tex`) and publication figures.

## Reproducibility
```bash
python scripts/verify_numerical_reproducibility.py
python scripts/run_all_remaining_simulations.py
```
