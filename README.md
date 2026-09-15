# OmniFed: Capability- and Drift-Aware Federated Learning

OmniFed (CapDrift-FL) is a high-performance federated learning system designed for heterogeneous edge AI environments.
It dynamically budgets client local workloads based on measured processing capability while adaptively gating client drift on non-IID data distributions.

## Key Features
- **Capability-Aware Workload Scheduling:** Smooths client processing throughput (steps/sec) and assigns differentiated local step ceilings to prevent straggler idle waiting.
- **Directional Drift Gating:** Periodically evaluates client model trajectory against global consensus displacements and halts surplus local training early if drift exceeds threshold.
- **Contribution-Bounded Aggregation:** Employs median-calibrated sample-weighted clipping to eliminate outsized parameter updates from high-throughput nodes.
- **Enterprise Edge Deployment:** Native Flower strategy adapter tested on physical multi-node Cisco UCS C220 M4 rackmount servers running Proxmox VE LXC containers.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
