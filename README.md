# OmniFed / CapDrift-FL
### Capability- and Drift-Aware Federated Learning for Heterogeneous Edge Systems
**DePaul University — School of Computing**  
**NET 470: Cloud Networking Technologies — Fall 2026**  
**Authors:** William Ileka & Brendan Kechtban  
**Professor:** Dr. Nazli Siasi  

---

## Overview

OmniFed (evaluating the **CapDrift-FL** controller) is a federated learning framework built to handle real hardware disparities and non-IID data skew across edge and fog environments.

In standard synchronous federated learning (like FedAvg), every client runs the exact same local workload. When you mix fast GPUs with multi-core server CPUs, faster machines finish their work in seconds and sit completely idle waiting for slower workers to reach the barrier. While giving faster machines more local training steps eliminates that idle waste, doing so under non-IID data causes high-throughput clients to overfit to their local data, dragging the global model off course (client drift) and dominating aggregation.

CapDrift-FL addresses this by decoupling local execution into three coordinated mechanisms:
1. **Runtime Capability Budgeting:** Dynamically estimates each client's processing throughput (steps/sec) with exponential moving average (EMA) smoothing to set achievable local step ceilings for a target round duration.
2. **Surplus Directional Drift Gating:** Requires all participating clients to complete a mandatory minimum workload ($S_{\min} = 5$), then evaluates surplus local computation blocks using full-model cosine distance against recent global trajectory updates. If an update drifts off course ($d_i > \tau_d$), surplus training is stopped immediately.
3. **Contribution-Bounded Aggregation:** Applies median-calibrated norm clipping to sample-weighted client updates at the coordinator, preventing fast nodes from overpowering global consensus.

The system was evaluated on a physical bare-metal testbed consisting of three enterprise Cisco UCS C220 M4 rackmount servers running Proxmox VE 8.2 LXC containers alongside an NVIDIA GeForce RTX 5070 Ti workstation coordinator.

---

## Physical Testbed Architecture

```
                     +------------------------------------------------+
                     |          CENTRAL COORDINATOR / HOST            |
                     |       Host Workstation (192.168.30.102)        |
                     |  - Hardware: NVIDIA RTX 5070 Ti (CUDA 13.0)    |
                     |  - Coordinator: Flower Strategy (gRPC :8080)   |
                     |  - Local Role: Central Evaluator & Client 4    |
                     +-----------------------+------------------------+
                                             |
                        +--------------------+--------------------+
                        | (gRPC / LAN)       | (gRPC / LAN)       | (gRPC / LAN)
                        | Subnet: .20.0/24   | Subnet: .10.0/24   | Subnet: .40.0/24
                        v                    v                    v
               +-----------------+  +-----------------+  +-----------------+
               |   EDGE HOST 1   |  |   EDGE HOST 2   |  |   EDGE HOST 3   |
               |  (Cisco C220)   |  |  (Cisco C220)   |  |  (Cisco C220)   |
               +-----------------+  +-----------------+  +-----------------+
               | Proxmox pve01   |  | Proxmox pve02   |  | Proxmox pve03   |
               | CT 201 (Ubuntu) |  | CT 202 (Ubuntu) |  | CT 203 (Ubuntu) |
               | Dual Intel Xeon |  | Dual Intel Xeon |  | Dual Intel Xeon |
               | 192.168.20.100  |  | 192.168.10.100  |  | 192.168.40.100  |
               +-----------------+  +-----------------+  +-----------------+
```

### Hardware Breakdown
- **Workstation Coordinator:** AMD Ryzen 5 5600X (6 Cores / 12 Threads @ 3.7 GHz), 64 GB DDR4 RAM, NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7 VRAM).
- **Cisco UCS C220 M4 Host 1 (`pve01`):** Dual Intel Xeon E5-2680 v4 (28 Cores / 56 Threads @ 2.40 GHz, 64 GB ECC RAM).
- **Cisco UCS C220 M4 Host 2 (`pve02`):** Dual Intel Xeon E5-2620 v4 (16 Cores / 32 Threads @ 2.10 GHz, 48 GB ECC RAM).
- **Cisco UCS C220 M4 Host 3 (`pve03`):** Dual Intel Xeon E5-2620 v4 (16 Cores / 32 Threads @ 2.10 GHz, 48 GB ECC RAM).
- **Network Fabric:** Dedicated Gigabit Ethernet switch interconnecting the 4 subnets.

---

## Quickstart

### 1. Environment Setup
```powershell
# Navigate into the repository
cd OmniFed

# Activate Python 3.10+ virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the 5-Way Ablation Benchmark
To reproduce the component ablation study across FedAvg, Capability-Only, Drift-Only, Capability+Drift, and Full CapDrift-FL:
```powershell
python scripts/run_capdrift_ablation.py
```

### 3. Run External Competitor Benchmarks
To compare CapDrift-FL against FedAvg, FedProx, and FedNova:
```powershell
python scripts/run_competitor_benchmark.py
```

### 4. Run the Physical Cluster Launcher
To dispatch training across the Proxmox LXC containers and the workstation coordinator:
```powershell
.\scripts\launch_entire_cluster.ps1 -Rounds 6
```

### 5. Launch the Diagnostic Studio GUI
To inspect live classification, Grad-CAM attention heatmaps, and telemetry:
```powershell
.\scripts\run_diagnostic_studio.ps1
# Or launch the web studio directly:
python src/app.py
```

---

## Repository Structure

```
OmniFed/
|-- config/
|   |-- default_config.json        # Hyperparameters for training and controller
|   +-- cluster_nodes.json         # Node registry (IPs, container IDs, hardware tiers)
|-- src/
|   |-- model.py                   # Dynamic model factory (MobileNetV3-Small, ResNet-18)
|   |-- dataset.py                 # Chest X-ray preprocessing & non-IID partitioner
|   |-- server.py                  # Flower coordinator implementation
|   |-- client.py                  # Edge client training loop
|   |-- cluster_launcher.py        # Automated cluster launcher across SSH & local workers
|   |-- inference.py               # Grad-CAM heatmap visualization and evaluation
|   |-- node_manager.py            # Node health check and ping latency tracking
|   |-- app.py                     # FastAPI telemetry and diagnostics web service
|   +-- research/
|       |-- capdrift_controller.py     # Three-stage controller (capability, drift, clipping)
|       |-- capdrift_flower_adapter.py # Wire adapter bridging controller to Flower strategy
|       |-- heterogeneous_fleet.py     # Hardware profile definitions and telemetry tracking
|       +-- non_iid.py                 # Dirichlet label skew partition generator
|-- scripts/
|   |-- run_capdrift_ablation.py       # Matched 5-way ablation experiment suite
|   |-- run_competitor_benchmark.py    # Comparison vs FedNova, FedProx, FedAvg
|   |-- run_non_iid_skew_sweep.py      # Dirichlet skew sweep (alpha in {0.1, 0.3, 0.5})
|   |-- run_cluster_5client_benchmark.py # Physical 5-client benchmark across Proxmox LXC
|   |-- run_cluster_scaling_benchmark.py # Multi-node scaling benchmark (1 to 5 nodes)
|   |-- launch_entire_cluster.ps1      # Orchestrates Proxmox LXC execution over SSH
|   +-- deploy_cluster_code.py         # Automated deployment utility for edge containers
|-- results/
|   |-- fl_metrics.json                # Recorded baseline training metrics
|   |-- federated_privacy_capture.pcap # Wireshark trace verifying raw data isolation
|   |-- cluster_runs/                  # Cisco UCS 5-client cluster telemetry logs
|   +-- research/                      # Telemetry logs (.json / .csv) for all paper tables
+-- docs/
    |-- main.tex                       # IEEE conference manuscript (LaTeX source)
    |-- OVERLEAF_CapDrift_FL.tex       # Overleaf synchronization source
    +-- figures/                       # High-resolution publication figures (Section III-V)
```

---

## Experimental Findings Summary

### 1. Physical 5-Client Cluster (Cisco UCS + RTX 5070 Ti)
- **Idle Waste Reduction:** Cut barrier synchronization idle waiting from **3.41 s down to 1.05 s** per round (a 69.1% reduction).
- **Execution-Time Utilization:** Increased active compute utilization from **80.5% (FedAvg) to 98.2% (CapDrift-FL)**.
- **Data Locality:** Wireshark captures on the Proxmox bridge interface confirmed that 100% of wire traffic on port 8080 consisted of serialized protobuf parameter arrays. Raw medical scans remained isolated on client local storage.

### 2. 5-Way Control Component Ablation ($\alpha = 0.5$, 8 rounds)
- **FedAvg Baseline:** Reached 67.97% peak and dropped to 60.13% final accuracy (loss 0.6027), wasting 167.1 s in idle waiting.
- **Capability-Only:** Boosted initial throughput and peaked at 88.89% in Round 5, but overfitted to biased partitions in later rounds, dropping to 79.74% final accuracy (loss 0.5191).
- **CapDrift-FL (Full):** Gated 28 divergent surplus blocks and clipped 1 outlier update, maintaining **88.89% final accuracy** with the lowest loss (**0.3128**) and completing in 73.3 s.

### 3. Comparison with Competitor Baselines
- **FedNova:** Normalizes step counts during aggregation to resolve objective inconsistency, but without proactive drift gating it fluctuated wildly between rounds (falling to 66.67% in Round 2 and 71.90% in Round 4), taking 126.1 s to finish.
- **FedProx ($\mu = 0.01$):** Proximal regularization restricted updates near the global model, stabilizing training at 71.24% peak accuracy, but remained slowed down by uniform straggler waiting.
- **CapDrift-FL:** Finished 41.8% faster than FedNova (73.3 s vs. 126.1 s) with monotonic convergence and superior class balance.

---

## Course & Academic Context

This project was developed for **NET 470: Cloud Networking Technologies** in the School of Computing at DePaul University (Fall Quarter 2026).  
**Course Professor:** Dr. Nazli Siasi

```bibtex
@article{ileka2026capdrift,
  author      = {William Ileka and Brendan Kechtban},
  title       = {CapDrift-FL: Capability- and Drift-Aware Federated Learning for Heterogeneous Edge AI Systems},
  journal     = {NET 470 Cloud Networking Technologies Project Report},
  institution = {School of Computing, DePaul University},
  year        = {2026}
}
```
