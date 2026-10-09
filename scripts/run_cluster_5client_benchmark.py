"""
Physical Proxmox 5-Client Cluster Benchmark Orchestrator.
Coordinates real federated learning across 5 LXC containers distributed over
the 3 physical Cisco UCS fog servers (pve01, pve02, pve03) and the RTX 5070 Ti coordinator.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 5 Physical LXC Clients distributed across 3 Cisco UCS hosts:
PHYSICAL_5_FLEET = [
    # Host 1: pve01 (Dual Xeon E5-2680 v4) - 2 Clients
    {"host": "192.168.20.100", "ctid": 201, "node_id": "hospital_01_pve01", "data_dir": "/opt/fl_client/data/node_1"},
    {"host": "192.168.20.100", "ctid": 205, "node_id": "hospital_02_pve01", "data_dir": "/opt/fl_client/data/node_2"},

    # Host 2: pve02 (Dual Xeon E5-2620 v4) - 2 Clients
    {"host": "192.168.10.100", "ctid": 202, "node_id": "hospital_03_pve02", "data_dir": "/opt/fl_client/data/node_9"},
    {"host": "192.168.10.100", "ctid": 213, "node_id": "hospital_04_pve02", "data_dir": "/opt/fl_client/data/node_10"},

    # Host 3: pve03 (Dual Xeon E5-2620 v4) - 1 Client
    {"host": "192.168.40.100", "ctid": 203, "node_id": "hospital_05_pve03", "data_dir": "/opt/fl_client/data/node_15"}
]


def stop_all_cluster_clients():
    """Stops any lingering client python processes on all containers."""
    for c in PHYSICAL_5_FLEET:
        host = c["host"]
        ctid = c["ctid"]
        cmd = f"pct exec {ctid} -- pkill -f 'cluster_client_entry.py' 2>/dev/null || true"
        subprocess.run(["ssh", "-o", "BatchMode=yes", f"root@{host}", cmd], capture_output=True)


def launch_cluster_clients(server_ip: str, port: int, mode: str):
    """Launches the 5 distributed edge clients concurrently via SSH."""
    for c in PHYSICAL_5_FLEET:
        host = c["host"]
        ctid = c["ctid"]
        node_id = c["node_id"]
        data_dir = c["data_dir"]

        client_cmd = (
            f"nohup /opt/fl_client/venv/bin/python3 /opt/fl_client/cluster_client_entry.py "
            f"--server {server_ip}:{port} "
            f"--node_id {node_id} "
            f"--data_dir {data_dir} "
            f"--mode {mode} "
            f"> /opt/fl_client/client_run.log 2>&1 &"
        )
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", f"root@{host}", f"pct exec {ctid} -- bash -c '{client_cmd}'"]
        subprocess.run(ssh_cmd, check=True)
        print(f"  -> Dispatched Client [{node_id}] to Host {host} (CT {ctid})...")


def run_benchmark(mode: str = "capdrift", rounds: int = 3, server_ip: str = "192.168.30.106", port: int = 8080):
    print("=" * 80)
    print(f"  PHYSICAL CLUSTER EXPERIMENT: 5 LXC NODES ACROSS 3 CISCO UCS SERVERS")
    print(f"  Mode: {mode.upper()} | Rounds: {rounds} | Coordinator: {server_ip}:{port}")
    print("=" * 80)

    # 1. Clean previous client runs
    print("\n[1/4] Stopping any previous client instances across Proxmox cluster...")
    stop_all_cluster_clients()
    time.sleep(1)

    # 2. Launch Coordinator on 5070 Ti Workstation
    print(f"\n[2/4] Starting Central Coordinator on RTX 5070 Ti ({mode.upper()})...")
    server_cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        "src/research/cluster_server_entry.py",
        "--rounds", str(rounds),
        "--min_clients", "5",
        "--port", str(port),
        "--mode", mode
    ]
    server_proc = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT))
    time.sleep(3)  # Wait for coordinator gRPC port to bind

    # 3. Launch 5 edge clients concurrently
    print(f"\n[3/4] Launching 5 Physical Edge Clients concurrently across pve01, pve02, pve03...")
    t_start = time.perf_counter()
    launch_cluster_clients(server_ip, port, mode)

    # 4. Wait for completion
    print(f"\n[4/4] All 5 physical edge nodes training over real gRPC LAN wire...")
    server_proc.wait()
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print(f"  PHYSICAL CLUSTER RUN ({mode.upper()}) COMPLETED IN {total_time:.2f}s!")
    print("=" * 80)

    # Read and print summary
    summary_path = PROJECT_ROOT / "results" / "cluster_runs" / f"cluster_telemetry_{mode}_5clients.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\nSummary of {len(data)} completed communication rounds on physical cluster:")
        for r in data:
            print(f"  Round {r['round']:02d}: Acc={r['accuracy']}%, Loss={r['loss']}, "
                  f"RoundTime={r['round_duration_sec']}s, IdleStragglerWaste={r['straggler_waste_sec']}s, "
                  f"Util={r['cluster_utilization_pct']}%, Clipped={r['num_clipped']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["fedavg", "capdrift"], default="capdrift")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--server_ip", type=str, default="192.168.30.106")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    run_benchmark(mode=args.mode, rounds=args.rounds, server_ip=args.server_ip, port=args.port)
