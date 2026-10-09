"""
Multi-Scale Physical Cluster Benchmark Orchestrator.
Supports scaling across 5, 10, and 20 real Proxmox LXC containers across
the 3 Cisco UCS fog servers (pve01, pve02, pve03) and the RTX 5070 Ti coordinator.
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Full 20-container physical fleet across the 3 Cisco UCS servers:
FLEET_20 = [
    # pve01 (Dual Xeon E5-2680 v4) - 8 Nodes
    {"host": "192.168.20.100", "ctid": 201, "node_id": "hospital_01_pve01", "data_dir": "/opt/fl_client/data/node_1"},
    {"host": "192.168.20.100", "ctid": 205, "node_id": "hospital_02_pve01", "data_dir": "/opt/fl_client/data/node_2"},
    {"host": "192.168.20.100", "ctid": 206, "node_id": "hospital_03_pve01", "data_dir": "/opt/fl_client/data/node_3"},
    {"host": "192.168.20.100", "ctid": 207, "node_id": "hospital_04_pve01", "data_dir": "/opt/fl_client/data/node_4"},
    {"host": "192.168.20.100", "ctid": 208, "node_id": "hospital_05_pve01", "data_dir": "/opt/fl_client/data/node_5"},
    {"host": "192.168.20.100", "ctid": 209, "node_id": "hospital_06_pve01", "data_dir": "/opt/fl_client/data/node_6"},
    {"host": "192.168.20.100", "ctid": 210, "node_id": "hospital_07_pve01", "data_dir": "/opt/fl_client/data/node_7"},
    {"host": "192.168.20.100", "ctid": 211, "node_id": "hospital_08_pve01", "data_dir": "/opt/fl_client/data/node_8"},

    # pve02 (Dual Xeon E5-2620 v4) - 6 Nodes
    {"host": "192.168.10.100", "ctid": 202, "node_id": "hospital_09_pve02", "data_dir": "/opt/fl_client/data/node_9"},
    {"host": "192.168.10.100", "ctid": 213, "node_id": "hospital_10_pve02", "data_dir": "/opt/fl_client/data/node_10"},
    {"host": "192.168.10.100", "ctid": 214, "node_id": "hospital_11_pve02", "data_dir": "/opt/fl_client/data/node_11"},
    {"host": "192.168.10.100", "ctid": 215, "node_id": "hospital_12_pve02", "data_dir": "/opt/fl_client/data/node_12"},
    {"host": "192.168.10.100", "ctid": 216, "node_id": "hospital_13_pve02", "data_dir": "/opt/fl_client/data/node_13"},
    {"host": "192.168.10.100", "ctid": 217, "node_id": "hospital_14_pve02", "data_dir": "/opt/fl_client/data/node_14"},

    # pve03 (Dual Xeon E5-2620 v4) - 6 Nodes
    {"host": "192.168.40.100", "ctid": 203, "node_id": "hospital_15_pve03", "data_dir": "/opt/fl_client/data/node_15"},
    {"host": "192.168.40.100", "ctid": 223, "node_id": "hospital_16_pve03", "data_dir": "/opt/fl_client/data/node_16"},
    {"host": "192.168.40.100", "ctid": 224, "node_id": "hospital_17_pve03", "data_dir": "/opt/fl_client/data/node_17"},
    {"host": "192.168.40.100", "ctid": 225, "node_id": "hospital_18_pve03", "data_dir": "/opt/fl_client/data/node_18"},
    {"host": "192.168.40.100", "ctid": 226, "node_id": "hospital_19_pve03", "data_dir": "/opt/fl_client/data/node_19"},
    {"host": "192.168.40.100", "ctid": 227, "node_id": "hospital_20_pve03", "data_dir": "/opt/fl_client/data/node_20"},
]


def get_selected_fleet(num_clients: int) -> List[Dict]:
    if num_clients == 5:
        # 2 on pve01, 2 on pve02, 1 on pve03
        return [
            FLEET_20[0], FLEET_20[1],  # pve01: 201, 205
            FLEET_20[8], FLEET_20[9],  # pve02: 202, 213
            FLEET_20[14]               # pve03: 203
        ]
    elif num_clients == 10:
        # 4 on pve01, 3 on pve02, 3 on pve03
        return [
            FLEET_20[0], FLEET_20[1], FLEET_20[2], FLEET_20[3], # pve01: 201, 205, 206, 207
            FLEET_20[8], FLEET_20[9], FLEET_20[10],              # pve02: 202, 213, 214
            FLEET_20[14], FLEET_20[15], FLEET_20[16]            # pve03: 203, 223, 224
        ]
    elif num_clients == 20:
        return FLEET_20
    else:
        return FLEET_20[:num_clients]


def stop_fleet_clients(fleet: List[Dict]):
    """Stops lingering client python instances across hosts."""
    by_host = collections.defaultdict(list)
    for c in fleet:
        by_host[c["host"]].append(c["ctid"])
    for host, ctids in by_host.items():
        cmd = " ; ".join(f"pct exec {ctid} -- pkill -f 'cluster_client_entry.py' 2>/dev/null || true" for ctid in ctids)
        subprocess.run(["ssh", "-o", "BatchMode=yes", f"root@{host}", cmd], capture_output=True)


def launch_fleet_clients(fleet: List[Dict], server_ip: str, port: int, mode: str):
    """Launches all selected clients via host-batched SSH commands."""
    by_host = collections.defaultdict(list)
    for c in fleet:
        by_host[c["host"]].append(c)

    for host, clients in by_host.items():
        commands = []
        for c in clients:
            ctid = c["ctid"]
            node_id = c["node_id"]
            data_dir = c["data_dir"]
            cmd = (
                f"nohup /opt/fl_client/venv/bin/python3 /opt/fl_client/cluster_client_entry.py "
                f"--server {server_ip}:{port} "
                f"--node_id {node_id} "
                f"--data_dir {data_dir} "
                f"--mode {mode} "
                f"> /opt/fl_client/client_run.log 2>&1 &"
            )
            commands.append(f"pct exec {ctid} -- bash -c '{cmd}'")
        
        full_batch_cmd = " ; ".join(commands)
        subprocess.run(["ssh", "-o", "BatchMode=yes", f"root@{host}", full_batch_cmd], check=True)
        print(f"  -> Dispatched {len(clients)} container jobs to Host {host}...")


def run_experiment(num_clients: int = 10, mode: str = "capdrift", rounds: int = 2, server_ip: str = "192.168.30.106", port: int = 8080):
    selected_fleet = get_selected_fleet(num_clients)
    print("=" * 85)
    print(f"  PHYSICAL CLUSTER SCALING BENCHMARK: N = {len(selected_fleet)} REAL LXC NODES")
    print(f"  Mode: {mode.upper()} | Rounds: {rounds} | Coordinator: {server_ip}:{port}")
    print("=" * 85)

    # 1. Stop any previous clients
    print(f"\n[1/4] Ensuring clean state across {len(selected_fleet)} containers...")
    stop_fleet_clients(FLEET_20)
    time.sleep(1)

    # 2. Launch Coordinator on 5070 Ti Workstation
    print(f"\n[2/4] Starting Central Coordinator on RTX 5070 Ti ({mode.upper()})...")
    server_cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        "src/research/cluster_server_entry.py",
        "--rounds", str(rounds),
        "--min_clients", str(len(selected_fleet)),
        "--port", str(port),
        "--mode", mode
    ]
    server_proc = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT))
    time.sleep(3)

    # 3. Launch clients
    print(f"\n[3/4] Launching {len(selected_fleet)} Physical Edge Clients concurrently...")
    t_start = time.perf_counter()
    launch_fleet_clients(selected_fleet, server_ip, port, mode)

    # 4. Wait for completion
    print(f"\n[4/4] All {len(selected_fleet)} physical edge nodes training over real gRPC LAN wire...")
    server_proc.wait()
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 85)
    print(f"  CLUSTER RUN ({mode.upper()}, N={len(selected_fleet)}) FINISHED IN {total_time:.2f}s!")
    print("=" * 85)

    summary_path = PROJECT_ROOT / "results" / "cluster_runs" / f"cluster_telemetry_{mode}_{len(selected_fleet)}clients.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            print(f"  Round {r['round']:02d}: Acc={r['accuracy']}%, Loss={r['loss']}, "
                  f"Duration={r['round_duration_sec']}s, IdleStragglerWaste={r['straggler_waste_sec']}s, "
                  f"ClusterUtil={r['cluster_utilization_pct']}%, Clipped={r['num_clipped']}/{len(selected_fleet)}")
    return total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, choices=[5, 10, 20], default=10)
    parser.add_argument("--mode", type=str, choices=["fedavg", "capdrift"], default="capdrift")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--server_ip", type=str, default="192.168.30.106")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    run_experiment(
        num_clients=args.num_clients,
        mode=args.mode,
        rounds=args.rounds,
        server_ip=args.server_ip,
        port=args.port
    )
