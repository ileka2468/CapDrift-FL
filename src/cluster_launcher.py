"""
Dynamic Multi-Node Cluster Orchestrator for OmniFed Framework.
Reads active nodes from config/cluster_nodes.json, dynamically coordinates
central aggregation on RTX 5070 Ti, and dispatches remote edge training jobs.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from node_manager import node_manager
from utils import get_hardware_info


def free_port_if_occupied(port: int):
    """Detect and terminate any stale process lingering on the coordinator port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = (s.connect_ex(('127.0.0.1', port)) == 0)
    if in_use:
        print(f"[Warning] Port {port} is occupied. Attempting to clear stale coordinator...")
        try:
            if sys.platform == "win32":
                res = subprocess.run(
                    f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"',
                    shell=True, capture_output=True, text=True
                )
                pids = [p.strip() for p in res.stdout.strip().splitlines() if p.strip().isdigit()]
                for pid in set(pids):
                    pid_int = int(pid)
                    if pid_int != os.getpid():
                        print(f"  -> Terminating lingering process PID {pid} on port {port}...")
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            else:
                subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
            time.sleep(1.5)
        except Exception as e:
            print(f"  -> Could not auto-clear port {port}: {e}")


def main():
    parser = argparse.ArgumentParser(description="OmniFed Dynamic Cluster Orchestrator")
    parser.add_argument("--rounds", type=int, default=6, help="Federated communication rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--port", type=int, default=8080, help="gRPC coordinator port")
    parser.add_argument("--server_ip", type=str, default="192.168.30.102", help="Coordinator LAN IP")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", help="Model architecture")
    args = parser.parse_args()

    print("=================================================================")
    print("   OMNIFED: DYNAMIC MULTI-NODE CLUSTER ORCHESTRATOR             ")
    print("=================================================================")

    # Ensure port is clean before starting
    free_port_if_occupied(args.port)

    all_nodes = node_manager.get_all_nodes()
    enabled_nodes = [n for n in all_nodes if n.get("enabled", True)]
    min_clients = len(enabled_nodes)

    print(f"\n[1/4] Discovered {len(all_nodes)} total nodes in registry ({min_clients} active/enabled):")
    for n in enabled_nodes:
        print(f"  * [{n['id']}] {n['name']} @ {n['host_ip']} ({n['compute']})")

    if min_clients == 0:
        print("Error: No enabled nodes found in config/cluster_nodes.json")
        sys.exit(1)

    # 1. Start Central Coordinator
    print(f"\n[2/4] Starting Central Coordinator on RTX 5070 Ti (port {args.port}, min_clients={min_clients}, arch={args.arch})...")
    test_data = str(Path(args.data_dir) / "global_test") if (Path(args.data_dir) / "global_test").exists() else args.data_dir
    server_cmd = [
        sys.executable,
        "src/server.py",
        "--rounds", str(args.rounds),
        "--min_clients", str(min_clients),
        "--port", str(args.port),
        "--arch", args.arch,
        "--data_dir", test_data
    ]
    server_proc = subprocess.Popen(server_cmd)
    try:
        time.sleep(2)
        if server_proc.poll() is not None:
            print(f"[Error] Coordinator failed to launch (exit code: {server_proc.returncode}).")
            return

        # 2. Trigger Remote Edge Nodes
        print("\n[3/4] Dispatching Edge Client Jobs to Active Nodes...")
        local_nodes = []
        for n in enabled_nodes:
            node_id = n["id"]
            c_type = n.get("connection_type", "local")

            if c_type in ("ssh_lxc", "ssh_direct"):
                print(f"  -> Dispatching to remote node: {n['name']} ({n['host_ip']})...")
                norm_dir = Path(args.data_dir).name.lower()
                if "signature" in norm_dir or "forgery" in norm_dir:
                    remote_data_dir = f"/opt/fl_client/datasets/signature_forgery/{node_id}"
                elif "dna" in norm_dir:
                    remote_data_dir = f"/opt/fl_client/datasets/dna_cgr/{node_id}"
                elif "handwriting" in norm_dir:
                    remote_data_dir = f"/opt/fl_client/datasets/handwriting/{node_id}"
                else:
                    remote_data_dir = n.get("data_dir", f"/opt/fl_client/data/{node_id}")

                cmd_info = node_manager.generate_join_command(
                    node_id=node_id,
                    coordinator_ip=args.server_ip,
                    port=args.port,
                    epochs=args.epochs,
                    arch=args.arch,
                    data_dir=remote_data_dir
                )
                # Execute dispatch command over SSH
                try:
                    subprocess.run(cmd_info["ssh_dispatch_command"], shell=True, timeout=10)
                except Exception as e:
                    print(f"     Warning: Dispatch failed for {node_id}: {e}")
            elif c_type == "local":
                local_nodes.append(n)

        # 3. Launch Local Clients (e.g. Node 4 on CUDA Workstation)
        if local_nodes:
            for ln in local_nodes:
                print(f"\n[4/4] Starting Local Edge Client on {ln['name']} ({ln['compute']})...")
                candidate_node_data = Path(args.data_dir) / ln["id"]
                if candidate_node_data.exists():
                    active_data_dir = str(candidate_node_data)
                else:
                    active_data_dir = ln.get("data_dir", f"{args.data_dir}/{ln['id']}")

                client_cmd = [
                    sys.executable,
                    ln.get("client_script", "src/client.py"),
                    "--server", f"127.0.0.1:{args.port}",
                    "--node_id", ln["id"],
                    "--data_dir", active_data_dir,
                    "--arch", args.arch,
                    "--epochs", str(args.epochs)
                ]
                client_proc = subprocess.run(client_cmd)

        # Wait for coordinator
        server_proc.wait()

        # Automatically refresh report plots
        print("\nGenerating updated report charts...")
        subprocess.run([sys.executable, "scripts/plot_results.py"])

        print("\n=================================================================")
        print("   OMNIFED FEDERATED RUN COMPLETE ACROSS ALL REGISTERED NODES!   ")
        print("=================================================================")
    except KeyboardInterrupt:
        print("\n[OmniFed] Cluster training aborted by user.")
    finally:
        if server_proc.poll() is None:
            print("[OmniFed] Cleaning up Central Coordinator process...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
