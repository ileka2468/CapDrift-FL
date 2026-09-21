"""
Dynamic Node Registry and Onboarding Engine for OmniFed Framework.
Manages physical and virtual cluster nodes, tests network readiness,
and dynamically generates 1-click join commands for distributed edge workers.
"""

import json
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "cluster_nodes.json"


class NodeManager:
    """Manages dynamic cluster worker registration and onboarding commands."""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.nodes: List[Dict] = self._load_nodes()

    def _load_nodes(self) -> List[Dict]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[NodeManager] Warning: Error reading {self.config_path}: {e}")
        return []

    def save_nodes(self) -> None:
        """Persists the node registry to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.nodes, f, indent=4)

    def get_all_nodes(self) -> List[Dict]:
        return self.nodes

    def get_node(self, node_id: str) -> Optional[Dict]:
        for n in self.nodes:
            if n["id"] == node_id:
                return n
        return None

    def add_node(
        self,
        node_id: str,
        name: str,
        host_ip: str,
        compute: str = "CPU (AVX2)",
        connection_type: str = "ssh_lxc",
        ssh_user: Optional[str] = "root",
        container_id: Optional[str] = None,
        data_dir: str = "/opt/fl_client/data",
        client_script: str = "/opt/fl_client/client.py",
        python_bin: str = "/opt/fl_client/venv/bin/python3",
        enabled: bool = True
    ) -> Dict:
        """Registers a new physical, virtual, or remote node into the cluster."""
        # Check if ID already exists
        for idx, n in enumerate(self.nodes):
            if n["id"] == node_id:
                # Update existing
                self.nodes[idx] = {
                    "id": node_id,
                    "name": name,
                    "host_ip": host_ip,
                    "ssh_user": ssh_user,
                    "container_id": container_id,
                    "data_dir": data_dir,
                    "client_script": client_script,
                    "python_bin": python_bin,
                    "compute": compute,
                    "connection_type": connection_type,
                    "enabled": enabled
                }
                self.save_nodes()
                return self.nodes[idx]

        new_node = {
            "id": node_id,
            "name": name,
            "host_ip": host_ip,
            "ssh_user": ssh_user,
            "container_id": container_id,
            "data_dir": data_dir,
            "client_script": client_script,
            "python_bin": python_bin,
            "compute": compute,
            "connection_type": connection_type,
            "enabled": enabled
        }
        self.nodes.append(new_node)
        self.save_nodes()
        return new_node

    def remove_node(self, node_id: str) -> bool:
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n["id"] != node_id]
        if len(self.nodes) < before:
            self.save_nodes()
            return True
        return False

    def toggle_node(self, node_id: str, enabled: bool) -> bool:
        node = self.get_node(node_id)
        if node:
            node["enabled"] = enabled
            self.save_nodes()
            return True
        return False

    def generate_join_command(
        self,
        node_id: str,
        coordinator_ip: str = "192.168.30.102",
        port: int = 8080,
        epochs: int = 1,
        arch: str = "mobilenet_v3_small",
        data_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generates 1-click execution commands for any remote worker to join the coordinator.
        """
        node = self.get_node(node_id)
        if data_dir is None:
            data_dir = node["data_dir"] if node else f"data/{node_id}"
        python_bin = node["python_bin"] if node else "python3"
        client_script = node["client_script"] if node else "client.py"

        # 1. Native CLI command to paste directly onto the remote machine
        bash_cmd = f"{python_bin} {client_script} --server {coordinator_ip}:{port} --node_id {node_id} --data_dir {data_dir} --arch {arch} --epochs {epochs}"

        # 2. Automated SSH dispatch command from coordinator
        if node and node.get("connection_type") == "ssh_lxc" and node.get("container_id"):
            ssh_host = node["host_ip"]
            user = node.get("ssh_user", "root")
            ct = node["container_id"]
            ssh_cmd = (
                f"ssh -o StrictHostKeyChecking=no {user}@{ssh_host} "
                f"\"pct exec {ct} -- bash -c 'nohup {bash_cmd} > /opt/fl_client/client.log 2>&1 &'\""
            )
        elif node and node.get("connection_type") == "ssh_direct":
            ssh_host = node["host_ip"]
            user = node.get("ssh_user", "root")
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no {user}@{ssh_host} \"nohup {bash_cmd} > client.log 2>&1 &\""
        else:
            ssh_cmd = bash_cmd

        # 3. Quick curl bootstrap snippet for brand-new workers
        curl_bootstrap = f"curl -s http://{coordinator_ip}:5000/api/bootstrap/{node_id} | bash"

        return {
            "node_id": node_id,
            "direct_command": bash_cmd,
            "ssh_dispatch_command": ssh_cmd,
            "bootstrap_snippet": curl_bootstrap,
            "target_server": f"{coordinator_ip}:{port}"
        }

    def test_node_connectivity(self, node_id: str) -> Dict:
        """Tests live socket ping to the node IP."""
        node = self.get_node(node_id)
        if not node:
            return {"status": "not_found", "latency_ms": -1}

        ip = node["host_ip"]
        if ip in ("127.0.0.1", "localhost"):
            return {"status": "online", "latency_ms": 0.2, "type": "local"}

        start = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.8)
            sock.connect((ip, 22))  # Test SSH port
            sock.close()
            latency = (time.perf_counter() - start) * 1000.0
            return {"status": "online", "latency_ms": round(latency, 1), "type": "ssh"}
        except Exception:
            # Fallback ping test
            return {"status": "unreachable", "latency_ms": -1, "type": "unreachable"}


node_manager = NodeManager()

if __name__ == "__main__":
    print(f"Loaded {len(node_manager.get_all_nodes())} nodes from registry:")
    for n in node_manager.get_all_nodes():
        print(f" - [{n['id']}] {n['name']} ({n['host_ip']}) | {n['compute']}")
    print("\nExample Join Command for node_1:")
    print(node_manager.generate_join_command("node_1")["ssh_dispatch_command"])
