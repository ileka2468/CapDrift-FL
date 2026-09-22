"""
Fast Cluster Code Deployer.
Distributes capdrift_flower_adapter.py and cluster_client_entry.py
into /opt/fl_client/ on all active Proxmox LXC containers.
"""

import subprocess
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Target hosts and containers
CLUSTER_HOSTS = {
    "192.168.20.100": [201, 205, 206, 207, 208, 209, 210, 211], # pve01
    "192.168.10.100": [202, 213, 214, 215, 216, 217],           # pve02
    "192.168.40.100": [203, 223, 224, 225, 226, 227],           # pve03
}

FILES_TO_DEPLOY = [
    PROJECT_ROOT / "src" / "research" / "capdrift_flower_adapter.py",
    PROJECT_ROOT / "src" / "research" / "cluster_client_entry.py",
    PROJECT_ROOT / "src" / "model.py",
    PROJECT_ROOT / "src" / "dataset.py",
    PROJECT_ROOT / "src" / "utils.py"
]


def deploy_code_to_cluster():
    print("==========================================================================")
    print("  DEPLOYING CAPDRIFT-FL CODE TO PROXMOX LXC CLUSTER CONTAINERS")
    print("==========================================================================")

    # 1. Create a local tar archive of code files
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        archive_path = Path(tmp.name)

    with tarfile.open(archive_path, "w:gz") as tar:
        for f in FILES_TO_DEPLOY:
            if f.exists():
                tar.add(f, arcname=f.name)
                print(f"  + Added: {f.name}")

    # 2. Upload to each host and push to LXC containers
    for host_ip, ctids in CLUSTER_HOSTS.items():
        print(f"\n---> Syncing Host: {host_ip} ({len(ctids)} containers)...")
        remote_tar = "/tmp/fl_code_update.tar.gz"
        
        # SCP archive
        scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", str(archive_path), f"root@{host_ip}:{remote_tar}"]
        subprocess.run(scp_cmd, check=True)

        # Distribute into each container
        for ctid in ctids:
            check_running = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", f"root@{host_ip}", f"pct status {ctid}"],
                capture_output=True, text=True
            )
            if "status: running" in check_running.stdout:
                dist_cmd = (
                    f"pct push {ctid} {remote_tar} /tmp/fl_code_update.tar.gz && "
                    f"pct exec {ctid} -- tar -xzf /tmp/fl_code_update.tar.gz -C /opt/fl_client/ && "
                    f"pct exec {ctid} -- rm -f /tmp/fl_code_update.tar.gz"
                )
                subprocess.run(["ssh", "-o", "BatchMode=yes", f"root@{host_ip}", dist_cmd], check=True)
                print(f"    [CT {ctid}] Updated code in /opt/fl_client/")
            else:
                print(f"    [CT {ctid}] (stopped, skipped)")

    archive_path.unlink(missing_ok=True)
    print("\n==========================================================================")
    print("  CODE DEPLOYMENT COMPLETE ACROSS ALL RUNNING PROXMOX CONTAINERS")
    print("==========================================================================")


if __name__ == "__main__":
    deploy_code_to_cluster()
