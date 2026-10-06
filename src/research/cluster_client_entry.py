"""
Standalone Edge Client Entrypoint for Proxmox Heterogeneous Cluster.
Deployed into /opt/fl_client/ on Proxmox LXC containers.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import flwr as fl

# Ensure local imports resolve
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
OPT_DIR = Path("/opt/fl_client")
if OPT_DIR.exists() and str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

from capdrift_flower_adapter import CapDriftFlowerClient, ClusterControllerConfig
from model import get_model


def get_data_loader(data_dir: str, batch_size: int = 16):
    data_path = Path(data_dir)
    train_dir = data_path / "train" if (data_path / "train").exists() else data_path
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(str(train_dir), transform=tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, len(dataset.classes)


def main():
    parser = argparse.ArgumentParser(description="Proxmox FL Edge Client")
    parser.add_argument("--server", type=str, default="192.168.30.106:8080", help="gRPC coordinator IP:port")
    parser.add_argument("--node_id", type=str, default="node_1", help="Client Node ID")
    parser.add_argument("--data_dir", type=str, default="/opt/fl_client/data/node_1", help="Data directory")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", help="Architecture")
    parser.add_argument("--mode", type=str, choices=["fedavg", "capdrift", "capability_only", "drift_only"], default="capdrift")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--target_time", type=float, default=12.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)
        print(f"[{args.node_id}] CPU Execution with {threads} threads")

    loader, num_classes = get_data_loader(args.data_dir, batch_size=args.batch_size)
    print(f"[{args.node_id}] Loaded dataset with {len(loader.dataset)} samples across {num_classes} classes")

    model = get_model(architecture=args.arch, num_classes=num_classes, pretrained=True).to(device)

    # Controller configurations
    cfg = ClusterControllerConfig(target_round_time=args.target_time)
    if args.mode == "fedavg":
        cfg.use_capability_budgeting = False
        cfg.use_drift_gating = False
        cfg.use_clipping = False
    elif args.mode == "capability_only":
        cfg.use_capability_budgeting = True
        cfg.use_drift_gating = False
        cfg.use_clipping = False
    elif args.mode == "drift_only":
        cfg.use_capability_budgeting = False
        cfg.use_drift_gating = True
        cfg.use_clipping = False
    elif args.mode == "capdrift":
        cfg.use_capability_budgeting = True
        cfg.use_drift_gating = True
        cfg.use_clipping = True

    client = CapDriftFlowerClient(
        node_id=args.node_id,
        data_loader=loader,
        model=model,
        device=device,
        config=cfg
    )

    print(f"[{args.node_id}] Connecting to Coordinator at {args.server} (Mode: {args.mode})...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )


if __name__ == "__main__":
    main()
