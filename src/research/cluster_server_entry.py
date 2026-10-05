"""
Central FL Coordinator Entrypoint for Proxmox Heterogeneous Cluster.
Runs on the NVIDIA GeForce RTX 5070 Ti Workstation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import flwr as fl

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import get_model, get_parameters, set_parameters, get_model_size_mb
from src.research.capdrift_flower_adapter import CapDriftFlowerStrategy, ClusterControllerConfig


class CentralEvaluator:
    def __init__(self, data_dir: str = "data/global_test", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        test_dir = Path(data_dir) / "test" if (Path(data_dir) / "test").exists() else Path(data_dir)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(str(test_dir), transform=tf)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        print(f"[Coordinator Evaluator] Active device: {self.device} ({len(dataset)} validation samples)")

    def evaluate(self, model: nn.Module) -> Tuple[float, float, float, float, float]:
        model.to(self.device)
        model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * len(y)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        acc = float((all_preds == all_targets).mean()) if total_samples > 0 else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
        return avg_loss, acc, float(prec), float(rec), float(f1)


def main():
    parser = argparse.ArgumentParser(description="Proxmox FL Central Coordinator")
    parser.add_argument("--rounds", type=int, default=5, help="FL Rounds")
    parser.add_argument("--min_clients", type=int, default=5, help="Minimum connected clients")
    parser.add_argument("--port", type=int, default=8080, help="gRPC Port")
    parser.add_argument("--mode", type=str, choices=["fedavg", "capdrift"], default="capdrift")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--data_dir", type=str, default="data/global_test")
    parser.add_argument("--target_time", type=float, default=12.0)
    args = parser.parse_args()

    print("==========================================================================")
    print(f"  PHYSICAL CLUSTER COORDINATOR: NVIDIA RTX 5070 Ti ({args.mode.upper()})")
    print(f"  Listening on 0.0.0.0:{args.port} | Expected Clients: {args.min_clients} | Rounds: {args.rounds}")
    print("==========================================================================")

    evaluator = CentralEvaluator(data_dir=args.data_dir, device="cuda")
    base_model = get_model(architecture=args.arch, num_classes=2, pretrained=True)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(base_model))

    cfg = ClusterControllerConfig(target_round_time=args.target_time)
    if args.mode == "fedavg":
        cfg.use_capability_budgeting = False
        cfg.use_drift_gating = False
        cfg.use_clipping = False
    elif args.mode == "capdrift":
        cfg.use_capability_budgeting = True
        cfg.use_drift_gating = True
        cfg.use_clipping = True

    strategy = CapDriftFlowerStrategy(
        evaluator=evaluator,
        base_model=base_model,
        config=cfg,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_params
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )

    # Save summary
    out_dir = Path("results/cluster_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"cluster_telemetry_{args.mode}_{args.min_clients}clients.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(strategy.history, f, indent=2)
    print(f"\n[Saved Cluster Run Telemetry] -> {summary_path}")


if __name__ == "__main__":
    main()
