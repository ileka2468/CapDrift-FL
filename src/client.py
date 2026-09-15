"""
Flower Federated Learning Client (Hospital Edge Node)
Deployable on Proxmox VE Nodes (Intel Xeon CPU) or 5070 Ti Workstation (CUDA).

Features:
- Automatic hardware acceleration detection (CUDA vs AVX2 Multi-threading)
- Local training epoch timer for hardware heterogeneity and straggler analysis
- Private local data partition enforcement (zero image transmission)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import flwr as fl

from model import get_model, set_parameters, get_parameters
from dataset import load_partition
from utils import get_hardware_info


class HospitalClient(fl.client.NumPyClient):
    def __init__(
        self,
        node_id: str,
        data_dir: str,
        arch: str = "mobilenet_v3_small",
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        force_cpu: bool = False
    ):
        self.node_id = node_id
        self.arch = arch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        
        # Hardware setup
        if torch.cuda.is_available() and not force_cpu:
            self.device = torch.device("cuda")
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device = torch.device("cpu")
            # Maximize Xeon CPU utilization with OpenMP/MKL threads
            threads = os.cpu_count() or 4
            torch.set_num_threads(threads)
            self.device_name = f"CPU ({threads} threads, AVX2)"
            
        print(f"[{self.node_id}] Initialized on compute engine: {self.device_name}")
        
        # Load local partition
        self.train_loader, self.val_loader = load_partition(
            data_dir=data_dir,
            batch_size=self.batch_size,
            use_synthetic_fallback=True
        )
        
        # Discover number of classes dynamically
        num_classes = 2
        meta_path = Path("results/model_metadata.json")
        if hasattr(self.train_loader.dataset, "classes") and len(self.train_loader.dataset.classes) > 1:
            num_classes = len(self.train_loader.dataset.classes)
        elif hasattr(self.train_loader.dataset, "dataset") and hasattr(self.train_loader.dataset.dataset, "classes") and len(self.train_loader.dataset.dataset.classes) > 1:
            num_classes = len(self.train_loader.dataset.dataset.classes)
        elif Path(data_dir).exists():
            from dataset import scan_dataset_classes
            classes, n_cls = scan_dataset_classes(data_dir)
            if n_cls > 1:
                num_classes = n_cls
        elif meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    num_classes = meta.get("num_classes", 2)
            except Exception:
                pass

        print(f"[{self.node_id}] Model Architecture: {self.arch} ({num_classes} classes)")
        self.model = get_model(architecture=self.arch, num_classes=num_classes, pretrained=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        # Synchronize global weights from coordinator
        set_parameters(self.model, parameters)
        self.model.train()
        
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        start_time = time.perf_counter()
        total_loss = 0.0
        num_samples = 0
        
        print(f"\n[{self.node_id}] Beginning local training round ({self.epochs} epochs)...")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
            total_loss += epoch_loss
            
        training_duration = time.perf_counter() - start_time
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        print(f"[{self.node_id}] Training completed in {training_duration:.2f}s | Avg Loss: {avg_loss:.4f}")
        
        metrics = {
            "loss": avg_loss,
            "training_duration_sec": training_duration,
            "device": self.device_name,
            "node_id": self.node_id
        }
        
        return get_parameters(self.model), num_samples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        set_parameters(self.model, parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += labels.size(0)
                
        avg_loss = loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, total, {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Flower Client (Hospital Node)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Coordinator gRPC host:port")
    parser.add_argument("--node_id", type=str, default="node_1", help="Identifier (e.g. node_1, node_2)")
    parser.add_argument("--data_dir", type=str, default="data/node_1", help="Path to local data partition")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is detected")
    args = parser.parse_args()

    client = HospitalClient(
        node_id=args.node_id,
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_cpu=args.force_cpu
    )
    
    print(f"Connecting to Central Coordinator at {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )


if __name__ == "__main__":
    main()
