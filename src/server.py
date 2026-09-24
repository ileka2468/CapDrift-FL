"""
Flower Federated Learning Server (Coordinator Node)
Running on NVIDIA GeForce RTX 5070 Ti Workstation.

Orchestrates FedAvg rounds, performs CUDA-accelerated global validation,
measures network weight payloads, and tracks hardware straggler metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

from model import get_model, set_parameters, get_parameters, get_model_size_mb
from dataset import load_partition
from utils import get_hardware_info, calculate_network_savings, save_metrics_to_disk


class Evaluator:
    """Evaluates the global model on the holdout validation set using CUDA on the 5070 Ti."""
    def __init__(self, data_dir: str = "data/global_test", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        _, self.test_loader = load_partition(data_dir, batch_size=32, use_synthetic_fallback=True, is_eval_only=True)
        self.criterion = nn.CrossEntropyLoss()
        print(f"[Coordinator Evaluator] Active device: {self.device} ({len(self.test_loader.dataset)} evaluation samples across all classes)")

    def evaluate(self, model: nn.Module) -> Tuple[float, float, float, float, float]:
        model.to(self.device)
        model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        start_t = time.perf_counter()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        eval_time = time.perf_counter() - start_t
        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = float((all_preds == all_targets).mean()) if total_samples > 0 else 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )
        
        return avg_loss, accuracy, float(precision), float(recall), float(f1)


class CustomFedAvg(FedAvg):
    """
    FedAvg strategy with automated telemetry, bandwidth tracking,
    and CUDA-accelerated central evaluation on the RTX 5070 Ti.
    """
    def __init__(
        self,
        evaluator: Evaluator,
        model_size_mb: float,
        arch: str = "mobilenet_v3_small",
        num_classes: int = 2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.model_size_mb = model_size_mb
        self.arch = arch
        self.num_classes = num_classes
        self.history: List[Dict[str, Union[int, float]]] = []
        self.base_model = get_model(architecture=arch, num_classes=num_classes, pretrained=False)
        self.best_accuracy = 0.0

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized evaluation callback invoked after FedAvg model aggregation."""
        if server_round == 0:
            return None  # Skip round 0 initial parameters
            
        ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(self.base_model, ndarrays)
        
        loss, acc, prec, rec, f1 = self.evaluator.evaluate(self.base_model)
        
        print("\n" + "=" * 60)
        print(f" [ROUND {server_round} GLOBAL EVALUATION on RTX 5070 Ti]")
        print(f"  Loss:      {loss:.4f}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("=" * 60 + "\n")
        
        # Log metrics
        self.history.append({
            "round": server_round,
            "loss": round(loss, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
        save_metrics_to_disk(self.history)
        
        # Save model weights
        os.makedirs("results", exist_ok=True)
        torch.save(self.base_model.state_dict(), "results/global_model.pth")
        if acc >= self.best_accuracy:
            self.best_accuracy = acc
            torch.save(self.base_model.state_dict(), "results/best_model.pth")
            print(f"  [Checkpoint] Saved new best model checkpoint to results/best_model.pth ({acc * 100:.2f}% acc)")
        
        return loss, {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Flower Coordinator on 5070 Ti")
    parser.add_argument("--rounds", type=int, default=6, help="Number of FL rounds")
    parser.add_argument("--min_clients", type=int, default=4, help="Minimum connected clients")
    parser.add_argument("--port", type=str, default="8080", help="gRPC coordinator port")
    parser.add_argument("--data_dir", type=str, default="data/global_test", help="Test dataset path")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", help="Model architecture")
    args = parser.parse_args()

    print("================================================================")
    print("   CENTRAL FL COORDINATOR: NVIDIA RTX 5070 Ti WORKSTATION        ")
    print("================================================================")
    
    hw = get_hardware_info()
    print(f"Host Compute: {hw.get('gpu_name')} | System RAM: {hw.get('ram_gb')} GB")
    
    # Discover number of classes dynamically
    num_classes = 2
    if Path(args.data_dir).exists():
        from dataset import scan_dataset_classes
        classes, n_cls = scan_dataset_classes(args.data_dir)
        if n_cls > 1:
            num_classes = n_cls
    elif Path("results/model_metadata.json").exists():
        try:
            with open("results/model_metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                num_classes = meta.get("num_classes", 2)
        except Exception:
            pass

    # Initialize initial parameters
    init_model = get_model(architecture=args.arch, num_classes=num_classes, pretrained=True)
    init_weights = get_parameters(init_model)
    init_parameters = ndarrays_to_parameters(init_weights)
    model_size_mb = get_model_size_mb(init_model)
    
    print(f"Model Architecture: {args.arch} ({num_classes} classes) | Payload Size: {model_size_mb:.2f} MB")
    print(f"Server binding to 0.0.0.0:{args.port} (Waiting for Proxmox edge nodes)...")
    
    evaluator = Evaluator(data_dir=args.data_dir, device="cuda")
    
    min_fit = args.min_clients
    strategy = CustomFedAvg(
        evaluator=evaluator,
        model_size_mb=model_size_mb,
        arch=args.arch,
        num_classes=num_classes,
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # We evaluate centrally on the 5070 Ti
        min_fit_clients=min_fit,
        min_available_clients=min_fit,
        initial_parameters=init_parameters,
    )
    
    server_address = f"0.0.0.0:{args.port}"
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

# Timeout buffer: set coordinator socket timeout to 120s for slow edge nodes
