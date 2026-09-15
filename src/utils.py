"""
System telemetry, hardware profiling, network bandwidth calculation,
and visualization utilities for NET 470 project deliverables.
"""

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Any, List
import psutil
import torch


def get_hardware_info() -> Dict[str, Any]:
    """Inspect and return current host compute capabilities (GPU vs CPU)."""
    info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "None (CPU Compute Node)"
        
    return info


def calculate_network_savings(
    raw_dataset_size_mb: float,
    model_payload_mb: float,
    num_clients: int,
    num_rounds: int
) -> Dict[str, float]:
    """
    Computes communication bandwidth trade-offs:
      - Centralized: raw images uploaded to central cloud server.
      - Federated: model weights sent down (server->client) and up (client->server) per round.
    """
    centralized_total_mb = raw_dataset_size_mb
    
    # 2 transfers per round (Downlink + Uplink) per client
    fl_round_mb = 2 * model_payload_mb * num_clients
    fl_total_mb = fl_round_mb * num_rounds
    
    saved_mb = max(0.0, centralized_total_mb - fl_total_mb)
    savings_pct = (saved_mb / centralized_total_mb) * 100.0 if centralized_total_mb > 0 else 0.0
    
    return {
        "centralized_mb": round(centralized_total_mb, 2),
        "fl_round_mb": round(fl_round_mb, 2),
        "fl_total_mb": round(fl_total_mb, 2),
        "saved_mb": round(saved_mb, 2),
        "savings_percentage": round(savings_pct, 2)
    }


def save_metrics_to_disk(
    metrics_history: List[Dict[str, Any]],
    output_path: str = "results/fl_metrics.json"
) -> None:
    """Save training and evaluation history to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=4)
    print(f"[Telemetry] Saved run metrics to {output_path}")


if __name__ == "__main__":
    hw = get_hardware_info()
    print("=== System Hardware Telemetry ===")
    for k, v in hw.items():
        print(f"  {k}: {v}")
        
    # Bandwidth calculation for 11,712 Kaggle images (~1,200 MB) vs MobileNetV3 (5.8 MB payload * 2 transfers)
    net_eval = calculate_network_savings(
        raw_dataset_size_mb=1200.0,
        model_payload_mb=5.8,
        num_clients=3,
        num_rounds=6
    )
    print("\n=== Bandwidth Analysis (Syllabus Metric 1) ===")
    print(f"  Centralized Dataset Transfer: {net_eval['centralized_mb']} MB")
    print(f"  Federated Total (6 Rounds):   {net_eval['fl_total_mb']} MB")
    print(f"  Network Bandwidth Saved:      {net_eval['savings_percentage']}% ({net_eval['saved_mb']} MB)")
