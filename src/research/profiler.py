"""
Hardware Capability & Straggler Profiler for OmniFed.
Measures client compute throughput (samples/sec), communication latency (RTT),
and calculates straggler idle penalties across heterogeneous edge-fog nodes.
"""

import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ClientProfile:
    node_id: str
    device_type: str  # "cuda" | "cpu"
    device_name: str
    num_threads: int
    throughput_samples_per_sec: float
    batch_latency_ms: float
    network_rtt_ms: float
    compute_to_comm_ratio: float
    timestamp: float


class HardwareProfiler:
    """
    Automated hardware benchmarking engine run by clients during registration
    or at round initialization to inform capability-aware scheduling.
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def benchmark_compute_throughput(
        self,
        arch: str = "mobilenet_v3_small",
        num_classes: int = 2,
        batch_size: int = 32,
        warmup_steps: int = 5,
        benchmark_steps: int = 20
    ) -> Tuple[float, float]:
        """
        Executes real forward + backward passes with optimizer step
        to measure true empirical throughput in samples/second.
        """
        # Build benchmark model
        if arch == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dummy_x = torch.randn(batch_size, 3, 224, 224, device=self.device)
        dummy_y = torch.randint(0, num_classes, (batch_size,), device=self.device)

        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            out = model(dummy_x)
            loss = criterion(out, dummy_y)
            loss.backward()
            optimizer.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(benchmark_steps):
            optimizer.zero_grad()
            out = model(dummy_x)
            loss = criterion(out, dummy_y)
            loss.backward()
            optimizer.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.perf_counter() - t0
        total_samples = batch_size * benchmark_steps
        samples_per_sec = total_samples / max(1e-6, total_time)
        batch_latency_ms = (total_time / benchmark_steps) * 1000.0

        return samples_per_sec, batch_latency_ms

    def measure_network_rtt(self, host: str, port: int = 8080, timeout: float = 1.0) -> float:
        """Measures round-trip TCP handshake latency in milliseconds."""
        if host in ("127.0.0.1", "localhost"):
            return 0.5

        latencies = []
        for _ in range(3):
            try:
                t0 = time.perf_counter()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((host, port))
                s.close()
                latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                latencies.append(10.0)  # Default fallback if port closed

        return float(np.mean(latencies)) if latencies else 10.0

    def generate_profile(
        self,
        node_id: str,
        coordinator_host: str = "127.0.0.1",
        coordinator_port: int = 8080,
        model_size_mb: float = 5.8
    ) -> ClientProfile:
        """Constructs a comprehensive ClientProfile dataclass for the active node."""
        dev_type = self.device.type
        if dev_type == "cuda":
            dev_name = torch.cuda.get_device_name(0)
            n_threads = 1
        else:
            dev_name = f"CPU ({os.cpu_count()} cores)"
            n_threads = torch.get_num_threads()

        throughput, batch_ms = self.benchmark_compute_throughput()
        rtt_ms = self.measure_network_rtt(coordinator_host, coordinator_port)

        # Theoretical network transfer time assuming 100 Mbps edge link
        comm_time_sec = (2.0 * model_size_mb * 8.0) / 100.0 + (rtt_ms / 1000.0)
        # Compute time for 100 samples
        compute_time_sec = 100.0 / max(1.0, throughput)
        rho = compute_time_sec / max(1e-6, comm_time_sec)

        return ClientProfile(
            node_id=node_id,
            device_type=dev_type,
            device_name=dev_name,
            num_threads=n_threads,
            throughput_samples_per_sec=round(throughput, 2),
            batch_latency_ms=round(batch_ms, 2),
            network_rtt_ms=round(rtt_ms, 2),
            compute_to_comm_ratio=round(rho, 3),
            timestamp=time.time()
        )


def compute_straggler_telemetry(client_durations: Dict[str, float]) -> Dict[str, float]:
    """
    Given execution durations of all clients in a round:
    Calculates total round duration, straggler bottleneck identity,
    and cumulative idle wait time wasted across all faster nodes.
    """
    if not client_durations:
        return {"round_time_sec": 0.0, "total_idle_time_sec": 0.0, "straggler_overhead_percent": 0.0}

    durations = list(client_durations.values())
    max_duration = max(durations)
    min_duration = min(durations)

    idle_times = [max_duration - d for d in durations]
    total_idle = sum(idle_times)

    # Percentage of total client runtime spent idling
    total_active_time = sum(durations)
    idle_pct = (total_idle / (total_active_time + total_idle)) * 100.0 if (total_active_time + total_idle) > 0 else 0.0

    return {
        "round_time_sec": round(max_duration, 3),
        "fastest_client_duration_sec": round(min_duration, 3),
        "slowest_client_duration_sec": round(max_duration, 3),
        "total_idle_time_sec": round(total_idle, 3),
        "straggler_overhead_percent": round(idle_pct, 1)
    }


if __name__ == "__main__":
    profiler = HardwareProfiler()
    profile = profiler.generate_profile("local_test_node")
    print(f"Profile: {asdict(profile)}")
