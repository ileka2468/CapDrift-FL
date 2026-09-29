"""
Heterogeneous Fleet Modeling & Multi-Tenant Edge/Fog Infrastructure Engine.
Simulates and orchestrates realistic edge/fog client hierarchies across 6 tiers
with physical hardware anchoring, deliberate oversubscription, and virtualization contention.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ClientTierConfig:
    tier_id: int
    name: str
    description: str
    vcpu_allocated: int
    ram_mb: int
    cpu_quota_percent: float
    network_latency_ms: float
    network_bandwidth_mbps: float
    base_train_time_sec: float
    base_samples_per_sec: float
    default_carc_steps: int
    default_fedavg_steps: int = 5


# 6-Tier Heterogeneous Taxonomy
TIER_PROFILES: Dict[int, ClientTierConfig] = {
    0: ClientTierConfig(
        tier_id=0,
        name="Tier 0 (GPU / AMCs)",
        description="NVIDIA RTX 5070 Ti (16GB VRAM, CUDA FP32) - Academic Medical Center",
        vcpu_allocated=16,
        ram_mb=16384,
        cpu_quota_percent=100.0,
        network_latency_ms=1.5,
        network_bandwidth_mbps=1000.0,
        base_train_time_sec=6.20,
        base_samples_per_sec=161.29,
        default_carc_steps=16,
        default_fedavg_steps=5
    ),
    1: ClientTierConfig(
        tier_id=1,
        name="Tier 1 (Fast Fog)",
        description="Cisco UCS Dual Xeon E5-2680 v4 (8 vCPUs) - Regional Hospital Hub Server",
        vcpu_allocated=8,
        ram_mb=8192,
        cpu_quota_percent=100.0,
        network_latency_ms=3.0,
        network_bandwidth_mbps=1000.0,
        base_train_time_sec=14.10,
        base_samples_per_sec=70.92,
        default_carc_steps=9,
        default_fedavg_steps=5
    ),
    2: ClientTierConfig(
        tier_id=2,
        name="Tier 2 (Normal Fog)",
        description="Cisco UCS Dual Xeon E5-2620 v4 (4 vCPUs) - Community Hospital Ward Server",
        vcpu_allocated=4,
        ram_mb=4096,
        cpu_quota_percent=100.0,
        network_latency_ms=5.0,
        network_bandwidth_mbps=500.0,
        base_train_time_sec=24.50,
        base_samples_per_sec=40.82,
        default_carc_steps=5,
        default_fedavg_steps=5
    ),
    3: ClientTierConfig(
        tier_id=3,
        name="Tier 3 (Slow Edge)",
        description="Cisco UCS Xeon Slice (2 vCPUs, 50% quota) - Mobile Clinic / Station",
        vcpu_allocated=2,
        ram_mb=2048,
        cpu_quota_percent=50.0,
        network_latency_ms=18.0,
        network_bandwidth_mbps=100.0,
        base_train_time_sec=44.20,
        base_samples_per_sec=22.62,
        default_carc_steps=3,
        default_fedavg_steps=5
    ),
    4: ClientTierConfig(
        tier_id=4,
        name="Tier 4 (Constrained Edge)",
        description="Cisco UCS Xeon Slice (1 vCPU, 25% quota) - Handheld Point-of-Care",
        vcpu_allocated=1,
        ram_mb=1024,
        cpu_quota_percent=25.0,
        network_latency_ms=45.0,
        network_bandwidth_mbps=25.0,
        base_train_time_sec=78.50,
        base_samples_per_sec=12.74,
        default_carc_steps=2,
        default_fedavg_steps=5
    ),
    5: ClientTierConfig(
        tier_id=5,
        name="Tier 5 (Network Straggler)",
        description="Cisco UCS Xeon Slice + Netem (2 vCPUs, 200ms RTT, 5Mbps) - Rural Satellite",
        vcpu_allocated=2,
        ram_mb=2048,
        cpu_quota_percent=50.0,
        network_latency_ms=200.0,
        network_bandwidth_mbps=5.0,
        base_train_time_sec=86.80,
        base_samples_per_sec=11.52,
        default_carc_steps=2,
        default_fedavg_steps=5
    )
}


# Fleet compositions across client scale N
FLEET_SCALES_COMPOSITION: Dict[int, Dict[int, int]] = {
    4:  {0: 1, 1: 1, 2: 2, 3: 0, 4: 0, 5: 0},
    20: {0: 1, 1: 4, 2: 7, 3: 5, 4: 2, 5: 1},
    40: {0: 1, 1: 5, 2: 12, 3: 14, 4: 6, 5: 2},
    60: {0: 1, 1: 6, 2: 16, 3: 22, 4: 11, 5: 4},
    80: {0: 1, 1: 8, 2: 20, 3: 30, 4: 15, 5: 6},
    100: {0: 1, 1: 10, 2: 25, 3: 36, 4: 20, 5: 8}
}

TOTAL_PHYSICAL_VCPUS = 120  # Cisco UCS 3-host cluster (56 + 32 + 32)


def get_fleet_clients(num_clients: int) -> List[ClientTierConfig]:
    """Returns the ordered list of client tier configs for a federation of size num_clients."""
    if num_clients not in FLEET_SCALES_COMPOSITION:
        # Interpolate or extrapolate distribution
        base_dist = FLEET_SCALES_COMPOSITION[100]
        scale_factor = num_clients / 100.0
        dist = {tier: max(1 if tier <= 2 else 0, int(round(count * scale_factor))) for tier, count in base_dist.items()}
        # Adjust sum
        current_sum = sum(dist.values())
        diff = num_clients - current_sum
        dist[3] += diff
    else:
        dist = FLEET_SCALES_COMPOSITION[num_clients]

    clients = []
    for tier_id, count in dist.items():
        for _ in range(count):
            clients.append(TIER_PROFILES[tier_id])
    return clients


def calculate_cluster_oversubscription(clients: List[ClientTierConfig]) -> Tuple[float, float]:
    """
    Computes total requested vCPUs and the oversubscription factor
    relative to the 120 physical vCPUs across the Cisco UCS hosts.
    """
    # Exclude Tier 0 (runs on workstation RTX 5070 Ti)
    fog_clients = [c for c in clients if c.tier_id != 0]
    total_requested_vcpus = sum(c.vcpu_allocated for c in fog_clients)
    oversubscription_ratio = total_requested_vcpus / float(TOTAL_PHYSICAL_VCPUS)
    return total_requested_vcpus, oversubscription_ratio


def compute_contention_penalty(oversubscription_ratio: float) -> float:
    """
    Calculates the nonlinear system degradation factor from CPU runqueue contention,
    L3 cache thrashing, and memory bus pressure when oversubscribing the 3 physical hosts.
    """
    if oversubscription_ratio <= 1.0:
        # Comfort zone (0 overcommit): linear scaling with negligible virtualization overhead
        return 1.0 + 0.05 * oversubscription_ratio
    else:
        # Saturated zone: context switches, TLB shootdowns, and scheduling runqueue wait
        excess = oversubscription_ratio - 1.0
        penalty = 1.05 + 0.32 * (excess ** 1.35) + 0.12 * (excess ** 2.0)
        return penalty


def simulate_round_timings(
    num_clients: int,
    strategy: str = "CARC",
    target_budget_sec: float = 24.0,
    seed: int = 42
) -> Dict[str, float]:
    """
    Simulates physical execution of 1 federated communication round across
    the high-density heterogeneous testbed.
    """
    np.random.seed(seed + num_clients)
    clients = get_fleet_clients(num_clients)
    total_requested_vcpus, overcommit_ratio = calculate_cluster_oversubscription(clients)
    contention = compute_contention_penalty(overcommit_ratio)

    client_durations = []
    idle_times = []
    steps_executed = []
    samples_processed = []

    # Samples per client per step (batch size 32)
    b = 32

    for c in clients:
        # Real-world jitter ~3%
        jitter = np.random.normal(1.0, 0.03)
        effective_speed = (c.base_samples_per_sec / contention) * jitter

        if strategy == "FEDAVG":
            # All clients take uniform 5 steps
            steps = 5
            # Execution time: (steps * b) / speed + comm
            comp_time = (steps * b) / max(1.0, effective_speed)
            comm_time = (c.network_latency_ms * 2 / 1000.0) + (11.6 * 8 / c.network_bandwidth_mbps)
            duration = comp_time + comm_time
            steps_executed.append(steps)
            samples_processed.append(steps * b)
            client_durations.append(duration)

        elif strategy == "CARC":
            # Steps are allocated dynamically according to throughput and target round budget
            target_t = target_budget_sec
            # Allocate steps subject to budget
            comm_time = (c.network_latency_ms * 2 / 1000.0) + (11.6 * 8 / c.network_bandwidth_mbps)
            available_comp_t = max(2.0, target_t - comm_time)
            raw_k = int(np.floor((available_comp_t * effective_speed) / b))
            # Clip between min 2 and max 25
            steps = max(2, min(25, raw_k))
            comp_time = (steps * b) / max(1.0, effective_speed)
            duration = comp_time + comm_time
            steps_executed.append(steps)
            samples_processed.append(steps * b)
            client_durations.append(duration)

    round_latency = max(client_durations)
    for dur in client_durations:
        idle_times.append(round_latency - dur)

    total_samples = sum(samples_processed)
    throughput_samples_per_sec = total_samples / round_latency
    total_idle_waste_sec = sum(idle_times)

    return {
        "num_clients": num_clients,
        "strategy": strategy,
        "round_latency_sec": round(round_latency, 2),
        "total_idle_waste_sec": round(total_idle_waste_sec, 2),
        "avg_idle_per_client_sec": round(total_idle_waste_sec / num_clients, 2),
        "total_samples": total_samples,
        "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
        "overcommit_ratio": round(overcommit_ratio, 2),
        "contention_penalty": round(contention, 2),
        "min_client_duration": round(min(client_durations), 2),
        "max_client_duration": round(max(client_durations), 2),
        "mean_steps": round(float(np.mean(steps_executed)), 2)
    }
