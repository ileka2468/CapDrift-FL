"""Paired MobileNetV3-Small benchmark on four tiers under a fixed latency model.

FedAvg, FedProx, FedNova (historical step-normalized variant), and CapDrift-Legacy
use the original optimization rules. FedNova-Momentum normalizes by the full
local SGDM coefficient sum. CapDrift-Tuned accepts a frozen development-selected
configuration; CapDrift-FL keeps the existing projection recipe.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.capdrift_controller import (
    CapDriftConfig,
    CapDriftClientTrainer,
    CapDriftCoordinator,
)
from research.competitor_split import filename_group_ids, split_indices
from model import get_model, get_model_size_mb
from research.non_iid import partition_dataset_dirichlet
from research.heterogeneous_fleet import TIER_PROFILES, ClientTierConfig

OUTPUT_DIR = PROJECT_ROOT / "results" / "research" / "competitor_projection"
DATA_RELATIVE_PATH = Path("data/global_test/train")
DEFAULT_SEEDS = [42, 101, 2024, 7, 888]
DEFAULT_METHODS = ("FedAvg", "FedProx", "FedNova", "CapDrift-Legacy", "CapDrift-FL")
METHODS = {name: 0.01 if name == "FedProx" else 0.0 for name in
           (*DEFAULT_METHODS, "FedNova-Momentum", "CapDrift-Tuned")}
CAPDRIFT_METHODS = ("CapDrift-FL", "CapDrift-Tuned")
TIER_IDS = [0, 1, 2, 4]
BATCH_SIZE = 16
MIN_STEPS = 5
MAX_STEPS = 40
COMPUTE_DEADLINE_SEC = 6.5
LOCAL_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


class NonFiniteMetricError(RuntimeError):
    def __init__(self, round_idx: int, accuracy: float, loss: float):
        self.round_idx = round_idx
        super().__init__(f"round {round_idx}: accuracy={accuracy}, loss={loss}")


def network_duration(tier: ClientTierConfig, model_size_mb: float) -> float:
    # Keep the historical one-payload convention (MiB plugged into the nominal
    # Mbps rate) and RTT term so the published benchmark's anchor is unchanged.
    return tier.network_latency_ms * 2.0 / 1000.0 + model_size_mb * 8.0 / tier.network_bandwidth_mbps


def reference_round_duration(client_tiers: List[ClientTierConfig], model_size_mb: float) -> float:
    return max(MIN_STEPS * BATCH_SIZE / tier.base_samples_per_sec +
               network_duration(tier, model_size_mb) for tier in client_tiers)


def build_config(
    method_name: str, *, server_momentum: float, server_lr: float,
    anneal_start_round: int, anneal_power: float,
) -> CapDriftConfig:
    shared = dict(T_target_sec=COMPUTE_DEADLINE_SEC, S_min=MIN_STEPS)
    if method_name in ("FedAvg", "FedProx"):
        return CapDriftConfig(**shared, S_max=MIN_STEPS, enable_capability=False,
                              enable_drift_gating=False, enable_clipping=False)
    if method_name in ("FedNova", "FedNova-Momentum"):
        return CapDriftConfig(**shared, S_max=MAX_STEPS, enable_capability=True,
                              enable_drift_gating=False, enable_clipping=False)
    shared.update(S_max=MAX_STEPS, enable_capability=True, enable_drift_gating=True,
                  enable_clipping=True, dynamic_clipping=True, tau_drift=0.85, kappa_c=1.5)
    if method_name == "CapDrift-Legacy":
        return CapDriftConfig(**shared)
    if method_name == "CapDrift-FL":
        return CapDriftConfig(
            **shared, drift_mode="projection", drift_parameter_prefix="classifier.3",
            scale_local_lr=True, server_momentum=server_momentum, server_lr=server_lr,
            anneal_start_round=anneal_start_round, anneal_power=anneal_power,
        )
    if method_name == "CapDrift-Tuned":
        raise ValueError("CapDrift-Tuned requires an explicit selected controller config")
    raise ValueError(f"Unknown method: {method_name}")


class PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, images_tensor, labels_tensor, group_ids=None, recognized_filenames=None):
        self.images = images_tensor
        self.labels = labels_tensor
        self.targets = labels_tensor
        self.group_ids = group_ids
        self.recognized_filenames = recognized_filenames

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def freeze_batchnorm_stats(module: nn.Module, _inputs) -> None:
    """Use inherited BN running stats despite the local loop calling .train()."""
    module.training = False


def fednova_momentum_normalizer(steps: int, momentum: float) -> float:
    """Sum local SGDM gradient coefficients with zero initial buffer.

    This recurrence avoids cancellation as momentum approaches 1. See
    Wang et al., NeurIPS 2020, section 4.1, Momentum SGD as Local Solver.
    """
    if steps < 0 or not 0 <= momentum < 1:
        raise ValueError("steps must be nonnegative and momentum in [0, 1)")
    coefficient = total = 0.0
    for _ in range(steps):
        coefficient = momentum * coefficient + 1.0
        total += coefficient
    return total


def evaluate_accuracy(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = (correct / total * 100.0) if total > 0 else 0.0
    avg_loss = (total_loss / total) if total > 0 else 0.0
    return acc, avg_loss


def training_targets(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """Read class IDs in subset order without fetching any images."""
    if isinstance(dataset, Subset):
        return training_targets(dataset.dataset)[dataset.indices]
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long, device="cpu")
    if isinstance(dataset, torch.utils.data.TensorDataset) and len(dataset.tensors) == 2:
        return dataset.tensors[1].to(dtype=torch.long, device="cpu")
    raise ValueError("local prior adjustment requires dataset class metadata")


def run_competitor_trial(
    method_name: str,
    train_dataset: torch.utils.data.Dataset,
    val_loader: DataLoader,
    client_indices: List[List[int]],
    client_tiers: List[ClientTierConfig],
    num_rounds: int = 8,
    mu_prox: float = 0.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    *,
    config: Optional[CapDriftConfig] = None,
    rng_policy: str = "paired",
    freeze_bn: bool = False,
    channels_last: bool = False,
    fused_sgd: bool = False,
    local_lr: float = 0.01,
    baseline_local_lr: float = 0.01,
    schedule_total_rounds: Optional[int] = None,
    normalize_momentum_updates: bool = False,
    measure_balanced_accuracy: bool = False,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = num_rounds if schedule_total_rounds is None else schedule_total_rounds
    if not 1 <= num_rounds <= horizon:
        raise ValueError("schedule_total_rounds must cover all trial rounds")
    if normalize_momentum_updates and (method_name not in CAPDRIFT_METHODS or config is None):
        raise ValueError("normalized CapDrift requires an explicit CapDrift controller configuration")
    if normalize_momentum_updates and config.scale_local_lr:
        raise ValueError("momentum-aware ablations must use the common unscaled local LR")
    trial_start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_clients = len(client_indices)
    shuffle_generators = [torch.Generator() if rng_policy == "paired" else None
                          for _ in client_indices]
    client_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=BATCH_SIZE, shuffle=True, generator=g)
        for idxs, g in zip(client_indices, shuffle_generators)
    ]
    sample_counts = {f"client_{i}": len(idxs) for i, idxs in enumerate(client_indices)}
    total_samples = sum(sample_counts.values())

    global_model = get_model("mobilenet_v3_small", num_classes=2, pretrained=True).to(device)
    if channels_last:
        global_model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    model_size_mb = get_model_size_mb(global_model)
    expected_duration = reference_round_duration(client_tiers, model_size_mb)

    cfg = config if config is not None else build_config(
        method_name, server_momentum=0.9, server_lr=1.0,
        anneal_start_round=15, anneal_power=1.0,
    )
    if cfg.gradient_mass_blend and (normalize_momentum_updates or method_name not in CAPDRIFT_METHODS):
        raise ValueError("bounded mass blending is a separate CapDrift aggregation mode")
    client_class_counts = None
    if cfg.local_prior_strength:
        if method_name not in CAPDRIFT_METHODS:
            raise ValueError("local prior adjustment is a CapDrift option")
        targets = training_targets(train_dataset)
        if targets.shape != (len(train_dataset),) or ((targets < 0) | (targets > 1)).any():
            raise ValueError("expected one binary label per training image")
        client_class_counts = [torch.bincount(targets[idxs], minlength=2).tolist() for idxs in client_indices]

    coordinator = CapDriftCoordinator(cfg)
    clients = {f"client_{i}": CapDriftClientTrainer(f"client_{i}", cfg) for i in range(num_clients)}

    for i, cid in enumerate(clients):
        tier = client_tiers[i]
        tier_steps_per_sec = tier.base_samples_per_sec / BATCH_SIZE
        clients[cid].q_hat = tier_steps_per_sec
        coordinator.client_q_hat[cid] = tier_steps_per_sec

    round_telemetry = []
    cumulative_time = 0.0
    cumulative_idle = 0.0
    total_gating_events = 0
    total_clipping_events = 0
    total_projections = 0
    eval_shadow = None

    print(f"\n=======================================================")
    print(f"  Competitor Benchmark: {method_name} (Rounds={num_rounds}, seed={seed})")
    print(f"=======================================================")

    for r in range(1, num_rounds + 1):
        global_weights = [p.data.clone() for p in global_model.parameters()]
        client_results = []
        client_durations = []

        for i in range(num_clients):
            cid = f"client_{i}"
            client = clients[cid]
            loader = client_loaders[i]
            tier = client_tiers[i]

            if shuffle_generators[i] is not None:
                # Match both minibatch order and local model stochasticity for
                # this client/round, independent of other clients' step counts.
                round_seed = (int(seed) * 1_000_003 + r * num_clients + i) % 2**63
                shuffle_generators[i].manual_seed(round_seed)
                torch.manual_seed(round_seed)

            local_model = copy.deepcopy(global_model)
            if freeze_bn:
                for module in local_model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.register_forward_pre_hook(freeze_batchnorm_stats)
            optimizer = torch.optim.SGD(
                local_model.parameters(), lr=local_lr if method_name in CAPDRIFT_METHODS else baseline_local_lr,
                momentum=LOCAL_MOMENTUM, weight_decay=WEIGHT_DECAY, fused=True if fused_sgd else None,
            )
            step_duration_sec = BATCH_SIZE / tier.base_samples_per_sec

            # Special training loop for FedProx
            if method_name == "FedProx":
                s_cap, _ = client.determine_step_budget()
                data_iter = iter(loader)
                local_model.train()
                for _ in range(s_cap):
                    try:
                        images, labels = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        images, labels = next(data_iter)
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    if mu_prox > 0:
                        prox = sum(torch.sum((p_loc - p_glob.to(device)) ** 2) for p_loc, p_glob in zip(local_model.parameters(), global_weights))
                        loss = loss + 0.5 * mu_prox * prox
                    loss.backward()
                    optimizer.step()

                tau_tr = step_duration_sec * s_cap
                delta_i = [p.data.clone() - w_g for p, w_g in zip(local_model.parameters(), global_weights)]
                res = {
                    "client_id": cid,
                    "delta_i": delta_i,
                    "q_i": s_cap / max(1e-4, tau_tr),
                    "steps_completed": s_cap,
                    "drift_evaluations": [],
                    "gated_early": False,
                    "tau_tr": tau_tr
                }
            else:
                res = client.train_round(
                    local_model=local_model,
                    global_model_weights=global_weights,
                    delta_G_prev=coordinator.delta_G_prev,
                    train_loader=loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    simulated_step_duration_sec=step_duration_sec,
                    **({"server_round": r, "total_rounds": horizon}
                       if method_name in CAPDRIFT_METHODS else {}),
                    **({"train_class_counts": client_class_counts[i]} if client_class_counts is not None else {}),
                )

            compute_time = res["tau_tr"]
            if not math.isfinite(compute_time) or not 0 <= compute_time <= COMPUTE_DEADLINE_SEC + 1e-9:
                raise RuntimeError(f"{method_name} round {r} ({cid}) violates the {COMPUTE_DEADLINE_SEC}s compute deadline: {compute_time:.9f}s")
            network_time = network_duration(tier, model_size_mb)
            tot_dur = compute_time + network_time
            res["total_client_duration"] = tot_dur
            client_durations.append(tot_dur)
            client_results.append(res)

            if res["gated_early"]:
                total_gating_events += 1

        round_duration = max(client_durations)
        if not math.isclose(round_duration, expected_duration, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"{method_name} round {r} modeled time {round_duration:.9f}s differs from reference {expected_duration:.9f}s")
        round_idle = sum(round_duration - d for d in client_durations)
        cumulative_time += round_duration
        cumulative_idle += round_idle

        # Server Aggregation Strategy
        if method_name in ("FedNova", "FedNova-Momentum"):
            # The historical variant must retain the step-count normalizer.
            # The canonical SGDM variant follows the authors' implementation:
            # tau_eff = sum p_i a_i; global delta = tau_eff * sum p_i delta_i/a_i.
            normalizers = [
                fednova_momentum_normalizer(res["steps_completed"], LOCAL_MOMENTUM)
                if method_name == "FedNova-Momentum" else res["steps_completed"]
                for res in client_results
            ]
            tau_eff = sum(sample_counts[res["client_id"]] / total_samples * norm
                          for res, norm in zip(client_results, normalizers))
            with torch.no_grad():
                for param_idx, p in enumerate(global_model.parameters()):
                    agg_delta = sum(
                        (sample_counts[res["client_id"]] / total_samples) *
                        (res["delta_i"][param_idx] / norm) *
                        tau_eff
                        for res, norm in zip(client_results, normalizers) if norm > 0
                    )
                    p.data.add_(agg_delta)
            coord_telem = {"clipping_events": [], "clipping_cap_c": 0.0}
        else:
            normalized = (
                {res["client_id"]: fednova_momentum_normalizer(res["steps_completed"], LOCAL_MOMENTUM)
                 * (res["lr_scale"] if cfg.gradient_mass_blend else 1.0)
                 for res in client_results}
                if normalize_momentum_updates or cfg.gradient_mass_blend else None
            )
            coord_telem = coordinator.aggregate_round(
                server_round=r,
                global_model=global_model,
                client_results=client_results,
                sample_counts=sample_counts,
                **({"total_rounds": horizon} if method_name in CAPDRIFT_METHODS else {}),
                **({"local_gradient_masses": normalized} if normalized is not None else {}),
            )

        num_clipped = sum(1 for e in coord_telem.get("clipping_events", []) if e.get("is_clipped"))
        num_projections = sum(
            1 for res in client_results for event in res.get("drift_evaluations", [])
            if event.get("orthogonal_scale", 1.0) < 1.0
        )
        total_clipping_events += num_clipped
        total_projections += num_projections

        eval_parameters = getattr(coordinator, "averaged_parameters", None)
        evaluation_fields = {}
        if eval_parameters is None:
            val_acc, val_loss = evaluate_accuracy(global_model, val_loader, criterion, device)
        else:
            # The training iterate and its BN buffers continue unchanged;
            # only the evaluation weight shadow receives the EMA parameters.
            if eval_shadow is None:
                eval_shadow = copy.deepcopy(global_model)
            with torch.no_grad():
                for shadow, average in zip(eval_shadow.parameters(), eval_parameters):
                    shadow.copy_(average)
            val_acc, val_loss = evaluate_accuracy(eval_shadow, val_loader, criterion, device)
            raw_acc, raw_loss = evaluate_accuracy(global_model, val_loader, criterion, device)
            if not (math.isfinite(raw_acc) and math.isfinite(raw_loss)):
                raise NonFiniteMetricError(r, raw_acc, raw_loss)
            evaluation_fields = {"raw_val_accuracy": raw_acc, "raw_val_loss": raw_loss}
        if not (math.isfinite(val_acc) and math.isfinite(val_loss)):
            raise NonFiniteMetricError(r, val_acc, val_loss)
        utilization = (sum(client_durations) / (num_clients * round_duration)) * 100.0 if round_duration > 0 else 0.0
        compute_utilization = (sum(res["tau_tr"] for res in client_results) /
                               (num_clients * round_duration)) * 100.0
        steps_per_client = {res["client_id"]: res["steps_completed"] for res in client_results}

        round_info = {
            "round": r,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "round_duration_sec": round_duration,
            "cumulative_time_sec": cumulative_time,
            "round_idle_sec": round_idle,
            "cluster_utilization_pct": utilization,
            "compute_utilization_pct": compute_utilization,
            "compute_sec_per_client": {res["client_id"]: res["tau_tr"] for res in client_results},
            "client_duration_sec": {res["client_id"]: res["total_client_duration"] for res in client_results},
            "steps_per_client": steps_per_client,
            "lr_scale_per_client": {res["client_id"]: res.get("lr_scale", 1.0) for res in client_results},
            "projection_evaluations": num_projections,
            "num_clipped": num_clipped,
            **evaluation_fields,
        }
        if "server_lr" in coord_telem:
            round_info["effective_server_lr"] = coord_telem["server_lr"]
        if normalize_momentum_updates or cfg.gradient_mass_blend:
            round_info["effective_gradient_mass"] = coord_telem["effective_gradient_mass"]
        if cfg.gradient_mass_blend:
            round_info["mass_adjustment_per_client"] = {
                event["cid"]: event["mass_adjustment"] for event in coord_telem["clipping_events"]
            }
        if measure_balanced_accuracy and r == num_rounds:
            evaluation_model = eval_shadow if eval_parameters is not None else global_model
            class_counts, class_hits = [0, 0], [0, 0]
            with torch.inference_mode():
                for images, labels in val_loader:
                    labels = labels.to(device)
                    pred = evaluation_model(images.to(device)).argmax(dim=1)
                    for k in range(2):
                        mask = labels == k
                        class_counts[k] += mask.sum().item()
                        class_hits[k] += (mask & (pred == labels)).sum().item()
            if not all(class_counts):
                raise ValueError("balanced accuracy requires both classes")
            by_class = [100.0 * hits / count for hits, count in zip(class_hits, class_counts)]
            round_info["balanced_accuracy"] = sum(by_class) / 2.0
            round_info["class_sample_counts"] = class_counts
            round_info["class_correct_counts"] = class_hits
        round_telemetry.append(round_info)

        print(
            f"   Round {r:02d} | Acc: {val_acc:5.2f}% | Loss: {val_loss:.4f} | "
            f"Duration: {round_duration:5.2f}s | Idle: {round_idle:5.2f}s | "
            f"Util: {utilization:4.1f}% | Steps: {list(steps_per_client.values())} | "
            f"Clipped: {num_clipped} | Projected: {num_projections}"
        )

    accuracies = [h["val_accuracy"] for h in round_telemetry]
    return {
        "method": method_name,
        "seed": seed,
        "num_rounds": num_rounds,
        "peak_accuracy": max(accuracies),
        "final_accuracy": accuracies[-1],
        "final_loss": round_telemetry[-1]["val_loss"],
        "total_time_sec": cumulative_time,
        "total_idle_sec": cumulative_idle,
        "mean_utilization_pct": float(np.mean([h["cluster_utilization_pct"] for h in round_telemetry])),
        "mean_compute_utilization_pct": float(np.mean([h["compute_utilization_pct"] for h in round_telemetry])),
        "max_client_compute_time_sec": max(max(h["compute_sec_per_client"].values()) for h in round_telemetry),
        "wall_time_sec": time.perf_counter() - trial_start,
        "total_gating_events": total_gating_events,
        "total_clipping_events": total_clipping_events,
        "total_projection_evaluations": total_projections,
        "history": round_telemetry
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="ImageFolder root (auto-detects local or parent workspace data)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="Reuse completed trials from an incomplete run with identical settings")
    parser.add_argument("--continue-on-nonfinite", action="store_true", help="Record numerical failures and attempt remaining trials")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-policy", choices=("image", "filename-group"), default="image",
                        help="Group-holdout policy uses inferred filename IDs and exact content, not confirmed patient IDs")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--rng-policy", choices=["paired", "legacy"], default="paired")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last model layout for all methods")
    parser.add_argument("--freeze-bn", action="store_true", help="Freeze pretrained BN running statistics in all methods")
    parser.add_argument("--fused-sgd", action="store_true", help="Use fused CUDA SGD for all methods")
    parser.add_argument("--cpu-threads", type=int, default=4, help="PyTorch intra-op threads, including image preprocessing")
    parser.add_argument("--local-lr", type=float, default=0.01, help="CapDrift-FL base local learning rate")
    parser.add_argument("--baseline-local-lr", type=float, default=0.01, help="Shared fixed learning rate for comparison methods")
    parser.add_argument("--server-lr", type=float, default=1.0, help="CapDrift-FL server learning rate")
    parser.add_argument("--server-momentum", type=float, default=0.9, help="CapDrift-FL server momentum")
    parser.add_argument("--anneal-start-round", type=int, default=15, help="Keep full ceiling through this round")
    parser.add_argument("--anneal-power", type=float, default=1.0)
    parser.add_argument("--tuned-config", type=Path,
                        help="Frozen development-selection manifest; required with CapDrift-Tuned")
    return parser


def load_dataset(root: Path) -> Tuple[PreloadedDataset, List[str], str]:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Preloading dataset into memory: {root}", flush=True)
    raw_dataset = ImageFolder(str(root), transform=tf)
    if len(raw_dataset.classes) != 2:
        raise ValueError("Expected exactly two ImageFolder classes")
    images = torch.empty((len(raw_dataset), 3, 224, 224))
    labels = torch.tensor(raw_dataset.targets, dtype=torch.long)
    digest = hashlib.sha256()
    paths, image_hashes = [], []
    for i, (path, label) in enumerate(raw_dataset.samples):
        file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        digest.update(json.dumps([Path(path).relative_to(root).as_posix(), label, file_hash]).encode("utf-8"))
        paths.append(path)
        image_hashes.append(file_hash)
        images[i] = raw_dataset[i][0]
    group_ids, recognized = filename_group_ids(paths, image_hashes)
    return PreloadedDataset(images, labels, group_ids, recognized), raw_dataset.classes, digest.hexdigest()


def write_json(path: Path, data: Dict) -> None:
    temp = path.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    temp.replace(path)


def split_fingerprint(train_indices, val_indices) -> str:
    payload = json.dumps([train_indices, val_indices], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_tuned_manifest(path: Path):
    """Read one frozen controller configuration and its development provenance."""
    payload = path.read_bytes()
    manifest = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(manifest, dict) or not all(k in manifest for k in (
            "name", "controller_config", "local_lr", "provenance")):
        raise ValueError("manifest requires name, controller_config, local_lr and provenance")
    if (not isinstance(manifest["name"], str) or not manifest["name"]
            or not isinstance(manifest["controller_config"], dict)
            or not isinstance(manifest["provenance"], dict)):
        raise ValueError("manifest config and provenance must be objects and name must not be empty")
    local_lr = manifest["local_lr"]
    if isinstance(local_lr, bool) or not isinstance(local_lr, (float, int)) or not math.isfinite(local_lr) or local_lr <= 0:
        raise ValueError("manifest local_lr must be positive and finite")
    config = CapDriftConfig(**manifest["controller_config"])
    default_off = {"local_prior_strength", "gradient_mass_blend"}
    if asdict(config).keys() - default_off - manifest["controller_config"].keys():
        raise ValueError("manifest controller_config must specify all controller fields")
    return config, manifest, hashlib.sha256(payload).hexdigest()


def validate_tuned_provenance(manifest: Dict, args, data_sha256: str, outer_split_sha256: str):
    provenance = manifest["provenance"]
    expected = {
        "data_sha256": data_sha256,
        "outer_split_policy": args.split_policy,
        "outer_split_seed": args.split_seed,
        "outer_fraction": args.holdout_fraction,
        "outer_split_sha256": outer_split_sha256,
        "dirichlet_alpha": args.alpha,
        "partition_seed": args.partition_seed,
    }
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    if provenance.get("finalists", {}).get("rounds") != args.rounds:
        mismatches.append("finalists.rounds")
    environment = provenance.get("environment", {})
    for key, value in {
        "paired_rng": args.rng_policy == "paired",
        "freeze_bn": args.freeze_bn,
        "channels_last": args.channels_last,
        "fused_sgd": args.fused_sgd,
    }.items():
        if key in environment and environment[key] != value:
            mismatches.append(f"environment.{key}")
    if mismatches:
        raise ValueError(f"tuned manifest protocol mismatch: {', '.join(mismatches)}")


def summarize_trials(name: str, trials: List[Dict]) -> Dict:
    summary = {"method": name, "trials": trials}
    metrics = {
        "final_acc": ("final_accuracy", 2),
        "peak_acc": ("peak_accuracy", 2),
        "loss": ("final_loss", 4),
        "time_sec": ("total_time_sec", 2),
        "idle_sec": ("total_idle_sec", 2),
        "util_pct": ("mean_utilization_pct", 2),
        "compute_util_pct": ("mean_compute_utilization_pct", 2),
        "wall_time_sec": ("wall_time_sec", 2),
        "gating_events": ("total_gating_events", 2),
        "clipping_events": ("total_clipping_events", 2),
        "projection_evaluations": ("total_projection_evaluations", 2),
    }
    for key, (trial_key, digits) in metrics.items():
        values = [trial[trial_key] for trial in trials]
        summary[f"mean_{key}"] = round(float(np.mean(values)), digits)
        summary[f"std_{key}"] = round(float(np.std(values)), digits)
    return summary


def validate_resume(previous: Dict, expected: Dict, results: Dict) -> None:
    ignored = {"complete", "failed_trials"}
    keys = previous.keys() | expected.keys()
    mismatches = sorted(key for key in keys - ignored if previous.get(key) != expected.get(key))
    if mismatches:
        raise ValueError(f"resume protocol mismatch: {', '.join(mismatches)}")
    if previous["complete"]:
        raise ValueError("the previous run is already marked complete")
    if results.keys() - set(expected["methods"]):
        raise ValueError("unexpected method in previously saved results")
    for name, method in results.items():
        seen = set()
        for trial in method["trials"]:
            seed = trial["seed"]
            if (seed in seen or seed not in expected["seeds"] or trial["method"] != name
                    or trial["num_rounds"] != expected["num_rounds"]
                    or len(trial["history"]) != expected["num_rounds"]
                    or not math.isfinite(trial["final_accuracy"])
                    or not math.isfinite(trial["final_loss"])
                    or not math.isclose(trial["total_time_sec"], expected["modeled_reference_total_time_sec"])):
                raise ValueError(f"invalid saved trial for {name}, seed {seed}")
            seen.add(seed)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.anneal_start_round < 0 or args.cpu_threads < 1:
        parser.error("rounds and cpu-threads must be positive and anneal-start-round nonnegative")
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.alpha, args.anneal_power, args.local_lr, args.baseline_local_lr, args.server_lr)):
        parser.error("alpha, anneal-power and learning rates must be positive and finite")
    if not 0.0 <= args.server_momentum < 1.0:
        parser.error("server-momentum must be in [0, 1)")
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("holdout-fraction must be in (0, 1)")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("methods and seeds must not contain duplicates")
    if any(not 0 <= seed < 2**32 for seed in [args.split_seed, args.partition_seed, *args.seeds]):
        parser.error("seeds must be in [0, 2**32)")
    tuned = tuned_manifest = tuned_hash = None
    if ("CapDrift-Tuned" in args.methods) != (args.tuned_config is not None):
        parser.error("--tuned-config is required exactly when CapDrift-Tuned is selected")
    if args.tuned_config is not None:
        try:
            tuned, tuned_manifest, tuned_hash = read_tuned_manifest(args.tuned_config)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.fused_sgd and device.type != "cuda":
        parser.error("--fused-sgd requires CUDA")
    print(f"Device: {device}; deterministic algorithms: {args.deterministic}; RNG: {args.rng_policy}")

    data_dir = args.data_dir
    if data_dir is None:
        candidates = [PROJECT_ROOT / DATA_RELATIVE_PATH, PROJECT_ROOT.parent / DATA_RELATIVE_PATH]
        data_dir = next((path for path in candidates if path.is_dir()), None)
        if data_dir is None:
            parser.error(f"ImageFolder data not found in {candidates}; pass --data-dir")
    data_dir = data_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "competitor_benchmark_results.json"
    metadata_path = output_dir / "competitor_benchmark_metadata.json"
    csv_path = output_dir / "competitor_benchmark_summary.csv"
    if args.resume and not metadata_path.is_file():
        parser.error("--resume requires metadata from a previous incomplete run")
    if not args.resume and any(path.exists() for path in [json_path, metadata_path, csv_path]):
        parser.error("benchmark artifacts already exist; choose a new --output-dir to preserve them")
    full_dataset, classes, data_hash = load_dataset(data_dir)
    try:
        train_indices, val_indices = split_indices(
            full_dataset, policy=args.split_policy, seed=args.split_seed,
            holdout_fraction=args.holdout_fraction,
        )
        outer_split_sha = split_fingerprint(train_indices, val_indices)
        if tuned_manifest is not None:
            validate_tuned_provenance(tuned_manifest, args, data_hash, outer_split_sha)
    except (ValueError, TypeError, AttributeError) as exc:
        parser.error(str(exc))

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    print(f"Loaded {len(full_dataset)} images: {len(train_subset)} train and {len(val_subset)} validation.")

    num_clients = len(TIER_IDS)
    client_sub_indices = partition_dataset_dirichlet(
        train_subset, num_clients=num_clients, alpha=args.alpha, seed=args.partition_seed,
    )
    client_indices = [[train_indices[idx] for idx in sub] for sub in client_sub_indices]
    if sorted(i for client in client_indices for i in client) != sorted(train_indices) or any(not client for client in client_indices):
        raise RuntimeError("Client partitions must cover training samples exactly once and have no empty clients")
    client_tiers = [TIER_PROFILES[i] for i in TIER_IDS]
    configs = {name: tuned if name == "CapDrift-Tuned" else build_config(
        name, server_momentum=args.server_momentum, server_lr=args.server_lr,
        anneal_start_round=args.anneal_start_round, anneal_power=args.anneal_power,
    ) for name in args.methods}
    payload_mb = get_model_size_mb(get_model("mobilenet_v3_small", num_classes=2, pretrained=False))
    modeled_round = reference_round_duration(client_tiers, payload_mb)
    print(f"Modeled time anchor: {modeled_round:.9f} s/round, {args.rounds * modeled_round:.2f} s total.")

    def class_counts(indices):
        counts = np.bincount(full_dataset.labels[indices].numpy(), minlength=len(classes))
        return dict(zip(classes, counts.tolist()))

    metadata = {
        "schema_version": 3,
        "complete": False,
        "methods": args.methods,
        "seeds": args.seeds,
        "num_rounds": args.rounds,
        "num_clients": num_clients,
        "alpha": args.alpha,
        "split_seed": args.split_seed,
        "split_policy": args.split_policy,
        "holdout_fraction": args.holdout_fraction,
        "outer_split_sha256": outer_split_sha,
        "partition_seed": args.partition_seed,
        "rng_policy": args.rng_policy,
        "data_dir": str(data_dir),
        "dataset_content_sha256": data_hash,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_class_counts": class_counts(train_indices),
        "val_class_counts": class_counts(val_indices),
        "client_class_counts": {f"client_{i}": class_counts(idx) for i, idx in enumerate(client_indices)},
        "split_partition_sha256": hashlib.sha256(json.dumps(
            [train_indices, val_indices, client_indices]).encode("utf-8")).hexdigest(),
        "model": "mobilenet_v3_small",
        "weights": torchvision.models.MobileNet_V3_Small_Weights.DEFAULT.name,
        "model_payload_mib": payload_mb,
        "training_batch_size": BATCH_SIZE,
        "local_optimizer": {
            "name": "SGD", "baseline_lr": args.baseline_local_lr, "capdrift_lr": args.local_lr,
            "momentum": LOCAL_MOMENTUM, "weight_decay": WEIGHT_DECAY, "fused": args.fused_sgd,
        },
        "freeze_bn": args.freeze_bn,
        "batchnorm_protocol": (
            "Pretrained running statistics frozen for all local training and validation"
            if args.freeze_bn else "Original: local BN uses minibatch statistics; server retains pretrained running statistics"
        ),
        "fednova_protocol": "Historical step-normalized SGDM variant, not momentum-aware FedNova normalization",
        "controller_configs": {name: asdict(c) for name, c in configs.items()},
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "deterministic_algorithms": args.deterministic,
        "channels_last": args.channels_last,
        "cpu_threads": args.cpu_threads,
        "tier_ids": TIER_IDS,
        "tiers": [asdict(tier) for tier in client_tiers],
        "compute_deadline_sec": COMPUTE_DEADLINE_SEC,
        "modeled_reference_round_duration_sec": modeled_round,
        "modeled_reference_total_time_sec": args.rounds * modeled_round,
        "time_protocol": "max_client(compute + 2 latency_ms / 1000 + 8 payload_mib / bandwidth_mbps)",
        "cluster_utilization_protocol": "sum(client compute + communication time) / (clients * round time)",
        "compute_utilization_protocol": "sum(client compute time) / (clients * round time)",
        "metric_protocol": "Per-trial final round (not best round); population SD across listed seeds",
    }
    if "FedNova-Momentum" in args.methods:
        metadata["fednova_momentum_protocol"] = (
            "Author implementation: a_i = sum of local SGDM buffer coefficients "
            "(round-reset, no Nesterov), tau_eff = sum p_i a_i; delta = tau_eff sum p_i delta_i/a_i"
        )
    if args.split_policy == "filename-group":
        train_groups = {full_dataset.group_ids[i] for i in train_indices}
        val_groups = {full_dataset.group_ids[i] for i in val_indices}
        if not train_groups.isdisjoint(val_groups):
            raise RuntimeError("overlap between train and held-out filename groups")
        metadata["filename_groups"] = {
            "protocol": "Heuristic filename subject proxies merged with exact-byte aliases",
            "verified_patient_ids": False,
            "recognized_filenames": full_dataset.recognized_filenames,
            "training_groups": len(train_groups),
            "holdout_groups": len(val_groups),
            "overlap_groups": 0,
        }
    if tuned_manifest is not None:
        metadata["capdrift_tuned_manifest"] = tuned_manifest
        metadata["capdrift_tuned_manifest_sha256"] = tuned_hash
        metadata["local_optimizer"]["capdrift_tuned_lr"] = tuned_manifest["local_lr"]
        metadata["metric_protocol"] += "; tuned uses EMA shadow if configured, with raw metrics recorded"
    all_results = {}
    if args.resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if json_path.is_file():
            all_results = json.loads(json_path.read_text(encoding="utf-8"))
        validate_resume(previous, metadata, all_results)
    else:
        write_json(metadata_path, metadata)

    failed_trials = []
    summary_data = []

    for name in args.methods:
        print(f"\n=======================================================")
        print(f"  Benchmarking {name} across {len(args.seeds)} seeds ({args.rounds} rounds)...")
        print(f"=======================================================")
        completed = {trial["seed"]: trial for trial in all_results.get(name, {}).get("trials", [])}
        for seed in args.seeds:
            if seed in completed:
                print(f"   Reusing validated completed trial: {name}, seed {seed}")
                continue
            try:
                completed[seed] = run_competitor_trial(
                    method_name=name,
                    train_dataset=full_dataset,
                    val_loader=val_loader,
                    client_indices=client_indices,
                    client_tiers=client_tiers,
                    num_rounds=args.rounds,
                    mu_prox=METHODS[name],
                    device=device,
                    seed=seed,
                    config=configs[name],
                    rng_policy=args.rng_policy,
                    freeze_bn=args.freeze_bn,
                    channels_last=args.channels_last,
                    fused_sgd=args.fused_sgd,
                    local_lr=tuned_manifest["local_lr"] if name == "CapDrift-Tuned" else args.local_lr,
                    baseline_local_lr=args.baseline_local_lr,
                )
            except NonFiniteMetricError as exc:
                failed_trials.append({"method": name, "seed": seed, "round": exc.round_idx, "diagnostic": str(exc)})
                metadata["failed_trials"] = failed_trials
                write_json(metadata_path, metadata)
                print(f"   Numerical failure: {name}, seed {seed}, {exc}")
                if not args.continue_on_nonfinite:
                    raise
                continue
            all_results[name] = summarize_trials(name, [completed[i] for i in args.seeds if i in completed])
            write_json(json_path, all_results)

        data = all_results.get(name)

        def summary_stat(key, digits):
            return f"{data[f'mean_{key}']:.{digits}f} ± {data[f'std_{key}']:.{digits}f}" if data else "N/A"

        summary_data.append({
            "Method": name,
            "Successful Trials": f"{len(completed)}/{len(args.seeds)}",
            "Final Acc (%) [Mean ± Std]": summary_stat("final_acc", 2),
            "Peak Acc (%) [Mean ± Std]": summary_stat("peak_acc", 2),
            "Final Loss": summary_stat("loss", 4),
            "Total Time (s)": summary_stat("time_sec", 2),
            "Idle Waste (s)": summary_stat("idle_sec", 2),
            "Mean Util (%)": summary_stat("util_pct", 2),
            "Mean Compute Util (%)": summary_stat("compute_util_pct", 2),
            "Mean Wall Time (s)": summary_stat("wall_time_sec", 2),
            "Gated Events": summary_stat("gating_events", 1),
            "Clipped Events": summary_stat("clipping_events", 1),
            "Projected Blocks": summary_stat("projection_evaluations", 1),
        })

    write_json(json_path, all_results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_data[0].keys()))
        writer.writeheader()
        writer.writerows(summary_data)
    metadata["complete"] = not failed_trials
    write_json(metadata_path, metadata)
    print(f"\n[Saved Competitor Results] -> {json_path}")
    print(f"[Saved Competitor CSV] -> {csv_path}")
    print(f"[Saved Benchmark Metadata] -> {metadata_path}")

    # Print markdown summary
    print("\n" + "=" * 90)
    print(f"  COMPETITOR BENCHMARK SUMMARY ({args.rounds} rounds, {len(args.seeds)} seeds)")
    print("=" * 90)
    print("| Method | Successful Trials | Final Acc (%) | Peak Acc (%) | Final Loss | Modeled Time (s) | Idle Waste (s) | Active Util (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in summary_data:
        print(f"| **{r['Method']}** | {r['Successful Trials']} | **{r['Final Acc (%) [Mean ± Std]']}** | {r['Peak Acc (%) [Mean ± Std]']} | {r['Final Loss']} | {r['Total Time (s)']} | {r['Idle Waste (s)']} | {r['Mean Util (%)']} |")
    print("=" * 90)
    return 2 if failed_trials else 0


if __name__ == "__main__":
    raise SystemExit(main())
