"""
Unit and Integration Tests for CapDrift-FL Controller.
Verifies:
1. Stage 1: Capability Step Budgeting
2. Stage 2: Block-Wise Cosine Drift Gating (proving surplus halts on drift)
3. Stage 3: Coordinator-Side Contribution Clipping (proving norm bounding & direction preservation)
4. End-to-end integration round across heterogeneous clients.
"""

import copy
import math
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "research"))

from capdrift_controller import (
    CapDriftConfig,
    CapDriftClientTrainer,
    CapDriftCoordinator,
    compute_directional_drift,
    flatten_params
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc(x)


class TestCapDriftController(unittest.TestCase):

    def test_stage1_capability_budgeting(self):
        """Verify Eq. (6): S_cap_i(t) = min(S_max, max(1, floor(q_hat * T)))"""
        cfg = CapDriftConfig(
            T_target_sec=30.0,
            S_max=50,
            S_min=5,
            S_0=5,
            enable_capability=True
        )
        client = CapDriftClientTrainer("client_1", cfg)
        
        # Before any observations, should use S_0
        s_cap, s_req = client.determine_step_budget()
        self.assertEqual(s_cap, 5)
        self.assertEqual(s_req, 5)
        
        # Set throughput q_hat = 0.5 steps/sec -> 0.5 * 30 = 15 steps
        client.q_hat = 0.5
        s_cap, s_req = client.determine_step_budget()
        self.assertEqual(s_cap, 15)
        self.assertEqual(s_req, 5)
        
        # Set throughput q_hat = 3.0 steps/sec -> 3.0 * 30 = 90 -> capped at S_max = 50
        client.q_hat = 3.0
        s_cap, s_req = client.determine_step_budget()
        self.assertEqual(s_cap, 50)
        self.assertEqual(s_req, 5)
        
        # Set slow throughput q_hat = 0.05 steps/sec -> 0.05 * 30 = 1.5 -> floor = 1
        client.q_hat = 0.05
        s_cap, s_req = client.determine_step_budget()
        self.assertEqual(s_cap, 1)
        self.assertEqual(s_req, 1) # min(5, 1) = 1

    def test_stage2_directional_drift_calculation(self):
        """Verify Eq. (5): d_i(t, k) = 1 - cos(Delta_i, Delta_G)"""
        # Perfect alignment (cos = 1.0 -> d = 0.0)
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([2.0, 4.0, 6.0])
        d = compute_directional_drift(v1, v2)
        self.assertAlmostEqual(d, 0.0, places=5)
        
        # Orthogonal (cos = 0.0 -> d = 1.0)
        v_ortho = torch.tensor([-2.0, 1.0, 0.0])
        d_ortho = compute_directional_drift(v1, v_ortho)
        self.assertAlmostEqual(d_ortho, 1.0, places=5)
        
        # Opposite direction (cos = -1.0 -> d = 2.0)
        v_opp = torch.tensor([-1.0, -2.0, -3.0])
        d_opp = compute_directional_drift(v1, v_opp)
        self.assertAlmostEqual(d_opp, 2.0, places=5)
        
        # Numerical threshold epsilon
        v_zero = torch.tensor([1e-10, 0.0, 0.0])
        self.assertIsNone(compute_directional_drift(v_zero, v1, epsilon=1e-8))

    def test_stage2_drift_gating_halts_surplus_training(self):
        """
        Verify that when a client's local update opposes the global trajectory,
        drift gating (d_i > tau_d) halts training after S_req, preventing surplus steps.
        """
        cfg = CapDriftConfig(
            T_target_sec=30.0,
            S_max=20,
            S_min=5,
            K_block=5,
            tau_drift=0.50, # requires cosine sim >= 0.50
            enable_capability=True,
            enable_drift_gating=True
        )
        client = CapDriftClientTrainer("client_drift", cfg)
        client.q_hat = 0.6  # 0.6 * 30 = 18 steps budget (surplus is 18 - 5 = 13 steps)
        
        model = TinyModel()
        global_weights = [p.data.clone() for p in model.parameters()]
        
        # Synthetic dataset where gradient drives model in positive direction
        X = torch.randn(64, 8)
        y = torch.randint(0, 2, (64,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=8)
        
        # Provide previous global displacement in opposite direction
        # so local update will have negative cosine similarity (drift > 1.0 > tau_drift=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # Perform 1 step to see direction
        model.zero_grad()
        loss = criterion(model(X[:8]), y[:8])
        loss.backward()
        sample_grad = flatten_params([p.grad for p in model.parameters()])
        
        # Construct global displacement that opposes local gradient update
        # Since update is -lr * grad, local displacement Delta_i will align with -grad.
        # So we set delta_G to align with +grad to force cos(Delta_i, Delta_G) < 0 (i.e. d > 1.0)
        opposing_delta_G = sample_grad.clone()
        
        res = client.train_round(
            local_model=model,
            global_model_weights=global_weights,
            delta_G_prev=opposing_delta_G,
            train_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu")
        )
        
        # Client MUST complete the required S_min = 5 steps
        # But MUST be gated early before reaching S_cap = 18!
        self.assertEqual(res["S_req"], 5)
        self.assertEqual(res["S_cap"], 18)
        self.assertTrue(res["gated_early"], "Surplus training should have been gated early by drift check!")
        self.assertEqual(res["steps_completed"], 5, f"Expected 5 steps due to gating, got {res['steps_completed']}")

    def test_stage3_coordinator_clipping(self):
        """
        Verify Eq. (7): Bounding contributions G_i(t) to norm cap c while preserving direction.
        """
        cfg = CapDriftConfig(
            kappa_c=1.5,
            server_lr=1.0,
            enable_clipping=True
        )
        coord = CapDriftCoordinator(cfg)
        global_model = TinyModel()
        
        # Setup 3 dummy client results with different update norms
        # Client 0: Small update (norm 0.2)
        # Client 1: Medium update (norm 0.5)
        # Client 2: Massive outlier update (norm 5.0)
        c0_delta = [torch.full_like(p, 0.05) for p in global_model.parameters()]
        c1_delta = [torch.full_like(p, 0.12) for p in global_model.parameters()]
        c2_delta = [torch.full_like(p, 1.25) for p in global_model.parameters()]
        
        client_results = [
            {"client_id": "c0", "delta_i": c0_delta, "q_i": 1.0},
            {"client_id": "c1", "delta_i": c1_delta, "q_i": 1.0},
            {"client_id": "c2", "delta_i": c2_delta, "q_i": 1.0},
        ]
        sample_counts = {"c0": 100, "c1": 100, "c2": 100}
        
        telemetry = coord.aggregate_round(
            server_round=1,
            global_model=global_model,
            client_results=client_results,
            sample_counts=sample_counts
        )
        
        cap_c = telemetry["clipping_cap_c"]
        events = telemetry["clipping_events"]
        
        # Verify clipping behavior
        # c2 had a massive norm, so it should be clipped
        c2_event = [e for e in events if e["cid"] == "c2"][0]
        self.assertTrue(c2_event["is_clipped"])
        self.assertAlmostEqual(c2_event["bounded_norm"], cap_c, places=4)
        self.assertLess(c2_event["scale_factor"], 1.0)
        
        # c0 was small, so it should NOT be clipped
        c0_event = [e for e in events if e["cid"] == "c0"][0]
        self.assertFalse(c0_event["is_clipped"])
        self.assertAlmostEqual(c0_event["scale_factor"], 1.0, places=5)
        self.assertAlmostEqual(c0_event["bounded_norm"], c0_event["raw_norm"], places=5)

    def test_end_to_end_controller_round(self):
        """
        Verify end-to-end multi-client execution with heterogeneous throughput.
        """
        cfg = CapDriftConfig(
            T_target_sec=10.0,
            S_max=20,
            S_min=3,
            K_block=2,
            tau_drift=0.50,
            kappa_c=1.5,
            server_lr=1.0,
            enable_capability=True,
            enable_drift_gating=True,
            enable_clipping=True
        )
        
        global_model = TinyModel()
        coord = CapDriftCoordinator(cfg)
        
        # 3 clients with different speeds: Fast, Medium, Slow
        clients = {
            "fast_gpu": CapDriftClientTrainer("fast_gpu", cfg),
            "med_fog": CapDriftClientTrainer("med_fog", cfg),
            "slow_edge": CapDriftClientTrainer("slow_edge", cfg)
        }
        clients["fast_gpu"].q_hat = 1.8   # 1.8 * 10 = 18 steps
        clients["med_fog"].q_hat = 0.8    # 0.8 * 10 = 8 steps
        clients["slow_edge"].q_hat = 0.2  # 0.2 * 10 = 2 steps (below S_min -> 2 steps)
        
        X = torch.randn(64, 8)
        y = torch.randint(0, 2, (64,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=8)
        criterion = nn.CrossEntropyLoss()
        
        # Round 1
        client_results = []
        global_weights = [p.data.clone() for p in global_model.parameters()]
        
        for cid, cl in clients.items():
            loc_model = copy.deepcopy(global_model)
            opt = torch.optim.SGD(loc_model.parameters(), lr=0.01)
            res = cl.train_round(
                local_model=loc_model,
                global_model_weights=global_weights,
                delta_G_prev=coord.delta_G_prev,
                train_loader=loader,
                criterion=criterion,
                optimizer=opt,
                device=torch.device("cpu")
            )
            client_results.append(res)
            
        telemetry = coord.aggregate_round(
            server_round=1,
            global_model=global_model,
            client_results=client_results,
            sample_counts={"fast_gpu": 200, "med_fog": 100, "slow_edge": 50}
        )
        
        self.assertIsNotNone(coord.delta_G_prev)
        self.assertGreater(torch.norm(coord.delta_G_prev).item(), 0.0)
        self.assertEqual(len(telemetry["clipping_events"]), 3)
        print("\n [TEST SUCCESS] Round 1 CapDrift Telemetry verified successfully!")


if __name__ == "__main__":
    unittest.main()
