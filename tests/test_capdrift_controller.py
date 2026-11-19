"""
Unit and Integration Tests for CapDrift-FL Controller.
Verifies:
1. Stage 1: Capability Step Budgeting
2. Stage 2: Legacy Cosine Drift Gating and Classifier-Head Surplus Projection
3. Stage 3: Coordinator-Side Contribution Clipping (proving norm bounding & direction preservation)
4. Local LR scaling, server momentum, and late-round workload annealing
5. End-to-end integration round across heterogeneous clients.
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


class TinyClassifier(nn.Module):
    """Keeps the backbone and final-head gradients independent of the weights."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(), nn.Linear(1, 2, bias=False)
        )
        nn.init.zeros_(self.classifier[3].weight)

    def forward(self, x):
        return self.classifier(x) + self.backbone * x


class LinearLoss(nn.Module):
    def forward(self, output, target):
        return (output * target).sum()


def constant_gradient_loader(head_grad):
    return DataLoader(TensorDataset(torch.ones(1, 1), torch.tensor([head_grad])))


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
            drift_mode="gate",
            enable_capability=True,
            enable_drift_gating=True
        )
        client = CapDriftClientTrainer("client_drift", cfg)
        client.q_hat = 0.6  # 0.6 * 30 = 18 steps budget (surplus is 18 - 5 = 13 steps)
        
        model = TinyClassifier()
        global_weights = [p.detach().clone() for p in model.parameters()]

        # Each SGD step has displacement proportional to [3, 1, 2]; oppose it exactly.
        opposing_delta_G = torch.tensor([-3.0, -1.0, -2.0])
        
        res = client.train_round(
            local_model=model,
            global_model_weights=global_weights,
            delta_G_prev=opposing_delta_G,
            train_loader=constant_gradient_loader((-1.0, -2.0)),
            criterion=LinearLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
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


class TestOptimizedCapDrift(unittest.TestCase):
    def test_projection_attenuates_only_surplus_head_updates(self):
        """The required block, backbone, and signed parallel displacement are untouched."""
        for head_update in ((1.0, 2.0), (-1.0, 2.0)):
            with self.subTest(head_update=head_update):
                cfg = CapDriftConfig(
                    S_min=1, S_max=4, S_0=4, K_block=2,
                    drift_mode="projection", drift_parameter_prefix="classifier.3",
                )
                trainer = CapDriftClientTrainer("tiny", cfg)
                model = TinyClassifier()
                result = trainer.train_round(
                    local_model=model,
                    global_model_weights=[p.detach().clone() for p in model.parameters()],
                    delta_G_prev=torch.tensor([1.0, 0.0]),
                    train_loader=constant_gradient_loader(tuple(-x for x in head_update)),
                    criterion=LinearLoss(),
                    optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
                    device=torch.device("cpu"),
                )
                self.assertEqual(result["steps_completed"], 4)
                self.assertFalse(result["gated_early"])
                cosine = head_update[0] / math.hypot(*head_update)
                scale = max(0.0, cosine)
                expected_head = torch.tensor([[4 * head_update[0]], [head_update[1] * (1 + 3 * scale)]])
                expected_backbone = torch.tensor([4 * sum(head_update)])
                torch.testing.assert_close(result["delta_i"][0], expected_backbone)
                torch.testing.assert_close(result["delta_i"][1], expected_head)
                torch.testing.assert_close(model.classifier[3].weight, expected_head)
                self.assertEqual(len(result["drift_evaluations"]), 2)
                for event in result["drift_evaluations"]:
                    self.assertAlmostEqual(event["cosine_similarity"], cosine, places=6)
                    self.assertAlmostEqual(event["orthogonal_scale"], scale, places=6)

    def test_projection_bootstraps_without_consensus(self):
        cfg = CapDriftConfig(
            S_min=1, S_max=3, S_0=3, K_block=2,
            drift_mode="projection", drift_parameter_prefix="classifier.3",
        )
        for consensus in (None, torch.zeros(2)):
            with self.subTest(consensus=consensus):
                model = TinyClassifier()
                result = CapDriftClientTrainer("tiny", cfg).train_round(
                    model, [p.detach().clone() for p in model.parameters()],
                    consensus, constant_gradient_loader((-1.0, -2.0)),
                    LinearLoss(), torch.optim.SGD(model.parameters(), lr=1.0), torch.device("cpu"),
                )
                self.assertEqual(result["steps_completed"], 3)
                self.assertEqual(result["drift_evaluations"], [])
                torch.testing.assert_close(result["delta_i"][1], torch.tensor([[3.0], [6.0]]))

    def test_step_budget_rescales_each_optimizer_group(self):
        cfg = CapDriftConfig(S_min=2, S_max=8, S_0=8, scale_local_lr=True, enable_drift_gating=False)
        for slow in (False, True):
            with self.subTest(slow=slow):
                trainer = CapDriftClientTrainer("tiny", cfg)
                if slow:
                    trainer.q_hat = cfg.S_min / cfg.T_target_sec
                model = TinyClassifier()
                optimizer = torch.optim.SGD([
                    {"params": [model.backbone], "lr": 0.4},
                    {"params": model.classifier[3].parameters(), "lr": 0.2},
                ])
                result = trainer.train_round(
                    model, [p.detach().clone() for p in model.parameters()],
                    None, constant_gradient_loader((-1.0, -2.0)), LinearLoss(), optimizer,
                    torch.device("cpu"),
                )
                steps = 2 if slow else 8
                scale = 1.0 if slow else 0.5
                self.assertEqual(result["steps_completed"], steps)
                self.assertAlmostEqual(result["lr_scale"], scale)
                self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.4, 0.2])
                torch.testing.assert_close(result["delta_i"][0], torch.tensor([steps * 0.4 * scale * 3]))
                torch.testing.assert_close(result["delta_i"][1], torch.tensor([[steps * 0.2 * scale], [steps * 0.4 * scale]]))

    def test_server_momentum_and_prior_head_displacement(self):
        cfg = CapDriftConfig(
            enable_clipping=False, server_momentum=0.9, server_lr=0.5,
            drift_parameter_prefix="classifier.3",
        )
        coordinator = CapDriftCoordinator(cfg)
        model = TinyClassifier()
        samples = {"tiny": 10}

        def aggregate(round_number, backbone, head):
            return coordinator.aggregate_round(round_number, model, [{
                "client_id": "tiny", "q_i": 1.0,
                "delta_i": [torch.tensor([backbone]), torch.tensor(head).reshape(2, 1)],
            }], samples)

        aggregate(1, 4.0, [2.0, -4.0])
        torch.testing.assert_close(coordinator.delta_G_prev, torch.tensor([1.0, -2.0]))
        telemetry = aggregate(2, 0.0, [0.0, 2.0])
        # v_2 = 0.9 * [4, 2, -4] + [0, 0, 2]; applied step = 0.5 * v_2.
        torch.testing.assert_close(coordinator.delta_G_prev, torch.tensor([0.9, -0.8]))
        torch.testing.assert_close(flatten_params(list(model.parameters())), torch.tensor([3.8, 1.9, -2.8]))
        self.assertAlmostEqual(telemetry["global_displacement_norm"], math.sqrt(1.8**2 + 0.9**2 + 0.8**2), places=6)

    def test_annealed_workload_respects_deadline_even_with_stale_throughput(self):
        cfg = CapDriftConfig(
            S_min=5, S_max=40, S_0=40, T_target_sec=6.5,
            anneal_start_round=15, anneal_power=0.5,
        )
        trainer = CapDriftClientTrainer("tiny", cfg)
        trainer.q_hat = 1000.0
        self.assertEqual(trainer.determine_step_budget(server_round=1, total_rounds=25), (40, 5))
        self.assertEqual(trainer.determine_step_budget(server_round=15, total_rounds=25), (40, 5))
        self.assertEqual(trainer.determine_step_budget(server_round=16, total_rounds=25), (24, 5))
        self.assertEqual(trainer.determine_step_budget(server_round=20, total_rounds=25), (17, 5))
        self.assertEqual(trainer.determine_step_budget(server_round=25, total_rounds=25), (5, 5))
        linear = CapDriftClientTrainer("linear", CapDriftConfig(S_0=40, S_max=40, anneal_start_round=15))
        self.assertEqual(linear.determine_step_budget(server_round=20, total_rounds=25), (8, 5))
        model = TinyClassifier()
        result = trainer.train_round(
            model, [p.detach().clone() for p in model.parameters()], None,
            constant_gradient_loader((-1.0, -2.0)), LinearLoss(),
            torch.optim.SGD(model.parameters(), lr=0.01), torch.device("cpu"),
            simulated_step_duration_sec=1.256, server_round=1, total_rounds=25,
        )
        self.assertEqual(result["S_cap"], 5)
        self.assertEqual(result["steps_completed"], 5)
        self.assertAlmostEqual(result["tau_tr"], 6.28)
        self.assertLessEqual(result["tau_tr"], cfg.T_target_sec)

    def test_deadline_shorter_than_one_step_leaves_weights_and_ema_unchanged(self):
        cfg = CapDriftConfig(S_min=5, S_max=40, T_target_sec=6.5, scale_local_lr=True)
        trainer = CapDriftClientTrainer("tiny", cfg)
        trainer.q_hat = 1000.0
        model = TinyClassifier()
        result = trainer.train_round(
            model, [p.detach().clone() for p in model.parameters()], None,
            DataLoader(TensorDataset(torch.empty(0, 1), torch.empty(0, 2))), LinearLoss(),
            torch.optim.SGD(model.parameters(), lr=0.01), torch.device("cpu"),
            simulated_step_duration_sec=6.51,
        )
        self.assertEqual((result["S_cap"], result["S_req"], result["steps_completed"]), (0, 0, 0))
        self.assertEqual((result["tau_tr"], result["q_i"], result["lr_scale"]), (0.0, 0.0, 1.0))
        self.assertEqual(result["q_hat"], 1000.0)
        for displacement in result["delta_i"]:
            torch.testing.assert_close(displacement, torch.zeros_like(displacement))


if __name__ == "__main__":
    unittest.main()
