"""Deterministic, allocation-only and scalar tests for the tuned controller."""

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research.capdrift_controller import (
    CapDriftClientTrainer,
    CapDriftConfig,
    CapDriftCoordinator,
)
from research.heterogeneous_fleet import TIER_PROFILES


class SumLoss(nn.Module):
    def forward(self, output, target):
        return -(output * target).sum()


def scalar_model():
    model = nn.Linear(1, 1, bias=False)
    model.double()
    nn.init.zeros_(model.weight)
    return model


def contribution(client_id, step_count, displacement):
    return {
        "client_id": client_id,
        "q_i": 1.0,
        "steps_completed": step_count,
        "delta_i": [torch.tensor([[displacement]], dtype=torch.float64)],
    }


class TestTunedController(unittest.TestCase):
    def test_configuration_guards_invalid_powers_schedule_and_average(self):
        for invalid in (
            {"budget_lr_power": -0.5},
            {"client_update_normalization_power": float("nan")},
            {"client_update_normalization_power": -0.5},
            {"server_lr_decay_start_round": -1},
            {"server_lr_final_ratio": 1.1},
            {"server_lr_final_ratio": float("nan")},
            {"server_average_decay": 1.0},
            {"server_average_decay": -0.1},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CapDriftConfig(**invalid)

    def test_full_budget_schedule_preserves_all_tier_steps_and_compute_deadline(self):
        cfg = CapDriftConfig(
            T_target_sec=6.5, S_min=5, S_max=40,
            anneal_start_round=None, enable_drift_gating=True, drift_mode="projection",
            scale_local_lr=True, budget_lr_power=0.75,
            server_lr_decay_start_round=15, server_lr_final_ratio=0.25,
        )
        tiers = [TIER_PROFILES[idx] for idx in (0, 1, 2, 4)]
        trainers = [CapDriftClientTrainer(str(tier.tier_id), cfg) for tier in tiers]
        for trainer, tier in zip(trainers, tiers):
            trainer.q_hat = tier.base_samples_per_sec / 16
        for rnd in (1, 15, 16, 25):
            with self.subTest(round=rnd):
                caps = []
                for trainer, tier in zip(trainers, tiers):
                    step_dt = 16 / tier.base_samples_per_sec
                    cap, required = trainer.determine_step_budget(rnd, 25, step_dt)
                    self.assertEqual(required, 5)
                    self.assertLessEqual(cap * step_dt, 6.5)
                    caps.append(cap)
                self.assertEqual(caps, [40, 28, 16, 5])
        # A stale/saturated estimate still can't force work past the hard deadline.
        trainers[-1].q_hat = 1000.0
        self.assertEqual(trainers[-1].determine_step_budget(25, 25, 16 / 12.74), (5, 5))

    def test_budget_power_rescales_work_but_restores_optimizer_lr(self):
        cfg = CapDriftConfig(
            S_min=2, S_max=8, S_0=8, enable_drift_gating=False,
            scale_local_lr=True, budget_lr_power=0.75,
        )
        dataset = TensorDataset(torch.ones(1, 1, dtype=torch.float64), torch.ones(1, 1))
        model = scalar_model()
        trainer = CapDriftClientTrainer("client", cfg)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        result = trainer.train_round(
            model, [model.weight.detach().clone()], None, DataLoader(dataset),
            SumLoss(), opt, torch.device("cpu"),
        )
        ratio = (2 / 8) ** 0.75
        self.assertEqual(result["steps_completed"], 8)
        self.assertAlmostEqual(result["lr_scale"], ratio)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)
        self.assertAlmostEqual(float(result["delta_i"][0]), ratio * 0.1 * 8)

    def test_effective_sample_weights_are_normalized_then_clipped(self):
        for clipping, expected in [(False, 2.2), (True, 1.5)]:
            with self.subTest(clipping=clipping):
                cfg = CapDriftConfig(
                    S_min=5, client_update_normalization_power=0.5,
                    server_lr=1.0, enable_clipping=clipping, kappa_c=1.0,
                )
                model = scalar_model()
                telemetry = CapDriftCoordinator(cfg).aggregate_round(
                    1, model, [contribution("slow", 5, 1.0), contribution("fast", 20, 3.0)],
                    {"slow": 1, "fast": 3},
                )
                self.assertAlmostEqual(float(model.weight.detach()), expected)
                # Raw: n*sqrt(5/steps) gives 1 and 1.5, normalized to 0.4 and 0.6.
                events = {event["cid"]: event for event in telemetry["clipping_events"]}
                self.assertAlmostEqual(events["slow"]["aggregation_weight"], 0.4)
                self.assertAlmostEqual(events["fast"]["aggregation_weight"], 0.6)
                self.assertAlmostEqual(events["slow"]["raw_norm"], 0.4)
                self.assertAlmostEqual(events["fast"]["raw_norm"], 1.8)
                if clipping:
                    self.assertAlmostEqual(events["fast"]["bounded_norm"], 1.1)

    def test_no_work_client_does_not_dilute_effective_sample_weights(self):
        cfg = CapDriftConfig(
            client_update_normalization_power=0.5, enable_clipping=True, kappa_c=1.0,
        )
        model = scalar_model()
        coordinator = CapDriftCoordinator(cfg)
        inactive_clients = []
        sample_counts = {"active": 1}
        for i in range(3):
            client_id = f"inactive-{i}"
            inactive = contribution(client_id, 0, 0.0)
            inactive["q_i"] = 0.0
            inactive_clients.append(inactive)
            sample_counts[client_id] = 99
        telemetry = coordinator.aggregate_round(
            1, model, [*inactive_clients, contribution("active", 5, 2.0)], sample_counts,
        )
        events = {event["cid"]: event for event in telemetry["clipping_events"]}
        for inactive in inactive_clients:
            client_id = inactive["client_id"]
            self.assertEqual(events[client_id]["aggregation_weight"], 0.0)
            self.assertNotIn(client_id, coordinator.client_q_hat)
        self.assertEqual(events["active"]["aggregation_weight"], 1.0)
        self.assertEqual(telemetry["clipping_cap_c"], 2.0)
        self.assertEqual(float(model.weight.detach()), 2.0)

    def test_cosine_server_gain_is_applied_to_actual_and_reported_displacement(self):
        cfg = CapDriftConfig(
            enable_clipping=False, server_lr=2.0,
            server_lr_decay_start_round=2, server_lr_final_ratio=0.2,
        )
        model = scalar_model()
        coordinator = CapDriftCoordinator(cfg)
        for rnd in range(1, 7):
            with self.subTest(round=rnd):
                frac = max(0, rnd - 2) / 4
                rate = 2 * (0.2 + 0.8 * (1 + math.cos(math.pi * frac)) / 2)
                before = float(model.weight.detach())
                telemetry = coordinator.aggregate_round(
                    rnd, model, [contribution("client", 5, 1.0)], {"client": 1},
                    total_rounds=6,
                )
                self.assertAlmostEqual(telemetry["server_lr"], rate)
                self.assertAlmostEqual(float(model.weight.detach()) - before, rate)
                self.assertAlmostEqual(float(coordinator.delta_G_prev), rate)

    def test_optional_evaluation_ema_does_not_modify_training_or_consensus_state(self):
        cfg = CapDriftConfig(
            enable_clipping=False, server_lr=1.0, server_average_decay=0.75,
            server_average_start_round=2,
        )
        model = scalar_model()
        coordinator = CapDriftCoordinator(cfg)
        for rnd, delta, expected_raw, expected_average in (
            (1, 2.0, 2.0, None), (2, 4.0, 6.0, 6.0), (3, 8.0, 14.0, 8.0),
        ):
            with self.subTest(round=rnd):
                coordinator.aggregate_round(
                    rnd, model, [contribution("client", 5, delta)], {"client": 1},
                    total_rounds=3,
                )
                if expected_average is None:
                    self.assertIsNone(coordinator.averaged_parameters)
                else:
                    self.assertAlmostEqual(float(coordinator.averaged_parameters[0]), expected_average)
                self.assertAlmostEqual(float(model.weight.detach()), expected_raw)
                self.assertAlmostEqual(float(coordinator.delta_G_prev), delta)


if __name__ == "__main__":
    unittest.main()
