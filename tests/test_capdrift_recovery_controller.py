"""CPU-only recovery checks for class priors and capped solver-mass mixing."""

import contextlib
from dataclasses import asdict
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_competitor_benchmark as bench


class RecordingCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, logits, labels):
        self.logits = logits.detach().clone()
        return super().forward(logits, labels)


def scalar_model():
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    return model


def contribution(cid, displacement):
    return {"client_id": cid, "q_i": 1.0,
            "delta_i": [torch.tensor([[displacement]], dtype=torch.float64)]}


class TestTrainingClassPrior(unittest.TestCase):
    def test_centered_smoothed_prior_only_changes_training_logits(self):
        model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
        nn.init.zeros_(model.weight)
        cfg = bench.CapDriftConfig(S_min=1, S_max=1, S_0=1, local_prior_strength=0.75)
        trainer = bench.CapDriftClientTrainer("majority", cfg)
        examples = torch.ones((1, 1), dtype=torch.float64)
        labels = torch.tensor([0])
        criterion = RecordingCrossEntropy()
        result = trainer.train_round(
            model, [model.weight.detach().clone()], None,
            DataLoader(TensorDataset(examples, labels)), criterion,
            torch.optim.SGD(model.parameters(), lr=0.2), torch.device("cpu"),
            train_class_counts=[20, 0],
        )

        # The unseen local class has finite mass without adding an inference bias.
        log_counts = torch.tensor([24.0, 4.0], dtype=torch.float64).log()
        adjusted = 0.75 * (log_counts - log_counts.mean())
        torch.testing.assert_close(criterion.logits, adjusted.unsqueeze(0))
        expected_raw = -0.2 * (adjusted.softmax(0) - torch.tensor([1.0, 0.0]))
        self.assertEqual(result["steps_completed"], 1)
        torch.testing.assert_close(result["delta_i"][0].reshape(-1), expected_raw)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(examples), expected_raw.unsqueeze(0))

    def test_enabled_prior_requires_local_counts(self):
        model = scalar_model()
        trainer = bench.CapDriftClientTrainer(
            "missing", bench.CapDriftConfig(S_min=1, S_max=1, S_0=1, local_prior_strength=1.0),
        )
        x = torch.ones((1, 1), dtype=torch.float64)
        y = torch.zeros(1, dtype=torch.long)
        with self.assertRaises(ValueError):
            trainer.train_round(model, [model.weight.detach().clone()], None,
                                DataLoader(TensorDataset(x, y)), nn.CrossEntropyLoss(),
                                torch.optim.SGD(model.parameters(), lr=0.1), torch.device("cpu"))
        self.assertEqual(model.weight.item(), 0.0)

    def test_invalid_prior_strength_and_blend_rejected(self):
        invalid_fields = (
            {"local_prior_strength": -0.1}, {"local_prior_strength": float("inf")},
            {"local_prior_strength": float("nan")}, {"gradient_mass_blend": -0.1},
            {"gradient_mass_blend": 1.1}, {"gradient_mass_blend": float("nan")},
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                bench.CapDriftConfig(**fields)


class TestCappedGradientMassBlend(unittest.TestCase):
    def test_capped_interpolation_is_before_dynamic_clipping(self):
        model = scalar_model()
        contributions = [contribution("low", 2), contribution("middle", 3), contribution("high", 80)]
        coordinator = bench.CapDriftCoordinator(
            bench.CapDriftConfig(enable_clipping=True, kappa_c=1.0, gradient_mass_blend=0.5),
        )
        result = coordinator.aggregate_round(
            1, model, contributions, {"low": 1, "middle": 8, "high": 1},
            local_gradient_masses={"low": 1.0, "middle": 10.0, "high": 100.0},
        )

        # Mean mass 18.1 -> ratios 18.1, 1.81, 0.181 -> cap to 2, 1.81, 0.5.
        # Blend halfway with 1.0 and clip the resulting contributions, not the
        # raw deltas or the unbounded normalization ratios.
        expected_norm = [0.3, 3.372, 6.0]
        self.assertAlmostEqual(result["effective_gradient_mass"], 18.1)
        for event, expected in zip(result["clipping_events"], expected_norm):
            self.assertAlmostEqual(event["raw_norm"], expected)
        self.assertAlmostEqual(result["clipping_cap_c"], 3.372)
        self.assertAlmostEqual(model.weight.item(), 7.044)
        self.assertEqual([e["is_clipped"] for e in result["clipping_events"]],
                         [False, False, True])

    def test_blend_requires_masses_but_canonical_route_stays_unchanged(self):
        model = scalar_model()
        updates = [contribution("a", 2), contribution("b", 12)]
        counts = {"a": 1, "b": 1}
        blend = bench.CapDriftCoordinator(bench.CapDriftConfig(gradient_mass_blend=0.5))
        with self.assertRaises(ValueError):
            blend.aggregate_round(1, model, updates, counts)
        self.assertEqual(model.weight.item(), 0.0)
        self.assertEqual(blend.round_history, [])

        masses = {"a": 1.0, "b": 3.0}
        mixed = bench.CapDriftCoordinator(
            bench.CapDriftConfig(enable_clipping=False, gradient_mass_blend=0.5),
        )
        mixed.aggregate_round(1, model, updates, counts, local_gradient_masses=masses)
        self.assertAlmostEqual(model.weight.item(), 6.5)

        canonical_model = scalar_model()
        canonical = bench.CapDriftCoordinator(bench.CapDriftConfig(enable_clipping=False))
        # The zero-blend, explicitly normalized path must not cap ratios.
        canonical.aggregate_round(
            1, canonical_model, updates, counts, local_gradient_masses={"a": 1.0, "b": 9.0},
        )
        self.assertAlmostEqual(canonical_model.weight.item(), 5.0 + 10 / 3)


class TestRecoveryBenchmark(unittest.TestCase):
    def test_frozen_manifest_can_omit_only_new_default_off_controls(self):
        legacy = asdict(bench.CapDriftConfig())
        del legacy["local_prior_strength"]
        del legacy["gradient_mass_blend"]
        manifest = {"name": "legacy-v3", "controller_config": legacy,
                    "local_lr": 0.002, "provenance": {}}
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "selected.json"
            raw = (json.dumps(manifest, indent=2) + "\n").encode()
            path.write_bytes(raw)
            config, serialized, digest = bench.read_tuned_manifest(path)
            self.assertEqual(config.local_prior_strength, 0.0)
            self.assertEqual(config.gradient_mass_blend, 0.0)
            self.assertEqual(serialized, manifest)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(path.read_bytes(), raw)

            del legacy["budget_lr_power"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                bench.read_tuned_manifest(path)

    def test_benchmark_uses_training_counts_and_actual_lr_scaled_sgdm_mass(self):
        tiers = [bench.TIER_PROFILES[idx] for idx in bench.TIER_IDS]
        clients = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9]]
        labels = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 0, 1])
        train_features = torch.zeros((10, 1), dtype=torch.float64)
        train = bench.PreloadedDataset(train_features, labels)
        val = DataLoader(TensorDataset(train_features, torch.zeros_like(labels)))
        model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
        cfg = bench.CapDriftConfig(
            T_target_sec=6.5, S_min=5, S_max=40,
            local_prior_strength=1.0, gradient_mass_blend=0.5,
            scale_local_lr=True, enable_drift_gating=False,
        )
        steps = [40, 28, 16, 5]
        lr_scales = [0.4, 0.5, 0.6, 1.0]
        observed = {}
        original_aggregate = bench.CapDriftCoordinator.aggregate_round

        def fake_train(trainer, *, simulated_step_duration_sec, train_class_counts=None, **kwargs):
            idx = int(trainer.client_id.removeprefix("client_"))
            observed[idx] = list(train_class_counts)
            return {"client_id": trainer.client_id,
                    "delta_i": [torch.zeros_like(model.weight)],
                    "q_i": 1 / simulated_step_duration_sec, "steps_completed": steps[idx],
                    "lr_scale": lr_scales[idx],
                    "tau_tr": steps[idx] * simulated_step_duration_sec, "gated_early": False}

        def record_aggregate(coordinator, *args, **kwargs):
            observed["masses"] = kwargs["local_gradient_masses"]
            return original_aggregate(coordinator, *args, **kwargs)

        with (
            patch.object(bench, "get_model", return_value=model),
            patch.object(bench, "get_model_size_mb", return_value=5.797981262207031),
            patch.object(bench, "evaluate_accuracy", return_value=(50.0, 0.7)),
            patch.object(bench.CapDriftClientTrainer, "train_round", fake_train),
            patch.object(bench.CapDriftCoordinator, "aggregate_round", record_aggregate),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            trial = bench.run_competitor_trial(
                "CapDrift-Tuned", train, val, clients, tiers, config=cfg,
                num_rounds=1, device=torch.device("cpu"),
            )

        self.assertEqual([observed[i] for i in range(4)], [[0, 1], [1, 1], [0, 3], [2, 2]])
        for i, (count, scale) in enumerate(zip(steps, lr_scales)):
            self.assertAlmostEqual(observed["masses"][f"client_{i}"],
                                   scale * bench.fednova_momentum_normalizer(count, bench.LOCAL_MOMENTUM))
        self.assertAlmostEqual(25 * trial["total_time_sec"], 205.6197213692418)
        self.assertLessEqual(trial["max_client_compute_time_sec"], 6.5)


if __name__ == "__main__":
    unittest.main()
