"""CPU-only checks for tuning isolation and the strengthened competitor."""

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_competitor_benchmark as benchmark


class GroupedLabels:
    """A mixed-size fixture that does not load images during splitting."""

    def __init__(self):
        self.targets = []
        self.group_ids = []
        for class_id in range(2):
            for group in range(12):
                size = 1 + (group % 4)
                self.targets.extend([class_id] * size)
                self.group_ids.extend([f"{class_id}:{group}"] * size)

    def __len__(self):
        return len(self.targets)


class TestSplitIsolation(unittest.TestCase):
    def assert_isolated(self, dataset, train, held, universe):
        self.assertEqual(len(set(train)), len(train))
        self.assertEqual(len(set(held)), len(held))
        self.assertEqual(set(train) | set(held), set(universe))
        self.assertFalse(set(train) & set(held))
        train_groups = {dataset.group_ids[idx] for idx in train}
        held_groups = {dataset.group_ids[idx] for idx in held}
        self.assertFalse(train_groups & held_groups)

    def test_historical_image_split_retains_seeded_index_order(self):
        ds = GroupedLabels()
        split = np.random.RandomState(42).permutation(len(ds)).tolist()
        train, held = benchmark.split_indices(ds, policy="image", seed=42, holdout_fraction=0.2)
        self.assertEqual((train, held), (split[:48], split[48:]))

    def test_filename_aliases_and_byte_duplicates_join_transitively(self):
        paths = (
            "PNEUMONIA/person2_virus_1.jpeg",
            "PNEUMONIA/person2_bacteria_8.jpeg",
            "PNEUMONIA/person3_virus_7.jpeg",
            "NORMAL/IM-002-0001.jpeg",
            "NORMAL/IM-002-0002.jpeg",
            "NORMAL/NORMAL2-IM-002-0001.jpeg",
            "NORMAL/unrecognized.png",
        )
        groups, recognized = benchmark.filename_group_ids(paths, ("a", "b", "a", "c", "d", "e", "e"))
        self.assertEqual(recognized, 6)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertEqual(groups[3], groups[4])
        self.assertEqual(groups[5], groups[6])
        self.assertEqual(len(set(groups)), 3)

    def test_group_holdout_is_label_balanced_reproducible_and_nested(self):
        ds = GroupedLabels()
        train, held = benchmark.split_indices(ds, policy="filename-group", seed=42)
        self.assert_isolated(ds, train, held, range(len(ds)))
        self.assertEqual(benchmark.split_indices(ds, policy="filename-group", seed=42), (train, held))
        for label in (0, 1):
            self.assertIn(label, {ds.targets[i] for i in train})
            observed = sum(ds.targets[i] == label for i in held)
            # Six images per class is the 20% target. Whole groups are indivisible.
            self.assertLessEqual(abs(observed - 6), 1)

        dev_train, dev_held = benchmark.split_indices(
            ds, policy="filename-group", seed=107, indices=train
        )
        self.assert_isolated(ds, dev_train, dev_held, train)
        self.assertFalse(set(held) & (set(dev_train) | set(dev_held)))


class TestMomentumAwareFedNova(unittest.TestCase):
    def test_normalizer_matches_integrated_sgd_momentum(self):
        for steps in (0, 1, 5, 16, 40):
            for momentum in (0.0, 0.2, 0.9, 1.0 - 1e-8):
                with self.subTest(steps=steps, momentum=momentum):
                    w = nn.Parameter(torch.zeros((), dtype=torch.float64))
                    optimizer = torch.optim.SGD([w], lr=0.05, momentum=momentum)
                    for _ in range(steps):
                        optimizer.zero_grad(set_to_none=True)
                        w.backward()  # Constant unit gradient; PyTorch handles the buffer.
                        optimizer.step()
                    self.assertTrue(math.isclose(
                        benchmark.fednova_momentum_normalizer(steps, momentum),
                        -float(w.detach()) / 0.05, rel_tol=1e-9, abs_tol=1e-9,
                    ))

    def test_heterogeneous_momentum_normalization_matches_analytic_update(self):
        """Clients emit the same integrated scalar gradient at unequal step counts."""
        tier_ids = (0, 1, 2, 4)
        tiers = [benchmark.TIER_PROFILES[i] for i in tier_ids]
        client_steps = (40, 28, 16, 5)
        # Deliberately unequal sample counts test the weighting term.
        client_indices = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9]]
        dataset = TensorDataset(torch.zeros(10, 1), torch.zeros(10, dtype=torch.long))
        model = nn.Linear(1, 1, bias=False).double()
        nn.init.zeros_(model.weight)
        payload_mib = 5.797981262207031  # fixed benchmark payload convention

        def effective_steps(steps):
            buffer, total = 0.0, 0.0
            for _ in range(steps):
                buffer = 1 + benchmark.LOCAL_MOMENTUM * buffer
                total += buffer
            return total

        def fake_train(trainer, *args, simulated_step_duration_sec, **kwargs):
            i = int(trainer.client_id.removeprefix("client_"))
            step_count = client_steps[i]
            return {
                "client_id": trainer.client_id,
                "delta_i": [torch.tensor([[effective_steps(step_count)]], dtype=torch.float64)],
                "q_i": 1 / simulated_step_duration_sec,
                "steps_completed": step_count,
                "tau_tr": step_count * simulated_step_duration_sec,
                "gated_early": False,
            }

        with (
            patch.object(benchmark, "get_model", return_value=model),
            patch.object(benchmark, "get_model_size_mb", return_value=payload_mib),
            patch.object(benchmark, "evaluate_accuracy", return_value=(50.0, 0.7)),
            patch.object(benchmark.CapDriftClientTrainer, "train_round", fake_train),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            trial = benchmark.run_competitor_trial(
                "FedNova-Momentum", dataset, None, client_indices, tiers,
                device=torch.device("cpu"), num_rounds=1,
            )

        effective = sum(
            len(ids) / len(dataset) * effective_steps(steps)
            for ids, steps in zip(client_indices, client_steps)
        )
        self.assertAlmostEqual(float(model.weight.detach()), effective, places=10)
        self.assertAlmostEqual(trial["total_time_sec"] * 25, 205.6197213692418)
        self.assertLessEqual(trial["max_client_compute_time_sec"], 6.5)
        utilization = 100 * sum(
            steps * benchmark.BATCH_SIZE / tier.base_samples_per_sec
            + benchmark.network_duration(tier, payload_mib)
            for steps, tier in zip(client_steps, tiers)
        ) / (len(tiers) * trial["total_time_sec"])
        self.assertAlmostEqual(trial["mean_utilization_pct"], utilization, places=10)


class TestTunedBenchmark(unittest.TestCase):
    def test_evaluation_shadow_keeps_raw_training_and_full_utilization(self):
        dataset = TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))
        model = nn.Linear(1, 1, bias=False).double()
        nn.init.zeros_(model.weight)
        payload_mib = 5.797981262207031
        cfg = benchmark.CapDriftConfig(
            S_min=5, S_max=40, T_target_sec=6.5, drift_mode="projection",
            enable_clipping=False, server_average_decay=0.75, server_average_start_round=22,
        )
        tiers = [benchmark.TIER_PROFILES[i] for i in (0, 1, 2, 4)]

        def fake_train(trainer, local_model, global_model_weights, delta_G_prev,
                       train_loader, criterion, optimizer, device,
                       simulated_step_duration_sec, server_round, total_rounds):
            self.assertAlmostEqual(float(local_model.weight.detach()), server_round - 1)
            self.assertEqual(total_rounds, 25)
            steps, _ = trainer.determine_step_budget(
                server_round, total_rounds, simulated_step_duration_sec
            )
            return {
                "client_id": trainer.client_id,
                "delta_i": [torch.ones((1, 1), dtype=torch.float64)],
                "q_i": 1 / simulated_step_duration_sec,
                "steps_completed": steps,
                "tau_tr": steps * simulated_step_duration_sec,
                "gated_early": False,
            }

        def fake_evaluate(scoring_model, *_):
            return 50.0, float(scoring_model.weight.detach())

        with (
            patch.object(benchmark, "get_model", return_value=model),
            patch.object(benchmark, "get_model_size_mb", return_value=payload_mib),
            patch.object(benchmark, "evaluate_accuracy", fake_evaluate),
            patch.object(benchmark.CapDriftClientTrainer, "train_round", fake_train),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            trial = benchmark.run_competitor_trial(
                "CapDrift-Tuned", dataset, None, [[0], [1], [2], [3]], tiers,
                config=cfg, device=torch.device("cpu"), num_rounds=25,
            )
        for row in trial["history"]:
            self.assertEqual(list(row["steps_per_client"].values()), [40, 28, 16, 5])
            self.assertAlmostEqual(row["round_duration_sec"], trial["total_time_sec"] / 25)
        final_round = trial["history"][-1]
        self.assertAlmostEqual(final_round["raw_val_loss"], 25.0)
        self.assertAlmostEqual(trial["final_loss"], 23.265625)
        self.assertAlmostEqual(float(model.weight.detach()), 25.0)
        self.assertAlmostEqual(trial["total_time_sec"], 205.6197213692418)
        self.assertLessEqual(trial["max_client_compute_time_sec"], 6.5)


if __name__ == "__main__":
    unittest.main()
