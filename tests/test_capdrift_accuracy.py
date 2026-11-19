import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_capdrift_accuracy as study

bench = study.bench


class TestAccuracyProtocol(unittest.TestCase):
    def test_excludes_prior_groups_aliases_and_source_duplicates(self):
        old = [(Path("person12_bacteria_21.jpeg"), 1, "old-bytes")]
        source = [
            (Path("person12_virus_22.jpeg"), 1, "other-bytes"),
            (Path("renamed.jpeg"), 1, "old-bytes"),
            (Path("person50_virus_30.jpeg"), 1, "new-bytes"),
            (Path("person50_bacteria_32.jpeg"), 1, "new-bytes"),
            (Path("person50_bacteria_33.jpeg"), 1, "new-distinct-bytes"),
            (Path("NORMAL2-IM-0005-0001.jpeg"), 0, "normal-bytes"),
        ]
        rows, groups, _ = study.eligible_source_rows(old, source)
        self.assertEqual(rows, [source[i] for i in (2, 4, 5)])
        self.assertEqual(groups, 2)

    def test_rank_is_final_accuracy_first_then_loss(self):
        variants = {
            "with_better_accuracy": {"final_accuracy": 94.0, "final_loss": 0.4},
            "best_ce": {"final_accuracy": 91.0, "final_loss": 0.1},
            "same_acc_lower_ce": {"final_accuracy": 94.0, "final_loss": 0.3},
        }
        self.assertEqual(sorted(variants, key=lambda x: study.rank_key(x, variants[x])),
                         ["same_acc_lower_ce", "with_better_accuracy", "best_ce"])
        self.assertEqual(study.score([{"final_accuracy": 89.0, "peak_accuracy": 100,
                                      "final_loss": 0.35}])["final_accuracy"], 89)


class TestGradientMass(unittest.TestCase):
    @staticmethod
    def update(cid, delta):
        return {"client_id": cid, "q_i": 1.0,
                "delta_i": [torch.tensor([[delta]], dtype=torch.double)]}

    def test_clipping_bounds_actual_normalized_contribution(self):
        model = nn.Linear(1, 1, bias=False).double()
        nn.init.zeros_(model.weight)
        coordinator = bench.CapDriftCoordinator(bench.CapDriftConfig(kappa_c=1.25))
        result = coordinator.aggregate_round(
            1, model, [self.update("x", 10), self.update("y", 60), self.update("z", 100)],
            {"x": 2, "y": 1, "z": 1}, local_gradient_masses={"x": 2, "y": 3, "z": 10},
        )
        self.assertAlmostEqual(result["effective_gradient_mass"], 4.25)
        self.assertEqual([event["is_clipped"] for event in result["clipping_events"]],
                         [False, True, False])
        self.assertAlmostEqual(model.weight.item(), 34.53125)

    def test_zero_mass_is_excluded_and_tempered_weighting_is_incompatible(self):
        model = nn.Linear(1, 1, bias=False).double()
        nn.init.zeros_(model.weight)
        updates = [self.update("x", 3), self.update("y", 0)]
        counts, masses = {"x": 2, "y": 90}, {"x": 1.5, "y": 0.0}
        coordinator = bench.CapDriftCoordinator(bench.CapDriftConfig(enable_clipping=False))
        result = coordinator.aggregate_round(1, model, updates, counts, local_gradient_masses=masses)
        self.assertAlmostEqual(model.weight.item(), 3)
        self.assertEqual(len(result["clipping_events"]), 1)
        incompatible = bench.CapDriftCoordinator(
            bench.CapDriftConfig(client_update_normalization_power=0.5))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            incompatible.aggregate_round(1, model, updates, counts, local_gradient_masses=masses)

    def test_no_ablations_matches_fednova_and_balanced_metric_is_reported(self):
        cfg = bench.build_config(study.REFERENCE, server_momentum=0, server_lr=1,
                                 anneal_start_round=15, anneal_power=1)
        clients = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9]]
        tiers = [bench.TIER_PROFILES[t] for t in bench.TIER_IDS]
        features = torch.full((10, 1), 1.0, dtype=torch.double)
        data = TensorDataset(features, torch.tensor([0] * 7 + [1] * 3))
        models = {}

        def fake_train(trainer, *, simulated_step_duration_sec, **kwargs):
            rank = int(trainer.client_id[-1])
            count = [40, 28, 16, 5][rank]
            return {"client_id": trainer.client_id,
                    "delta_i": [torch.tensor([[rank + 0.23], [0.0]], dtype=torch.double)],
                    "q_i": 1 / simulated_step_duration_sec, "steps_completed": count,
                    "tau_tr": count * simulated_step_duration_sec, "gated_early": False}

        def run(name, normalized):
            model = nn.Linear(1, 2, bias=False).double()
            nn.init.zeros_(model.weight)
            models[name] = model
            with (
                patch.object(bench, "get_model", return_value=model),
                patch.object(bench, "get_model_size_mb", return_value=5.797981262207031),
                patch.object(bench.CapDriftClientTrainer, "train_round", fake_train),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return bench.run_competitor_trial(
                    name, data, DataLoader(data, batch_size=4), clients, tiers,
                    num_rounds=1, device=torch.device("cpu"),
                    config=cfg if normalized else None, normalize_momentum_updates=normalized,
                    measure_balanced_accuracy=True,
                )
        base = run(study.REFERENCE, False)
        candidate = run("CapDrift-Tuned", True)
        torch.testing.assert_close(models[study.REFERENCE].weight, models["CapDrift-Tuned"].weight,
                                   rtol=0, atol=0)
        self.assertEqual(base["final_accuracy"], candidate["final_accuracy"])
        self.assertEqual(candidate["final_accuracy"], 70.0)
        self.assertEqual(candidate["history"][-1]["balanced_accuracy"], 50.0)
        self.assertEqual(candidate["history"][-1]["class_correct_counts"], [7, 0])
        self.assertAlmostEqual(25 * candidate["total_time_sec"], 205.6197213692418)


if __name__ == "__main__":
    unittest.main()
