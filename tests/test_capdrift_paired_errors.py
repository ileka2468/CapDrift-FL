"""CPU checks for paired predictions and disjoint threshold assessment."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_capdrift_paired_errors as paired
import run_capdrift_paired_replay as replay


def rows(margins, labels):
    return [{"id": i, "label": y, "margin": m}
            for i, (m, y) in enumerate(zip(margins, labels, strict=True))]


class TestPairedErrors(unittest.TestCase):
    def test_exact_argmax_tie_favors_class_zero(self):
        self.assertEqual(paired.score(rows([0., 0.01], [0, 1])), 2)

    def test_exclusive_error_confidence_and_opposite_disagreements(self):
        cap = rows([.08, -3., 1., -1.], [0, 1, 1, 1])
        nova = rows([-.1, 3., -1., -1.], [0, 1, 1, 1])
        result = paired.compare(cap, nova)
        self.assertEqual((result["rival_only_correct"], result["cap_only_correct"]), (2, 1))
        self.assertEqual(result["rival_only_near"], 1)
        self.assertEqual(result["rival_only_confident_ge_80pct"], 1)
        self.assertEqual(result["rival_only_by_label"], [1, 1])
        with self.assertRaises(ValueError):
            paired.compare(cap, nova[::-1])

    def test_threshold_does_not_select_on_assessment(self):
        # Half A wants to favor 0; half B wants to favor 1.
        inputs = rows([.06, -.06], [0, 1])
        a, b = paired.threshold_study(inputs, [[0], [1]])
        self.assertEqual((a["threshold"], b["threshold"]), (.1, -.1))
        self.assertEqual((a["adjusted_hits"], b["adjusted_hits"]), (0, 0))
        self.assertEqual(paired.choose_threshold(rows([.4, -.4], [1, 0])), 0.)
        with self.assertRaises(ValueError):
            paired.threshold_study(inputs, [[0], [0, 1]])

    def test_replay_threshold_changes_decisions_not_cross_entropy(self):
        inputs = rows([120., -.05, -120.], [1, 1, 0])
        a, b = replay.measure(inputs, 0.), replay.measure(inputs, -.1)
        self.assertEqual((a["hits"], b["hits"]), (2, 3))
        self.assertEqual(b["balanced_accuracy"], 100)
        self.assertEqual(a["ce"], b["ce"])
        self.assertTrue(math.isfinite(a["ce"]))


if __name__ == "__main__":
    unittest.main()
