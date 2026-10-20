"""Test cho evals/metrics.py — mọi kỳ vọng đều tính tay được từ dữ liệu nhỏ."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import metrics  # noqa: E402


class TestPrecisionAtK(unittest.TestCase):
    def test_half_hits(self):
        # top-5 = a b c d e; evidence {b, d} -> 2/5
        self.assertAlmostEqual(
            metrics.precision_at_k(["a", "b", "c", "d", "e"], {"b", "d"}, 5), 0.4
        )

    def test_smaller_k(self):
        # top-3 = a b c; evidence {b, d} -> 1/3
        self.assertAlmostEqual(
            metrics.precision_at_k(["a", "b", "c", "d", "e"], {"b", "d"}, 3), 1 / 3
        )

    def test_ranked_shorter_than_k(self):
        # mẫu số là số vị trí thực có: 1/2
        self.assertAlmostEqual(metrics.precision_at_k(["a", "b"], {"a"}, 5), 0.5)

    def test_empty_and_zero_k(self):
        self.assertEqual(metrics.precision_at_k([], {"a"}, 5), 0.0)
        self.assertEqual(metrics.precision_at_k(["a"], {"a"}, 0), 0.0)


class TestRecallAtK(unittest.TestCase):
    def test_hit_inside_k(self):
        self.assertEqual(metrics.recall_at_k(["a", "b", "c"], {"c"}, 3), 1.0)

    def test_hit_outside_k(self):
        self.assertEqual(metrics.recall_at_k(["a", "b", "c"], {"c"}, 2), 0.0)

    def test_empty_evidence(self):
        self.assertEqual(metrics.recall_at_k(["a", "b"], [], 5), 0.0)


class TestPercentile(unittest.TestCase):
    def test_linear_interpolation(self):
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertAlmostEqual(metrics.percentile(values, 0), 0.0)
        self.assertAlmostEqual(metrics.percentile(values, 50), 15.0)
        self.assertAlmostEqual(metrics.percentile(values, 95), 28.5)
        self.assertAlmostEqual(metrics.percentile(values, 100), 30.0)

    def test_single_and_empty(self):
        self.assertEqual(metrics.percentile([7.0], 95), 7.0)
        self.assertEqual(metrics.percentile([], 95), 0.0)


class TestAbstainRates(unittest.TestCase):
    ROWS = [
        {"group": "has_answer", "abstained": False, "abstain_correct": True},
        {"group": "has_answer", "abstained": True, "abstain_correct": False},
        {"group": "no_answer", "abstained": True, "abstain_correct": True},
        {"group": "no_answer", "abstained": False, "abstain_correct": False},
        {"group": "no_answer", "abstained": True, "abstain_correct": True},
    ]

    def test_abstain_correct_rate_by_group(self):
        self.assertAlmostEqual(
            metrics.abstain_correct_rate(self.ROWS, "no_answer"), 2 / 3
        )
        self.assertAlmostEqual(
            metrics.abstain_correct_rate(self.ROWS, "has_answer"), 0.5
        )

    def test_missing_group_returns_none(self):
        self.assertIsNone(metrics.abstain_correct_rate(self.ROWS, "injection"))

    def test_false_abstain_rate(self):
        # 1 trong 2 câu has_answer bị abstain oan
        self.assertAlmostEqual(metrics.false_abstain_rate(self.ROWS), 0.5)

    def test_mean(self):
        self.assertAlmostEqual(metrics.mean([1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(metrics.mean([]), 0.0)


if __name__ == "__main__":
    unittest.main()
