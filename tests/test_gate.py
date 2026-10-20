"""Test gate: ngưỡng lọc, thứ tự kiểm tra, sắp xếp passed và cờ abstain."""

from __future__ import annotations

import unittest

from src.decision.gate import GateConfig, apply_gate
from src.decision.scorer import Scores
from src.schema import Candidate, Chunk


def _candidate(doc_id: str, sim: float = 0.9) -> Candidate:
    chunk = Chunk(id=f"{doc_id}#0", doc_id=doc_id, text=f"nội dung {doc_id}", start=0, end=10, index=0)
    return Candidate(chunk=chunk, score=sim)


CFG = GateConfig(relevance=0.45, evidence=0.55, injection=0.70, contradiction=0.70, top_k=5)


class TestGate(unittest.TestCase):
    def test_evidence_low_rejected(self):
        cand = _candidate("a")
        result = apply_gate([cand], [Scores(0.9, 0.30, 0.0, 0.0)], CFG)
        self.assertEqual(result.passed, [])
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].reason, "evidence")
        self.assertTrue(result.abstain)

    def test_injection_high_rejected_even_with_high_evidence(self):
        cand = _candidate("a")
        result = apply_gate([cand], [Scores(0.95, 0.95, 0.0, 0.90)], CFG)
        self.assertEqual(result.passed, [])
        self.assertEqual(result.rejected[0].reason, "injection")

    def test_injection_checked_before_contradiction(self):
        cand = _candidate("a")
        result = apply_gate([cand], [Scores(0.9, 0.9, 0.95, 0.95)], CFG)
        self.assertEqual(result.rejected[0].reason, "injection")
        self.assertEqual(result.conflicts, [])

    def test_contradiction_goes_to_conflicts(self):
        cand = _candidate("a")
        result = apply_gate([cand], [Scores(0.9, 0.9, 0.85, 0.0)], CFG)
        self.assertEqual(result.passed, [])
        self.assertEqual(result.rejected, [])
        self.assertEqual(len(result.conflicts), 1)
        self.assertIs(result.conflicts[0][0], cand)
        self.assertTrue(result.abstain)

    def test_relevance_low_rejected(self):
        cand = _candidate("a")
        result = apply_gate([cand], [Scores(0.40, 0.90, 0.0, 0.0)], CFG)
        self.assertEqual(result.rejected[0].reason, "relevance")

    def test_all_rejected_means_abstain(self):
        cands = [_candidate("a"), _candidate("b")]
        scores = [Scores(0.1, 0.1, 0.0, 0.0), Scores(0.9, 0.2, 0.0, 0.0)]
        result = apply_gate(cands, scores, CFG)
        self.assertEqual(result.passed, [])
        self.assertTrue(result.abstain)

    def test_passed_sorted_by_evidence_then_relevance(self):
        cands = [_candidate("low"), _candidate("high"), _candidate("mid")]
        scores = [
            Scores(0.90, 0.60, 0.0, 0.0),
            Scores(0.80, 0.95, 0.0, 0.0),
            Scores(0.99, 0.60, 0.0, 0.0),
        ]
        result = apply_gate(cands, scores, CFG)
        order = [c.chunk.doc_id for c, _ in result.passed]
        # evidence 0.95 trước; hai đoạn evidence 0.60 thì relevance cao hơn (0.99) trước.
        self.assertEqual(order, ["high", "mid", "low"])
        self.assertFalse(result.abstain)

    def test_top_k_cut(self):
        cfg = GateConfig(relevance=0.45, evidence=0.55, injection=0.70, contradiction=0.70, top_k=2)
        cands = [_candidate("a"), _candidate("b"), _candidate("c")]
        scores = [Scores(0.9, 0.9, 0.0, 0.0)] * 3
        result = apply_gate(cands, scores, cfg)
        self.assertEqual(len(result.passed), 2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            apply_gate([_candidate("a")], [], CFG)


if __name__ == "__main__":
    unittest.main()
