"""Smoke test cho evals/sweep_thresholds.py: fixture mini tự dựng trong thư mục tạm.

Chạy script bằng subprocess (đúng cách dùng CLI thật) với lưới rút gọn qua
--relevance-grid/--evidence-grid, kiểm tra: có file results JSONL + report MD,
số dòng JSONL bằng số tổ hợp trong lưới, mỗi dòng đủ trường chỉ số, và stdout
có công bố tổ hợp tốt nhất.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CORPUS = [
    {
        "id": "doc-a",
        "title": "Alpha catalog",
        "text": "The Alpha A1 printer costs 100 dollars and prints 20 pages per minute.",
    },
    {
        "id": "doc-b",
        "title": "Beta catalog",
        "text": "The Beta B2 scanner scans 30 pages per minute and costs 150 dollars.",
    },
    {
        "id": "doc-c",
        "title": "Printer buying guide",
        "text": (
            "Printers are office devices used to print documents. This guide "
            "compares printer types in general and lists no prices."
        ),
    },
]

GOLDEN = [
    {
        "id": "t-001",
        "question": "How much does the Alpha A1 printer cost?",
        "group": "has_answer",
        "expected": "100 dollars",
        "evidence_doc_ids": ["doc-a"],
    },
    {
        "id": "t-002",
        "question": "How much does the Beta B2 scanner cost?",
        "group": "has_answer",
        "expected": "150 dollars",
        "evidence_doc_ids": ["doc-b"],
    },
    {
        "id": "t-003",
        "question": "How much does the Gamma G9 printer cost?",
        "group": "no_answer",
        "expected": "abstain",
        "evidence_doc_ids": [],
    },
    {
        "id": "t-004",
        "question": "What is the price of the Alpha A1 printer in the buying guide?",
        "group": "distractor_no_evidence",
        "expected": "abstain",
        "evidence_doc_ids": [],
    },
]

RELEVANCE_GRID = "0.30,0.45,0.60"
EVIDENCE_GRID = "0.35,0.55,0.70"
N_COMBOS = 3 * 3

REQUIRED_KEYS = {
    "relevance", "evidence", "injection", "contradiction", "top_k",
    "abstain_correct", "abstain_correct_count", "abstain_group_total",
    "false_abstain", "false_abstain_count", "has_answer_total",
    "balanced_accuracy", "rank",
}


class TestSweepSmoke(unittest.TestCase):
    def test_sweep_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "corpus.jsonl"
            golden_path = tmp_path / "golden.jsonl"
            corpus_path.write_text(
                "".join(json.dumps(r) + "\n" for r in CORPUS), encoding="utf-8"
            )
            golden_path.write_text(
                "".join(json.dumps(r) + "\n" for r in GOLDEN), encoding="utf-8"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "evals" / "sweep_thresholds.py"),
                    "--golden", str(golden_path),
                    "--corpus", str(corpus_path),
                    "--out-dir", str(tmp_path / "out"),
                    "--relevance-grid", RELEVANCE_GRID,
                    "--evidence-grid", EVIDENCE_GRID,
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

            results_files = list((tmp_path / "out" / "results").glob("sweep-*.jsonl"))
            report_files = list((tmp_path / "out" / "reports").glob("sweep-*.md"))
            self.assertEqual(len(results_files), 1)
            self.assertEqual(len(report_files), 1)

            rows = [
                json.loads(line)
                for line in results_files[0].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), N_COMBOS)
            for row in rows:
                self.assertTrue(REQUIRED_KEYS.issubset(row.keys()))
                # Fixture: 2 câu nhóm abstain, 2 câu has_answer.
                self.assertEqual(row["abstain_group_total"], 2)
                self.assertEqual(row["has_answer_total"], 2)

            # Tổ hợp tốt nhất: hạng 1, có balanced accuracy cao nhất trong lưới.
            best = next(r for r in rows if r["rank"] == 1)
            self.assertEqual(
                best["balanced_accuracy"],
                max(r["balanced_accuracy"] for r in rows),
            )
            # ... và được in ra stdout cho người chạy thấy ngay.
            self.assertIn("best:", proc.stdout)
            self.assertIn(f"{best['balanced_accuracy']:.4f}", proc.stdout)

            report = report_files[0].read_text(encoding="utf-8")
            self.assertIn("CẢNH BÁO OVERFITTING", report)
            self.assertIn("Top 10 tổ hợp", report)


if __name__ == "__main__":
    unittest.main()
