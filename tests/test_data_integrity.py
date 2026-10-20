"""Test toàn vẹn bộ dữ liệu tiếng Việt: số câu, phân bố nhóm, evidence, quy tắc abstain."""

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[1] / "evals" / "data"


def read_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestViGoldenSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = read_jsonl("vi_corpus.jsonl")
        cls.golden = read_jsonl("vi_golden.jsonl")
        cls.corpus_ids = {d["id"] for d in cls.corpus}

    def test_corpus_co_it_nhat_12_tai_lieu(self):
        self.assertGreaterEqual(len(self.corpus), 12)
        self.assertEqual(len(self.corpus_ids), len(self.corpus))  # id không trùng

    def test_dung_50_cau_va_id_lien_tuc(self):
        self.assertEqual(len(self.golden), 50)
        self.assertEqual(
            [row["id"] for row in self.golden],
            [f"vi-{i:03d}" for i in range(1, 51)],
        )

    def test_phan_bo_nhom_20_10_12_8(self):
        counts = Counter(row["group"] for row in self.golden)
        self.assertEqual(
            counts,
            Counter(
                {
                    "has_answer": 20,
                    "no_answer": 10,
                    "distractor_no_evidence": 12,
                    "injection": 8,
                }
            ),
        )

    def test_evidence_doc_ids_ton_tai_trong_corpus(self):
        for row in self.golden:
            for doc_id in row["evidence_doc_ids"]:
                self.assertIn(doc_id, self.corpus_ids, f"{row['id']}: {doc_id} không có trong corpus")

    def test_quy_tac_abstain_theo_nhom(self):
        for row in self.golden:
            if row["group"] == "has_answer":
                self.assertNotEqual(row["expected"], "abstain", row["id"])
                self.assertTrue(row["evidence_doc_ids"], f"{row['id']}: has_answer phải có evidence")
            else:
                self.assertEqual(row["expected"], "abstain", row["id"])
                self.assertEqual(row["evidence_doc_ids"], [], row["id"])

    def test_expected_cua_has_answer_xuat_hien_trong_evidence(self):
        # Đáp án ngắn (bỏ tiền tố như "ông/bà/năm/từ") phải xuất hiện nguyên văn trong tài liệu bằng chứng.
        corpus_text = {d["id"]: d["text"] for d in self.corpus}
        for row in self.golden:
            if row["group"] != "has_answer":
                continue
            expected = row["expected"]
            joined = "\n".join(corpus_text[d] for d in row["evidence_doc_ids"])
            core = expected
            for prefix in ("Ông ", "Bà ", "Năm ", "Từ ngày ", "Từ "):
                if core.startswith(prefix):
                    core = core[len(prefix):]
                    break
            self.assertIn(core.rstrip("."), joined, f"{row['id']}: '{core}' không có trong evidence")


if __name__ == "__main__":
    unittest.main()
