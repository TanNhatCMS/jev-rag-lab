"""Test HashingEmbedder + NumpyStore: index corpus nhỏ, search phải trả đúng tài liệu kỳ vọng ở top-1."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.chunking.chunker import chunk_document
from src.embeddings.hashing import HashingEmbedder
from src.retrieval.retriever import Retriever
from src.schema import Document
from src.vectorDB.numpy_store import NumpyStore

DOCS = [
    Document(
        id="doc-bao-hanh",
        title="Bảo hành",
        text="Chính sách bảo hành: máy in được bảo hành hai mươi bốn tháng, màn hình bảo hành ba mươi sáu tháng.",
    ),
    Document(
        id="doc-gia-ca",
        title="Bảng giá",
        text="Bảng giá sản phẩm: máy in giá ba triệu đồng, máy photocopy giá mười chín triệu đồng.",
    ),
    Document(
        id="doc-nghi-le",
        title="Nghỉ lễ",
        text="Lịch nghỉ lễ: công ty nghỉ Tết từ ngày mười bốn tháng hai đến ngày hai mươi hai tháng hai.",
    ),
]


def build_retriever() -> Retriever:
    embedder = HashingEmbedder(dim=384)
    store = NumpyStore(dim=embedder.dim)
    chunks = [c for d in DOCS for c in chunk_document(d, size=800, overlap=100)]
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return Retriever(embedder, store, top_n=3)


class TestHashingEmbedder(unittest.TestCase):
    def test_deterministic_va_chuan_hoa_l2(self):
        emb = HashingEmbedder(dim=128)
        a = emb.embed(["máy in bảo hành"])
        b = emb.embed(["máy in bảo hành"])
        self.assertTrue(np.allclose(a, b))
        self.assertAlmostEqual(float(np.linalg.norm(a[0])), 1.0, places=6)

    def test_shape_dung_dim(self):
        emb = HashingEmbedder(dim=64)
        self.assertEqual(emb.embed(["một", "hai ba"]).shape, (2, 64))


class TestNumpyStoreSearch(unittest.TestCase):
    def test_search_tra_dung_doc_o_top1(self):
        retriever = build_retriever()
        cases = [
            ("bảo hành máy in bao nhiêu tháng", "doc-bao-hanh"),
            ("giá máy photocopy bao nhiêu tiền", "doc-gia-ca"),
            ("nghỉ Tết từ ngày nào", "doc-nghi-le"),
        ]
        for query, expected_doc in cases:
            with self.subTest(query=query):
                results = retriever.retrieve(query)
                self.assertTrue(results)
                self.assertEqual(results[0].chunk.doc_id, expected_doc)
                self.assertEqual(results[0].rank, 0)

    def test_search_top_n_gioi_han_so_luong(self):
        retriever = build_retriever()
        self.assertLessEqual(len(retriever.retrieve("máy in", top_n=2)), 2)

    def test_save_load_khoi_phuc_dung(self):
        retriever = build_retriever()
        before = retriever.retrieve("bảo hành máy in")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "store")
            retriever.store.save(path)
            loaded = NumpyStore.load(path)
        after = Retriever(retriever.embedder, loaded, top_n=3).retrieve("bảo hành máy in")
        self.assertEqual(
            [c.chunk.id for c in before],
            [c.chunk.id for c in after],
        )
        self.assertAlmostEqual(before[0].score, after[0].score, places=9)


if __name__ == "__main__":
    unittest.main()
