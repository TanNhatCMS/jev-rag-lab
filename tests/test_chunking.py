"""Test chunking: kích thước tối đa, overlap giữa các chunk liên tiếp, vị trí khớp text gốc."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import chunk_document, chunk_text
from src.schema import Document


class TestChunkText(unittest.TestCase):
    def test_kich_thuoc_khong_vuot_size(self):
        text = "abcdefghij" * 100  # 1000 ký tự
        parts = chunk_text(text, size=200, overlap=50)
        self.assertTrue(parts)
        for _, _, piece in parts:
            self.assertLessEqual(len(piece), 200)

    def test_vi_tri_khop_text_goc(self):
        text = "".join(chr(ord("a") + (i % 26)) for i in range(500))
        for start, end, piece in chunk_text(text, size=120, overlap=30):
            self.assertEqual(text[start:end], piece)
            self.assertEqual(end - start, len(piece))

    def test_overlap_giua_cac_chunk_lien_tiep(self):
        text = "x" * 1000
        parts = chunk_text(text, size=200, overlap=50)
        for (s1, e1, _), (s2, _, _) in zip(parts, parts[1:]):
            # chunk sau bắt đầu sớm hơn điểm kết thúc của chunk trước đúng `overlap` ký tự
            self.assertEqual(e1 - s2, 50)
            self.assertEqual(s2 - s1, 150)

    def test_text_ngan_hon_size_chi_mot_chunk(self):
        parts = chunk_text("xin chào", size=800, overlap=100)
        self.assertEqual(parts, [(0, 8, "xin chào")])

    def test_text_rong_tra_list_rong(self):
        self.assertEqual(chunk_text("", size=100, overlap=10), [])

    def test_overlap_lon_hon_size_bi_loi(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=10, overlap=10)


class TestChunkDocument(unittest.TestCase):
    def test_chunk_document_gan_id_va_vi_tri(self):
        doc = Document(id="d1", title="T", text="a" * 300)
        chunks = chunk_document(doc, size=100, overlap=20)
        self.assertEqual([c.id for c in chunks], [f"d1#{i}" for i in range(len(chunks))])
        self.assertTrue(all(c.doc_id == "d1" for c in chunks))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        for c in chunks:
            self.assertEqual(doc.text[c.start : c.end], c.text)


if __name__ == "__main__":
    unittest.main()
