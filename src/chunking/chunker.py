"""Chunker theo ký tự với overlap cố định (vòng 1: đơn giản để biến thí nghiệm nằm ở tầng gate)."""

from __future__ import annotations

from src.schema import Chunk, Document


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[tuple[int, int, str]]:
    """Cắt text thành các đoạn dài tối đa `size` ký tự, gối nhau `overlap` ký tự.

    Trả về list (start, end, đoạn) với text[start:end] == đoạn.
    """
    if size <= 0:
        raise ValueError("size phải > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap phải thỏa 0 <= overlap < size")
    if not text:
        return []
    step = size - overlap
    parts: list[tuple[int, int, str]] = []
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        parts.append((start, end, text[start:end]))
        if end == len(text):
            break
    return parts


def chunk_document(doc: Document, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Cắt một Document thành các Chunk có id dạng '<doc_id>#<thứ tự>'."""
    return [
        Chunk(
            id=f"{doc.id}#{i}",
            doc_id=doc.id,
            text=piece,
            start=start,
            end=end,
            index=i,
        )
        for i, (start, end, piece) in enumerate(chunk_text(doc.text, size, overlap))
    ]
