"""Protocol VectorStore: interface chung cho kho vector (NumpyStore, sau này FAISS...)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.schema import Candidate, Chunk


@runtime_checkable
class VectorStore(Protocol):
    """Kho vector tối thiểu cho first-stage retrieval."""

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Thêm các chunk kèm vector embedding tương ứng (theo cùng thứ tự)."""
        ...

    def search(self, query_vector: np.ndarray, top_n: int) -> list[Candidate]:
        """Trả tối đa top_n ứng viên có độ tương đồng cao nhất với query_vector."""
        ...

    def save(self, path: str) -> None:
        """Lưu toàn bộ kho xuống đĩa."""
        ...

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Nạp lại kho đã lưu bằng save()."""
        ...
