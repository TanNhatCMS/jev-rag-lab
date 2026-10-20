"""NumpyStore: vector store trong RAM bằng numpy — đủ cho corpus nhỏ của phòng lab.

Lưu/nạp: file .npz chứa ma trận vector + file .json chứa metadata của chunk đi kèm.
Vector được chuẩn hóa L2 khi thêm vào, nên tích vô hướng chính là cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.schema import Candidate, Chunk


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class NumpyStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._matrix = np.zeros((0, dim), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"vectors phải có shape ({len(chunks)}, {self.dim}), nhận {vectors.shape}")
        vectors = _normalize(np.asarray(vectors, dtype=np.float64))
        self._matrix = np.vstack([self._matrix, vectors]) if len(self.chunks) else vectors
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_n: int) -> list[Candidate]:
        if not self.chunks or top_n <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float64).ravel()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self._matrix @ q
        order = np.argsort(-scores)[:top_n]
        return [
            Candidate(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order)
        ]

    def save(self, path: str) -> None:
        base = Path(path)
        np.savez(base.with_suffix(".npz"), matrix=self._matrix)
        meta = [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "text": c.text,
                "start": c.start,
                "end": c.end,
                "index": c.index,
            }
            for c in self.chunks
        ]
        base.with_suffix(".json").write_text(
            json.dumps({"dim": self.dim, "chunks": meta}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "NumpyStore":
        base = Path(path)
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]))
        with np.load(base.with_suffix(".npz")) as data:
            store._matrix = data["matrix"]
        store.chunks = [Chunk(**row) for row in meta["chunks"]]
        if store._matrix.shape[0] != len(store.chunks):
            raise ValueError("File kho hỏng: số vector không khớp số chunk")
        return store
