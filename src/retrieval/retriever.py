"""Retriever: embed câu hỏi -> tìm top_n ứng viên trong VectorStore."""

from __future__ import annotations

from src.embeddings.base import Embedder
from src.schema import Candidate
from src.vectorDB.base import VectorStore


class Retriever:
    """First-stage retrieval thuần vector (chưa qua tầng quyết định/gate)."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_n: int = 20):
        self.embedder = embedder
        self.store = store
        self.top_n = top_n

    def retrieve(self, query: str, top_n: int | None = None) -> list[Candidate]:
        """Trả danh sách ứng viên xếp theo similarity giảm dần."""
        n = self.top_n if top_n is None else top_n
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, n)
