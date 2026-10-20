"""Phần dùng chung cho các pipeline: PipelineResult và hàm dựng Retriever từ corpus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.chunking.chunker import chunk_document
from src.decision.gate import Rejection
from src.decision.scorer import Scores
from src.embeddings.base import Embedder
from src.embeddings.hashing import HashingEmbedder
from src.embeddings.http import HttpEmbedder
from src.ingestion.loader import load_documents
from src.retrieval.retriever import Retriever
from src.schema import Candidate
from src.vectorDB.numpy_store import NumpyStore


@dataclass
class PipelineResult:
    """Kết quả một lượt hỏi của một pipeline, đủ dữ kiện cho eval harness so sánh."""

    answer: str | None
    abstained: bool
    used: list[Candidate]  # các đoạn thực tế đưa vào prompt generation
    rejected: list[Rejection]  # ứng viên bị loại, kèm lý do
    conflicts: list[tuple[Candidate, Scores]]  # ứng viên mâu thuẫn, tách riêng
    scores: list[Scores]  # điểm của mọi ứng viên đã retrieve ([] nếu không có scorer)
    timings_ms: dict[str, float]  # {"retrieval", "scoring", "generation"} tính bằng ms
    scorer_name: str


def empty_timings() -> dict[str, float]:
    """Khung timings mặc định, mọi giai đoạn 0 ms."""
    return {"retrieval": 0.0, "scoring": 0.0, "generation": 0.0}


def elapsed_ms(start: float) -> float:
    """Số ms trôi qua kể từ mốc time.perf_counter() truyền vào."""
    return (time.perf_counter() - start) * 1000.0


def load_config(config: dict | str | Path) -> dict:
    """Nhận dict config sẵn, hoặc đường dẫn tới config.yaml thì đọc ra dict."""
    if isinstance(config, dict):
        return config
    data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    return data


def build_embedder(config: dict) -> Embedder:
    """Dựng embedder theo mục `embedding` của config (hashing local hoặc http)."""
    emb_cfg = config.get("embedding") or {}
    dim = int(emb_cfg.get("dim", 384))
    provider = str(emb_cfg.get("provider", "hashing"))
    if provider == "hashing":
        return HashingEmbedder(dim=dim)
    if provider == "http":
        return HttpEmbedder.from_env(dim=dim)
    raise ValueError(f"embedding.provider không hỗ trợ: {provider}")


def build_retriever(corpus_path: str | Path, config: dict | str | Path) -> Retriever:
    """Nạp corpus -> chunk -> embed -> NumpyStore -> Retriever (đúng tham số trong config)."""
    cfg = load_config(config)
    chunk_cfg = cfg.get("chunking") or {}
    size = int(chunk_cfg.get("size", 800))
    overlap = int(chunk_cfg.get("overlap", 100))
    top_n = int((cfg.get("retrieval") or {}).get("top_n", 20))

    docs = load_documents(corpus_path)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, size=size, overlap=overlap))

    embedder = build_embedder(cfg)
    store = NumpyStore(dim=embedder.dim)
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.add(chunks, vectors)
    return Retriever(embedder=embedder, store=store, top_n=top_n)
