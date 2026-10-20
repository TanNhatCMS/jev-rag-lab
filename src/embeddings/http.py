"""HttpEmbedder: gọi endpoint embedding tương thích OpenAI qua urllib (stdlib).

LƯU Ý: chưa test được trong sandbox hiện tại (không có mạng, không có API key).
Chỉ chạy được khi đã điền EMBEDDING_API_BASE / EMBEDDING_API_KEY / EMBEDDING_MODEL trong .env.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np


class HttpEmbedder:
    """POST {base}/embeddings với {"model", "input": [...]} kiểu OpenAI."""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        # Chưa biết dim cho tới lần gọi đầu tiên nếu không truyền vào từ đầu.
        if self._dim is None:
            raise RuntimeError("Chưa rõ dim: truyền dim=... hoặc gọi embed() một lần trước")
        return self._dim

    @classmethod
    def from_env(cls, dim: int | None = None) -> "HttpEmbedder":
        """Tạo embedder từ biến môi trường EMBEDDING_API_BASE / EMBEDDING_API_KEY / EMBEDDING_MODEL."""
        base = os.environ.get("EMBEDDING_API_BASE", "")
        key = os.environ.get("EMBEDDING_API_KEY", "")
        model = os.environ.get("EMBEDDING_MODEL", "")
        if not (base and key and model):
            raise RuntimeError("Thiếu EMBEDDING_API_BASE / EMBEDDING_API_KEY / EMBEDDING_MODEL trong môi trường")
        return cls(base_url=base, api_key=key, model=model, dim=dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float64)
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        rows = sorted(body["data"], key=lambda r: r["index"])
        matrix = np.array([r["embedding"] for r in rows], dtype=np.float64)
        self._dim = matrix.shape[1]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
