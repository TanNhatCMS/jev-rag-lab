"""HashingEmbedder: embedding 'túi từ' bằng hashing trick — local, deterministic, zero-install.

Chỉ dùng làm embedder mặc định để chạy plumbing/eval khi chưa có model embedding thật;
KHÔNG phản ánh chất lượng embedding ngữ nghĩa (đó là lý do có HttpEmbedder và extra
sentence-transformers trong pyproject.toml).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Tách từ đơn giản: thường hóa + lấy dãy ký tự chữ/số (giữ được tiếng Việt có dấu)."""
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Mỗi token được băm tới 1 chiều của vector với dấu +/-; tổng lại rồi chuẩn hóa L2."""

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim phải > 0")
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vec[value % self.dim] += 1.0 if (value >> 63) == 0 else -1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self._embed_one(t) for t in texts])
