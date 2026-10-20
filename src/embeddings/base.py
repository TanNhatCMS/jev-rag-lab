"""Protocol Embedder: interface chung cho mọi cách sinh embedding."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Bất kỳ embedder nào cũng phải có số chiều `dim` và hàm `embed`."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả ma trận shape (len(texts), dim), mỗi hàng đã chuẩn hóa L2."""
        ...
