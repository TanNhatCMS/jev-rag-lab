"""Protocol LLM: interface chung cho mọi client sinh văn bản (generation + scorer)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Bất kỳ LLM nào cũng chỉ cần một hàm generate(prompt) -> str."""

    def generate(self, prompt: str) -> str:
        """Sinh văn bản từ prompt, trả về chuỗi kết quả nguyên văn."""
        ...
