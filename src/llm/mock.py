"""MockLLM: LLM giả deterministic phục vụ test và kiểm tra plumbing.

KHÔNG phải model thật: nó không hiểu câu hỏi. Nó chỉ trích nguyên văn câu đầu
tiên của đoạn ngữ cảnh đầu tiên tìm thấy trong prompt (định dạng do
src/prompts/generation.py dựng) và thêm hậu tố "[mock-llm]" để mọi đầu ra
đều truy vết được là dữ liệu giả. Không thấy ngữ cảnh thì trả "ABSTAIN".
"""

from __future__ import annotations

import re

# Dòng mở đầu một đoạn ngữ cảnh trong prompt generation: "[1] <nhãn>"
_BLOCK_START = re.compile(r"^\[\d+\]\s")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _first_context_sentence(prompt: str) -> str | None:
    """Tìm đoạn ngữ cảnh đầu tiên trong prompt và lấy câu đầu tiên của nó."""
    lines = prompt.splitlines()
    collecting: list[str] = []
    in_block = False
    for line in lines:
        if _BLOCK_START.match(line):
            if in_block:
                break  # đã sang đoạn thứ hai
            in_block = True
            continue
        if in_block:
            if line.startswith("CÂU HỎI"):
                break
            if line.strip():
                collecting.append(line.strip())
    if not collecting:
        return None
    text = " ".join(collecting)
    return _SENTENCE_SPLIT.split(text, maxsplit=1)[0].strip()


class MockLLM:
    """LLM giả: trả câu đầu tiên của đoạn ngữ cảnh đầu tiên + hậu tố [mock-llm]."""

    def generate(self, prompt: str) -> str:
        sentence = _first_context_sentence(prompt)
        if sentence is None:
            return "ABSTAIN"
        return f"{sentence} [mock-llm]"
