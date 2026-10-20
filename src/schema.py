"""Kiểu dữ liệu dùng chung cho toàn pipeline: Document -> Chunk -> Candidate."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Một tài liệu nguồn nguyên vẹn trước khi chunk."""

    id: str
    title: str
    text: str
    source: str = ""  # đường dẫn file hoặc nguồn gốc, để truy vết
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Một đoạn văn cắt ra từ Document; start/end là vị trí ký tự trong text gốc."""

    id: str
    doc_id: str
    text: str
    start: int
    end: int
    index: int  # thứ tự của chunk trong tài liệu (0-based)


@dataclass
class Candidate:
    """Một chunk ứng viên do retrieval trả về, kèm điểm similarity."""

    chunk: Chunk
    score: float
    rank: int = 0  # hạng trong danh sách ứng viên (0-based), gán khi sắp xếp
