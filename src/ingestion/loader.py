"""Loader: nạp Document từ thư mục .txt/.md hoặc từ file corpus JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from src.schema import Document


def load_jsonl_corpus(path: str | Path) -> list[Document]:
    """Đọc corpus JSONL: mỗi dòng là một object {"id", "title", "text"}."""
    path = Path(path)
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "title", "text"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: thiếu field '{key}'")
            docs.append(
                Document(
                    id=str(row["id"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    source=str(path),
                    metadata={k: v for k, v in row.items() if k not in ("id", "title", "text")},
                )
            )
    return docs


def load_text_dir(path: str | Path) -> list[Document]:
    """Đọc mọi file .txt/.md trong một thư mục; id = tên file không có đuôi, title = dòng đầu tiên."""
    path = Path(path)
    docs: list[Document] = []
    for file in sorted(path.iterdir()):
        if file.suffix.lower() not in (".txt", ".md") or not file.is_file():
            continue
        text = file.read_text(encoding="utf-8").strip()
        first_line = text.splitlines()[0].lstrip("# ").strip() if text else file.stem
        docs.append(
            Document(id=file.stem, title=first_line, text=text, source=str(file))
        )
    return docs


def load_documents(path: str | Path) -> list[Document]:
    """Điểm vào chung: thư mục -> load_text_dir, file .jsonl -> load_jsonl_corpus."""
    path = Path(path)
    if path.is_dir():
        return load_text_dir(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_corpus(path)
    raise ValueError(f"Không hỗ trợ nguồn dữ liệu: {path}")
