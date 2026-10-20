"""Tải HotpotQA (bản distractor, split validation) và lấy mẫu 200 câu cho eval tiếng Anh.

Cách tải: Hugging Face datasets-server rows API
(https://huggingface.co/api/datasets/rows), phân trang 100 dòng/lần.
Từ mẫu 200 câu (seed cố định 42) ghi ra 2 file cạnh script này:
- hotpotqa_subset.jsonl: {"id", "question", "answer", "type", "level", "evidence_titles"}
- hotpotqa_corpus.jsonl: {"id", "title", "text"} — mỗi đoạn context là một tài liệu,
  id là slug của tiêu đề bài viết.

LƯU Ý: script CẦN INTERNET. Sandbox hiện tại không có mạng nên KHÔNG chạy ở đây;
chạy trên máy có mạng bằng: python3 evals/data/build_hotpotqa.py
Chỉ dùng thư viện chuẩn (urllib + json + random).
"""

from __future__ import annotations

import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_API = "https://huggingface.co/api/datasets/rows"
DATASET = "hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"
PAGE_SIZE = 100
POOL_SIZE = 1000  # tải 1.000 dòng đầu làm pool, rồi lấy mẫu 200 — đủ tái lập và rẻ băng thông
SAMPLE_SIZE = 200
SEED = 42

OUT_DIR = Path(__file__).resolve().parent


def fetch_page(offset: int, length: int) -> list[dict]:
    """Tải một trang rows từ datasets-server, trả list các trường của từng dòng."""
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_API}?{params}", timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return [entry["row"] for entry in body["rows"]]


def slugify(title: str, seen: dict[str, int]) -> str:
    """Biến tiêu đề bài viết thành id tài liệu duy nhất, dạng slug ASCII."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "doc"
    seen[slug] = seen.get(slug, 0) + 1
    return slug if seen[slug] == 1 else f"{slug}-{seen[slug]}"


def main() -> None:
    pool: list[dict] = []
    offset = 0
    while len(pool) < POOL_SIZE:
        page = fetch_page(offset, min(PAGE_SIZE, POOL_SIZE - len(pool)))
        if not page:
            break
        pool.extend(page)
        offset += len(page)
    print(f"Đã tải pool {len(pool)} dòng từ HotpotQA {CONFIG}/{SPLIT}")

    sample = random.Random(SEED).sample(pool, k=min(SAMPLE_SIZE, len(pool)))

    seen: dict[str, int] = {}
    corpus: dict[str, dict] = {}
    subset_path = OUT_DIR / "hotpotqa_subset.jsonl"
    corpus_path = OUT_DIR / "hotpotqa_corpus.jsonl"
    with subset_path.open("w", encoding="utf-8") as f_sub:
        for row in sample:
            titles = row["context"]["title"]
            sentences = row["context"]["sentences"]
            for title, sents in zip(titles, sentences):
                if title not in corpus:
                    corpus[title] = {
                        "id": slugify(title, seen),
                        "title": title,
                        "text": " ".join(s.strip() for s in sents),
                    }
            f_sub.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "type": row["type"],
                        "level": row["level"],
                        "evidence_titles": list(row["supporting_facts"]["title"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with corpus_path.open("w", encoding="utf-8") as f_cor:
        for doc in corpus.values():
            f_cor.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Đã ghi {len(sample)} câu -> {subset_path}")
    print(f"Đã ghi {len(corpus)} tài liệu context -> {corpus_path}")


if __name__ == "__main__":
    main()
