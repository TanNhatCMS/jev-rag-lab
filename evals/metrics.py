"""Chỉ số eval thuần (không I/O, không phụ thuộc src/) — tính tay kiểm chứng được.

Định nghĩa dùng trong toàn bộ harness (xem thêm evals/README.md):

- precision@k: trong top-k đầu của danh sách doc_id đã xếp hạng, tỉ lệ vị trí
  thuộc tập evidence_doc_ids. Mẫu số = min(k, len(ranked)); danh sách rỗng -> 0.
- recall@k (theo spec 001, dạng hit-rate): 1.0 nếu có ÍT NHẤT 1 evidence doc
  trong top-k, ngược lại 0.0. Tập evidence rỗng -> 0.0.
- abstain_correct của một dòng kết quả: True khi pipeline abstain đúng như
  `expected` yêu cầu (expected == "abstain" thì phải abstain; ngược lại thì
  không được abstain). Runner ghi sẵn cờ này vào từng dòng kết quả.
- abstain sai (false abstain): dòng thuộc nhóm has_answer nhưng pipeline abstain.
- percentile: nội suy tuyến tính (cùng quy ước mặc định của numpy.percentile).
"""

from __future__ import annotations


def precision_at_k(
    ranked_doc_ids: list[str], evidence_doc_ids: list[str] | set[str], k: int
) -> float:
    """Tỉ lệ vị trí trong top-k thuộc tập evidence."""
    if k <= 0 or not ranked_doc_ids:
        return 0.0
    evidence = set(evidence_doc_ids)
    top = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in evidence)
    return hits / len(top)


def recall_at_k(
    ranked_doc_ids: list[str], evidence_doc_ids: list[str] | set[str], k: int
) -> float:
    """1.0 nếu có ít nhất 1 evidence doc trong top-k, else 0.0."""
    if k <= 0 or not evidence_doc_ids:
        return 0.0
    evidence = set(evidence_doc_ids)
    return 1.0 if any(doc_id in evidence for doc_id in ranked_doc_ids[:k]) else 0.0


def mean(values: list[float]) -> float:
    """Trung bình cộng; danh sách rỗng -> 0.0."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: list[float], p: float) -> float:
    """Phân vị p (0..100) bằng nội suy tuyến tính; danh sách rỗng -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def true_rate(rows: list[dict], key: str) -> float | None:
    """Tỉ lệ dòng có row[key] là True; None nếu không có dòng nào."""
    if not rows:
        return None
    return sum(1 for row in rows if row.get(key)) / len(rows)


def abstain_correct_rate(rows: list[dict], group: str) -> float | None:
    """Tỉ lệ abstain đúng trong một nhóm; None nếu nhóm không có dòng nào."""
    group_rows = [row for row in rows if row.get("group") == group]
    return true_rate(group_rows, "abstain_correct")


def false_abstain_rate(rows: list[dict]) -> float | None:
    """Tỉ lệ abstain sai: nhóm has_answer nhưng pipeline lại abstain."""
    answer_rows = [row for row in rows if row.get("group") == "has_answer"]
    if not answer_rows:
        return None
    return sum(1 for row in answer_rows if row.get("abstained")) / len(answer_rows)


def group_count(rows: list[dict], group: str) -> int:
    """Số dòng thuộc một nhóm (để báo cáo ghi kèm mẫu số)."""
    return sum(1 for row in rows if row.get("group") == group)
