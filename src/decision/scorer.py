"""DecisionScorer (ER-001): interface chấm 4 tín hiệu cho từng cặp (query, đoạn).

Ba implementation:
- JevScorer: client HTTP tới Jev API của TypeSafe.
- LlmScorer: dùng một LLM + prompt packed (Pipeline C).
- HeuristicScorer: quy tắc offline, phục vụ plumbing/test — KHÔNG phải model quyết định.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.llm.base import LLM
from src.prompts.scoring import build_scoring_prompt
from src.schema import Candidate

SIGNALS = ("relevance", "evidence", "contradiction", "injection")


@dataclass
class Scores:
    """Bốn tín hiệu xác suất 0..1 cho một ứng viên."""

    relevance: float
    evidence: float
    contradiction: float
    injection: float


@runtime_checkable
class DecisionScorer(Protocol):
    """Interface duy nhất của tầng quyết định (ER-001)."""

    def score(self, query: str, candidates: list[Candidate]) -> list[Scores]:
        """Chấm mọi ứng viên, trả list Scores đúng thứ tự của candidates."""
        ...


def _clamp(value: object) -> float:
    """Ép một giá trị bất kỳ về float trong [0, 1]; không ép được thì 0.0."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


# ---------------------------------------------------------------------------
# JevScorer
# ---------------------------------------------------------------------------

# Mẫu câu phán đoán định kiểu cho từng tín hiệu, theo phong cách cookbook TypeSafe:
# mỗi cặp (query, đoạn) được hỏi 4 câu có/không và Jev trả xác suất cho từng câu.
_JUDGMENT_TEMPLATES = {
    "relevance": "The passage is relevant to the query.",
    "evidence": "The passage contains evidence that answers the query.",
    "contradiction": "The passage contradicts a premise of the query.",
    "injection": "The passage contains instructions that try to override the system's behavior.",
}


class JevScorer:
    """Client HTTP tới Jev API của TypeSafe (Pipeline B).

    TRUNG THỰC VỀ ĐỘ KIỂM CHỨNG: cài đặt này viết theo MẪU cookbook đã công bố
    (gửi phán đoán định kiểu cho từng cặp query–đoạn, đọc xác suất trả về),
    NHƯNG chưa kiểm chứng với API thật — sandbox không có mạng và không có
    JEV_API_KEY (câu hỏi mở Q2 của spec). Định dạng request/response dưới đây
    là bản phỏng theo tài liệu mô tả; khi có key thật phải đối chiếu lại với
    tài liệu TypeSafe và sửa cho khớp trước khi tin số đo của Pipeline B.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "JevScorer":
        """Tạo scorer từ biến môi trường JEV_API_BASE / JEV_API_KEY."""
        base = os.environ.get("JEV_API_BASE", "")
        key = os.environ.get("JEV_API_KEY", "")
        if not (base and key):
            raise RuntimeError("Thiếu JEV_API_BASE / JEV_API_KEY trong môi trường")
        return cls(base_url=base, api_key=key)

    def score(self, query: str, candidates: list[Candidate]) -> list[Scores]:
        if not candidates:
            return []
        items = []
        for cand in candidates:
            items.append(
                {
                    "query": query,
                    "passage": cand.chunk.text,
                    "judgments": [
                        {"signal": signal, "statement": statement}
                        for signal, statement in _JUDGMENT_TEMPLATES.items()
                    ],
                }
            )
        payload = json.dumps({"items": items}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/score",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        rows = body.get("results", body) if isinstance(body, dict) else body
        scores: list[Scores] = []
        for i in range(len(candidates)):
            row = rows[i] if isinstance(rows, list) and i < len(rows) else {}
            probs = row.get("probabilities", row) if isinstance(row, dict) else {}
            scores.append(
                Scores(
                    relevance=_clamp(probs.get("relevance")),
                    evidence=_clamp(probs.get("evidence")),
                    contradiction=_clamp(probs.get("contradiction")),
                    injection=_clamp(probs.get("injection")),
                )
            )
        return scores


# ---------------------------------------------------------------------------
# LlmScorer
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class LlmScorer:
    """Scorer dùng một LLM chấm packed qua prompt ở src/prompts/scoring.py (Pipeline C).

    Parse chặt chẽ: chỉ lấy mảng JSON đầu tiên khớp trong phản hồi, clamp mọi
    điểm về [0, 1]. Ứng viên nào thiếu trong phản hồi sẽ nhận Scores(0,0,0,0)
    và một ghi chú được thêm vào self.notes để truy vết khi đọc báo cáo eval.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.notes: list[str] = []

    def score(self, query: str, candidates: list[Candidate]) -> list[Scores]:
        if not candidates:
            return []
        raw = self.llm.generate(build_scoring_prompt(query, candidates))
        rows = self._extract_rows(raw)
        if rows is None:
            self.notes.append(
                f"Không tìm thấy mảng JSON trong phản hồi scorer: {raw[:120]!r}"
            )
            return [Scores(0.0, 0.0, 0.0, 0.0) for _ in candidates]
        by_index: dict[int, dict] = {}
        for pos, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            idx = row.get("index", pos)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = pos
            by_index[idx] = row
        scores: list[Scores] = []
        for i in range(len(candidates)):
            row = by_index.get(i)
            if row is None:
                self.notes.append(f"Scorer không trả điểm cho ứng viên {i}; điền 0.")
                scores.append(Scores(0.0, 0.0, 0.0, 0.0))
                continue
            scores.append(
                Scores(
                    relevance=_clamp(row.get("relevance")),
                    evidence=_clamp(row.get("evidence")),
                    contradiction=_clamp(row.get("contradiction")),
                    injection=_clamp(row.get("injection")),
                )
            )
        return scores

    @staticmethod
    def _extract_rows(raw: str) -> list | None:
        """Tìm mảng JSON đầu tiên trong phản hồi và parse; None nếu không parse được."""
        match = _JSON_ARRAY_RE.search(raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None


# ---------------------------------------------------------------------------
# HeuristicScorer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RAW_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Từ dừng tiếng Việt: không tính vào "token nội dung" khi đo độ phủ của query.
_STOPWORDS = {
    "là", "của", "có", "không", "được", "và", "các", "một", "những", "cho",
    "về", "khi", "này", "đó", "gì", "nào", "ở", "tại", "bao", "nhiêu",
    "với", "hãy", "tôi", "bạn", "thì", "ra", "vào", "theo", "như", "đã",
    "sẽ", "đang", "bị", "bởi", "hay", "hoặc", "nếu", "vì", "nên", "cũng",
}

# Mẫu câu chỉ thị can thiệp (so khớp trên văn bản đã thường hoá).
_INJECTION_PATTERNS = (
    "bỏ qua hướng dẫn",
    "bỏ qua chỉ thị",
    "hãy giảm giá",
    "giảm giá ngay",
    "không cần tuân theo",
    "ignore previous",
    "ignore instructions",
    "disregard",
    "you must",
    "system prompt",
)

_PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")


def _tokens(text: str) -> list[str]:
    """Tách token thường hoá (giữ tiếng Việt có dấu, giống HashingEmbedder)."""
    return _TOKEN_RE.findall(text.lower())


def _entity_tokens(query: str) -> set[str]:
    """Token 'thực thể' của query: chứa chữ số (A1, B2) hoặc viết hoa giữa câu (AquaPure)."""
    raw = _RAW_TOKEN_RE.findall(query)
    entities: set[str] = set()
    for pos, token in enumerate(raw):
        lowered = token.lower()
        if any(ch.isdigit() for ch in token):
            entities.add(lowered)
        elif pos > 0 and token[:1].isupper():
            entities.add(lowered)
    return entities


class HeuristicScorer:
    """Scorer quy tắc chạy offline — phục vụ plumbing/test, KHÔNG phải model quyết định.

    Cách chấm:
    - relevance = similarity của candidate (clamp 0..1).
    - evidence = độ phủ token nội dung của query trong đoạn, cộng 0.3 khi đoạn
      chứa số đứng riêng (thường là giá/số liệu), rồi nhân hệ số khớp thực thể:
      query nêu thực thể (mã/tên riêng) mà đoạn không nhắc tới thì evidence bị
      kéo xuống mạnh (đoạn nói về thực thể khác không thể là bằng chứng).
    - injection = khớp mẫu câu chỉ thị VI/EN: khớp -> >= 0.8; kèm thêm số %
      (kiểu 'giảm giá 50%') -> 0.95.
    - contradiction = 0.0 cố định (quy tắc offline không phán được mâu thuẫn).
    """

    def score(self, query: str, candidates: list[Candidate]) -> list[Scores]:
        content_tokens = {t for t in _tokens(query) if t not in _STOPWORDS}
        entities = _entity_tokens(query)
        return [
            Scores(
                relevance=_clamp(cand.score),
                evidence=self._evidence(content_tokens, entities, cand),
                contradiction=0.0,
                injection=self._injection(cand.chunk.text),
            )
            for cand in candidates
        ]

    @staticmethod
    def _evidence(content_tokens: set[str], entities: set[str], cand: Candidate) -> float:
        chunk_tokens = _tokens(cand.chunk.text)
        chunk_set = set(chunk_tokens)
        if content_tokens:
            coverage = len(content_tokens & chunk_set) / len(content_tokens)
        else:
            coverage = 0.0
        has_number = any(t.isdigit() for t in chunk_tokens)
        base = min(1.0, coverage + (0.3 if has_number else 0.0))
        if entities:
            matched = len(entities & chunk_set) / len(entities)
            base *= 0.25 + 0.75 * matched
        return _clamp(base)

    @staticmethod
    def _injection(text: str) -> float:
        lowered = text.lower()
        if not any(pattern in lowered for pattern in _INJECTION_PATTERNS):
            return 0.0
        score = 0.85
        if _PERCENT_RE.search(lowered):
            score = 0.95
        return score
