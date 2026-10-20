"""Gate: lọc ứng viên theo ngưỡng 4 tín hiệu (ER-003: ngưỡng nằm ở config.yaml).

Thứ tự kiểm tra cho mỗi ứng viên:
  1. injection > ngưỡng      -> loại, lý do "injection"
  2. contradiction > ngưỡng  -> tách vào conflicts (không qua, không tính loại)
  3. relevance < ngưỡng      -> loại, lý do "relevance"
  4. evidence < ngưỡng       -> loại, lý do "evidence"
Qua cả 4 thì vào passed, sắp theo evidence giảm dần rồi relevance giảm dần,
cắt ở top_k. passed rỗng -> abstain = True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.decision.scorer import Scores
from src.schema import Candidate


@dataclass
class GateConfig:
    """Ngưỡng gate + số đoạn tối đa qua cổng (top_k lấy từ mục retrieval của config)."""

    relevance: float = 0.45
    evidence: float = 0.55
    injection: float = 0.70
    contradiction: float = 0.70
    top_k: int = 5

    @classmethod
    def from_config(cls, config: dict) -> "GateConfig":
        """Nạp từ dict config đã đọc từ config.yaml (mục `gate` + `retrieval.top_k`)."""
        gate = config.get("gate") or {}
        retrieval = config.get("retrieval") or {}
        return cls(
            relevance=float(gate.get("relevance", 0.45)),
            evidence=float(gate.get("evidence", 0.55)),
            injection=float(gate.get("injection", 0.70)),
            contradiction=float(gate.get("contradiction", 0.70)),
            top_k=int(retrieval.get("top_k", 5)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GateConfig":
        """Đọc thẳng từ file config.yaml."""
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_config(config)


@dataclass
class Rejection:
    """Một ứng viên bị gate loại, kèm điểm và lý do để truy vết trong báo cáo."""

    candidate: Candidate
    scores: Scores | None
    reason: str  # "injection" | "relevance" | "evidence" | "similarity" (baseline)


@dataclass
class GateResult:
    """Kết quả qua cổng: passed đã xếp hạng, rejected kèm lý do, conflicts tách riêng."""

    passed: list[tuple[Candidate, Scores]] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    conflicts: list[tuple[Candidate, Scores]] = field(default_factory=list)
    abstain: bool = True


def apply_gate(
    candidates: list[Candidate], scores: list[Scores], cfg: GateConfig
) -> GateResult:
    """Áp ngưỡng gate lên danh sách ứng viên và điểm tương ứng (cùng thứ tự)."""
    if len(candidates) != len(scores):
        raise ValueError("candidates và scores phải cùng độ dài")
    result = GateResult()
    for cand, sc in zip(candidates, scores):
        if sc.injection > cfg.injection:
            result.rejected.append(Rejection(cand, sc, "injection"))
        elif sc.contradiction > cfg.contradiction:
            result.conflicts.append((cand, sc))
        elif sc.relevance < cfg.relevance:
            result.rejected.append(Rejection(cand, sc, "relevance"))
        elif sc.evidence < cfg.evidence:
            result.rejected.append(Rejection(cand, sc, "evidence"))
        else:
            result.passed.append((cand, sc))
    result.passed.sort(key=lambda pair: (-pair[1].evidence, -pair[1].relevance))
    result.passed = result.passed[: cfg.top_k]
    result.abstain = len(result.passed) == 0
    return result
