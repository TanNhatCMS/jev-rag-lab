"""Pipeline có gate (B và C): retrieve top_n -> scorer chấm -> gate -> LLM.

Một class duy nhất cho cả hai pipeline đối chứng: B dùng JevScorer,
C dùng LlmScorer (hoặc HeuristicScorer khi chạy offline). Khác biệt duy nhất
giữa B và C là scorer truyền vào và `name` ghi vào kết quả (biến kiểm soát
của thí nghiệm — xem mục "Biến kiểm soát" trong spec 001).
"""

from __future__ import annotations

import time

from src.decision.gate import GateConfig, apply_gate
from src.decision.scorer import DecisionScorer
from src.llm.base import LLM
from src.pipelines.common import PipelineResult, elapsed_ms, empty_timings
from src.prompts.generation import ABSTAIN_TOKEN, build_generation_prompt
from src.retrieval.retriever import Retriever


class GatedPipeline:
    """RAG có tầng quyết định: abstain khi không ứng viên nào qua được gate."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        scorer: DecisionScorer,
        gate_config: GateConfig,
        name: str,
        top_n: int | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.scorer = scorer
        self.gate_config = gate_config
        self.name = name
        self.top_n = top_n

    def run(self, query: str) -> PipelineResult:
        timings = empty_timings()

        start = time.perf_counter()
        candidates = self.retriever.retrieve(query, top_n=self.top_n)
        timings["retrieval"] = elapsed_ms(start)

        start = time.perf_counter()
        scores = self.scorer.score(query, candidates)
        timings["scoring"] = elapsed_ms(start)

        gate = apply_gate(candidates, scores, self.gate_config)
        used = [cand for cand, _ in gate.passed]
        if gate.abstain:
            return PipelineResult(
                answer=None,
                abstained=True,
                used=[],
                rejected=gate.rejected,
                conflicts=gate.conflicts,
                scores=scores,
                timings_ms=timings,
                scorer_name=self.name,
            )

        prompt = build_generation_prompt(
            query, [(c.chunk.doc_id, c.chunk.text) for c in used]
        )
        start = time.perf_counter()
        raw = self.llm.generate(prompt)
        timings["generation"] = elapsed_ms(start)

        answer = raw.strip()
        if answer == ABSTAIN_TOKEN:
            return PipelineResult(
                answer=None,
                abstained=True,
                used=used,
                rejected=gate.rejected,
                conflicts=gate.conflicts,
                scores=scores,
                timings_ms=timings,
                scorer_name=self.name,
            )
        return PipelineResult(
            answer=answer,
            abstained=False,
            used=used,
            rejected=gate.rejected,
            conflicts=gate.conflicts,
            scores=scores,
            timings_ms=timings,
            scorer_name=self.name,
        )
