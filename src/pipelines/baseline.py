"""Pipeline A — Baseline RAG thường: vector search -> lọc similarity -> LLM.

Không có tầng quyết định: ứng viên có similarity >= threshold được lấy top_k
theo đúng thứ tự retrieval đưa vào prompt. Abstain khi không ứng viên nào qua
threshold, hoặc khi LLM trả về đúng chuỗi "ABSTAIN".
"""

from __future__ import annotations

import time

from src.decision.gate import Rejection
from src.llm.base import LLM
from src.pipelines.common import PipelineResult, elapsed_ms, empty_timings
from src.prompts.generation import ABSTAIN_TOKEN, build_generation_prompt
from src.retrieval.retriever import Retriever


class BaselinePipeline:
    """RAG thường làm đối chứng (Pipeline A trong spec 001)."""

    name = "A_baseline"

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        threshold: float = 0.35,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k

    @classmethod
    def from_config(cls, retriever: Retriever, llm: LLM, config: dict) -> "BaselinePipeline":
        """Dựng từ dict config: threshold và top_k lấy từ mục `retrieval`."""
        retr = config.get("retrieval") or {}
        return cls(
            retriever=retriever,
            llm=llm,
            threshold=float(retr.get("similarity_threshold", 0.35)),
            top_k=int(retr.get("top_k", 5)),
        )

    def run(self, query: str) -> PipelineResult:
        timings = empty_timings()

        start = time.perf_counter()
        candidates = self.retriever.retrieve(query)
        timings["retrieval"] = elapsed_ms(start)

        kept = [c for c in candidates if c.score >= self.threshold][: self.top_k]
        rejected = [
            Rejection(c, None, "similarity")
            for c in candidates
            if c.score < self.threshold
        ]
        if not kept:
            return PipelineResult(
                answer=None,
                abstained=True,
                used=[],
                rejected=rejected,
                conflicts=[],
                scores=[],
                timings_ms=timings,
                scorer_name="none",
            )

        prompt = build_generation_prompt(
            query, [(c.chunk.doc_id, c.chunk.text) for c in kept]
        )
        start = time.perf_counter()
        raw = self.llm.generate(prompt)
        timings["generation"] = elapsed_ms(start)

        answer = raw.strip()
        if answer == ABSTAIN_TOKEN:
            return PipelineResult(
                answer=None,
                abstained=True,
                used=kept,
                rejected=rejected,
                conflicts=[],
                scores=[],
                timings_ms=timings,
                scorer_name="none",
            )
        return PipelineResult(
            answer=answer,
            abstained=False,
            used=kept,
            rejected=rejected,
            conflicts=[],
            scores=[],
            timings_ms=timings,
            scorer_name="none",
        )
