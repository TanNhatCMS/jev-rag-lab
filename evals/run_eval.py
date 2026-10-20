#!/usr/bin/env python3
"""Eval harness (ER-004/005): chạy các pipeline trên golden set, ghi kết quả + báo cáo.

Chạy từ gốc repo:
    python3 evals/run_eval.py [--golden ...] [--corpus ...] [--config ...]
                              [--out-dir ...] [--shadow]

Mỗi câu hỏi cho ra một dòng JSON trong evals/results/<run_id>.jsonl cho mỗi
pipeline chạy được, và một báo cáo Markdown trong evals/reports/<run_id>.md
nhúng toàn bộ config (ER-003). Pipeline B (Jev thật) thiếu JEV_API_KEY thì
ghi SKIPPED vào báo cáo — tuyệt đối không fake kết quả (ER-008).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

_EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _EVALS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_EVALS_DIR))

import metrics  # noqa: E402
from src.decision.gate import GateConfig, apply_gate  # noqa: E402
from src.decision.scorer import HeuristicScorer, JevScorer  # noqa: E402
from src.llm.mock import MockLLM  # noqa: E402
from src.pipelines.baseline import BaselinePipeline  # noqa: E402
from src.pipelines.common import build_retriever, load_config  # noqa: E402
from src.pipelines.gated import GatedPipeline  # noqa: E402

ABSTAIN_GROUPS = ("no_answer", "distractor_no_evidence", "injection")
STAGES = ("retrieval", "scoring", "generation", "total")


def _normalize(text: str) -> str:
    """Chuẩn hoá để so khớp đáp án: chữ thường + gộp khoảng trắng."""
    return " ".join(text.lower().split())


def load_golden(path: Path) -> list[dict]:
    """Đọc golden set JSONL thành list dict."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pipelines(retriever, config: dict) -> list[dict]:
    """Dựng các pipeline khả dụng; pipeline không dựng được ghi status skipped.

    Trả list dict: {"label", "pipeline" | None, "status", "note"}.
    """
    gate_config = GateConfig.from_config(config)
    entries: list[dict] = [
        {
            "label": "A-baseline",
            "pipeline": BaselinePipeline.from_config(retriever, MockLLM(), config),
            "status": "ran",
            "note": "BaselinePipeline + MockLLM",
        },
        {
            "label": "C-heuristic",
            "pipeline": GatedPipeline(
                retriever,
                MockLLM(),
                HeuristicScorer(),
                gate_config,
                name="C-heuristic",
            ),
            "status": "ran",
            "note": "GatedPipeline + HeuristicScorer (quy tắc offline) + MockLLM",
        },
    ]
    try:
        jev_scorer = JevScorer.from_env()
    except RuntimeError:
        entries.append(
            {
                "label": "B-jev",
                "pipeline": None,
                "status": "skipped",
                "note": "SKIPPED — thiếu JEV_API_KEY (Q2)",
            }
        )
    else:
        entries.append(
            {
                "label": "B-jev",
                "pipeline": GatedPipeline(
                    retriever, MockLLM(), jev_scorer, gate_config, name="B-jev"
                ),
                "status": "ran",
                "note": "GatedPipeline + JevScorer (API thật) + MockLLM",
            }
        )
    return entries


def run_question(entry: dict, question: dict) -> dict | None:
    """Chạy một pipeline cho một câu hỏi, trả dòng kết quả (None nếu pipeline lỗi)."""
    pipeline = entry["pipeline"]
    if pipeline is None:
        return None
    result = pipeline.run(question["question"])
    expected = question["expected"]
    expect_abstain = expected == "abstain"
    evidence = set(question.get("evidence_doc_ids") or [])
    used_doc_ids = [c.chunk.doc_id for c in result.used]
    if expect_abstain:
        abstain_correct = result.abstained
    else:
        abstain_correct = not result.abstained
    timings = {k: round(float(v), 3) for k, v in result.timings_ms.items()}
    return {
        "id": question["id"],
        "group": question["group"],
        "expected": expected,
        "pipeline": entry["label"],
        "abstained": result.abstained,
        "answer": result.answer,
        "used_doc_ids": used_doc_ids,
        "rejected_reasons": [r.reason for r in result.rejected],
        "injection_excluded": any(r.reason == "injection" for r in result.rejected),
        "timings_ms": timings,
        "abstain_correct": abstain_correct,
        "evidence_hit": (not expect_abstain)
        and any(doc_id in evidence for doc_id in used_doc_ids),
        "answer_match": (not expect_abstain)
        and result.answer is not None
        and _normalize(expected) in _normalize(result.answer),
    }


def shadow_for_question(retriever, scorer, gate_config, question: str) -> dict:
    """ER-005: chấm + gate trên kết quả retrieval của pipeline A, chỉ ghi — không lọc."""
    candidates = retriever.retrieve(question)
    scores = scorer.score(question, candidates)
    gate = apply_gate(candidates, scores, gate_config)
    return {
        "shadow_would_abstain": gate.abstain,
        "shadow_passed_doc_ids": [c.chunk.doc_id for c, _ in gate.passed],
    }


def _fmt_rate(rate: float | None, rows: list[dict], group: str, key: str) -> str:
    """Định dạng tỉ lệ kèm mẫu số, ví dụ '80.0% (8/10)'."""
    if rate is None:
        return "—"
    subset = [r for r in rows if r.get("group") == group]
    hits = sum(1 for r in subset if r.get(key))
    return f"{rate * 100:.1f}% ({hits}/{len(subset)})"


def _latency_table(rows: list[dict]) -> list[str]:
    """Bảng độ trễ TB/p50/p95 theo giai đoạn cho một pipeline."""
    lines = [
        "| Giai đoạn | TB (ms) | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|",
    ]
    for stage in STAGES:
        if stage == "total":
            values = [sum(r["timings_ms"].values()) for r in rows]
        else:
            values = [r["timings_ms"].get(stage, 0.0) for r in rows]
        lines.append(
            f"| {stage} | {metrics.mean(values):.2f} | "
            f"{metrics.percentile(values, 50):.2f} | {metrics.percentile(values, 95):.2f} |"
        )
    return lines


def write_report(
    path: Path,
    run_id: str,
    args: argparse.Namespace,
    config: dict,
    golden_rows: list[dict],
    entries: list[dict],
    all_rows: list[dict],
    retrieval_stats: dict,
    shadow: bool,
) -> None:
    """Sinh báo cáo Markdown: snapshot config, trạng thái pipeline, bảng chỉ số,
    giới hạn trung thực (ER-008) và kết luận thuần mô tả số."""
    groups_present = sorted({q["group"] for q in golden_rows})
    lines: list[str] = [
        f"# Báo cáo eval — {run_id}",
        "",
        f"- Thời điểm chạy: {datetime.now().isoformat(timespec='seconds')}",
        f"- Golden set: `{args.golden}` ({len(golden_rows)} câu; nhóm: "
        + ", ".join(f"{g}={metrics.group_count(golden_rows, g)}" for g in groups_present)
        + ")",
        f"- Corpus: `{args.corpus}`",
        f"- Shadow mode (ER-005): {'bật' if shadow else 'tắt'}",
        "",
        "## Snapshot config (ER-003)",
        "",
        "```yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip(),
        "```",
        "",
        "## Trạng thái pipeline",
        "",
        "| Pipeline | Trạng thái | Ghi chú |",
        "|---|---|---|",
    ]
    for entry in entries:
        lines.append(f"| {entry['label']} | {entry['status']} | {entry['note']} |")
    lines += [
        "",
        "## Chỉ số tầng retrieval (first-stage, chung cho mọi pipeline)",
        "",
        f"- Precision@5 (TB trên {retrieval_stats['n']} câu has_answer): "
        f"{retrieval_stats['precision@5']:.4f}",
        f"- Recall@20 (TB trên {retrieval_stats['n']} câu has_answer): "
        f"{retrieval_stats['recall@20']:.4f}",
        "",
        "## Chỉ số theo pipeline",
        "",
        "| Pipeline | Precision@5 (retrieval) | Recall@20 (retrieval) | "
        "Abstain đúng no_answer | Abstain đúng distractor | Abstain đúng injection | "
        "Abstain sai (has_answer) | Answer match (has_answer) | Evidence hit (has_answer) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        rows = [r for r in all_rows if r["pipeline"] == entry["label"]]
        if not rows:
            lines.append(
                f"| {entry['label']} | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            "| {label} | {p5:.4f} | {r20:.4f} | {na} | {di} | {ij} | {fa} | {am} | {eh} |".format(
                label=entry["label"],
                p5=retrieval_stats["precision@5"],
                r20=retrieval_stats["recall@20"],
                na=_fmt_rate(
                    metrics.abstain_correct_rate(rows, "no_answer"),
                    rows, "no_answer", "abstain_correct",
                ),
                di=_fmt_rate(
                    metrics.abstain_correct_rate(rows, "distractor_no_evidence"),
                    rows, "distractor_no_evidence", "abstain_correct",
                ),
                ij=_fmt_rate(
                    metrics.abstain_correct_rate(rows, "injection"),
                    rows, "injection", "abstain_correct",
                ),
                fa=_fmt_rate(
                    metrics.false_abstain_rate(rows), rows, "has_answer", "abstained"
                ),
                am=_fmt_rate(
                    metrics.true_rate(
                        [r for r in rows if r["group"] == "has_answer"], "answer_match"
                    ),
                    rows, "has_answer", "answer_match",
                ),
                eh=_fmt_rate(
                    metrics.true_rate(
                        [r for r in rows if r["group"] == "has_answer"], "evidence_hit"
                    ),
                    rows, "has_answer", "evidence_hit",
                ),
            )
        )
    lines += ["", "## Độ trễ theo pipeline", ""]
    for entry in entries:
        rows = [r for r in all_rows if r["pipeline"] == entry["label"]]
        lines.append(f"### {entry['label']}")
        lines.append("")
        if rows:
            lines += _latency_table(rows)
        else:
            lines.append("(không chạy — xem mục Trạng thái pipeline)")
        lines.append("")
    if shadow:
        shadow_rows = [r for r in all_rows if "shadow_would_abstain" in r]
        seen: dict[str, dict] = {}
        for r in shadow_rows:
            seen.setdefault(r["id"], r)
        unique = list(seen.values())
        would = sum(1 for r in unique if r["shadow_would_abstain"])
        lines += [
            "## Shadow mode (ER-005)",
            "",
            f"- Scorer dùng cho shadow: HeuristicScorer (offline) + gate theo config.",
            f"- Số câu shadow sẽ abstain nếu áp gate lên retrieval của A: "
            f"{would}/{len(unique)}.",
            "- Phân phối chi tiết theo từng câu nằm ở trường `shadow_would_abstain` /",
            "  `shadow_passed_doc_ids` trong file kết quả JSONL — dùng để hiệu chuẩn",
            "  ngưỡng sau này, KHÔNG phải kết quả của một pipeline thật.",
            "",
        ]
    lines += [
        "## Giới hạn của vòng chạy này (ER-008 — đọc trước khi dùng số)",
        "",
        "- **Embedder là HashingEmbedder**: vector băm túi từ (bag-of-words), không có",
        "  ngữ nghĩa — similarity chỉ phản ánh độ chồng lấp từ vựng giữa câu hỏi và đoạn.",
        "- **LLM là MockLLM**: không hiểu câu hỏi; nó trích nguyên văn câu đầu tiên của",
        "  đoạn ngữ cảnh đầu tiên và thêm hậu tố `[mock-llm]`. Chỉ số `answer_match` vì",
        "  thế chỉ kiểm tra plumbing trích xuất, KHÔNG đo chất lượng trả lời của model thật.",
        "- **Scorer của C-heuristic là HeuristicScorer**: chấm bằng quy tắc (độ phủ token",
        "  của query, khớp thực thể, mẫu câu injection); tín hiệu `contradiction` luôn 0.0",
        "  vì quy tắc offline không phán được mâu thuẫn. Đây không phải model quyết định.",
        "- **Pipeline B (Jev API thật) chưa chạy** — thiếu JEV_API_KEY (câu hỏi mở Q2).",
        "- **Dữ liệu tổng hợp và nhỏ**: corpus + golden set tự tạo (50 câu VI / 10 câu EN",
        "  ở vòng này); mọi tỉ lệ đều có mẫu số nhỏ, không suy rộng ra dữ liệu thật.",
        "- **Độ trễ đo offline trên máy local** với các thành phần giả/quy tắc — không",
        "  đại diện cho độ trễ của API thật (Jev/LLM/embedding qua mạng).",
        "",
        "## Kết luận (mô tả số, chưa kết luận GO/NO-GO)",
        "",
        "Vòng chạy này dùng toàn thành phần offline (HashingEmbedder, HeuristicScorer,",
        "MockLLM) nên **chưa đủ điều kiện kết luận GO/NO-GO** theo tiêu chí của spec 001.",
        "Giá trị của vòng này: xác nhận harness chạy đầu-cuối, ghi được số plumbing ban",
        "đầu và dữ liệu shadow để hiệu chuẩn ngưỡng. Quan sát mô tả từ bảng trên:",
        "",
    ]
    for entry in entries:
        rows = [r for r in all_rows if r["pipeline"] == entry["label"]]
        if not rows:
            lines.append(f"- {entry['label']}: không có số liệu ({entry['note']}).")
            continue
        abstain_groups = [g for g in ABSTAIN_GROUPS if metrics.group_count(rows, g)]
        desc = ", ".join(
            f"{g} {_fmt_rate(metrics.abstain_correct_rate(rows, g), rows, g, 'abstain_correct')}"
            for g in abstain_groups
        )
        lines.append(
            f"- {entry['label']}: abstain đúng theo nhóm — {desc}; "
            f"abstain sai {_fmt_rate(metrics.false_abstain_rate(rows), rows, 'has_answer', 'abstained')}; "
            f"answer_match {_fmt_rate(metrics.true_rate([r for r in rows if r['group'] == 'has_answer'], 'answer_match'), rows, 'has_answer', 'answer_match')}."
        )
    lines += [
        "",
        "*Mọi con số trên chỉ mô tả hành vi của các thành phần offline trên dữ liệu",
        "tổng hợp — cấm diễn giải thành kết luận về chất lượng của Jev hay model thật.*",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy eval RAG (jev-rag-lab)")
    parser.add_argument("--golden", default=str(_EVALS_DIR / "data" / "vi_golden.jsonl"))
    parser.add_argument("--corpus", default=str(_EVALS_DIR / "data" / "vi_corpus.jsonl"))
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_EVALS_DIR))
    parser.add_argument("--shadow", action="store_true", help="Bật shadow mode (ER-005)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    golden_rows = load_golden(Path(args.golden))
    retriever = build_retriever(args.corpus, config)
    entries = build_pipelines(retriever, config)

    # Chỉ số tầng retrieval trên các câu has_answer (có evidence để đối chiếu).
    p5_values, r20_values = [], []
    per_question_candidates: dict[str, list] = {}
    for q in golden_rows:
        candidates = retriever.retrieve(q["question"])
        per_question_candidates[q["id"]] = candidates
        evidence = q.get("evidence_doc_ids") or []
        if evidence:
            ranked = [c.chunk.doc_id for c in candidates]
            p5_values.append(metrics.precision_at_k(ranked, evidence, 5))
            r20_values.append(metrics.recall_at_k(ranked, evidence, 20))
    retrieval_stats = {
        "precision@5": metrics.mean(p5_values),
        "recall@20": metrics.mean(r20_values),
        "n": len(p5_values),
    }

    gate_config = GateConfig.from_config(config)
    shadow_scorer = HeuristicScorer()
    all_rows: list[dict] = []
    for q in golden_rows:
        shadow_fields = None
        if args.shadow:
            shadow_fields = shadow_for_question(
                retriever, shadow_scorer, gate_config, q["question"]
            )
        for entry in entries:
            row = run_question(entry, q)
            if row is None:
                continue
            if shadow_fields:
                row.update(shadow_fields)
            all_rows.append(row)

    dataset = Path(args.golden).stem.split("_")[0]
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M')}-{dataset}"
    out_dir = Path(args.out_dir)
    results_path = out_dir / "results" / f"{run_id}.jsonl"
    report_path = out_dir / "reports" / f"{run_id}.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(
        report_path, run_id, args, config, golden_rows, entries,
        all_rows, retrieval_stats, args.shadow,
    )
    print(f"run_id: {run_id}")
    print(f"results: {results_path} ({len(all_rows)} dòng)")
    print(f"report:  {report_path}")
    for entry in entries:
        print(f"  {entry['label']}: {entry['status']} — {entry['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
