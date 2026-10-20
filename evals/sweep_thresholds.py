#!/usr/bin/env python3
"""Quét hiệu chuẩn ngưỡng gate (relevance x evidence) trên golden set.

Chạy từ gốc repo:
    python3 evals/sweep_thresholds.py [--golden ...] [--corpus ...]
        [--config ...] [--out-dir ...]
        [--relevance-grid 0.30,0.35,...] [--evidence-grid 0.35,0.40,...]

Cách làm: chạy retrieval + HeuristicScorer MỘT lần cho mỗi câu hỏi (điểm
không phụ thuộc ngưỡng), rồi áp apply_gate với từng tổ hợp ngưỡng lên cùng
điểm đó. injection/contradiction giữ theo config, top_k theo mục retrieval.

Chỉ số mỗi tổ hợp:
- abstain đúng: tỉ lệ câu thuộc nhóm phải-abstain (no_answer /
  distractor_no_evidence / injection) mà gate abstain.
- abstain sai: tỉ lệ câu has_answer mà gate lại abstain.
- balanced accuracy = (abstain đúng + (1 - abstain sai)) / 2.

CẢNH BÁO TRUNG THỰC (ER-008): sweep và "đánh giá" dùng CÙNG một golden set —
tổ hợp tốt nhất trên lưới này có nguy cơ overfitting vào đúng 50 câu đó.
Số ở đây chỉ là GỢI Ý BAN ĐẦU để chọn ngưỡng đem đi kiểm chứng trên dữ liệu
mới (validation), tuyệt đối không phải kết quả validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _EVALS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_EVALS_DIR))

from src.decision.gate import GateConfig, apply_gate  # noqa: E402
from src.decision.scorer import HeuristicScorer  # noqa: E402
from src.pipelines.common import build_retriever, load_config  # noqa: E402

ABSTAIN_GROUPS = ("no_answer", "distractor_no_evidence", "injection")
ANSWER_GROUP = "has_answer"

DEFAULT_RELEVANCE_GRID = [round(0.30 + 0.05 * i, 2) for i in range(7)]  # 0.30..0.60
DEFAULT_EVIDENCE_GRID = [round(0.35 + 0.05 * i, 2) for i in range(8)]  # 0.35..0.70


def load_golden(path: Path) -> list[dict]:
    """Đọc golden set JSONL thành list dict (giống run_eval.load_golden)."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_grid(text: str | None, default: list[float]) -> list[float]:
    """Parse lưới ngưỡng dạng '0.30,0.35,...' từ CLI; None thì dùng mặc định."""
    if not text:
        return list(default)
    values = sorted({round(float(part), 4) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("Lưới ngưỡng rỗng")
    return values


def sweep(
    scored: list[tuple[str, list, list]],
    relevance_grid: list[float],
    evidence_grid: list[float],
    base: GateConfig,
) -> list[dict]:
    """Áp mọi tổ hợp ngưỡng lên điểm đã chấm sẵn; trả list kết quả đã xếp hạng.

    `scored`: list (group, candidates, scores) cho từng câu hỏi.
    Xếp hạng: balanced accuracy giảm dần; hoà thì abstain sai thấp hơn thắng;
    vẫn hoà thì relevance rồi evidence thấp hơn đứng trước (ổn định, tái lập).
    """
    n_abstain = sum(1 for group, _, _ in scored if group in ABSTAIN_GROUPS)
    n_answer = sum(1 for group, _, _ in scored if group == ANSWER_GROUP)
    combos: list[dict] = []
    for relevance in relevance_grid:
        for evidence in evidence_grid:
            cfg = GateConfig(
                relevance=relevance,
                evidence=evidence,
                injection=base.injection,
                contradiction=base.contradiction,
                top_k=base.top_k,
            )
            correct = 0
            false_abstain = 0
            for group, candidates, scores in scored:
                abstained = apply_gate(candidates, scores, cfg).abstain
                if group in ABSTAIN_GROUPS and abstained:
                    correct += 1
                elif group == ANSWER_GROUP and abstained:
                    false_abstain += 1
            correct_rate = correct / n_abstain if n_abstain else 0.0
            false_rate = false_abstain / n_answer if n_answer else 0.0
            combos.append(
                {
                    "relevance": relevance,
                    "evidence": evidence,
                    "injection": base.injection,
                    "contradiction": base.contradiction,
                    "top_k": base.top_k,
                    "abstain_correct": round(correct_rate, 4),
                    "abstain_correct_count": correct,
                    "abstain_group_total": n_abstain,
                    "false_abstain": round(false_rate, 4),
                    "false_abstain_count": false_abstain,
                    "has_answer_total": n_answer,
                    "balanced_accuracy": round(
                        (correct_rate + (1.0 - false_rate)) / 2.0, 4
                    ),
                }
            )
    combos.sort(
        key=lambda c: (
            -c["balanced_accuracy"],
            c["false_abstain"],
            c["relevance"],
            c["evidence"],
        )
    )
    for rank, combo in enumerate(combos, start=1):
        combo["rank"] = rank
    return combos


def find_current(combos: list[dict], relevance: float, evidence: float) -> dict | None:
    """Tìm tổ hợp trùng ngưỡng hiện tại trong config (so khớp sau khi làm tròn)."""
    for combo in combos:
        if (
            round(combo["relevance"], 4) == round(relevance, 4)
            and round(combo["evidence"], 4) == round(evidence, 4)
        ):
            return combo
    return None


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def write_report(
    path: Path,
    run_id: str,
    args: argparse.Namespace,
    config: dict,
    golden_rows: list[dict],
    combos: list[dict],
    current: dict | None,
    relevance_grid: list[float],
    evidence_grid: list[float],
) -> None:
    """Sinh báo cáo Markdown: lưới đã quét, top 10, vị trí ngưỡng hiện tại,
    và cảnh báo overfitting (bắt buộc, ER-008)."""
    base = GateConfig.from_config(config)
    n_abstain = sum(1 for q in golden_rows if q["group"] in ABSTAIN_GROUPS)
    n_answer = sum(1 for q in golden_rows if q["group"] == ANSWER_GROUP)
    best = combos[0]
    lines: list[str] = [
        f"# Báo cáo quét ngưỡng gate — {run_id}",
        "",
        f"- Thời điểm chạy: {datetime.now().isoformat(timespec='seconds')}",
        f"- Golden set: `{args.golden}` ({len(golden_rows)} câu; "
        f"nhóm abstain={n_abstain}, has_answer={n_answer})",
        f"- Corpus: `{args.corpus}`",
        "- Scorer: HeuristicScorer (quy tắc offline) — điểm chấm một lần, "
        "áp lại gate cho từng tổ hợp ngưỡng.",
        f"- Lưới relevance: {relevance_grid}",
        f"- Lưới evidence: {evidence_grid}",
        f"- Cố định theo config: injection={base.injection}, "
        f"contradiction={base.contradiction}, top_k={base.top_k}",
        f"- Tổng số tổ hợp: {len(combos)}",
        "",
        "## CẢNH BÁO OVERFITTING (đọc trước khi dùng số)",
        "",
        "Sweep này hiệu chuẩn và đánh giá trên **cùng một golden set**. Tổ hợp",
        "tốt nhất dưới đây vì thế là ngưỡng hợp với đúng những câu này nhất —",
        "chưa có gì đảm bảo nó giữ được hiệu quả trên câu hỏi mới. Chỉ dùng",
        "kết quả này như **gợi ý ban đầu** để chọn ngưỡng, rồi PHẢI kiểm chứng",
        "lại trên một tập validation độc lập (câu hỏi chưa từng dùng để chọn",
        "ngưỡng) trước khi coi ngưỡng là đã chốt. Mẫu số cũng nhỏ "
        f"(abstain n={n_abstain}, has_answer n={n_answer}) nên một câu đổi",
        "kết quả là đủ làm tỉ lệ nhảy vài điểm phần trăm.",
        "",
        "## Tổ hợp tốt nhất trên lưới",
        "",
        f"- relevance = {best['relevance']:.2f}, evidence = {best['evidence']:.2f} "
        f"→ abstain đúng {_pct(best['abstain_correct'])} "
        f"({best['abstain_correct_count']}/{best['abstain_group_total']}), "
        f"abstain sai {_pct(best['false_abstain'])} "
        f"({best['false_abstain_count']}/{best['has_answer_total']}), "
        f"balanced accuracy = {best['balanced_accuracy']:.4f}.",
        "",
        "## Top 10 tổ hợp (balanced accuracy giảm dần, hoà: abstain sai thấp hơn)",
        "",
        "| Hạng | relevance | evidence | Abstain đúng | Abstain sai | Balanced accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for combo in combos[:10]:
        lines.append(
            f"| {combo['rank']} | {combo['relevance']:.2f} | {combo['evidence']:.2f} "
            f"| {_pct(combo['abstain_correct'])} "
            f"({combo['abstain_correct_count']}/{combo['abstain_group_total']}) "
            f"| {_pct(combo['false_abstain'])} "
            f"({combo['false_abstain_count']}/{combo['has_answer_total']}) "
            f"| {combo['balanced_accuracy']:.4f} |"
        )
    lines += ["", "## Vị trí của ngưỡng hiện tại trong config", ""]
    if current is not None:
        lines.append(
            f"Ngưỡng hiện tại (config.yaml: relevance = {base.relevance:.2f}, "
            f"evidence = {base.evidence:.2f}) đứng **hạng {current['rank']}/"
            f"{len(combos)}** trên lưới: abstain đúng "
            f"{_pct(current['abstain_correct'])} "
            f"({current['abstain_correct_count']}/{current['abstain_group_total']}), "
            f"abstain sai {_pct(current['false_abstain'])} "
            f"({current['false_abstain_count']}/{current['has_answer_total']}), "
            f"balanced accuracy = {current['balanced_accuracy']:.4f}."
        )
    else:
        lines.append(
            f"Ngưỡng hiện tại (config.yaml: relevance = {base.relevance:.2f}, "
            f"evidence = {base.evidence:.2f}) **không nằm trong lưới đã quét** "
            "nên không có hạng để so sánh."
        )
    lines += [
        "",
        "*Nhắc lại: mọi con số trên là hành vi của HeuristicScorer (quy tắc",
        "offline) trên dữ liệu tổng hợp — không phải chất lượng của Jev hay",
        "model thật, và chưa qua validation độc lập.*",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quét hiệu chuẩn ngưỡng gate")
    parser.add_argument("--golden", default=str(_EVALS_DIR / "data" / "vi_golden.jsonl"))
    parser.add_argument("--corpus", default=str(_EVALS_DIR / "data" / "vi_corpus.jsonl"))
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--out-dir", default=str(_EVALS_DIR))
    parser.add_argument("--relevance-grid", default=None,
                        help="Danh sách phân tách bằng dấu phẩy; mặc định 0.30..0.60 bước 0.05")
    parser.add_argument("--evidence-grid", default=None,
                        help="Danh sách phân tách bằng dấu phẩy; mặc định 0.35..0.70 bước 0.05")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    base = GateConfig.from_config(config)
    relevance_grid = parse_grid(args.relevance_grid, DEFAULT_RELEVANCE_GRID)
    evidence_grid = parse_grid(args.evidence_grid, DEFAULT_EVIDENCE_GRID)
    golden_rows = load_golden(Path(args.golden))
    retriever = build_retriever(args.corpus, config)
    scorer = HeuristicScorer()

    scored: list[tuple[str, list, list]] = []
    for q in golden_rows:
        candidates = retriever.retrieve(q["question"])
        scores = scorer.score(q["question"], candidates)
        scored.append((q["group"], candidates, scores))

    combos = sweep(scored, relevance_grid, evidence_grid, base)
    current = find_current(combos, base.relevance, base.evidence)

    run_id = f"sweep-{datetime.now().strftime('%Y%m%d-%H%M')}"
    out_dir = Path(args.out_dir)
    results_path = out_dir / "results" / f"{run_id}.jsonl"
    report_path = out_dir / "reports" / f"{run_id}.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for combo in combos:
            f.write(json.dumps(combo, ensure_ascii=False) + "\n")
    write_report(
        report_path, run_id, args, config, golden_rows, combos, current,
        relevance_grid, evidence_grid,
    )

    best = combos[0]
    print(f"run_id: {run_id}")
    print(f"results: {results_path} ({len(combos)} tổ hợp)")
    print(f"report:  {report_path}")
    print(
        "best: relevance={:.2f} evidence={:.2f} -> abstain đúng {:.1f}% "
        "({}/{}), abstain sai {:.1f}% ({}/{}), balanced accuracy = {:.4f}".format(
            best["relevance"],
            best["evidence"],
            best["abstain_correct"] * 100,
            best["abstain_correct_count"],
            best["abstain_group_total"],
            best["false_abstain"] * 100,
            best["false_abstain_count"],
            best["has_answer_total"],
            best["balanced_accuracy"],
        )
    )
    if current is not None:
        print(
            "current (config): relevance={:.2f} evidence={:.2f} -> hạng {}/{}; "
            "balanced accuracy = {:.4f}".format(
                current["relevance"],
                current["evidence"],
                current["rank"],
                len(combos),
                current["balanced_accuracy"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())