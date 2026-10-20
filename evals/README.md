# Eval harness — jev-rag-lab

Bộ máy đo cho spec 001: chạy các pipeline trên golden set, ghi kết quả thô
(JSONL) và báo cáo so sánh (Markdown). Mọi ngưỡng lấy từ `../config.yaml`
(ER-003); mỗi báo cáo đều nhúng snapshot config đầy đủ để tái lập.

## Cách chạy

Từ gốc repo, zero-install (stdlib + numpy + pyyaml, `python3` hệ thống):

```bash
# Vòng mặc định: golden tiếng Việt (50 câu) + shadow mode
python3 evals/run_eval.py --shadow

# Vòng tiếng Anh (mini set 10 câu)
python3 evals/run_eval.py --golden evals/data/en_mini.jsonl \
    --corpus evals/data/en_corpus.jsonl --shadow
```

Tham số: `--golden` (mặc định `evals/data/vi_golden.jsonl`), `--corpus` (mặc
định `evals/data/vi_corpus.jsonl`), `--config` (mặc định `config.yaml` ở gốc
repo), `--out-dir` (mặc định `evals/`), `--shadow` (bật chế độ shadow ER-005).

Đầu ra:

- `evals/results/<run_id>.jsonl` — mỗi dòng là một (câu hỏi × pipeline):
  `id, group, expected, pipeline, abstained, answer, used_doc_ids,
  rejected_reasons, injection_excluded, timings_ms` và 3 cờ chấm
  `abstain_correct, evidence_hit, answer_match` (định nghĩa dưới đây).
  `run_id` dạng `YYYYMMDD-HHmm-<dataset>`.
- `evals/reports/<run_id>.md` — báo cáo người đọc: snapshot config, trạng
  thái pipeline, bảng chỉ số, độ trễ, mục giới hạn và kết luận mô tả số.

## Pipeline nào chạy được offline

| Pipeline | Thành phần | Điều kiện |
|---|---|---|
| `A-baseline` | BaselinePipeline + MockLLM | Luôn chạy (offline) |
| `C-heuristic` | GatedPipeline + **HeuristicScorer** (quy tắc) + MockLLM | Luôn chạy (offline). Đây là bản plumbing của Pipeline C, KHÔNG phải model nhỏ thật |
| `B-jev` | GatedPipeline + JevScorer (API TypeSafe) | Cần `JEV_API_BASE` + `JEV_API_KEY`; thiếu thì báo cáo ghi `SKIPPED — thiếu JEV_API_KEY (Q2)` và harness tiếp tục với 2 pipeline còn lại — không fake số |

## Ý nghĩa từng chỉ số

Định nghĩa chính xác nằm ở `evals/metrics.py` (hàm thuần, có test tính tay).

- **Precision@5** (tầng retrieval): trong top-5 doc đầu tiên của first-stage,
  tỉ lệ thuộc `evidence_doc_ids`; lấy trung bình trên các câu `has_answer`.
  Mẫu số là số vị trí thực có nếu danh sách ngắn hơn 5.
- **Recall@20** (tầng retrieval, dạng hit-rate theo spec): 1.0 nếu có ít nhất
  một evidence doc trong top-20, ngược lại 0.0; trung bình trên `has_answer`.
  Đo trần của hệ: gate không cứu được đoạn bằng chứng không được retrieve.
- **Abstain đúng theo nhóm**: với nhóm mà đáp án đúng là abstain (`no_answer`,
  `distractor_no_evidence`, `injection`) — tỉ lệ câu pipeline thực sự abstain.
  Cờ gốc là `abstain_correct` trong JSONL: True khi hành vi abstain khớp với
  `expected` (nhóm `has_answer`: True khi KHÔNG abstain).
- **Abstain sai**: tỉ lệ câu `has_answer` bị pipeline abstain oan — cái giá
  của gate quá nghiêm.
- **Answer match** (chỉ trên `has_answer`): `expected` có xuất hiện trong
  `answer` sau chuẩn hoá (chữ thường, gộp khoảng trắng) hay không. Với
  MockLLM — vốn chỉ trích câu đầu của đoạn đầu — chỉ số này chỉ kiểm tra
  plumbing trích xuất, không đo chất lượng trả lời.
- **Evidence hit** (chỉ trên `has_answer`): có ít nhất một evidence doc trong
  `used_doc_ids` (các đoạn thực tế đưa vào prompt).
- **Injection excluded**: có ít nhất một ứng viên bị gate loại với lý do
  `injection` hay không (chủ yếu có nghĩa với nhóm `injection`).
- **Độ trễ**: trung bình / p50 / p95 theo giai đoạn `retrieval`, `scoring`,
  `generation` và tổng — đo offline trên máy local, không đại diện API thật.

## Shadow mode (ER-005)

`--shadow`: với mỗi câu hỏi, HeuristicScorer + gate được chạy thêm trên kết
quả retrieval của pipeline A ở chế độ **chỉ ghi** — không thay đổi đáp án của
bất kỳ pipeline nào. Mỗi dòng kết quả có thêm `shadow_would_abstain` và
`shadow_passed_doc_ids`. Dùng dữ liệu này để xem phân phối quyết định của
gate và hiệu chuẩn ngưỡng trong `config.yaml` trước khi tin bất kỳ con số
abstain nào.

## Cách đọc báo cáo

1. Đọc mục **Trạng thái pipeline** trước: pipeline nào `skipped` thì mọi ô số
   của nó là `—`, đừng diễn giải.
2. Đọc mục **Giới hạn của vòng chạy này**: HashingEmbedder là túi từ băm
   (không ngữ nghĩa), MockLLM là trích dẫn mẫu, HeuristicScorer là quy tắc —
   số của vòng offline chỉ mô tả plumbing trên dữ liệu tổng hợp (ER-008).
3. So sánh A vs C trong bảng chỉ số: chênh lệch abstain đúng/sai giữa hai
   pipeline là hành vi của *quy tắc heuristic + ngưỡng hiện tại*, không phải
   bằng chứng về Jev hay model nhỏ thật. Chưa vòng nào đủ điều kiện kết luận
   GO/NO-GO.

## Chạy lại khi có key/model thật

Đặt biến môi trường (không ghi key vào file nào trong repo — ER-007; có thể
để trong `.env` đã gitignore rồi `export` tay, hoặc export trực tiếp):

```bash
export JEV_API_BASE="..."        # endpoint Jev API (TypeSafe)
export JEV_API_KEY="..."         # key Jev -> mở khoá pipeline B-jev
export LLM_API_BASE="..."        # endpoint OpenAI-compatible cho LLM thật
export LLM_API_KEY="..."
export LLM_MODEL="..."
export EMBEDDING_API_BASE="..."  # endpoint embedding thật (đặt thêm
export EMBEDDING_API_KEY="..."   #   embedding.provider: http trong config.yaml)
export EMBEDDING_MODEL="..."
```

Khi có `JEV_API_KEY`, lần chạy kế tiếp tự động thêm hàng `B-jev` với trạng
thái `ran`. Lưu ý trung thực: `JevScorer` hiện viết theo mẫu cookbook đã công
bố và **chưa kiểm chứng với API thật** — phải đối chiếu định dạng
request/response với tài liệu TypeSafe trước khi tin số của Pipeline B.
