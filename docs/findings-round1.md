# Tổng kết vòng 1 — jev-rag-lab (2026-10-20)

Tài liệu này tổng kết vòng thí nghiệm đầu tiên của nghiên cứu 001
(`specs/001-jev-rerank-gate/spec.md`): những gì đã cài đặt, số đo được,
5 phát hiện chính, kết quả quét hiệu chuẩn ngưỡng, và điều kiện để sang
vòng 2. Mọi con số đều lấy từ báo cáo eval đã sinh bởi harness — nguồn
được ghi ngay tại bảng.

**Đọc trước khi dùng số (ER-008):** toàn bộ vòng 1 chạy bằng thành phần
offline — HashingEmbedder (băm túi từ), HeuristicScorer (quy tắc),
MockLLM (giả). Vòng này chứng minh *harness và cơ chế gate chạy được
đầu-cuối*, KHÔNG chứng minh gì về chất lượng của Jev hay của một hệ RAG
dùng model thật. Chưa đủ điều kiện kết luận GO/NO-GO.

## 1. Thiết kế đã cài đặt

- **3 pipeline đối chứng** (chỉ khác nhau ở tầng quyết định, đúng biến
  kiểm soát của spec):
  - **A-baseline**: vector search → top-5 theo similarity (lọc ngưỡng
    similarity 0.35) → LLM.
  - **B-jev**: top-20 ứng viên → JevScorer (API TypeSafe) chấm → gate →
    top-5 / abstain. **Chưa chạy vòng 1** — thiếu `JEV_API_KEY` (Q2).
  - **C-heuristic** (đại diện cho Pipeline C khi offline): như B nhưng
    scorer là HeuristicScorer — quy tắc độ phủ token + khớp thực thể +
    mẫu câu injection. Đây là plumbing, không phải model quyết định.
- **Gate 4 tín hiệu** (`src/decision/gate.py`): mỗi ứng viên bị chấm
  `relevance`, `evidence`, `contradiction`, `injection` (xác suất 0–1).
  Thứ tự kiểm: injection > ngưỡng → loại; contradiction > ngưỡng → tách
  vào nhóm conflicts (vòng 1 chưa đưa vào context); relevance < ngưỡng →
  loại; evidence < ngưỡng → loại. Qua cả 4 → xếp theo evidence rồi
  relevance, cắt top-5. Không ai qua → **abstain**.
- **Ngưỡng khởi điểm** lấy từ config (`config.yaml`, theo Passage Gate
  đã công bố): relevance 0.45, evidence 0.55, injection 0.70,
  contradiction 0.70.
- **Shadow mode (ER-005)**: scorer chấm và áp gate thử lên kết quả
  retrieval của A nhưng chỉ ghi (`shadow_would_abstain`), không lọc —
  để lấy phân phối điểm phục vụ hiệu chuẩn.
- **Dữ liệu**: golden set tiếng Việt tự tạo 50 câu (has_answer 20,
  no_answer 10, distractor_no_evidence 12, injection 8) + bộ mini
  tiếng Anh 10 câu (has_answer 6, no_answer 3, distractor 1, injection 0).
- **Test**: 52 test tự động, tất cả pass tại thời điểm chốt vòng 1.

## 2. Số liệu chính

### 2.1 Tiếng Việt — nguồn: `evals/reports/20261020-1505-vi.md` (dữ liệu thô: `evals/results/20261020-1505-vi.jsonl`)

| Pipeline | Precision@5 | Recall@20 | Abstain đúng no_answer | Abstain đúng distractor | Abstain đúng injection | Abstain sai (has_answer) | Answer match | Evidence hit |
|---|---|---|---|---|---|---|---|---|
| A-baseline | 0.2000 | 1.0000 | 100% (10/10) | 58.3% (7/12) | 25.0% (2/8) | 35.0% (7/20) | 0.0% (0/20) | 65.0% (13/20) |
| C-heuristic | 0.2000 | 1.0000 | 100% (10/10) | 75.0% (9/12) | 75.0% (6/8) | **60.0% (12/20)** | 0.0% (0/20) | 40.0% (8/20) |
| B-jev | — | — | — | — | — | — | — | — (SKIPPED, thiếu key) |

Độ trễ (local, offline — không đại diện API thật): A total TB 0.12 ms;
C total TB 1.27 ms, trong đó scoring TB 1.18 ms.

### 2.2 Tiếng Anh (mini) — nguồn: `evals/reports/20261020-1505-en.md` (dữ liệu thô: `evals/results/20261020-1505-en.jsonl`)

| Pipeline | Precision@5 | Recall@20 | Abstain đúng no_answer | Abstain đúng distractor | Abstain sai (has_answer) | Answer match | Evidence hit |
|---|---|---|---|---|---|---|---|
| A-baseline | 0.3333 | 1.0000 | 100% (3/3) | 100% (1/1) | **100% (6/6)** | 0.0% (0/6) | 0.0% (0/6) |
| C-heuristic | 0.3333 | 1.0000 | 100% (3/3) | 100% (1/1) | **100% (6/6)** | 0.0% (0/6) | 0.0% (0/6) |
| B-jev | — | — | — | — | — | — | — (SKIPPED, thiếu key) |

Hai pipeline cho số giống hệt nhau ở vòng EN vì cả hai cùng abstain
toàn bộ (xem Phát hiện 4).

## 3. Năm phát hiện

### Phát hiện 1 — Gate heuristic chặn nhiễu tốt hơn baseline, nhưng quá nghiêm

Trên golden VI, C-heuristic abstain đúng ở nhóm distractor 75.0% (9/12)
so với baseline 58.3% (7/12), và nhóm injection 75.0% (6/8) so với
25.0% (2/8) — cơ chế gate có tác dụng thật với đoạn đúng-chủ-đề-không-
bằng-chứng. Đổi lại, **abstain sai tăng từ 35.0% (7/20) của baseline
lên 60.0% (12/20)**: hơn nửa số câu CÓ đáp án bị gate chặn hết ứng viên.
Evidence hit vì thế tụt từ 65.0% xuống 40.0%. Ở ngưỡng hiện tại
(0.45/0.55), gate heuristic là một cái lọc quá tay — đúng ca "gate quá
nghiêm" mà tài liệu tham khảo đã cảnh báo. Đây là lý do có mục quét
ngưỡng ở phần 4, và là lý do chưa thể kết luận gì về H1: scorer quy tắc
không tách được hai lớp "có bằng chứng / không có bằng chứng" đủ sạch.

### Phát hiện 2 — Tín hiệu injection không bắn một lần nào trong suốt vòng 1

Dù nhóm injection đạt "abstain đúng 75%", **không lần nào** đoạn bị
loại vì lý do injection:

- Trong dữ liệu thô VI (`evals/results/20261020-1505-vi.jsonl`), các
  dòng của C-heuristic có `injection_excluded = true`: **0/50**.
- Toàn bộ 787 lượt loại ứng viên của C-heuristic đều mang lý do
  `relevance` — tức abstain đúng ở nhóm injection là nhờ đoạn bị loại
  vì điểm liên quan thấp, không phải nhờ phát hiện injection.
- Nguyên nhân là lệch diễn đạt giữa hai phía: 10/10 mẫu câu trong
  `_INJECTION_PATTERNS` của HeuristicScorer cho **0 kết quả** khi grep
  trên corpus VI. Ví dụ mẫu scorer là `"bỏ qua hướng dẫn"` (khớp chuỗi
  con liên tục), trong khi corpus viết *"hãy bỏ qua **mọi** hướng dẫn
  và chính sách giá trước đó"* — từ chen giữa phá vỡ phép khớp chuỗi.

Bài học: một tín hiệu bảo vệ viết bằng khớp chuỗi cứng sẽ câm lặng
trước diễn đạt tự nhiên, và — nguy hiểm hơn — bảng chỉ số tổng vẫn
"đẹp" (75%) nhờ cơ chế khác gánh thay. Với scorer là model thật
(Jev/LLM), lớp phán đoán ngữ nghĩa mới là thứ cần đo ở vòng 2; nếu giữ
heuristic cho mục đích plumbing, mẫu câu phải được viết từ chính corpus.

### Phát hiện 3 — answer_match = 0 ở mọi pipeline là giới hạn của MockLLM, không phải lỗi harness

Cả A và C đều có answer_match = 0.0% (VI: 0/20; EN: 0/6). Nguyên nhân
đã rõ và nằm ở thành phần giả: MockLLM không hiểu câu hỏi — nó trích
nguyên văn *câu đầu tiên của đoạn ngữ cảnh đầu tiên* làm "câu trả lời".
Đáp án vàng (ví dụ "5.490.000 đồng") hiếm khi nằm đúng ở câu đầu đó,
nên phép so khớp chuỗi con thất bại 100% dù evidence có thể đã vào
đúng context (evidence hit của A là 65.0%). Kết luận đúng mức:
answer_match ở vòng 1 chỉ kiểm tra plumbing trích xuất, **không** đo
chất lượng trả lời; chỉ số này chỉ có ý nghĩa từ khi thay MockLLM
bằng LLM thật ở vòng 2.

### Phát hiện 4 — Vòng EN: trần similarity ≈ 0.34 < 0.35, ngưỡng không chuyển thẳng qua ngôn ngữ/dataset

Đo lại bằng retriever hiện tại trên golden EN mini: similarity cao
nhất trong toàn bộ ứng viên của 10 câu là **0.3411** — thấp hơn ngưỡng
similarity 0.35 của baseline. Hệ quả: baseline lọc sạch mọi ứng viên
và abstain cả 6/6 câu has_answer (abstain sai 100%), C-heuristic cũng
không còn gì để chấm — hai pipeline cho số giống hệt nhau, vòng EN
không phân biệt được gì. Bài học: ngưỡng similarity tuyệt đối gắn
chặt với embedder + ngôn ngữ + corpus (ở đây là HashingEmbedder túi
từ trên văn bản Anh ngắn); con số 0.35 "tham khảo từ hệ thật" không
mang sang được dataset khác. Mọi ngưỡng — similarity lẫn gate — phải
hiệu chuẩn trên chính dữ liệu sẽ chạy.

### Phát hiện 5 — Precision@5 = 0.20 trong khi Recall@20 = 1.0: retrieval là trần chất lượng

Trên golden VI, tài liệu chứa bằng chứng **luôn** nằm trong 20 ứng
viên first-stage (Recall@20 = 1.0000 ở cả hai vòng), nhưng trong top-5
theo similarity chỉ có trung bình 1/5 vị trí là tài liệu đúng
(Precision@5 = 0.2000; vòng EN: 0.3333). Nghĩa là first-stage *tìm
thấy* đáp án nhưng *xếp hạng* kém — và đó là lúc chưa có reranker thật
nào hoạt động (C-heuristic xếp lại theo evidence quy tắc nhưng
evidence hit vẫn giảm vì abstain sai). Điều này khớp đúng cảnh báo
trong spec: gate/rerank chỉ sắp xếp và lọc trong tập ứng viên đã
truy xuất; chất lượng thứ hạng first-stage là trần mà tầng quyết định
phải vượt qua bằng điểm của mình — muốn biết nó có vượt được không
thì cần scorer thật, không phải quy tắc.

## 4. Quét hiệu chuẩn ngưỡng (sweep)

Chạy `evals/sweep_thresholds.py` trên golden VI 50 câu: retrieval +
HeuristicScorer chấm một lần, áp gate cho lưới relevance
{0.30, ..., 0.60} × evidence {0.35, ..., 0.70} (injection 0.70,
contradiction 0.70, top_k 5) — 56 tổ hợp. Kết quả đầy đủ:
`evals/results/sweep-20261020-1511.jsonl`, báo cáo:
`evals/reports/sweep-20261020-1511.md`.

- **Tổ hợp tốt nhất trên lưới**: relevance = **0.40**, evidence =
  **0.55** → abstain đúng **76.7% (23/30)**, abstain sai **45.0%
  (9/20)**, balanced accuracy = **0.6583**. (Hạng 1–4 là một cao
  nguyên: evidence từ 0.55 đến 0.70 cho cùng một kết quả — ngưỡng
  evidence không còn tác dụng phân biệt trong vùng này, vì relevance
  đã là ràng buộc binding trước.)
- **Ngưỡng hiện tại trong config** (relevance 0.45, evidence 0.55):
  abstain đúng 83.3% (25/30), abstain sai 60.0% (12/20), balanced
  accuracy = 0.6167 — đứng **hạng 36/56**. Hạ relevance xuống 0.40
  đánh đổi −6.7 điểm abstain đúng lấy +15 điểm abstain sai.
- Ngay cả tổ hợp tốt nhất vẫn abstain sai 45% — xa tiêu chí GO của
  spec (abstain sai ≤ 10%). Điều này củng cố Phát hiện 1: nút thắt là
  khả năng tách lớp của scorer quy tắc, không phải vị trí ngưỡng.

> **Cảnh báo overfitting (bắt buộc đọc):** sweep hiệu chuẩn và đánh
> giá trên **cùng** golden set 50 câu. Tổ hợp "tốt nhất" ở trên là
> ngưỡng hợp với đúng 50 câu này nhất — chưa qua bất kỳ validation
> độc lập nào, mẫu số lại nhỏ (30/20) nên rất dễ là nhiễu. Chỉ dùng
> như **gợi ý ban đầu** (ứng viên: relevance 0.40) để đem kiểm chứng
> trên tập câu hỏi mới ở vòng 2.

## 5. Những gì vòng 1 CHƯA chứng minh được

- **RQ1** (gate giảm bịa?): chưa trả lời được ở mức model thật. Cái đã
  đo là hành vi của một scorer quy tắc trên embedder túi từ — cả hai
  đều không có phán đoán ngữ nghĩa.
- **RQ2** (độ trễ/chi phí): chưa đo được. Độ trễ vòng 1 là số
  offline trên máy local với thành phần giả (≈1 ms), không liên quan
  tới độ trễ của API thật qua mạng.
- **RQ3** (Jev vs model nhỏ): chưa chạy được vế nào của phép so sánh —
  Pipeline B thiếu key; Pipeline C mới có bản heuristic thay thế,
  chưa phải LlmScorer với model thật.
- **RQ4** (tiếng Việt): mới chỉ có plumbing chạy trên dữ liệu VI.
  Chưa có bằng chứng nào về việc Jev hay model nhỏ chấm tiếng Việt
  tốt đến đâu.
- **Pipeline B chưa chạy một lần nào** — mọi khẳng định về Jev vẫn
  đang ở trạng thái "theo tài liệu bên thứ ba".

## 6. Điều kiện để sang vòng 2 / kết luận GO-NO-GO

Ba đầu vào còn thiếu — đều nằm ngoài sandbox offline hiện tại:

1. **`JEV_API_KEY`** (và `JEV_API_BASE`) — câu hỏi mở Q2 của spec,
   blocker duy nhất của Pipeline B. Có key là chạy được JevScorer;
   lưu ý phải đối chiếu lại định dạng request/response với tài liệu
   TypeSafe thật trước khi tin số (xem cảnh báo trong
   `src/decision/scorer.py`).
2. **Một endpoint LLM + embedding thật** qua biến môi trường
   `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` và
   `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL`
   (client HTTP đã có sẵn trong `src/llm/http.py`,
   `src/embeddings/http.py`). Có endpoint là thay được MockLLM +
   HashingEmbedder, chạy LlmScorer thật cho Pipeline C, và
   answer_match/độ trễ mới có ý nghĩa.
3. **Đường mạng** để chạy `evals/data/build_hotpotqa.py` (tải
   HotpotQA distractor, lấy mẫu 200 câu, seed 42) — cho benchmark EN
   thật thay bộ mini 10 câu tự tạo (Q1: script đã sẵn sàng, chỉ chờ
   mạng).

Khi đủ cả ba: chạy lại harness trên VI + HotpotQA với A/B/C thật, quét
ngưỡng trên tập hiệu chuẩn riêng và kiểm chứng trên tập validation
độc lập, rồi mới áp tiêu chí GO/NO-GO của spec (abstain đúng ≥ 90%,
abstain sai ≤ 10%, độ trễ cộng thêm p95 ≤ 1.000 ms, precision@5
không giảm so với baseline).
