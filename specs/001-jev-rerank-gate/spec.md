# Đặc tả nghiên cứu: Tầng quyết định kiểu Jev — Rerank + Gate cho RAG

**Mã nghiên cứu**: `001-jev-rerank-gate`

**Ngày tạo**: 2026-10-20

**Trạng thái**: Vòng 1 (offline) đã chạy xong ngày 2026-10-20 — harness + 3 pipeline chạy đầu-cuối bằng thành phần offline (HashingEmbedder, HeuristicScorer, MockLLM), Pipeline B chưa chạy vì thiếu key. Báo cáo vòng 1: `evals/reports/20261020-1505-vi.md` và `-en.md`; tổng kết + quét hiệu chuẩn ngưỡng: `docs/findings-round1.md`. Chưa kết luận GO/NO-GO; còn câu hỏi mở Q1 (chờ đường mạng) và Q2 (chờ API key) ở mục "Làm rõ".

**Đầu vào**: 5 hình người dùng gửi ngày 2026-10-20, lưu tại `docs/source-images/`:
(1) `01-rag-vs-jev-rag.jpg` — sơ đồ "RAG vs Jev + RAG"; (2) `02-top-5-rag-architectures.jpg` —
"Top 5 RAG Architectures You Must Know in 2026"; (3) `03-top-9-places-to-use-jev.jpg` —
"Top 9 places to use Jev"; (4) `04-rag-project-structure.jpg` — cấu trúc project RAG Python;
(5) `05-langchain-rag-project-structure.jpg` — cấu trúc project LangChain + RAG.

## Phân tích 5 hình — bóc tách

### Hình 1 — "RAG vs Jev + RAG" (giả thuyết trung tâm của nghiên cứu)

Hai pipeline giống hệt nhau ở mọi khâu (Query → Embedding → Vector DB → LLM → Final Response).
Khác biệt duy nhất của nhánh phải là một **tầng quyết định nằm giữa retrieval và LLM**:

- Vector DB trả **Top-k Candidates** (danh sách ứng viên, thiên về recall) thay vì "Top-k Context" đưa thẳng cho LLM.
- **Jev (Rerank + Gate)** nhận đồng thời câu Query gốc và các ứng viên, chấm điểm từng cặp (query, đoạn văn): đoạn nào thực sự chứa bằng chứng trả lời, đoạn nào chỉ "cùng chủ đề".
- Ứng viên qua cổng mới vào LLM. Không ứng viên nào qua cổng → rẽ nhánh **Abstain: Not in Docs** — hệ thống từ chối trả lời thay vì để LLM đoán.

Điểm yếu của RAG thường mà hình này nhắm vào: điểm similarity cao chỉ nói "gần về ngữ nghĩa/từ khóa", **không** nói "đoạn này trả lời được câu hỏi". Đoạn đúng chủ đề nhưng không chứa đáp án vẫn lọt vào context — nguồn gốc chính của trả lời bịa (hallucination có căn cứ giả).

**Jev** (theo tài liệu TypeSafe và các bài phân tích đã tra cứu 2026-10-20) là một *model quyết định có cấu trúc*: thay vì sinh văn bản, nó trả xác suất/điểm cho các phán đoán dạng có/không đã định kiểu (ví dụ "đoạn C7 có giúp trả lời câu hỏi này không"), và chấm cả danh sách ứng viên trong **một** request. Ứng dụng tự quyết định hành động dựa trên ngưỡng điểm.

### Hình 2 — "Top 5 RAG Architectures 2026": vị trí của nghiên cứu này

| Kiến trúc | Ý tưởng | Quan hệ với nghiên cứu 001 |
|---|---|---|
| 01 Hybrid RAG | Vector search + BM25 → rank fusion (RRF) → Top-k | Biến thể retrieval đáng thử ở vòng 2 — first-stage mạnh hơn thì gate có ứng viên tốt hơn. Chưa thuộc vòng 1. |
| 02 Graph RAG | Truy vấn trên knowledge graph (Person–Company–Project–Technology–Location) | Chỉ phân tích, không cài đặt trong nghiên cứu này. |
| 03 Agentic RAG | Planner phân rã câu hỏi, chọn nguồn (vector/web/SQL), retry trong ngân sách | Chỉ phân tích. Lưu ý: tool-call gating (Hình 3, vị trí 3) là điểm nối tự nhiên nếu làm tiếp. |
| 04 Corrective RAG (CRAG) | Chấm bằng chứng trước khi tin: Correct → tinh lọc; Ambiguous → bổ sung; Incorrect → tìm nguồn khác | **Họ hàng gần nhất của tầng Gate** — khác biệt: CRAG chấm bằng LLM sinh văn bản, Jev chấm bằng model quyết định chuyên dụng. Pipeline C trong thí nghiệm (model nhỏ làm scorer) chính là một dạng CRAG rút gọn, dùng làm đối chứng. |
| 05 Multimodal RAG | Text + ảnh/biểu đồ + bảng → cùng một index | Chỉ phân tích. |

### Hình 3 — "Top 9 places to use Jev": phạm vi nghiên cứu chỉ lấy 4

| # | Vị trí | Trong nghiên cứu này? |
|---|---|---|
| 1 | Model routing (chọn small/medium/large LLM) | Không |
| 2 | Guardrails (chặn injection trước LLM) | **Có** — tín hiệu `injection` của gate; corpus eval có nhóm đoạn chứa chỉ thị injection. |
| 3 | Tool-call gating (allow/ask/deny) | Không |
| 4 | Inbox triage | Không |
| 5 | **Reranking** | **Có — lõi thí nghiệm.** |
| 6 | LLM evals (chấm đáp án nhanh, rẻ) | **Có, phụ** — dùng scorer chấm faithfulness của đáp án trong eval harness. |
| 7 | Bulk labeling | Không |
| 8 | Real-time control (~300ms) | Không |
| 9 | **Confidence gate** (> 0.9 tự làm; 0.5–0.9 xác nhận; < 0.5 chuyển người) | **Có — chính là nhánh Abstain**; trong thí nghiệm không có người duyệt giữa chừng nên vùng dưới ngưỡng = abstain. |

### Hình 4 + 5 — Cấu trúc project: khung xương của chính repo này

Repo nghiên cứu dùng cấu trúc Python của Hình 4 (bổ sung module `decision/` cho tầng quyết định — xem README). Hình 5 (LangChain) chỉ để tham khảo đối chiếu module; repo **không** khóa vào LangChain: pipeline viết bằng Python thuần + thư viện tối thiểu để biến số thí nghiệm nằm ở tầng quyết định, không nằm ở framework.

| Module trong hình | Vị trí trong repo |
|---|---|
| `ingestion/loader.py` | `src/ingestion/` |
| `chunking/chunker.py` | `src/chunking/` |
| `embeddings/embedder.py` | `src/embeddings/` |
| `vectorDB/vector_store.py` | `src/vectorDB/` (FAISS cho vòng 1 — chạy local, tái lập dễ) |
| `retrieval/retriever.py` | `src/retrieval/` |
| *(mới)* tầng quyết định | `src/decision/` — `scorer.py` (interface + các implementation) và `gate.py` (ngưỡng, rerank, abstain) |
| `prompts/prompt_templates.py` | `src/prompts/` |
| `llm/llm_client.py` | `src/llm/` |
| `api/routes.py` | `src/api/` (chỉ cần khi demo; eval chạy bằng CLI) |
| `config.yaml` / `.env` | `config.yaml` ở gốc repo + `.env` (không commit) cho API key |
| `tests/` | `tests/` |
| *(mới)* eval | `evals/` — golden set + script chạy eval + báo cáo |

## Câu hỏi nghiên cứu và giả thuyết

- **RQ1 — Hiệu quả chống bịa**: Gate theo bằng chứng có giảm tỷ lệ trả lời bịa so với RAG thường trên nhóm câu hỏi mà retrieval trả về đoạn đúng chủ đề nhưng không chứa đáp án không?
  - *Giả thuyết H1*: Có — pipeline có gate abstain đúng ≥ 90% trên nhóm này, trong khi baseline trả lời bịa ở đa số các câu đó.
- **RQ2 — Cái giá**: Độ trễ cộng thêm và chi phí mỗi lượt hỏi của tầng gate là bao nhiêu?
  - *Giả thuyết H2*: Packed scoring (1 request cho cả danh sách ứng viên) giữ độ trễ cộng thêm ≤ 1.000ms ở p95 với top-20 ứng viên.
- **RQ3 — Jev vs model nhỏ tự cài**: Trên cùng dữ liệu, Jev API (TypeSafe) chấm 4 tín hiệu tốt hơn hay kém hơn một model nhỏ (qua AI Gateway) được prompt để chấm cùng 4 tín hiệu?
  - *Giả thuyết H3*: Chênh lệch chất lượng nằm trong ±5 điểm phần trăm trên các chỉ số chính — nếu đúng, yếu tố quyết định chọn implementation sẽ là chi phí/độ trễ/quyền riêng tư dữ liệu, không phải chất lượng.
- **RQ4 — Tiếng Việt**: Các cơ chế trên giữ được bao nhiêu hiệu quả trên golden set tiếng Việt tự xây (tài liệu công bố về Jev gần như toàn tiếng Anh)?
  - *Không đặt giả thuyết trước* — đây là câu hỏi khám phá, kết quả âm tính cũng là kết quả có giá trị.

## Thiết kế thí nghiệm

### Ba pipeline đối chứng (chỉ khác nhau ở tầng quyết định)

- **Pipeline A — Baseline RAG thường**: embed query → vector search → top-k (k = 5) theo similarity → LLM. Ngưỡng similarity 0.35 (mặc định tham khảo từ hệ RAG thật đang chạy) — dưới ngưỡng coi như không tìm thấy.
- **Pipeline B — Jev + RAG**: vector search lấy top-N ứng viên (N = 20) → **Jev API** chấm 4 tín hiệu → gate theo ngưỡng → top-k qua cổng vào LLM; không ai qua cổng → abstain.
- **Pipeline C — Model nhỏ + RAG**: như B, nhưng scorer là model nhỏ qua AI Gateway, prompt chấm cùng 4 tín hiệu, cùng định dạng đầu ra (xác suất 0–1).

### Biến kiểm soát

- Cùng một corpus, cùng chunking, cùng model embedding, cùng vector store, cùng LLM generation và cùng prompt generation cho cả 3 pipeline. Biến độc lập duy nhất: (không gate / gate bằng Jev / gate bằng model nhỏ) và ngưỡng gate.
- Mọi lần chạy eval ghi lại: phiên bản config đầy đủ, seed nếu có, thời gian, chi phí token/API từng lượt.

### Dữ liệu eval

1. **Tiếng Anh (đối chiếu tài liệu công bố)**: tập con của một benchmark RAG công khai có sẵn đáp án + đoạn bằng chứng gán nhãn (ứng viên: NFCorpus hoặc HotpotQA — chốt ở Q1), lấy mẫu cố định, ghi rõ cách lấy mẫu để tái lập.
2. **Tiếng Việt (tự xây — phần riêng của nghiên cứu này)**: golden set ≥ 50 câu trên một corpus tài liệu tiếng Việt tự tạo/tổng hợp (không dùng dữ liệu người dùng thật), gồm 4 nhóm:
   - Câu có đáp án rõ trong corpus.
   - Câu **không** có đáp án trong corpus (đáp án đúng = abstain).
   - Câu có đoạn nhiễu đúng chủ đề nhưng không chứa đáp án (nhóm quyết định cho RQ1).
   - Câu mà đoạn khớp nhất chứa chỉ thị injection (đo tín hiệu `injection`).

### Bốn tín hiệu chấm (theo mô hình Passage Gate đã công bố)

Mỗi cặp (query, đoạn) được chấm xác suất 0–1 cho: `relevance` (liên quan), `evidence` (đoạn chứa bằng chứng trả lời được), `contradiction` (mâu thuẫn tiền đề câu hỏi), `injection` (chứa chỉ thị can thiệp hệ thống).

Ngưỡng khởi điểm (tham khảo cài đặt Passage Gate đã công bố — **điểm bắt đầu để quét ngưỡng, không phải số chốt**): loại khi `relevance < 0.45`; yêu cầu `evidence >= 0.55`; loại khi `injection > 0.70`; `contradiction > 0.70` → tách riêng ghi nhận (vòng 1 chưa đưa vào context).

### Chỉ số đo

- Precision@5 và Recall@20 của first-stage (để tách lỗi retrieval khỏi lỗi gate — gate không cứu được đoạn bằng chứng không được truy xuất).
- Tỷ lệ **abstain đúng** trên nhóm không-có-đáp án + nhóm nhiễu; tỷ lệ **abstain sai** (câu có đáp án nhưng bị chặn hết).
- Tỷ lệ trả lời bịa của baseline trên nhóm nhiễu (mẫu số để RQ1 so sánh).
- Độ trễ mỗi lượt (p50/p95), tách riêng: retrieval / scoring / generation.
- Chi phí mỗi lượt hỏi (token + phí API nếu có), quy về cùng đơn vị tiền.
- Với RQ3: thỏa thuận giữa hai scorer (tỷ lệ cùng quyết định pass/fail trên từng đoạn) + chất lượng từng scorer so với nhãn vàng.

## Yêu cầu hệ thí nghiệm

- **ER-001**: Tầng quyết định PHẢI nằm sau một interface duy nhất (`DecisionScorer` trong `src/decision/scorer.py`): đầu vào = câu hỏi + danh sách đoạn; đầu ra = 4 tín hiệu cho từng đoạn. Đổi implementation không được đòi sửa pipeline.
- **ER-002**: Scorer PHẢI chấm packed — toàn bộ ứng viên trong một request khi nhà cung cấp hỗ trợ; nếu implementation không hỗ trợ packed, phải ghi rõ số request thực tế vào log chi phí.
- **ER-003**: Mọi ngưỡng gate PHẢI nằm trong `config.yaml`, không hardcode trong code; mỗi lần chạy eval nhúng toàn bộ config vào file báo cáo.
- **ER-004**: Eval harness PHẢI chạy được cả 3 pipeline trên cùng một lệnh, cùng một bộ dữ liệu, và xuất báo cáo dạng bảng so sánh (Markdown) vào `evals/reports/`.
- **ER-005**: Harness PHẢI có chế độ **shadow**: gate chấm và ghi điểm nhưng không lọc — để đo phân phối điểm trước khi chốt ngưỡng.
- **ER-006**: Corpus và golden set tiếng Việt PHẢI là dữ liệu tổng hợp/tự tạo, có file mô tả nguồn gốc; tuyệt đối không đưa dữ liệu người dùng thật vào repo nghiên cứu.
- **ER-007**: Mọi API key (Jev, AI Gateway, LLM) chỉ đọc từ `.env` (đã gitignore), không bao giờ ghi vào config, log hay báo cáo.
- **ER-008**: Kết quả âm tính (giả thuyết sai) PHẢI được ghi vào báo cáo với cùng mức chi tiết như kết quả dương tính — không chọn lọc kết quả.

## Tiêu chí kết luận (go / no-go cho việc áp dụng vào sản phẩm)

- **GO** (đáng đưa tầng gate vào hệ RAG thật) khi đồng thời: abstain đúng ≥ 90% trên nhóm không-có-đáp án + nhiễu (gộp EN + VI); abstain sai ≤ 10%; độ trễ cộng thêm p95 ≤ 1.000ms; precision@5 không giảm so với baseline.
- **NO-GO** khi bất kỳ tiêu chí GO nào trượt ở cả hai ngôn ngữ sau khi đã quét ngưỡng hợp lý — kèm phân tích nguyên nhân (retrieval yếu / scorer yếu / ngưỡng không tách được hai lớp).
- **Kết luận riêng cho RQ3**: chọn implementation cho sản phẩm dựa trên bảng chất lượng–độ trễ–chi phí đã đo, không dựa trên tài liệu quảng bá của nhà cung cấp.
- Báo cáo kết luận cuối vòng 1 đặt tại `evals/reports/`, kèm toàn bộ số liệu thô để kiểm chứng lại.

## Làm rõ

Còn chờ quyết:

- **Q1**: Benchmark tiếng Anh vòng 1 dùng NFCorpus (ngắn, thiên truy xuất y sinh) hay tập con HotpotQA (câu hỏi đa bước, gần với ca rerank khó đã công bố)? → *Đã chốt hướng HotpotQA tập con 200 câu (theo khuyến nghị): script `evals/data/build_hotpotqa.py` đã viết sẵn (tải bản distractor, lấy mẫu seed 42), nhưng sandbox không có mạng nên CHƯA chạy được — chờ đường mạng.* Vòng 1 tạm dùng bộ mini EN 10 câu tự tạo.
- **Q2**: Jev API cần tài khoản/API key của TypeSafe — khi bắt đầu cài đặt Pipeline B, người dùng cung cấp key qua Secure Vault hay đăng ký tài khoản mới? → *Vẫn chờ key.* Đây là blocker duy nhất của Pipeline B: JevScorer đã cài đặt sẵn, Pipeline B bị SKIPPED trong mọi lần chạy vòng 1 vì thiếu `JEV_API_KEY`.
- **Q3**: Golden set tiếng Việt 50 câu có đủ cho vòng 1 không, hay nâng lên 100 câu ngay từ đầu (tốn công gán nhãn gấp đôi)? → **ĐÃ CHỐT (auto, theo khuyến nghị): 50 câu.** Golden set 50 câu đã xây xong (`evals/data/vi_golden.jsonl`: has_answer 20, no_answer 10, distractor_no_evidence 12, injection 8) và đã chạy vòng 1 + quét ngưỡng trên đó. Việc mở rộng lên 100 câu để dành cho vòng 2, khi cần tập hiệu chuẩn và tập validation độc lập.

Đã tự quyết từ ngữ cảnh: FAISS local cho vòng 1 (tái lập rẻ, thay vector store là biến phụ không thuộc RQ nào); không dùng LangChain (giữ biến thí nghiệm ở tầng quyết định); ngưỡng khởi điểm lấy theo cài đặt Passage Gate đã công bố rồi quét quanh điểm đó.

## Ngoài phạm vi

- Tích hợp vào bất kỳ hệ thống sản phẩm nào (kể cả khi kết luận GO — đó là dự án riêng, dùng lại spec 003 phía event-lead-si nếu cần).
- Cài đặt Graph RAG / Agentic RAG / Multimodal RAG (chỉ phân tích ở Hình 2).
- Dùng Jev cho model routing, tool-call gating, inbox triage, bulk labeling, real-time control (Hình 3).
- Huấn luyện/fine-tune model scorer riêng.
- Giao diện người dùng — nghiên cứu này chỉ cần CLI eval + báo cáo.

## Tài liệu tham khảo (đã tra cứu 2026-10-20)

- Jev AI — "RAG Evaluation: Check Retrieved Context with Jev" (jev-ai.pro/rag-evaluation): Jev chấm từng cặp query–passage trên shortlist sau first-stage search; cookbook TypeSafe (docs.typesafe.ai): classifying RAG passages, rerank, LLM guardrails.
- Daily Dose of Data Science — "Jev for RAG, clearly explained": luồng hybrid search → RRF top-20 ưu tiên recall → Jev chấm xác suất từng ứng viên → ngưỡng quyết định đoạn vào generation; lưu ý retrieval là trần chất lượng của cả hệ.
- Bài audit cài đặt "Passage Gate" và "jev-reranker" (thegeocommunity.com): 4 câu hỏi chấm cho mỗi ứng viên và các ngưỡng tham khảo dùng trong spec này; phân biệt rerank (sắp xếp tương đối) với gate (quyết định có cho vào generation hay không).
- Dev.to — "Jev in Depth: Can It Reshape Agent Search?": Jev thắng GPT-4o-mini nhưng kém GPT-5-mini ở bài lọc quan hệ đa bước (MuSiQue/HotpotQA) — cơ sở đặt giả thuyết H3 dè dặt và chọn Pipeline C làm đối chứng nghiêm túc.
- Analytics Vidhya — "Build a RAG Hallucination Filter: Using JEV for Best-of-N Checking": ghi nhận ca gate quá nghiêm chặn nhầm đáp án đúng và cơ chế chuyển cấp — cơ sở cho chỉ số abstain sai và chế độ shadow (ER-005).

*Lưu ý độ tin cậy: ngoài tài liệu TypeSafe, các nguồn trên là bài phân tích/audit của bên thứ ba, số liệu chưa kiểm chứng — repo này tồn tại chính để tự đo lại các khẳng định đó.*
