# jev-rag-lab

Repo **nghiên cứu** (không phải repo sản phẩm): kiểm chứng bằng thí nghiệm xem việc chèn một
**tầng quyết định kiểu Jev (rerank + gate)** giữa retrieval và LLM có thực sự làm RAG
tốt hơn hay không — ít trả lời bịa hơn, đổi lại bao nhiêu độ trễ và chi phí.

Xuất phát điểm: 5 hình phân tích ngày 2026-10-20 (lưu tại `docs/source-images/`):

1. `01-rag-vs-jev-rag.jpg` — sơ đồ RAG thường vs Jev + RAG (thêm tầng Rerank + Gate và nhánh Abstain).
2. `02-top-5-rag-architectures.jpg` — Hybrid / Graph / Agentic / Corrective (CRAG) / Multimodal RAG.
3. `03-top-9-places-to-use-jev.jpg` — 9 vị trí đặt model quyết định Jev trong hệ thống.
4. `04-rag-project-structure.jpg` — cấu trúc project RAG bằng Python (khung cho repo này).
5. `05-langchain-rag-project-structure.jpg` — biến thể LangChain (chỉ tham khảo, repo này không khóa vào LangChain).

## Câu hỏi nghiên cứu

- **RQ1**: Gate theo bằng chứng (evidence) có giảm được trả lời bịa so với RAG thường —
  khi retrieval trả về đoạn đúng chủ đề nhưng không chứa đáp án?
- **RQ2**: Cái giá phải trả là gì — độ trễ cộng thêm và chi phí mỗi lượt hỏi?
- **RQ3**: Jev API (TypeSafe, hosted) chấm tốt hơn hay kém hơn một model nhỏ tự cài
  (qua AI Gateway) trên cùng một bộ dữ liệu?
- **RQ4**: Các cơ chế trên hoạt động thế nào với **tiếng Việt**? (Tài liệu công bố về Jev
  gần như toàn tiếng Anh — đây là khoảng trống repo này tự đo.)

## Tài liệu

- Đặc tả nghiên cứu đầy đủ: [`specs/001-jev-rerank-gate/spec.md`](specs/001-jev-rerank-gate/spec.md)
  — phân tích 5 hình, thiết kế 3 pipeline đối chứng (baseline / Jev / model nhỏ),
  bộ dữ liệu eval, chỉ số đo và tiêu chí kết luận go/no-go.

## Cấu trúc repo (theo Hình 4, bổ sung module `decision/`)

```
src/
  ingestion/    Nạp tài liệu (PDF/CSV/web...)
  chunking/     Cắt văn bản thành chunk
  embeddings/   Sinh vector embedding
  vectorDB/     Thao tác vector store (FAISS/Chroma/pgvector)
  retrieval/    Truy xuất ứng viên (first-stage, ưu tiên recall)
  decision/     ★ Tầng quyết định: scorer (Jev / model nhỏ) + gate (ngưỡng, abstain)
  prompts/      Prompt templates
  llm/          Client gọi LLM generation
  api/          API endpoints (FastAPI)
  utils/        Hàm tiện ích dùng chung
tests/          Unit + integration tests
evals/          Golden set + script chạy eval, báo cáo kết quả
```

## Trạng thái

- 2026-10-20: Khởi tạo repo + đặc tả nghiên cứu 001 (bản nháp, còn câu hỏi mở trong mục "Làm rõ" của spec).
- 2026-10-20: **Vòng 1 (offline) hoàn tất** — scaffold + 3 pipeline + eval harness đã chạy đầu-cuối, 52/52 test pass. Headline (chỉ là hành vi của thành phần offline — HashingEmbedder + HeuristicScorer + MockLLM, không phải model thật): C-heuristic abstain đúng nhóm distractor 75% và nhóm injection 75% (baseline: 58.3% / 25%) nhưng abstain sai tới 60% câu có đáp án; Pipeline B chưa chạy (thiếu `JEV_API_KEY`). **Chưa kết luận GO/NO-GO.** Tổng kết đầy đủ: [`docs/findings-round1.md`](docs/findings-round1.md); số liệu thô tại `evals/reports/` + `evals/results/` (gồm cả quét hiệu chuẩn ngưỡng `sweep-*`).
- Bước tiếp theo (vòng 2) bị chặn bởi 3 đầu vào ngoài sandbox: `JEV_API_KEY`, một endpoint LLM + embedding thật, và đường mạng để lấy benchmark HotpotQA — chi tiết ở mục 6 của `docs/findings-round1.md`.

## Ghi chú

- Bản GitHub chưa chứa 5 file JPG trong `docs/source-images/` (công cụ đẩy file hiện tại chỉ xử lý file text) — bản local của repo có đủ 5 ảnh gốc.

- Kết quả nghiên cứu (nếu đạt tiêu chí go trong spec) mới là đầu vào để cân nhắc đưa tầng
  gate vào sản phẩm thật. Repo này không chứa dữ liệu người dùng và không kết nối hệ thống thật.
- Số liệu hiệu năng về Jev trích trong spec là từ tài liệu nhà cung cấp/bên thứ ba —
  chưa kiểm chứng; toàn bộ mục đích của repo là tự đo lại.
