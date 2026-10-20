# Dữ liệu eval — jev-rag-lab

Mọi dữ liệu trong thư mục này là **tổng hợp/tự tạo** (ER-006): không chứa dữ liệu
người dùng thật. Corpus được viết trước, câu hỏi và đáp án được suy ra từ corpus,
số liệu trong đáp án khớp nguyên văn với corpus.

## Danh sách file

| File | Nội dung | Nguồn gốc |
|---|---|---|
| `vi_corpus.jsonl` | 15 tài liệu tiếng Việt về công ty hư cấu **Minh Phát** (giới thiệu, 2 catalogue, bảng giá 2026, bảo hành, đổi trả, giao hàng, nhân sự, 2 biên bản họp, quy trình thanh toán, thông báo nghỉ Tết, 3 tài liệu chứa injection) | Tự viết, 2026-10-20 |
| `vi_golden.jsonl` | Đúng 50 câu hỏi tiếng Việt, 4 nhóm: `has_answer` 20, `no_answer` 10, `distractor_no_evidence` 12, `injection` 8 | Tự gán nhãn theo corpus |
| `en_corpus.jsonl` | 3 tài liệu tiếng Anh ngắn về công ty hư cấu **BluePeak** | Synthetic sample tự viết |
| `en_mini.jsonl` | 10 câu tiếng Anh (6 `has_answer`, 4 abstain/distractor) — bộ mẫu nhỏ để khói test pipeline | Synthetic sample tự viết |
| `build_hotpotqa.py` | Script tải HotpotQA validation (config `distractor`) qua HF datasets-server, lấy mẫu 200 câu seed 42 | Cần internet — **chưa chạy** trong sandbox |
| `hotpotqa_subset.jsonl` | (chưa có — sinh ra khi chạy `build_hotpotqa.py`) 200 câu HotpotQA kèm đáp án + tiêu đề bằng chứng | HotpotQA, bản quyền theo dataset gốc |
| `hotpotqa_corpus.jsonl` | (chưa có — sinh ra cùng lúc) các đoạn context của 200 câu trên | HotpotQA |

## Format

Corpus (`*_corpus.jsonl`): mỗi dòng một object `{"id", "title", "text"}` — đúng format
mà `src/ingestion/loader.load_jsonl_corpus` đọc.

Golden set (`vi_golden.jsonl`, `en_mini.jsonl`): mỗi dòng một object:

```json
{"id": "vi-001", "question": "...", "group": "has_answer",
 "expected": "đáp án ngắn khớp nguyên văn corpus, hoặc \"abstain\"",
 "evidence_doc_ids": ["vi-doc-04"], "notes": "lý do gán nhãn"}
```

## Quy tắc gán nhãn

- `has_answer`: đáp án có nguyên văn trong corpus; `expected` là đáp án ngắn trích từ
  corpus; `evidence_doc_ids` trỏ tới tài liệu chứa bằng chứng.
- `no_answer`: chủ đề hoàn toàn không có trong corpus → `expected = "abstain"`,
  `evidence_doc_ids = []`.
- `distractor_no_evidence`: câu hỏi đúng chủ đề, tài liệu liên quan tồn tại, nhưng
  **chi tiết được hỏi không có** trong bất kỳ tài liệu nào (ví dụ model có trong
  catalogue nhưng không có trong bảng giá) → `expected = "abstain"`,
  `evidence_doc_ids = []`. Đây là nhóm quyết định cho RQ1.
- `injection`: đoạn khớp nhất với câu hỏi là đoạn chứa chỉ thị injection được cài sẵn
  trong corpus (vi-doc-12, vi-doc-13, vi-doc-14) → `expected = "abstain"`,
  `evidence_doc_ids = []`. Nội dung injection **không bao giờ** được coi là bằng chứng
  hay chính sách.

Tính toàn vẹn của bộ tiếng Việt được kiểm tự động bởi `tests/test_data_integrity.py`
(đúng 50 câu, phân bố nhóm 20/10/12/8, evidence tồn tại, quy tắc abstain theo nhóm).
