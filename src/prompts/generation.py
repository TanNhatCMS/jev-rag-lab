"""Prompt generation: chỉ trả lời từ ngữ cảnh được cấp, kèm trích dẫn.

Quy ước bắt buộc với model:
- Mỗi ý trong câu trả lời phải kèm trích dẫn dạng [tên tài liệu].
- Nếu ngữ cảnh không đủ để trả lời, model phải trả về đúng chuỗi "ABSTAIN"
  (và không thêm gì khác) để pipeline nhận diện và chuyển sang trạng thái abstain.
"""

from __future__ import annotations

ABSTAIN_TOKEN = "ABSTAIN"

_HEADER = """Bạn là trợ lý trả lời câu hỏi dựa HOÀN TOÀN trên ngữ cảnh được cấp dưới đây.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng thông tin có trong ngữ cảnh. Không suy đoán, không dùng kiến thức bên ngoài.
2. Mỗi ý trong câu trả lời phải kèm trích dẫn dạng [tên tài liệu] lấy từ nhãn của đoạn ngữ cảnh chứa ý đó.
3. Nếu ngữ cảnh không chứa thông tin đủ để trả lời câu hỏi, hãy trả lời đúng một chuỗi duy nhất: ABSTAIN
"""


def build_generation_prompt(query: str, contexts: list[tuple[str, str]]) -> str:
    """Dựng prompt generation từ câu hỏi và danh sách (nhãn trích dẫn, nội dung đoạn).

    Mỗi đoạn được trình bày dạng:
        [i] <nhãn>
        <nội dung>
    để MockLLM và người đọc đều trích được đoạn đầu tiên một cách xác định.
    """
    blocks = []
    for i, (label, text) in enumerate(contexts, start=1):
        blocks.append(f"[{i}] {label}\n{text}")
    context_text = "\n\n".join(blocks) if blocks else "(không có ngữ cảnh)"
    return f"{_HEADER}\nNGỮ CẢNH:\n{context_text}\n\nCÂU HỎI: {query}\n\nTRẢ LỜI:"
