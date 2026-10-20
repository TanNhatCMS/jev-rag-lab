"""HttpLLM: gọi endpoint chat tương thích OpenAI qua urllib (stdlib).

LƯU Ý: chưa kiểm chứng với dịch vụ thật (sandbox không mạng/không key).
Chỉ chạy được khi đã có LLM_API_BASE / LLM_API_KEY / LLM_MODEL trong môi trường.
"""

from __future__ import annotations

import json
import os
import urllib.request


class HttpLLM:
    """POST {base}/chat/completions với {"model", "messages"} kiểu OpenAI."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "HttpLLM":
        """Tạo client từ biến môi trường LLM_API_BASE / LLM_API_KEY / LLM_MODEL."""
        base = os.environ.get("LLM_API_BASE", "")
        key = os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("LLM_MODEL", "")
        if not (base and key and model):
            raise RuntimeError(
                "Thiếu LLM_API_BASE / LLM_API_KEY / LLM_MODEL trong môi trường"
            )
        return cls(base_url=base, api_key=key, model=model)

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])
