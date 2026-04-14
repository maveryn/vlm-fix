from __future__ import annotations

import base64
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import find_dotenv, load_dotenv

from .base import BaseModel


def _image_to_base64_payload(image_path: str) -> Tuple[str, str]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)
    media_type, _ = mimetypes.guess_type(str(p))
    if media_type is None:
        media_type = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return media_type, b64


def _normalize_api_url(url: str) -> str:
    u = url.strip()
    if not u:
        return "https://api.anthropic.com/v1/messages"
    if "://" not in u:
        u = f"https://{u}"
    if u.endswith("/v1/messages"):
        return u
    if u.endswith("/v1"):
        return f"{u}/messages"
    if u.endswith("/"):
        u = u[:-1]
    return f"{u}/v1/messages"


def _extract_text_from_payload(payload: Dict[str, object]) -> str:
    content = payload.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            t = item.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts).strip()


def _extract_meta_from_payload(payload: Dict[str, object]) -> Dict[str, object]:
    usage = payload.get("usage", {})
    output_tokens = None
    if isinstance(usage, dict):
        val = usage.get("output_tokens")
        if isinstance(val, int):
            output_tokens = val
    stop_reason = payload.get("stop_reason")
    sr = None if stop_reason is None else str(stop_reason)
    return {
        "finish_reason": sr,
        "stop_reason": sr,
        "output_tokens": output_tokens,
    }


class Claude_VL(BaseModel):
    """Anthropic Claude vision wrapper for messages API (e.g., claude-sonnet-4-5)."""

    def __init__(
        self,
        model_path: str,
        vllm_batch_size: int = 1,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        api_url: str | None = None,
        timeout: float = 120.0,
        request_max_attempts: int = 5,
        retry_backoff_sec: float = 1.5,
        api_concurrency: int | None = None,
        anthropic_version: str = "2023-06-01",
        **_: object,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.vllm_batch_size = int(vllm_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.request_max_attempts = int(request_max_attempts)
        self.retry_backoff_sec = float(retry_backoff_sec)
        self.timeout = float(timeout)
        requested_concurrency = self.vllm_batch_size if api_concurrency is None else int(api_concurrency)
        self.api_concurrency = max(1, requested_concurrency)

        # Auto-load .env from the current working tree if present.
        dotenv_path = find_dotenv(filename=".env", usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise RuntimeError(
                f"Missing Anthropic API key. Set {api_key_env} in env or pass api_key to the model constructor."
            )

        raw_url = (
            api_url
            or os.environ.get("ANTHROPIC_API_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )
        self.api_url = _normalize_api_url(str(raw_url))
        self.session = requests.Session()
        self.headers = {
            "x-api-key": resolved_key,
            "anthropic-version": str(anthropic_version),
            "content-type": "application/json",
        }

    def _build_messages(self, message: List[Dict[str, str]]) -> List[Dict[str, object]]:
        content: List[Dict[str, object]] = []
        for item in message:
            item_type = item["type"]
            item_value = item["value"]
            if item_type == "text":
                txt = str(item_value)
                if txt:
                    content.append({"type": "text", "text": txt})
            elif item_type == "image":
                media_type, b64 = _image_to_base64_payload(str(item_value))
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported content type: {item_type}")
        return [{"role": "user", "content": content}]

    def _create_response(self, message: List[Dict[str, str]]) -> Tuple[str, Dict[str, object]]:
        payload: Dict[str, object] = {
            "model": self.model_path,
            "max_tokens": self.max_new_tokens,
            "messages": self._build_messages(message),
        }
        if self.temperature > 0:
            payload["temperature"] = self.temperature

        attempt = 0
        while True:
            try:
                resp = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                status = int(resp.status_code)
                if status >= 400:
                    # Retry on transient server/rate-limit errors.
                    if status in (408, 409, 429) or status >= 500:
                        attempt += 1
                        if attempt >= self.request_max_attempts:
                            resp.raise_for_status()
                        sleep_sec = min(20.0, self.retry_backoff_sec * (2 ** (attempt - 1)))
                        time.sleep(sleep_sec)
                        continue
                    resp.raise_for_status()

                data = resp.json()
                text = _extract_text_from_payload(data)
                meta = _extract_meta_from_payload(data)
                return text, meta
            except requests.RequestException:
                attempt += 1
                if attempt >= self.request_max_attempts:
                    raise
                sleep_sec = min(20.0, self.retry_backoff_sec * (2 ** (attempt - 1)))
                time.sleep(sleep_sec)

    def generate(self, message: List[Dict[str, str]], dataset: str | None = None) -> str:
        text, _ = self._create_response(message)
        return text

    def generate_batch(
        self, messages: List[List[Dict[str, str]]], dataset: str | None = None
    ) -> List[str]:
        preds, _ = self.generate_batch_with_meta(messages, dataset=dataset)
        return preds

    def generate_batch_with_meta(
        self, messages: List[List[Dict[str, str]]], dataset: str | None = None
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        if not messages:
            return [], []

        n = len(messages)
        preds: List[str] = [""] * n
        metas: List[Dict[str, object]] = [{} for _ in range(n)]
        workers = max(1, min(self.api_concurrency, n))

        if workers == 1:
            for i, msg in enumerate(messages):
                text, meta = self._create_response(msg)
                preds[i] = text
                metas[i] = meta
            return preds, metas

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_idx = {pool.submit(self._create_response, msg): i for i, msg in enumerate(messages)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                text, meta = fut.result()
                preds[i] = text
                metas[i] = meta
        return preds, metas
