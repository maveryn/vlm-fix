from __future__ import annotations

import base64
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from .base import BaseModel


def _image_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_text_from_response(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    outputs = getattr(resp, "output", None) or []
    chunks: List[str] = []
    for item in outputs:
        content = getattr(item, "content", None) or []
        for part in content:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())
    return "\n".join(chunks).strip()


def _extract_meta_from_response(resp) -> Dict[str, object]:
    usage = getattr(resp, "usage", None)
    output_tokens = None
    if usage is not None:
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)

    incomplete = getattr(resp, "incomplete_details", None)
    stop_reason = None if incomplete is None else str(incomplete)
    finish_reason = getattr(resp, "status", None)
    return {
        "finish_reason": None if finish_reason is None else str(finish_reason),
        "stop_reason": stop_reason,
        "output_tokens": None if output_tokens is None else int(output_tokens),
    }


class OpenAIChatGPT_VL(BaseModel):
    """OpenAI API wrapper for vision-capable GPT models used in the release package."""

    @staticmethod
    def _supports_reasoning_effort(model_path: str) -> bool:
        model_id = str(model_path).split("/")[-1].lower()
        return model_id.startswith("gpt-5")

    def __init__(
        self,
        model_path: str,
        vllm_batch_size: int = 1,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 6,
        request_max_attempts: int = 5,
        retry_backoff_sec: float = 1.5,
        api_concurrency: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.vllm_batch_size = int(vllm_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        env_reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT")
        resolved_reasoning_effort = reasoning_effort if reasoning_effort is not None else env_reasoning_effort
        if resolved_reasoning_effort is not None:
            normalized = str(resolved_reasoning_effort).strip().lower()
            if normalized in {"", "none", "off", "disable", "disabled", "null"}:
                resolved_reasoning_effort = None
            else:
                resolved_reasoning_effort = normalized
        if resolved_reasoning_effort is not None and not self._supports_reasoning_effort(model_path):
            # Guardrail: ignore reasoning effort for models that don't support it (e.g., gpt-4.1).
            resolved_reasoning_effort = None
        self.reasoning_effort = resolved_reasoning_effort
        self.request_max_attempts = int(request_max_attempts)
        self.retry_backoff_sec = float(retry_backoff_sec)
        requested_concurrency = self.vllm_batch_size if api_concurrency is None else int(api_concurrency)
        self.api_concurrency = max(1, requested_concurrency)

        # Auto-load .env from current working tree if present.
        # This lets OPENAI_API_KEY and related vars be picked up without shell export.
        dotenv_path = find_dotenv(filename=".env", usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise RuntimeError(
                f"Missing OpenAI API key. Set {api_key_env} in env or pass api_key to the model constructor."
            )

        client_kwargs: Dict[str, object] = {
            "api_key": resolved_key,
            "timeout": float(timeout),
            "max_retries": int(max_retries),
        }
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        org = os.environ.get("OPENAI_ORG_ID")
        project = os.environ.get("OPENAI_PROJECT_ID")
        if org:
            client_kwargs["organization"] = org
        if project:
            client_kwargs["project"] = project

        self.client = OpenAI(**client_kwargs)

    def _build_input(self, message: List[Dict[str, str]]) -> List[Dict[str, object]]:
        content: List[Dict[str, object]] = []
        for item in message:
            item_type = item["type"]
            item_value = item["value"]
            if item_type == "text":
                content.append({"type": "input_text", "text": str(item_value)})
            elif item_type == "image":
                content.append({"type": "input_image", "image_url": _image_to_data_url(str(item_value))})
            else:
                raise ValueError(f"Unsupported content type: {item_type}")
        return [{"role": "user", "content": content}]

    def _create_response(self, message: List[Dict[str, str]]) -> Tuple[str, Dict[str, object]]:
        payload: Dict[str, object] = {
            "model": self.model_path,
            "input": self._build_input(message),
            "max_output_tokens": self.max_new_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        # Keep deterministic default behavior aligned with the existing eval stack.
        if self.temperature > 0:
            payload["temperature"] = self.temperature

        attempt = 0
        while True:
            try:
                resp = self.client.responses.create(**payload)
                text = _extract_text_from_response(resp)
                meta = _extract_meta_from_response(resp)
                return text, meta
            except Exception as exc:  # pragma: no cover - API/network runtime path
                name = type(exc).__name__
                non_retryable = {
                    "BadRequestError",
                    "AuthenticationError",
                    "PermissionDeniedError",
                    "NotFoundError",
                    "ConflictError",
                }
                if name in non_retryable:
                    raise
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
