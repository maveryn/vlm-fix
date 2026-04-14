from __future__ import annotations

from typing import Any, Callable, Dict

DEFAULT_MAX_NEW_TOKENS = 4096


def _resolve_model_class(model_cls_name: str):
    if model_cls_name == "Qwen2_5_VL":
        from eval.models import Qwen2_5_VL

        return Qwen2_5_VL
    if model_cls_name == "InternVL":
        from eval.models import InternVL

        return InternVL
    if model_cls_name == "Molmo2_VL":
        from eval.models import Molmo2_VL

        return Molmo2_VL
    if model_cls_name == "OpenAIChatGPT_VL":
        from eval.models import OpenAIChatGPT_VL

        return OpenAIChatGPT_VL
    if model_cls_name == "Claude_VL":
        from eval.models import Claude_VL

        return Claude_VL
    raise ValueError(f"Unknown model class: {model_cls_name}")


def _factory(model_cls_name: str, model_path: str, defaults: Dict[str, Any]) -> Callable[..., object]:
    def _make(**overrides):
        cls = _resolve_model_class(model_cls_name)
        kwargs = dict(defaults)
        kwargs.update(overrides)
        kwargs["model_path"] = model_path
        return cls(**kwargs)

    return _make


def _common_defaults() -> Dict[str, Any]:
    return {
        "use_vllm": True,
        "require_vllm": True,
        "vllm_batch_size": 128,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": 0.0,
    }


def _openai_defaults() -> Dict[str, Any]:
    return {
        "vllm_batch_size": 1,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": 0.0,
    }


def _anthropic_defaults() -> Dict[str, Any]:
    return {
        "vllm_batch_size": 1,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": 0.0,
    }


OPEN_10_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-14B",
    "allenai/Molmo2-4B",
    "allenai/Molmo2-8B",
]

API_4_MODELS = [
    "gpt-4.1",
    "gpt-5.2",
    "claude-sonnet-4-0",
    "claude-sonnet-4-5",
]

MODEL_REGISTRY: Dict[str, Callable[..., object]] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen2.5-VL-3B-Instruct", _common_defaults()),
    "Qwen2.5-VL-3B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen2.5-VL-3B-Instruct", _common_defaults()),
    "Qwen/Qwen2.5-VL-7B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen2.5-VL-7B-Instruct", _common_defaults()),
    "Qwen2.5-VL-7B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen2.5-VL-7B-Instruct", _common_defaults()),
    "Qwen/Qwen3-VL-4B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen3-VL-4B-Instruct", _common_defaults()),
    "Qwen3-VL-4B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen3-VL-4B-Instruct", _common_defaults()),
    "Qwen/Qwen3-VL-8B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen3-VL-8B-Instruct", _common_defaults()),
    "Qwen3-VL-8B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen3-VL-8B-Instruct", _common_defaults()),
    "Qwen/Qwen3-VL-32B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen3-VL-32B-Instruct", {**_common_defaults(), "max_model_len": 8192}),
    "Qwen3-VL-32B-Instruct": _factory("Qwen2_5_VL", "Qwen/Qwen3-VL-32B-Instruct", {**_common_defaults(), "max_model_len": 8192}),
    "OpenGVLab/InternVL3_5-4B": _factory("InternVL", "OpenGVLab/InternVL3_5-4B", _common_defaults()),
    "InternVL3_5-4B": _factory("InternVL", "OpenGVLab/InternVL3_5-4B", _common_defaults()),
    "OpenGVLab/InternVL3_5-8B": _factory("InternVL", "OpenGVLab/InternVL3_5-8B", _common_defaults()),
    "InternVL3_5-8B": _factory("InternVL", "OpenGVLab/InternVL3_5-8B", _common_defaults()),
    "OpenGVLab/InternVL3_5-14B": _factory("InternVL", "OpenGVLab/InternVL3_5-14B", _common_defaults()),
    "InternVL3_5-14B": _factory("InternVL", "OpenGVLab/InternVL3_5-14B", _common_defaults()),
    "allenai/Molmo2-4B": _factory("Molmo2_VL", "allenai/Molmo2-4B", {**_common_defaults(), "max_num_batched_tokens": 65536}),
    "Molmo2-4B": _factory("Molmo2_VL", "allenai/Molmo2-4B", {**_common_defaults(), "max_num_batched_tokens": 65536}),
    "allenai/Molmo2-8B": _factory("Molmo2_VL", "allenai/Molmo2-8B", {**_common_defaults(), "max_num_batched_tokens": 65536}),
    "Molmo2-8B": _factory("Molmo2_VL", "allenai/Molmo2-8B", {**_common_defaults(), "max_num_batched_tokens": 65536}),
    "gpt-4.1": _factory("OpenAIChatGPT_VL", "gpt-4.1", _openai_defaults()),
    "openai/gpt-4.1": _factory("OpenAIChatGPT_VL", "gpt-4.1", _openai_defaults()),
    "gpt-5.2": _factory("OpenAIChatGPT_VL", "gpt-5.2", _openai_defaults()),
    "openai/gpt-5.2": _factory("OpenAIChatGPT_VL", "gpt-5.2", _openai_defaults()),
    "claude-sonnet-4-0": _factory("Claude_VL", "claude-sonnet-4-0", _anthropic_defaults()),
    "anthropic/claude-sonnet-4-0": _factory("Claude_VL", "claude-sonnet-4-0", _anthropic_defaults()),
    "claude-sonnet-4-5": _factory("Claude_VL", "claude-sonnet-4-5", _anthropic_defaults()),
    "anthropic/claude-sonnet-4-5": _factory("Claude_VL", "claude-sonnet-4-5", _anthropic_defaults()),
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())
