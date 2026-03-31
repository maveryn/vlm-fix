from __future__ import annotations

from typing import List, Dict


class BaseModel:
    """Minimal base model interface aligned with vlmeval inference usage."""

    def __init__(self) -> None:
        self.use_vllm = False
        self.vllm_batch_size = 1

    def generate(self, message: List[Dict[str, str]], dataset: str | None = None) -> str:
        raise NotImplementedError

    def generate_batch(self, messages: List[List[Dict[str, str]]], dataset: str | None = None) -> List[str]:
        return [self.generate(m, dataset=dataset) for m in messages]

    def generate_batch_with_meta(
        self, messages: List[List[Dict[str, str]]], dataset: str | None = None
    ) -> tuple[List[str], List[Dict[str, object]]]:
        preds = self.generate_batch(messages, dataset=dataset)
        metas: List[Dict[str, object]] = [{} for _ in preds]
        return preds, metas
