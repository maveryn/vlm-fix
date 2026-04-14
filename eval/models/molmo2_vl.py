from __future__ import annotations

import os
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import BaseModel


def _build_inline_prompt(message: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for item in message:
        item_type = item["type"]
        if item_type == "image":
            parts.append("<|image|>")
        elif item_type == "text":
            parts.append(item["value"])
        else:
            raise ValueError(f"Unsupported content type: {item_type}")
    return "\n".join([p for p in parts if p]).strip()


class Molmo2_VL(BaseModel):
    """Molmo2 wrapper for VLM-Fix benchmark inference, with vLLM-first execution."""

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        vllm_batch_size: int = 128,
        max_model_len: int = 32768,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        gpu_memory_utilization: float = 0.92,
        require_vllm: bool = False,
        max_num_batched_tokens: int = 65536,
        max_crops: int = 36,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.use_vllm = bool(use_vllm)
        self.vllm_batch_size = int(vllm_batch_size)
        self.max_model_len = int(max_model_len)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.require_vllm = bool(require_vllm)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.max_crops = int(max_crops)
        self.verbose = bool(verbose)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.llm = None
        self.model = None
        if self.use_vllm:
            mp_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
            if not mp_method:
                mp_method = "spawn"
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = mp_method
            try:
                import multiprocessing as mp

                mp.set_start_method(mp_method, force=True)
            except RuntimeError:
                pass
            try:
                from vllm import LLM

                tp_size = int(os.environ.get("VLM_FIX_TP_SIZE", "1"))
                if tp_size < 1:
                    tp_size = 1

                self.llm = LLM(
                    model=model_path,
                    max_num_seqs=self.vllm_batch_size,
                    max_model_len=self.max_model_len,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    trust_remote_code=True,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[warn] vLLM initialization failed for {model_path}, falling back to HF: {exc}")
                if self.require_vllm:
                    raise RuntimeError(f"vLLM initialization failed for {model_path}: {exc}") from exc
                self.use_vllm = False

        if not self.use_vllm:
            self._init_hf_model()

    def _init_hf_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def _build_chat_prompt(self, message: List[Dict[str, str]]) -> str:
        inline = _build_inline_prompt(message)
        chat = [{"role": "user", "content": inline}]
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, (list, tuple)):
            return str(prompt[0])
        return str(prompt)

    def _prepare_vllm_images(self, image_paths: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))
        return images

    def _generate_single_hf(self, message: List[Dict[str, str]]) -> str:
        from transformers import GenerationConfig

        prompt = self._build_chat_prompt(message)
        image_paths = [x["value"] for x in message if x["type"] == "image"]
        images = [Image.open(p).convert("RGB") for p in image_paths]

        inputs = self.processor.process(
            images=images if images else None,
            text=prompt,
            images_kwargs={"max_crops": self.max_crops},
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=self.max_new_tokens, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
            )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        decoded = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return str(decoded)

    def generate(self, message: List[Dict[str, str]], dataset: str | None = None) -> str:
        if self.use_vllm:
            return self.generate_batch([message], dataset=dataset)[0]
        return self._generate_single_hf(message)

    def generate_batch(self, messages: List[List[Dict[str, str]]], dataset: str | None = None) -> List[str]:
        if not self.use_vllm:
            return [self._generate_single_hf(m) for m in messages]

        from vllm import SamplingParams

        requests = []
        for idx, message in enumerate(messages):
            image_paths = [x["value"] for x in message if x["type"] == "image"]
            req: Dict[str, object] = {
                "prompt": self._build_chat_prompt(message),
                "request_id": str(idx),
            }
            if image_paths:
                req["multi_modal_data"] = {"image": self._prepare_vllm_images(image_paths)}
            requests.append(req)

        sampling = SamplingParams(
            temperature=max(0.0, self.temperature),
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.llm.generate(requests, sampling_params=sampling, use_tqdm=False)

        out_map: Dict[str, str] = {}
        ordered: List[str] = []
        for out in outputs:
            rid = getattr(out, "request_id", None)
            text = out.outputs[0].text.strip() if out.outputs else ""
            ordered.append(text)
            if rid is not None:
                out_map[str(rid)] = text

        expected = {str(i) for i in range(len(messages))}
        if out_map and set(out_map.keys()) == expected:
            return [out_map[str(i)] for i in range(len(messages))]
        return ordered
