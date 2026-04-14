from __future__ import annotations

import os
from typing import Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import BaseModel


def _ensure_image_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://") or path.startswith("file://"):
        return path
    if os.path.exists(path):
        return "file://" + os.path.abspath(path)
    raise FileNotFoundError(path)


def _prepare_content(message: List[Dict[str, str]]) -> List[Dict[str, str]]:
    content = []
    for item in message:
        t = item["type"]
        v = item["value"]
        if t == "image":
            content.append({"type": "image", "image": _ensure_image_url(v)})
        elif t == "text":
            content.append({"type": "text", "text": v})
        else:
            raise ValueError(f"Unsupported content type: {t}")
    return content


class Qwen2_5_VL(BaseModel):
    """Minimal Qwen2.5-VL wrapper with vlmeval-style batch inference."""

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        vllm_batch_size: int = 128,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 32768,
        max_num_batched_tokens: int | None = None,
        require_vllm: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.use_vllm = bool(use_vllm)
        self.vllm_batch_size = int(vllm_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_model_len = int(max_model_len)
        self.max_num_batched_tokens = (
            int(max_num_batched_tokens) if max_num_batched_tokens is not None else None
        )
        self.require_vllm = bool(require_vllm)
        self.verbose = bool(verbose)

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        except TypeError:
            # Newer Transformers may drop fix_mistral_regex in tokenizer backend.
            self.processor = AutoProcessor.from_pretrained(model_path)

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

                llm_kwargs: Dict[str, object] = {
                    "model": model_path,
                    "max_num_seqs": self.vllm_batch_size,
                    "max_model_len": self.max_model_len,
                    "tensor_parallel_size": tp_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "trust_remote_code": True,
                }
                if self.max_num_batched_tokens is not None:
                    llm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
                # Qwen2.5-VL-3B expects tied output embeddings in vLLM init.
                # Applying tie_word_embeddings=False causes missing lm_head weights.
                if "qwen2.5-vl-3b-instruct" not in model_path.lower():
                    llm_kwargs["hf_overrides"] = {"tie_word_embeddings": False}

                self.llm = LLM(
                    **llm_kwargs,
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
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

    def _format_chat(self, message: List[Dict[str, str]]) -> List[Dict[str, object]]:
        return [{"role": "user", "content": _prepare_content(message)}]

    def _cast_inputs_to_model_dtype(self, inputs) -> None:
        model_dtype = next(self.model.parameters()).dtype
        for key, value in list(vars(inputs).items()):
            if torch.is_tensor(value) and value.dtype.is_floating_point and value.dtype != model_dtype:
                setattr(inputs, key, value.to(model_dtype))

    def _generate_batch_hf(self, messages: List[List[Dict[str, str]]]) -> List[str]:
        from qwen_vl_utils import process_vision_info

        outputs_all: List[str] = []
        # Conservative micro-batch to avoid OOM while still accelerating over per-item generation.
        micro_batch = max(1, min(8, len(messages)))
        for start in range(0, len(messages), micro_batch):
            chunk = messages[start : start + micro_batch]
            convos = [self._format_chat(m) for m in chunk]
            texts = []
            for convo in convos:
                text = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
                texts.append(text[0] if isinstance(text, (list, tuple)) else text)
            images, videos = process_vision_info(convos)

            inputs = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")
            self._cast_inputs_to_model_dtype(inputs)

            if self.temperature > 0:
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )
            else:
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated_trim = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            decoded = self.processor.tokenizer.batch_decode(
                generated_trim,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs_all.extend([x.strip() for x in decoded])
        return outputs_all

    def generate(self, message: List[Dict[str, str]], dataset: str | None = None) -> str:
        if self.use_vllm:
            return self.generate_batch_with_meta([message], dataset=dataset)[0][0]
        return self._generate_batch_hf([message])[0]

    def generate_batch(self, messages: List[List[Dict[str, str]]], dataset: str | None = None) -> List[str]:
        return self.generate_batch_with_meta(messages, dataset=dataset)[0]

    def generate_batch_with_meta(
        self, messages: List[List[Dict[str, str]]], dataset: str | None = None
    ) -> tuple[List[str], List[Dict[str, object]]]:
        if not self.use_vllm:
            texts = self._generate_batch_hf(messages)
            metas: List[Dict[str, object]] = []
            tokenizer = self.processor.tokenizer
            for text in texts:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                metas.append(
                    {
                        "finish_reason": None,
                        "stop_reason": None,
                        "output_tokens": int(len(token_ids)),
                    }
                )
            return texts, metas

        from qwen_vl_utils import process_vision_info
        from vllm import SamplingParams

        requests = []
        for idx, message in enumerate(messages):
            convo = self._format_chat(message)
            text = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            prompt = text[0] if isinstance(text, (list, tuple)) else text
            images, videos = process_vision_info(convo)

            req: Dict[str, object] = {
                "prompt": prompt,
                "request_id": str(idx),
            }
            if images:
                req["multi_modal_data"] = {"image": images}
            elif videos:
                req["multi_modal_data"] = {"video": videos}
            requests.append(req)

        sampling = SamplingParams(
            temperature=max(0.0, self.temperature),
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

        outputs = self.llm.generate(requests, sampling_params=sampling, use_tqdm=False)
        out_map: Dict[str, tuple[str, Dict[str, object]]] = {}
        ordered: List[str] = []
        ordered_meta: List[Dict[str, object]] = []
        for o in outputs:
            rid = getattr(o, "request_id", None)
            first = o.outputs[0] if o.outputs else None
            text = first.text.strip() if first is not None else ""
            token_ids = getattr(first, "token_ids", None) if first is not None else None
            meta = {
                "finish_reason": getattr(first, "finish_reason", None) if first is not None else None,
                "stop_reason": getattr(first, "stop_reason", None) if first is not None else None,
                "output_tokens": int(len(token_ids)) if token_ids is not None else None,
            }
            ordered.append(text)
            ordered_meta.append(meta)
            if rid is None:
                continue
            out_map[str(rid)] = (text, meta)

        expected = {str(i) for i in range(len(messages))}
        if out_map and set(out_map.keys()) == expected:
            texts = [out_map[str(i)][0] for i in range(len(messages))]
            metas = [out_map[str(i)][1] for i in range(len(messages))]
            return texts, metas
        return ordered, ordered_meta
