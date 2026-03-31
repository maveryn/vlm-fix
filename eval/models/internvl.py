from __future__ import annotations

import os
from typing import Dict, List, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: Sequence[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = False
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    tiles_per_row = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def _load_image(image_file: str, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


class InternVL(BaseModel):
    """InternVL wrapper aligned with the existing benchmark model interface."""

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = False,
        vllm_batch_size: int = 1,
        require_vllm: bool = False,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        gpu_memory_utilization: float = 0.92,
        load_in_8bit: bool = False,
        max_num: int = 6,
        total_max_num: int = 64,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.use_vllm = bool(use_vllm)
        self.vllm_batch_size = int(vllm_batch_size)
        self.require_vllm = bool(require_vllm)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_num = int(max_num)
        self.total_max_num = int(total_max_num)
        self.verbose = bool(verbose)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
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

                tp_size = int(os.environ.get("FIXATION_TP_SIZE", "1"))
                if tp_size < 1:
                    tp_size = 1

                self.llm = LLM(
                    model=model_path,
                    max_num_seqs=self.vllm_batch_size,
                    max_model_len=32768,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    trust_remote_code=True,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[warn] vLLM initialization failed for {model_path}, falling back to HF: {exc}")
                if self.require_vllm:
                    raise RuntimeError(f"vLLM initialization failed for {model_path}: {exc}") from exc
                self.use_vllm = False

        if not self.use_vllm:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=bool(load_in_8bit),
                device_map="auto",
            ).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_generation_config(self) -> Dict[str, object]:
        cfg: Dict[str, object] = {"do_sample": self.temperature > 0, "max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            cfg["temperature"] = self.temperature
            cfg["top_p"] = self.top_p
            if self.top_k >= 0:
                cfg["top_k"] = self.top_k
        return cfg

    def _format_prompt(self, message: List[Dict[str, str]]) -> str:
        image_num = len([x for x in message if x["type"] == "image"])
        if image_num == 0:
            return "\n".join([x["value"] for x in message if x["type"] == "text"]).strip()
        if image_num == 1:
            parts: List[str] = []
            for item in message:
                if item["type"] == "image":
                    parts.append("<image>")
                elif item["type"] == "text":
                    parts.append(item["value"])
            return "\n".join([p for p in parts if p]).strip()

        prompt = ""
        image_idx = 1
        for item in message:
            if item["type"] == "text":
                prompt += item["value"]
            elif item["type"] == "image":
                prompt += f"<Image-{image_idx}>"
                image_idx += 1
        prefix = "".join([f"Image-{i + 1}: <image>\n" for i in range(image_num)])
        placeholders = "".join([f"<Image-{i + 1}>" for i in range(image_num)])
        return (prefix + prompt).replace(placeholders, "").strip()

    def _build_vllm_prompt(self, message: List[Dict[str, str]]) -> str:
        plain_prompt = self._format_prompt(message)
        chat = [{"role": "user", "content": plain_prompt}]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, (list, tuple)):
            return str(prompt[0])
        return str(prompt)

    def _prepare_pixels(self, image_paths: List[str]) -> tuple[torch.Tensor | None, List[int]]:
        image_num = len(image_paths)
        if image_num == 0:
            return None, []

        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        if image_num == 1:
            pixel_values = _load_image(image_paths[0], max_num=max_num).to(self.device).to(torch.bfloat16)
            return pixel_values, [pixel_values.size(0)]

        num_patches_list: List[int] = []
        pixel_values_list: List[torch.Tensor] = []
        for image_path in image_paths:
            curr = _load_image(image_path, max_num=max_num).to(self.device).to(torch.bfloat16)
            num_patches_list.append(curr.size(0))
            pixel_values_list.append(curr)
        return torch.cat(pixel_values_list, dim=0), num_patches_list

    def _prepare_vllm_images(self, image_paths: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))
        return images

    def generate(self, message: List[Dict[str, str]], dataset: str | None = None) -> str:
        if self.use_vllm:
            return self.generate_batch([message], dataset=dataset)[0]

        prompt = self._format_prompt(message)
        image_paths = [x["value"] for x in message if x["type"] == "image"]
        pixel_values, num_patches_list = self._prepare_pixels(image_paths)

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=self._build_generation_config(),
                verbose=self.verbose,
            )

        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()

    def generate_batch(self, messages: List[List[Dict[str, str]]], dataset: str | None = None) -> List[str]:
        if not self.use_vllm:
            return [self.generate(m, dataset=dataset) for m in messages]

        from vllm import SamplingParams

        requests = []
        for idx, message in enumerate(messages):
            image_paths = [x["value"] for x in message if x["type"] == "image"]
            req: Dict[str, object] = {
                "prompt": self._build_vllm_prompt(message),
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
