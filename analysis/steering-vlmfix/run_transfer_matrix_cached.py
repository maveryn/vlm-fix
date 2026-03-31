#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SETUPS = [
    # within-game
    {"setup_id": "within_tictactoe", "source_game": "tictactoe", "target_game": "tictactoe", "style": "original"},
    {"setup_id": "within_reversi", "source_game": "reversi", "target_game": "reversi", "style": "original"},
    {"setup_id": "within_connect4", "source_game": "connect4", "target_game": "connect4", "style": "original"},
    {"setup_id": "within_dots_boxes", "source_game": "dots_boxes", "target_game": "dots_boxes", "style": "original"},
    # cross-game (Alice/Bob unified prompts)
    {"setup_id": "cross_tictactoe_to_connect4", "source_game": "tictactoe", "target_game": "connect4", "style": "alicebob"},
    {"setup_id": "cross_tictactoe_to_dots_boxes", "source_game": "tictactoe", "target_game": "dots_boxes", "style": "alicebob"},
    {"setup_id": "cross_reversi_to_connect4", "source_game": "reversi", "target_game": "connect4", "style": "alicebob"},
    {"setup_id": "cross_reversi_to_dots_boxes", "source_game": "reversi", "target_game": "dots_boxes", "style": "alicebob"},
]


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cached matrix steering runner for VLM-Fix transfer experiments. "
            "Loads model once, precomputes row cache once per (game, style), then sweeps repeats and layers."
        )
    )
    p.add_argument("--dataset", type=Path, default=Path("data/generated/vlm_fix/instances.parquet"))
    p.add_argument("--dataset-root", type=Path, default=Path("data/generated/vlm_fix"))
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--prompt-type", type=str, default="direct", choices=["direct", "cot"])
    p.add_argument("--render-variant", type=str, default="canonical")
    p.add_argument("--prompt-variant", type=str, default="standard")

    p.add_argument("--states-per-game", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument(
        "--setup-ids",
        type=str,
        default="",
        help="Optional comma-separated setup_id subset. Default: run all setups.",
    )

    p.add_argument("--last-n-layers", type=int, default=12)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--centroid-train-policy",
        type=str,
        default="source_correct",
        choices=["source_correct", "all_train"],
    )

    p.add_argument("--write-row-details", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/steering-vlmfix/outputs/transfer_matrix_cached"))
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run folder name inside --out-dir. If omitted, an auto name is used.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Prompt / extraction
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform_internvl(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio_internvl(
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


def _dynamic_preprocess_internvl(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
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

    target_aspect_ratio = _find_closest_aspect_ratio_internvl(
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


def _load_image_internvl(image_file: str, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform_internvl(input_size=input_size)
    images = _dynamic_preprocess_internvl(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def split_labels(s: str) -> List[str]:
    labels = [x.strip() for x in str(s).split("|") if x.strip()]
    if len(labels) != 2:
        raise ValueError(f"Expected 2 labels in valid_labels, got: {s}")
    return labels


def extract_answer(text: str, labels: Sequence[str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    label_map = {lbl.lower(): lbl for lbl in labels}
    exact = label_map.get(raw.lower())
    if exact is not None:
        return exact

    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    for candidate in reversed(boxed):
        tok = candidate.strip()
        if tok.lower() in label_map:
            return label_map[tok.lower()]

    low = raw.lower()
    matches: List[Tuple[int, str]] = []
    for lbl in labels:
        patt = r"\b" + re.escape(lbl.lower()) + r"\b"
        for m in re.finditer(patt, low):
            matches.append((m.start(), lbl))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]
    return ""


def rewrite_prompt_to_alicebob(prompt: str, valid_labels: str, answer: str) -> Tuple[str, str, str]:
    labels = split_labels(valid_labels)
    l1, l2 = labels[0], labels[1]
    ans_low = str(answer).strip().lower()
    if ans_low == l1.lower():
        answer_mapped = "Alice"
    elif ans_low == l2.lower():
        answer_mapped = "Bob"
    else:
        raise ValueError(f"Answer '{answer}' not in valid_labels '{valid_labels}'")

    player_sentence = f"There are two players: Alice and Bob. Alice is marked with {l1}, and Bob is marked with {l2}."
    players_pat = re.compile(
        r"\bPlayers are\s+" + re.escape(l1) + r"\s+and\s+" + re.escape(l2) + r"\.\s*",
        flags=re.IGNORECASE,
    )
    if players_pat.search(prompt):
        prompt_mod = players_pat.sub(player_sentence + " ", prompt, count=1)
    else:
        prompt_mod = f"{player_sentence} {prompt}"

    prompt_mod = re.sub(
        r"For this question,\s*Player 1\s*=\s*[^.]+?\s*and\s*Player 2\s*=\s*[^.]+?\.\s*",
        "",
        prompt_mod,
        flags=re.IGNORECASE,
    )

    pat = re.compile(r"Answer with only .*?Do not add any other text\.", flags=re.IGNORECASE)
    answer_instruction = "Answer with only Alice or Bob. Do not add any other text."
    if pat.search(prompt_mod):
        prompt_out = pat.sub(answer_instruction, prompt_mod, count=1)
    else:
        prompt_out = f"{prompt_mod} {answer_instruction}"

    return prompt_out, "Alice|Bob", answer_mapped


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------


def detect_family(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "qwen"
    if "molmo" in name:
        return "molmo"
    if "internvl" in name:
        return "internvl"
    return "other"


def make_processor(model_name: str, family: str):
    if family == "qwen":
        try:
            return AutoProcessor.from_pretrained(model_name, fix_mistral_regex=True)
        except TypeError:
            return AutoProcessor.from_pretrained(model_name)
    if family == "internvl":
        # Newer transformers versions may fail in AutoProcessor.from_pretrained for InternVL
        # because tokenizer fields expected by InternVLProcessor are not attached.
        from transformers.models.internvl.processing_internvl import InternVLProcessor

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        tokenizer.start_image_token = "<img>"
        tokenizer.end_image_token = "</img>"
        tokenizer.context_image_token = "<IMG_CONTEXT>"
        tokenizer.video_token = "<|video_pad|>"
        tokenizer.start_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.start_image_token)
        tokenizer.end_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.end_image_token)
        tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.context_image_token)
        tokenizer.video_token_id = tokenizer.convert_tokens_to_ids(tokenizer.video_token)

        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        video_processor = AutoVideoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return InternVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name: str):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    model.eval()
    return model


def get_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device):
        return model.device
    return next(model.parameters()).device


def move_inputs(inputs: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        t = v.to(device)
        if t.dtype.is_floating_point and t.dtype != dtype:
            t = t.to(dtype)
        out[k] = t
    return out


def build_inputs(
    processor,
    model,
    family: str,
    prompt: str,
    image_abs_path: str,
    image_text_order: str,
) -> Dict[str, torch.Tensor]:
    image = Image.open(image_abs_path).convert("RGB")

    if family == "qwen":
        from qwen_vl_utils import process_vision_info

        if str(image_text_order) == "text_first":
            content = [{"type": "text", "text": prompt}, {"type": "image", "image": image}]
        else:
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        convo = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        text = text[0] if isinstance(text, (list, tuple)) else text
        images, videos = process_vision_info(convo)
        proc = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        out = {k: v for k, v in proc.items() if torch.is_tensor(v)}
        out.pop("token_type_ids", None)
        return out

    if family == "internvl":
        # Match benchmark InternVL path (dynamic tiling + chat template expansion),
        # instead of InternVLProcessor direct text/image encoding.

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True, use_fast=False)

        question = f"<image>\n{prompt}" if str(image_text_order) == "image_first" else f"{prompt}\n<image>"
        pixel_values = _load_image_internvl(image_abs_path, max_num=6)
        n_patches = int(pixel_values.shape[0])

        mod = inspect.getmodule(model.__class__)
        get_tpl = getattr(mod, "get_conv_template", None)
        if get_tpl is None:
            raise RuntimeError("InternVL model module does not expose get_conv_template; cannot build chat prompt.")
        template = get_tpl(getattr(model, "template", "internvl_zh"))
        template.system_message = getattr(model, "system_message", template.system_message)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = "<img>" + "<IMG_CONTEXT>" * int(getattr(model, "num_image_token", 256)) * n_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        out = {k: v for k, v in model_inputs.items() if torch.is_tensor(v)}
        out["pixel_values"] = pixel_values
        # InternVL forward() path needs image_flags.
        out["image_flags"] = torch.ones((n_patches, 1), dtype=torch.long)
        out.pop("token_type_ids", None)
        return out

    inline = f"<|image|>\n{prompt}" if str(image_text_order) == "image_first" else f"{prompt}\n<|image|>"
    chat = [{"role": "user", "content": inline}]
    chat_prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    proc = processor(images=[image], text=chat_prompt, return_tensors="pt")
    out = {k: v for k, v in proc.items() if torch.is_tensor(v)}
    out.pop("token_type_ids", None)
    return out


def get_layer_stack(model):
    if hasattr(model, "model"):
        mm = model.model
        if hasattr(mm, "language_model") and hasattr(mm.language_model, "layers"):
            return mm.language_model.layers
        if hasattr(mm, "language_model") and hasattr(mm.language_model, "model") and hasattr(mm.language_model.model, "layers"):
            return mm.language_model.model.layers
        if hasattr(mm, "llm") and hasattr(mm.llm, "model") and hasattr(mm.llm.model, "layers"):
            return mm.llm.model.layers
        if hasattr(mm, "llm") and hasattr(mm.llm, "layers"):
            return mm.llm.layers
        if hasattr(mm, "transformer") and hasattr(mm.transformer, "h"):
            return mm.transformer.h
        if hasattr(mm, "transformer") and hasattr(mm.transformer, "blocks"):
            return mm.transformer.blocks
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    raise RuntimeError("Could not locate decoder layer stack.")


def forward_hidden_all_layers(model, inputs_cpu: Dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16) -> Tuple[int, Tuple[np.ndarray, ...]]:
    device = get_model_device(model)
    inputs = move_inputs(inputs_cpu, device=device, dtype=dtype)
    with torch.no_grad():
        out = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
    qidx = int(inputs["attention_mask"][0].sum().item()) - 1
    hs = tuple(h[0, qidx, :].detach().float().cpu().numpy() for h in out.hidden_states[1:])
    return qidx, hs


def _normalize_generate_ids(out_obj: Any) -> torch.Tensor:
    # HF generate() may return either tensor ids or an object with `.sequences`.
    if hasattr(out_obj, "sequences"):
        return out_obj.sequences
    return out_obj


def decode_new_tokens(
    tokenizer,
    out_obj: Any,
    prompt_ids: torch.Tensor | None,
    prompt_len: int,
) -> str:
    out_ids = _normalize_generate_ids(out_obj)
    if not torch.is_tensor(out_ids):
        return ""
    if out_ids.ndim != 2 or out_ids.shape[0] < 1:
        return ""

    seq = out_ids[0]

    # Preferred path: output contains prompt prefix + generated suffix.
    if prompt_ids is not None and torch.is_tensor(prompt_ids) and prompt_ids.ndim == 2 and prompt_ids.shape[0] >= 1:
        pref = prompt_ids[0]
        if seq.shape[0] >= pref.shape[0]:
            try:
                if torch.equal(seq[: pref.shape[0]].to(device=pref.device), pref):
                    gen_ids = seq[pref.shape[0] :]
                    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception:
                pass

    # Fallback A: legacy prompt_len slicing.
    if int(prompt_len) > 0 and seq.shape[0] > int(prompt_len):
        gen_ids = seq[int(prompt_len) :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Fallback B: some models return generated tokens only.
    return tokenizer.decode(seq, skip_special_tokens=True).strip()


def _internvl_chat_eos_token_id(model, tokenizer) -> int | None:
    try:
        mod = inspect.getmodule(model.__class__)
        get_tpl = getattr(mod, "get_conv_template", None)
        template_name = getattr(model, "template", None)
        if get_tpl is None or template_name is None:
            return None
        tpl = get_tpl(template_name)
        sep = str(getattr(tpl, "sep", "")).strip()
        if not sep:
            return None
        tok_id = tokenizer.convert_tokens_to_ids(sep)
        if tok_id is None:
            return None
        tok_id = int(tok_id)
        return tok_id if tok_id >= 0 else None
    except Exception:
        return None


def generate_plain(
    model,
    processor,
    inputs_cpu: Dict[str, torch.Tensor],
    max_new_tokens: int,
    family: str | None = None,
) -> str:
    device = get_model_device(model)
    inputs = move_inputs(inputs_cpu, device=device, dtype=torch.bfloat16)
    gen_inputs = dict(inputs)
    # InternVL generate() does not accept image_flags even though forward() uses it.
    gen_inputs.pop("image_flags", None)
    prompt_len = int(inputs["input_ids"].shape[1])
    prompt_ids = inputs.get("input_ids")
    gen_kwargs: Dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
    }
    if str(family) == "internvl":
        eos_id = _internvl_chat_eos_token_id(model, processor.tokenizer)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = int(eos_id)
        if getattr(processor, "tokenizer", None) is not None and getattr(processor.tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = int(processor.tokenizer.eos_token_id)
    with torch.no_grad():
        out_obj = model.generate(
            **gen_inputs,
            **gen_kwargs,
        )
    return decode_new_tokens(
        processor.tokenizer,
        out_obj,
        prompt_ids=prompt_ids,
        prompt_len=prompt_len,
    )


def generate_with_patch(
    model,
    processor,
    inputs_cpu: Dict[str, torch.Tensor],
    patch_layer: int,
    query_idx: int,
    delta_vec: np.ndarray,
    alpha: float,
    max_new_tokens: int,
    family: str | None = None,
) -> str:
    layers = get_layer_stack(model)
    target = layers[int(patch_layer)]
    device = get_model_device(model)
    inputs = move_inputs(inputs_cpu, device=device, dtype=torch.bfloat16)
    gen_inputs = dict(inputs)
    gen_inputs.pop("image_flags", None)
    prompt_len = int(inputs["input_ids"].shape[1])
    applied = {"flag": False}

    def _hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if (not applied["flag"]) and int(hidden.shape[1]) == int(prompt_len):
            patched = hidden.clone()
            dv = torch.from_numpy(delta_vec).to(device=patched.device, dtype=patched.dtype)
            patched[0, int(query_idx), :] = patched[0, int(query_idx), :] + float(alpha) * dv
            applied["flag"] = True
            if isinstance(out, tuple):
                return (patched, *out[1:])
            return patched
        return out

    h = target.register_forward_hook(_hook)
    try:
        prompt_ids = inputs.get("input_ids")
        gen_kwargs: Dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": int(max_new_tokens),
        }
        if str(family) == "internvl":
            eos_id = _internvl_chat_eos_token_id(model, processor.tokenizer)
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)
            if getattr(processor, "tokenizer", None) is not None and getattr(processor.tokenizer, "eos_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = int(processor.tokenizer.eos_token_id)
        with torch.no_grad():
            out_obj = model.generate(
                **gen_inputs,
                **gen_kwargs,
            )
    finally:
        h.remove()
    return decode_new_tokens(
        processor.tokenizer,
        out_obj,
        prompt_ids=prompt_ids,
        prompt_len=prompt_len,
    )


# -----------------------------------------------------------------------------
# Data prep + cache
# -----------------------------------------------------------------------------


@dataclass
class RowCache:
    hidden: np.ndarray  # [n_selected_layers, d]
    qidx: int
    prompt_len: int
    prediction_base: str
    extracted_base: str
    correct_base: int


def load_game_rows(
    dataset: pd.DataFrame,
    game: str,
    prompt_type: str,
    render_variant: str,
    prompt_variant: str,
    selected_states: Sequence[int],
    style: str,
) -> pd.DataFrame:
    g = dataset[
        (dataset["game"] == game)
        & (dataset["prompt_type"] == prompt_type)
        & (dataset["render_variant"] == render_variant)
        & (dataset["prompt_variant"] == prompt_variant)
    ][[
        "state_id",
        "rule_variant",
        "question_target",
        "image_text_order",
        "prompt",
        "answer",
        "valid_labels",
        "image_path",
    ]].copy()

    sel_set = set(int(x) for x in selected_states)
    g = g[g["state_id"].isin(sel_set)].copy()
    g = g.sort_values(["state_id", "rule_variant", "question_target", "image_text_order"]).reset_index(drop=True)

    if style == "alicebob":
        def _map_row(r: pd.Series) -> pd.Series:
            p2, v2, a2 = rewrite_prompt_to_alicebob(str(r["prompt"]), str(r["valid_labels"]), str(r["answer"]))
            r["prompt"] = p2
            r["valid_labels"] = v2
            r["answer"] = a2
            return r

        g = g.apply(_map_row, axis=1).reset_index(drop=True)

    g = g.reset_index(names="row_id")
    return g


def build_state_manifest(
    dataset: pd.DataFrame,
    games: Sequence[str],
    prompt_type: str,
    render_variant: str,
    prompt_variant: str,
    states_per_game: int,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for game in games:
        m = dataset[
            (dataset["game"] == game)
            & (dataset["prompt_type"] == prompt_type)
            & (dataset["render_variant"] == render_variant)
            & (dataset["prompt_variant"] == prompt_variant)
        ]
        states = sorted(int(x) for x in m["state_id"].unique().tolist())
        if len(states) < int(states_per_game):
            raise RuntimeError(f"Game '{game}' has only {len(states)} states, requested {states_per_game}.")
        out[str(game)] = states[: int(states_per_game)]
    return out


def build_split_manifest(state_manifest: Dict[str, List[int]], repeats: int, seed_base: int, test_size: float) -> pd.DataFrame:
    rows: List[dict] = []
    for game, states in sorted(state_manifest.items()):
        arr = np.array([int(x) for x in states], dtype=np.int64)
        for rep in range(int(repeats)):
            seed = int(seed_base) + int(rep)
            tr, te = train_test_split(arr, test_size=float(test_size), random_state=seed)
            tr_set = set(int(x) for x in tr)
            for sid in arr.tolist():
                rows.append(
                    {
                        "game": str(game),
                        "repeat": int(rep),
                        "seed": int(seed),
                        "state_id": int(sid),
                        "split": "train" if int(sid) in tr_set else "test",
                    }
                )
    return pd.DataFrame(rows).sort_values(["game", "repeat", "state_id"]).reset_index(drop=True)


def precompute_cache(
    *,
    model,
    processor,
    family: str,
    rows: pd.DataFrame,
    dataset_root: Path,
    selected_layers: Sequence[int],
    max_new_tokens: int,
    desc: str,
) -> Dict[int, RowCache]:
    out: Dict[int, RowCache] = {}
    all_layers = sorted(int(x) for x in selected_layers)

    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=desc, unit="row"):
        rid = int(r["row_id"])
        image_abs = str((dataset_root / str(r["image_path"])).resolve())
        inp = build_inputs(
            processor=processor,
            model=model,
            family=family,
            prompt=str(r["prompt"]),
            image_abs_path=image_abs,
            image_text_order=str(r["image_text_order"]),
        )

        qidx, hs = forward_hidden_all_layers(model, inp)
        pred = generate_plain(
            model,
            processor,
            inp,
            max_new_tokens=int(max_new_tokens),
            family=family,
        )
        labels = split_labels(str(r["valid_labels"]))
        ex = extract_answer(pred, labels)
        corr = int(ex == str(r["answer"]))

        sel = np.stack([hs[int(li)] for li in all_layers], axis=0).astype(np.float32)
        out[rid] = RowCache(
            hidden=sel,
            qidx=int(qidx),
            prompt_len=int(inp["input_ids"].shape[1]),
            prediction_base=str(pred),
            extracted_base=str(ex),
            correct_base=int(corr),
        )

        if (len(out) % 16) == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return out


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def main() -> int:
    args = parse_args()
    setup_filter = [x.strip() for x in str(args.setup_ids).split(",") if x.strip()]
    if setup_filter:
        setup_set = set(setup_filter)
        setups = [s for s in SETUPS if str(s["setup_id"]) in setup_set]
        known = {str(s["setup_id"]) for s in SETUPS}
        unknown = sorted(list(setup_set - known))
        if unknown:
            raise ValueError(f"Unknown setup_id values: {unknown}. Known: {sorted(list(known))}")
        if not setups:
            raise ValueError("No setups selected after --setup-ids filtering.")
    else:
        setups = list(SETUPS)

    safe_model = _safe_model_name(str(args.model))
    run_name = str(args.run_name).strip()
    if not run_name:
        setup_tag = "all" if len(setups) == len(SETUPS) else "subset-" + "-".join(sorted(str(s["setup_id"]) for s in setups))
        alpha_tag = f"{float(args.alpha):g}".replace(".", "p")
        run_name = (
            f"{safe_model}__spg{int(args.states_per_game)}__rep{int(args.repeats)}"
            f"__last{int(args.last_n_layers)}__alpha{alpha_tag}"
            f"__{str(args.prompt_type)}_{str(args.render_variant)}_{str(args.prompt_variant)}__{setup_tag}"
        )
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / run_name
    if out_dir.exists():
        k = 2
        while (out_root / f"{run_name}__r{k}").exists():
            k += 1
        out_dir = out_root / f"{run_name}__r{k}"
    out_dir.mkdir(parents=True, exist_ok=False)

    dataset = pd.read_parquet(args.dataset)

    required_games = sorted({s["source_game"] for s in setups} | {s["target_game"] for s in setups})
    state_manifest = build_state_manifest(
        dataset=dataset,
        games=required_games,
        prompt_type=str(args.prompt_type),
        render_variant=str(args.render_variant),
        prompt_variant=str(args.prompt_variant),
        states_per_game=int(args.states_per_game),
    )

    split_manifest = build_split_manifest(
        state_manifest=state_manifest,
        repeats=int(args.repeats),
        seed_base=int(args.seed_base),
        test_size=float(args.test_size),
    )

    family = detect_family(args.model)
    processor = make_processor(args.model, family)
    model = load_model(args.model)
    if family == "internvl":
        try:
            image_tok = getattr(processor, "image_token", "<IMG_CONTEXT>")
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                model.img_context_token_id = int(tok.convert_tokens_to_ids(image_tok))
        except Exception:
            pass

    n_layers = len(get_layer_stack(model))
    n_last = int(args.last_n_layers)
    if n_last <= 0:
        raise ValueError("--last-n-layers must be > 0")
    n_last = min(n_last, n_layers)
    selected_layers = list(range(n_layers - n_last, n_layers))
    layer_to_pos = {int(li): i for i, li in enumerate(selected_layers)}

    # Build and cache all (game, style) datasets once.
    game_style_rows: Dict[Tuple[str, str], pd.DataFrame] = {}
    game_style_cache: Dict[Tuple[str, str], Dict[int, RowCache]] = {}

    needed_game_styles = sorted(
        {(s["source_game"], s["style"]) for s in setups} | {(s["target_game"], s["style"]) for s in setups}
    )

    for game, style in needed_game_styles:
        rows = load_game_rows(
            dataset=dataset,
            game=str(game),
            prompt_type=str(args.prompt_type),
            render_variant=str(args.render_variant),
            prompt_variant=str(args.prompt_variant),
            selected_states=state_manifest[str(game)],
            style=str(style),
        )
        game_style_rows[(str(game), str(style))] = rows
        game_style_cache[(str(game), str(style))] = precompute_cache(
            model=model,
            processor=processor,
            family=family,
            rows=rows,
            dataset_root=args.dataset_root,
            selected_layers=selected_layers,
            max_new_tokens=int(args.max_new_tokens),
            desc=f"precompute_{game}_{style}",
        )

    results_rows: List[dict] = []
    detail_rows: List[dict] = []

    total_jobs = len(setups) * int(args.repeats) * len(selected_layers)
    pbar_jobs = tqdm(total=total_jobs, desc="matrix_jobs", unit="job")

    for setup in setups:
        setup_id = str(setup["setup_id"])
        source_game = str(setup["source_game"])
        target_game = str(setup["target_game"])
        style = str(setup["style"])

        src_rows = game_style_rows[(source_game, style)]
        src_cache = game_style_cache[(source_game, style)]
        tgt_rows = game_style_rows[(target_game, style)]
        tgt_cache = game_style_cache[(target_game, style)]

        for rep in range(int(args.repeats)):
            seed = int(args.seed_base) + int(rep)

            src_train_states = set(
                int(x)
                for x in split_manifest[
                    (split_manifest["game"] == source_game)
                    & (split_manifest["repeat"] == int(rep))
                    & (split_manifest["split"] == "train")
                ]["state_id"].tolist()
            )
            tgt_train_states = set(
                int(x)
                for x in split_manifest[
                    (split_manifest["game"] == target_game)
                    & (split_manifest["repeat"] == int(rep))
                    & (split_manifest["split"] == "train")
                ]["state_id"].tolist()
            )
            tgt_test_states = set(
                int(x)
                for x in split_manifest[
                    (split_manifest["game"] == target_game)
                    & (split_manifest["repeat"] == int(rep))
                    & (split_manifest["split"] == "test")
                ]["state_id"].tolist()
            )

            src_train = src_rows[src_rows["state_id"].isin(src_train_states)].copy().reset_index(drop=True)
            tgt_train = tgt_rows[tgt_rows["state_id"].isin(tgt_train_states)].copy().reset_index(drop=True)
            tgt_test = tgt_rows[tgt_rows["state_id"].isin(tgt_test_states)].copy().reset_index(drop=True)

            if src_train.empty or tgt_train.empty or tgt_test.empty:
                raise RuntimeError(f"Empty split for setup={setup_id}, repeat={rep}")

            # Labels / mappings independent of layer.
            src_labels = sorted(set(str(x) for x in src_train["answer"].astype(str).to_list()))
            tgt_labels_train = sorted(set(str(x) for x in tgt_train["answer"].astype(str).to_list()))
            tgt_labels_test = sorted(set(str(x) for x in tgt_test["answer"].astype(str).to_list()))
            if len(src_labels) != 2 or len(tgt_labels_train) != 2 or len(tgt_labels_test) != 2:
                raise RuntimeError(
                    f"Binary label requirement failed: src={src_labels}, tgt_train={tgt_labels_train}, tgt_test={tgt_labels_test}"
                )
            if set(tgt_labels_train) != set(tgt_labels_test):
                raise RuntimeError(f"Target train/test labels mismatch: train={tgt_labels_train}, test={tgt_labels_test}")

            src_ans_to_idx = {lbl: i for i, lbl in enumerate(src_labels)}
            tgt_ans_to_idx = {lbl: i for i, lbl in enumerate(tgt_labels_train)}

            y_rule_src = (src_train["rule_variant"] == "inverse").astype(int).to_numpy()
            y_ans_src = np.array([src_ans_to_idx[str(x)] for x in src_train["answer"].astype(str).to_list()], dtype=np.int64)
            src_correct = np.array([bool(src_cache[int(rid)].correct_base) for rid in src_train["row_id"].astype(int).to_list()])

            y_rule_tgt = (tgt_train["rule_variant"] == "inverse").astype(int).to_numpy()
            y_ans_tgt = np.array([tgt_ans_to_idx[str(x)] for x in tgt_train["answer"].astype(str).to_list()], dtype=np.int64)

            for li in selected_layers:
                pos = layer_to_pos[int(li)]

                # Source centroids by (rule, answer)
                X_src = np.stack([src_cache[int(rid)].hidden[pos] for rid in src_train["row_id"].astype(int).to_list()], axis=0).astype(np.float32)
                centroids: Dict[int, Dict[int, np.ndarray]] = {}
                for rule_cls in [0, 1]:
                    idx_rule = np.where(y_rule_src == int(rule_cls))[0]
                    if idx_rule.size == 0:
                        raise RuntimeError(f"No source train rows for rule={rule_cls}, setup={setup_id}, repeat={rep}")
                    y_ans_rule = y_ans_src[idx_rule]
                    centroids_rule: Dict[int, np.ndarray] = {}
                    for ans_cls in [0, 1]:
                        if str(args.centroid_train_policy) == "all_train":
                            idx_use = idx_rule[y_ans_rule == int(ans_cls)]
                        else:
                            idx_use = idx_rule[(y_ans_rule == int(ans_cls)) & src_correct[idx_rule]]
                            if idx_use.size == 0:
                                idx_use = idx_rule[y_ans_rule == int(ans_cls)]
                        if idx_use.size == 0:
                            idx_use = idx_rule
                        centroids_rule[int(ans_cls)] = X_src[idx_use].mean(axis=0).astype(np.float32)
                    centroids[int(rule_cls)] = centroids_rule

                # Target routing probes on target train.
                X_tgt = np.stack([tgt_cache[int(rid)].hidden[pos] for rid in tgt_train["row_id"].astype(int).to_list()], axis=0).astype(np.float32)

                rule_probe = LogisticRegression(max_iter=2000, random_state=int(seed), class_weight="balanced")
                rule_probe.fit(X_tgt, y_rule_tgt)

                answer_probe_by_rule: Dict[int, Any] = {}
                for rule_cls in [0, 1]:
                    idx_rule = np.where(y_rule_tgt == int(rule_cls))[0]
                    if idx_rule.size == 0:
                        raise RuntimeError(f"No target train rows for rule={rule_cls}, setup={setup_id}, repeat={rep}")
                    y_ans_rule = y_ans_tgt[idx_rule]
                    uniq = np.unique(y_ans_rule)
                    if uniq.size >= 2:
                        ans_probe = LogisticRegression(max_iter=2000, random_state=int(seed), class_weight="balanced")
                        ans_probe.fit(X_tgt[idx_rule], y_ans_rule)
                        answer_probe_by_rule[int(rule_cls)] = ans_probe
                    else:
                        answer_probe_by_rule[int(rule_cls)] = int(uniq[0])

                # Evaluate on target test.
                eval_rows: List[dict] = []
                for _, r in tgt_test.iterrows():
                    rid = int(r["row_id"])
                    h = tgt_cache[rid].hidden[pos]
                    pred_rule = int(rule_probe.predict(h.reshape(1, -1))[0])
                    ans_probe = answer_probe_by_rule[int(pred_rule)]
                    if isinstance(ans_probe, LogisticRegression):
                        pred_ans = int(ans_probe.predict(h.reshape(1, -1))[0])
                    else:
                        pred_ans = int(ans_probe)

                    delta = centroids[int(pred_rule)][int(pred_ans)] - h

                    inp = build_inputs(
                        processor=processor,
                        model=model,
                        family=family,
                        prompt=str(r["prompt"]),
                        image_abs_path=str((args.dataset_root / str(r["image_path"])).resolve()),
                        image_text_order=str(r["image_text_order"]),
                    )
                    pred = generate_with_patch(
                        model=model,
                        processor=processor,
                        inputs_cpu=inp,
                        patch_layer=int(li),
                        query_idx=int(tgt_cache[rid].qidx),
                        delta_vec=delta,
                        alpha=float(args.alpha),
                        max_new_tokens=int(args.max_new_tokens),
                        family=family,
                    )
                    valid_labels = split_labels(str(r["valid_labels"]))
                    ex = extract_answer(pred, valid_labels)
                    corr = int(ex == str(r["answer"]))

                    rule_true = int(str(r["rule_variant"]) == "inverse")
                    ans_true = int(tgt_ans_to_idx[str(r["answer"])])
                    eval_rows.append(
                        {
                            "rule_variant": str(r["rule_variant"]),
                            "correct_base": int(tgt_cache[rid].correct_base),
                            "correct_patched": int(corr),
                            "probe_rule_hit": int(pred_rule == rule_true),
                            "probe_answer_hit": int(pred_ans == ans_true),
                            "prediction_patched": str(pred),
                            "extracted_patched": str(ex),
                        }
                    )

                eval_df = pd.DataFrame(eval_rows)
                if eval_df.empty:
                    raise RuntimeError(f"Empty eval df for setup={setup_id}, repeat={rep}, layer={li}")

                rec = {
                    "model": str(args.model),
                    "setup_id": setup_id,
                    "setup_style": style,
                    "source_game": source_game,
                    "target_game": target_game,
                    "repeat": int(rep),
                    "seed": int(seed),
                    "layer_0based": int(li),
                    "layer_paper": int(li) + 1,
                    "alpha": float(args.alpha),
                    "source_train_n": int(len(src_train)),
                    "target_train_n": int(len(tgt_train)),
                    "target_test_n": int(len(eval_df)),
                    "centroid_train_policy": str(args.centroid_train_policy),
                    "routing_rule_acc_target": float(eval_df["probe_rule_hit"].mean()),
                    "routing_answer_acc_target": float(eval_df["probe_answer_hit"].mean()),
                    "target_base_acc": float(eval_df["correct_base"].mean()),
                    "target_patched_acc": float(eval_df["correct_patched"].mean()),
                    "target_base_std_rule_acc": float(eval_df.loc[eval_df["rule_variant"] == "standard", "correct_base"].mean()),
                    "target_base_inv_rule_acc": float(eval_df.loc[eval_df["rule_variant"] == "inverse", "correct_base"].mean()),
                    "target_patched_std_rule_acc": float(eval_df.loc[eval_df["rule_variant"] == "standard", "correct_patched"].mean()),
                    "target_patched_inv_rule_acc": float(eval_df.loc[eval_df["rule_variant"] == "inverse", "correct_patched"].mean()),
                }
                results_rows.append(rec)

                if bool(args.write_row_details):
                    for rr in eval_rows:
                        detail_rows.append(
                            {
                                "setup_id": setup_id,
                                "repeat": int(rep),
                                "seed": int(seed),
                                "layer_0based": int(li),
                                "layer_paper": int(li) + 1,
                                **rr,
                            }
                        )

                pbar_jobs.update(1)
                if (len(results_rows) % 8) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    pbar_jobs.close()

    # Save outputs
    run_meta = {
        "model": str(args.model),
        "safe_model": _safe_model_name(str(args.model)),
        "prompt_type": str(args.prompt_type),
        "render_variant": str(args.render_variant),
        "prompt_variant": str(args.prompt_variant),
        "states_per_game": int(args.states_per_game),
        "repeats": int(args.repeats),
        "seed_base": int(args.seed_base),
        "test_size": float(args.test_size),
        "n_layers_total": int(n_layers),
        "selected_layers_0based": [int(x) for x in selected_layers],
        "selected_layers_paper": [int(x) + 1 for x in selected_layers],
        "alpha": float(args.alpha),
        "max_new_tokens": int(args.max_new_tokens),
        "centroid_train_policy": str(args.centroid_train_policy),
        "setup_filter": setup_filter,
        "setups": setups,
    }

    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (out_dir / "state_manifest.json").write_text(json.dumps(state_manifest, indent=2), encoding="utf-8")
    split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    runs_df = pd.DataFrame(results_rows)
    runs_df.to_csv(out_dir / "summary_runs.csv", index=False)

    agg = (
        runs_df
        .groupby(["setup_id", "setup_style", "source_game", "target_game", "layer_0based", "layer_paper"], as_index=False)
        .agg(
            n=("target_test_n", "sum"),
            repeats=("repeat", "nunique"),
            base_acc_mean=("target_base_acc", "mean"),
            base_acc_std=("target_base_acc", "std"),
            patched_acc_mean=("target_patched_acc", "mean"),
            patched_acc_std=("target_patched_acc", "std"),
            base_std_mean=("target_base_std_rule_acc", "mean"),
            base_inv_mean=("target_base_inv_rule_acc", "mean"),
            patched_std_mean=("target_patched_std_rule_acc", "mean"),
            patched_inv_mean=("target_patched_inv_rule_acc", "mean"),
            routing_rule_mean=("routing_rule_acc_target", "mean"),
            routing_ans_mean=("routing_answer_acc_target", "mean"),
        )
        .sort_values(["setup_id", "layer_0based"])
        .reset_index(drop=True)
    )
    agg.to_csv(out_dir / "summary_layer_setup_meanstd.csv", index=False)

    if bool(args.write_row_details):
        pd.DataFrame(detail_rows).to_csv(out_dir / "rows_eval_details.csv", index=False)

    print("[ok] wrote", out_dir)
    print(json.dumps({
        "runs": int(len(runs_df)),
        "setups": int(len(setups)),
        "repeats": int(args.repeats),
        "layers": int(len(selected_layers)),
    }, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
