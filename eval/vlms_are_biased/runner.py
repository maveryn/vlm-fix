from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from eval.model_registry import MODEL_REGISTRY


def _build_messages(chunk: pd.DataFrame) -> List[List[dict[str, str]]]:
    messages: List[List[dict[str, str]]] = []
    for _, row in chunk.iterrows():
        messages.append(
            [
                {"type": "image", "value": str(row["image_abs_path"])},
                {"type": "text", "value": str(row["prompt"])},
            ]
        )
    return messages


def _infer_single_model(
    model_name: str,
    instances: pd.DataFrame,
    batch_size: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")

    ctor = MODEL_REGISTRY[model_name]
    model = ctor(vllm_batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=0.0)

    rows: list[dict[str, object]] = []
    for start in tqdm(range(0, len(instances), batch_size), desc=f"Infer {model_name}", unit="batch"):
        chunk = instances.iloc[start : start + batch_size]
        messages = _build_messages(chunk)
        preds, metas = model.generate_batch_with_meta(messages, dataset="vlms_are_biased")

        for (_, src), pred, meta in zip(chunk.iterrows(), preds, metas):
            row = {
                "model": model_name,
                "index": int(src["index"]),
                "source_index": int(src["source_index"]),
                "dataset_split": str(src["dataset_split"]),
                "topic": str(src["topic"]),
                "sub_topic": str(src.get("sub_topic", "")),
                "id": str(src["id"]),
                "image_variant": str(src.get("image_variant", "original")),
                "prompt_style": str(src["prompt_style"]),
                "prompt": str(src["prompt"]),
                "ground_truth": str(src["ground_truth"]),
                "expected_bias": str(src["expected_bias"]),
                "prediction": str(pred),
                "image_abs_path": str(src["image_abs_path"]),
            }
            if isinstance(meta, dict):
                row["output_tokens"] = meta.get("output_tokens")
                row["finish_reason"] = meta.get("finish_reason")
                row["stop_reason"] = meta.get("stop_reason")
            else:
                row["output_tokens"] = None
                row["finish_reason"] = None
                row["stop_reason"] = None
            rows.append(row)

    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return pd.DataFrame(rows)


def run_inference(
    model_names: Iterable[str],
    instances: pd.DataFrame,
    batch_size: int,
    max_new_tokens: int,
    out_dir: str | Path,
) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        model_df = _infer_single_model(
            model_name=model_name,
            instances=instances,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        safe_model = model_name.replace("/", "_")
        model_df.to_parquet(out_path / f"{safe_model}_predictions.parquet", index=False)
        all_frames.append(model_df)

    if not all_frames:
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    return merged
