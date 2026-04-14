#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.model_registry import MODEL_REGISTRY
from eval.vlms_are_biased.dataset import load_prepared_instances, prepare_vlms_are_biased_dataset
from eval.vlms_are_biased.metrics import add_scores
from eval.vlms_are_biased.prompt_variants import TOPICS
from eval.vlms_are_biased.reporting import write_matrix_reports
from eval.vlms_are_biased.runner import run_inference


DEFAULT_MODELS = [
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


PROMPT_FILE_LABEL = {"original": "original", "item_alias": "alias"}


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text).strip()).strip("_")


def _write_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            payload = {
                "id": str(row.get("id", "")),
                "source_index": int(row.get("source_index", -1)),
                "topic": str(row.get("topic", "")),
                "sub_topic": str(row.get("sub_topic", "")),
                "image_variant": str(row.get("image_variant", "")),
                "prompt_style": str(row.get("prompt_style", "")),
                "prompt": str(row.get("prompt", "")),
                "model_response": str(row.get("prediction", "")),
                "extracted_answer": str(row.get("prediction_norm", "")),
                "ground_truth": str(row.get("ground_truth", "")),
                "expected_bias": str(row.get("expected_bias", "")),
                "is_correct": int(float(row.get("accuracy", 0.0)) > 0.5),
                "matches_bias": int(float(row.get("bias_ratio", 0.0)) > 0.5),
                "image_path": str(row.get("image_abs_path", "")),
                "output_tokens": None
                if pd.isna(row.get("output_tokens", None))
                else int(row.get("output_tokens")),
                "finish_reason": None
                if pd.isna(row.get("finish_reason", None))
                else str(row.get("finish_reason")),
                "stop_reason": None if pd.isna(row.get("stop_reason", None)) else str(row.get("stop_reason")),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full VLMs-Are-Biased matrix and export per-topic JSONL predictions + "
            "summary Excel (models x prompt styles x datasets)."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names from eval.model_registry.MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=list(TOPICS),
        help="Topic subset to evaluate.",
    )
    parser.add_argument(
        "--prompt-styles",
        nargs="+",
        default=["original", "item_alias"],
        choices=["original", "item_alias"],
        help="Prompt styles to evaluate.",
    )
    parser.add_argument(
        "--image-variants",
        nargs="+",
        default=["original"],
        choices=["original", "flipped"],
        help="Image variants to evaluate.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/vlms_are_biased_prepared",
        help="Prepared dataset directory (contains instances.parquet).",
    )
    parser.add_argument(
        "--prepare-if-missing",
        action="store_true",
        help="Auto-prepare dataset if instances.parquet is missing.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="main",
        help="HF split used when auto-preparing missing dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for vLLM generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max output tokens.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/vlms_are_biased",
        help="Root output for per-subset JSONL predictions.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for aggregate reports.",
    )
    parser.add_argument(
        "--results-xlsx",
        type=str,
        default="vlms_are_biased_summary.xlsx",
        help="Excel filename written under --results-dir.",
    )
    return parser.parse_args()


def _ensure_dataset(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    instances_path = dataset_dir / "instances.parquet"
    if instances_path.exists():
        return
    if not args.prepare_if_missing:
        raise FileNotFoundError(
            f"Missing prepared dataset at {instances_path}. "
            "Run eval/prepare_vlms_are_biased.py first, or pass --prepare-if-missing."
        )
    prepare_vlms_are_biased_dataset(
        out_dir=dataset_dir,
        dataset_split=str(args.dataset_split),
        topics=args.topics,
        prompt_styles=args.prompt_styles,
        overwrite=False,
    )


def _validate_models(models: Iterable[str]) -> None:
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}. Known keys: {sorted(MODEL_REGISTRY.keys())}")


def main() -> int:
    args = _parse_args()
    _validate_models(args.models)
    _ensure_dataset(args)

    runs_root = Path(args.runs_dir)
    raw_root = runs_root / "_raw_parquet"
    results_dir = Path(args.results_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    completed = 0
    total = len(args.models) * len(args.prompt_styles) * len(args.topics)
    total *= len(args.image_variants)
    print(
        f"[INFO] Starting matrix run: models={len(args.models)}, styles={len(args.prompt_styles)}, "
        f"image_variants={len(args.image_variants)}, topics={len(args.topics)}, combinations={total}",
        flush=True,
    )

    for model_name in args.models:
        safe_model = _safe_name(model_name)
        for image_variant in args.image_variants:
            for prompt_style in args.prompt_styles:
                instances = load_prepared_instances(
                    dataset_path=args.dataset_dir,
                    prompt_style=prompt_style,
                    topics=args.topics,
                    image_variant=image_variant,
                )
                raw_out_dir = raw_root / _safe_name(image_variant) / _safe_name(prompt_style) / safe_model
                try:
                    pred_df = run_inference(
                        model_names=[model_name],
                        instances=instances,
                        batch_size=int(args.batch_size),
                        max_new_tokens=int(args.max_new_tokens),
                        out_dir=raw_out_dir,
                    )
                    pred_df = add_scores(pred_df)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    for topic in args.topics:
                        completed += 1
                        summary_rows.append(
                            {
                                "model": model_name,
                                "image_variant": image_variant,
                                "prompt_style": prompt_style,
                                "topic": topic,
                                "n_total": 0,
                                "n_correct": 0,
                                "accuracy": None,
                                "bias_ratio": None,
                                "avg_output_tokens": None,
                                "jsonl_path": None,
                                "status": "failed",
                                "error": err,
                            }
                        )
                        print(
                            f"[FAIL {completed:03d}/{total}] model={model_name} image={image_variant} "
                            f"style={prompt_style} topic={topic} error={err}",
                            flush=True,
                        )
                    continue

                for topic in args.topics:
                    topic_df = pred_df[pred_df["topic"] == topic].copy()
                    if topic_df.empty:
                        continue

                    topic_safe = _safe_name(str(topic).lower())
                    prompt_file_label = PROMPT_FILE_LABEL.get(str(prompt_style), str(prompt_style))
                    out_jsonl = runs_root / topic_safe / (
                        f"{safe_model}_img-{image_variant}_prompt-{prompt_file_label}.jsonl"
                    )
                    _write_jsonl(topic_df, out_jsonl)

                    detail_cols = [
                        "model",
                        "image_variant",
                        "prompt_style",
                        "topic",
                        "id",
                        "source_index",
                        "ground_truth",
                        "expected_bias",
                        "prediction",
                        "prediction_norm",
                        "accuracy",
                        "bias_ratio",
                        "image_abs_path",
                    ]
                    detail_df = topic_df[detail_cols].copy()
                    detail_df["jsonl_path"] = str(out_jsonl)
                    detail_df["status"] = "ok"
                    detail_df["error"] = ""
                    detail_frames.append(detail_df)

                    n_total = int(len(topic_df))
                    n_correct = int(round(float(topic_df["accuracy"].sum())))
                    acc = float(topic_df["accuracy"].mean())
                    bias = float(topic_df["bias_ratio"].mean())
                    avg_tokens = (
                        float(topic_df["output_tokens"].dropna().mean())
                        if "output_tokens" in topic_df.columns and topic_df["output_tokens"].notna().any()
                        else None
                    )

                    summary_rows.append(
                        {
                            "model": model_name,
                            "image_variant": image_variant,
                            "prompt_style": prompt_style,
                            "topic": topic,
                            "n_total": n_total,
                            "n_correct": n_correct,
                            "accuracy": acc,
                            "bias_ratio": bias,
                            "avg_output_tokens": avg_tokens,
                            "jsonl_path": str(out_jsonl),
                            "status": "ok",
                            "error": "",
                        }
                    )
                    completed += 1
                    print(
                        f"[DONE {completed:03d}/{total}] model={model_name} image={image_variant} "
                        f"style={prompt_style} topic={topic} n={n_total} acc={acc:.4f} bias={bias:.4f}",
                        flush=True,
                    )

    summary_df = pd.DataFrame(summary_rows)
    summary_sort_cols = ["model", "prompt_style", "topic"]
    if "image_variant" in summary_df.columns:
        summary_sort_cols = ["model", "image_variant", "prompt_style", "topic"]
    summary_df = summary_df.sort_values(summary_sort_cols).reset_index(drop=True)
    if detail_frames:
        details_long = pd.concat(detail_frames, ignore_index=True)
        detail_sort_cols = ["model", "prompt_style", "topic", "source_index"]
        if "image_variant" in details_long.columns:
            detail_sort_cols = ["model", "image_variant", "prompt_style", "topic", "source_index"]
        details_long = details_long.sort_values(detail_sort_cols).reset_index(drop=True)
    else:
        details_long = pd.DataFrame(
            columns=[
                "model",
                "image_variant",
                "prompt_style",
                "topic",
                "id",
                "source_index",
                "ground_truth",
                "expected_bias",
                "prediction",
                "prediction_norm",
                "accuracy",
                "bias_ratio",
                "image_abs_path",
                "jsonl_path",
                "status",
                "error",
            ]
        )
    details_long.to_csv(results_dir / "vlms_are_biased_summary_long.csv", index=False)
    report_paths = write_matrix_reports(
        summary_df=summary_df,
        topics=args.topics,
        out_dir=results_dir,
        summary_matrix_csv_name="vlms_are_biased_summary_matrix.csv",
        bias_matrix_csv_name="vlms_are_biased_bias_matrix.csv",
        summary_xlsx_name=str(args.results_xlsx),
        combined_latex_path=None,
        details_csv_name=None,
    )

    config = {
        "models": list(args.models),
        "topics": list(args.topics),
        "prompt_styles": list(args.prompt_styles),
        "image_variants": list(args.image_variants),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "runs_dir": str(runs_root.resolve()),
        "results_dir": str(results_dir.resolve()),
        "results_xlsx": str(Path(report_paths["summary_xlsx"]).resolve()),
        "results_matrix_csv": str(Path(report_paths["summary_matrix_csv"]).resolve()),
        "results_bias_csv": str(Path(report_paths["bias_matrix_csv"]).resolve()),
        "latex_table": str(Path(report_paths["latex"]).resolve()) if "latex" in report_paths else None,
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "n_combinations": int(total),
        "n_combinations_completed": int(completed),
    }
    with (results_dir / "vlms_are_biased_matrix_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(
        "[INFO] Finished. "
        f"Summary CSV: {report_paths['summary_matrix_csv']} | "
        f"Bias CSV: {report_paths['bias_matrix_csv']} | "
        f"Excel: {report_paths['summary_xlsx']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
