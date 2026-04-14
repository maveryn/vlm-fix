#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.model_registry import MODEL_REGISTRY
from eval.vlms_are_biased.dataset import load_prepared_instances, prepare_vlms_are_biased_dataset
from eval.vlms_are_biased.metrics import add_scores, aggregate, save_reports
from eval.vlms_are_biased.prompt_variants import TOPICS
from eval.vlms_are_biased.reporting import write_matrix_reports
from eval.vlms_are_biased.runner import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLMs-Are-Biased evaluation with Original / ITEM-Alias prompt styles."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-VL-7B-Instruct"],
        help="Model names from eval.model_registry.MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/vlms_are_biased_prepared",
        help="Prepared dataset directory (contains instances.parquet).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="main",
        help="HF split used when auto-preparing missing dataset.",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="both",
        choices=["original", "item_alias", "both"],
        help="Prompt style to evaluate.",
    )
    parser.add_argument(
        "--image-variant",
        type=str,
        default="original",
        choices=["original", "flipped"],
        help="Image variant to evaluate.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=list(TOPICS),
        help="Topic subset to evaluate.",
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
        "--out",
        type=str,
        default="outputs/vlms_are_biased_eval",
        help="Output directory for predictions and reports.",
    )
    parser.add_argument(
        "--prepare-if-missing",
        action="store_true",
        help="Auto-prepare dataset if instances.parquet is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    unknown = [m for m in args.models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unsupported models: {unknown}. Known keys: {sorted(MODEL_REGISTRY.keys())}"
        )

    dataset_dir = Path(args.dataset_dir)
    instances_path = dataset_dir / "instances.parquet"
    if not instances_path.exists():
        if not args.prepare_if_missing:
            raise FileNotFoundError(
                f"Missing prepared dataset at {instances_path}. "
                "Run eval/prepare_vlms_are_biased.py first, or pass --prepare-if-missing."
            )

        styles = ["original", "item_alias"] if args.prompt_style == "both" else [args.prompt_style]
        prepare_vlms_are_biased_dataset(
            out_dir=dataset_dir,
            dataset_split=str(args.dataset_split),
            topics=args.topics,
            prompt_styles=styles,
            overwrite=False,
        )

    instances = load_prepared_instances(
        dataset_path=dataset_dir,
        prompt_style=str(args.prompt_style),
        topics=args.topics,
        image_variant=str(args.image_variant),
    )
    if len(instances) == 0:
        raise RuntimeError("No rows to evaluate after filtering. Check topics/prompt-style arguments.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = run_inference(
        model_names=args.models,
        instances=instances,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        out_dir=out_dir,
    )

    pred_scored = add_scores(pred_raw)
    summary, deltas, payload = aggregate(pred_scored)
    save_reports(out_dir=out_dir, pred_df=pred_scored, summary=summary, deltas=deltas, payload=payload)

    run_cfg = {
        "models": list(args.models),
        "dataset_dir": str(dataset_dir),
        "dataset_split": str(args.dataset_split),
        "prompt_style": str(args.prompt_style),
        "image_variant": str(args.image_variant),
        "topics": list(args.topics),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "prepare_if_missing": bool(args.prepare_if_missing),
        "n_instances": int(len(instances)),
    }
    report_paths = write_matrix_reports(
        summary_df=summary,
        topics=list(args.topics),
        out_dir=out_dir,
        summary_matrix_csv_name="summary_matrix.csv",
        bias_matrix_csv_name="bias_matrix.csv",
        summary_xlsx_name="summary.xlsx",
        combined_latex_path=out_dir / "summary_table.tex",
        details_csv_name="summary_long.csv",
    )
    run_cfg["formatted_reports"] = report_paths

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print("\nDeltas (vs original):")
    print(deltas.to_string(index=False))
    print(f"\nSaved reports to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
