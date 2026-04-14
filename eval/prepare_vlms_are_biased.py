#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.vlms_are_biased.dataset import prepare_vlms_are_biased_dataset
from eval.vlms_are_biased.prompt_variants import PROMPT_STYLES, TOPICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local VLMs-Are-Biased dataset cache (images + parquet) with "
            "Original and ITEM-Alias prompt styles."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/vlms_are_biased_prepared",
        help="Output directory for prepared dataset artifacts.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="main",
        help="HF split name (e.g., main, remove_background_q1q2).",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=list(TOPICS),
        help="Topics to include.",
    )
    parser.add_argument(
        "--prompt-styles",
        nargs="+",
        default=list(PROMPT_STYLES),
        choices=list(PROMPT_STYLES),
        help="Prompt styles to materialize.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared parquet files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    df = prepare_vlms_are_biased_dataset(
        out_dir=out_dir,
        dataset_split=str(args.dataset_split),
        topics=args.topics,
        prompt_styles=args.prompt_styles,
        overwrite=bool(args.overwrite),
    )

    counts = (
        df.groupby(["topic", "prompt_style"], as_index=False)
        .agg(n=("index", "size"))
        .sort_values(["topic", "prompt_style"])
    )

    run_cfg = {
        "dataset_split": str(args.dataset_split),
        "topics": list(args.topics),
        "prompt_styles": list(args.prompt_styles),
        "n_rows": int(len(df)),
    }
    with (out_dir / "prepare_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(counts.to_string(index=False))
    print(f"\nSaved: {out_dir / 'instances.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
