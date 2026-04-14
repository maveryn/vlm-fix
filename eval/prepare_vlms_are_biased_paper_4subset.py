#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.vlms_are_biased.dataset import prepare_vlms_are_biased_paper_4subset
from eval.vlms_are_biased.prompt_variants import TOPICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the 4-topic counting subset used in the paper from "
            "anvo25/vlms-are-biased, including original/item-alias prompts and "
            "original/flipped image variants."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/generated/vlms_are_biased_hf_original_4subset_322",
        help="Output directory for the prepared 4-topic paper subset.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="original",
        help="HF split name for the paper subset source. Defaults to `original`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild prepared images/parquets even if they already exist.",
    )
    parser.add_argument(
        "--skip-flipped",
        action="store_true",
        help="Do not pre-materialize flipped images.",
    )
    parser.add_argument(
        "--skip-variant-parquets",
        action="store_true",
        help="Do not write explicit per-combination parquet views for the 2x2 study matrix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_vlms_are_biased_paper_4subset(
        out_dir=Path(args.out_dir),
        dataset_split=str(args.dataset_split),
        overwrite=bool(args.overwrite),
        materialize_flipped=bool(not args.skip_flipped),
        write_variant_parquets=bool(not args.skip_variant_parquets),
    )

    run_cfg = {
        "dataset_split": str(args.dataset_split),
        "topics": list(TOPICS),
        "out_dir": str(args.out_dir),
        "overwrite": bool(args.overwrite),
        "materialize_flipped": bool(not args.skip_flipped),
        "write_variant_parquets": bool(not args.skip_variant_parquets),
        "summary": summary,
    }
    out_dir = Path(args.out_dir)
    with (out_dir / "prepare_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved paper subset to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
