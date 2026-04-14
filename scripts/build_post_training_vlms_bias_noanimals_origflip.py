#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build RLVR training parquet from VLMs-Are-Biased prepared data: "
            "exclude Animals, use original prompts, and duplicate each sample with "
            "original + flipped image variants."
        )
    )
    parser.add_argument(
        "--instances-parquet",
        type=Path,
        default=Path("dataset/vlms_are_biased_prepared/instances_original.parquet"),
        help="Prepared VLMs-Are-Biased parquet containing original prompts.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/vlms_are_biased_prepared"),
        help="Root dir for prepared images/ and images_flipped/.",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("dataset/post_training/VLBIAS_train_noanimals_origprompt_origflip.parquet"),
        help="Output RLVR training parquet path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("dataset/post_training/VLBIAS_train_noanimals_origprompt_origflip_summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260322,
        help="Shuffle seed for deterministic row order.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable row shuffling.",
    )
    parser.add_argument(
        "--resize-max-side",
        type=int,
        default=0,
        help=(
            "If > 0, write resized image copies (max(width,height)=this value) and point parquet "
            "rows to those resized files."
        ),
    )
    parser.add_argument(
        "--resized-image-root",
        type=Path,
        default=Path("dataset/vlms_are_biased_prepared_resized/max_side_512"),
        help="Root folder where resized images are written when --resize-max-side > 0.",
    )
    return parser.parse_args()


def _as_repo_relative(path: Path, repo_root: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(repo_root.resolve()))
    except ValueError:
        return str(p)


def _resolve_variant_path(
    row: pd.Series,
    *,
    image_variant: str,
    dataset_root: Path,
    # return absolute path; conversion to repo-relative happens later.
) -> Path:
    rel = str(row.get("image_rel_path", "")).strip()
    if not rel:
        raise ValueError(f"Missing image_rel_path for id={row.get('id')}")

    if image_variant == "original":
        cand = dataset_root / rel
    elif image_variant == "flipped":
        if rel.startswith("images/"):
            rel_flip = rel.replace("images/", "images_flipped/", 1)
        else:
            rel_flip = rel.replace("/images/", "/images_flipped/")
        cand = dataset_root / rel_flip
    else:
        raise ValueError(f"Unknown image_variant: {image_variant}")

    if not cand.exists():
        raise FileNotFoundError(f"Image not found for variant={image_variant}: {cand}")
    return cand.resolve()


def _resize_copy_if_needed(
    src_abs: Path,
    *,
    resize_max_side: int,
    dataset_root: Path,
    resized_root: Path,
    repo_root: Path,
    path_cache: Dict[str, str],
    size_cache: Dict[str, tuple[int, int]],
) -> str:
    src_key = str(src_abs)
    if src_key in path_cache:
        return path_cache[src_key]

    if resize_max_side <= 0:
        rel_path = _as_repo_relative(src_abs, repo_root=repo_root)
        with Image.open(src_abs) as im:
            size_cache[rel_path] = (int(im.width), int(im.height))
        path_cache[src_key] = rel_path
        return rel_path

    try:
        sub_rel = src_abs.relative_to(dataset_root.resolve())
    except ValueError:
        # Fallback: still keep deterministic file placement.
        sub_rel = Path(src_abs.name)
    dst_abs = (resized_root / sub_rel).resolve()
    dst_abs.parent.mkdir(parents=True, exist_ok=True)

    if not dst_abs.exists():
        with Image.open(src_abs) as im:
            w, h = int(im.width), int(im.height)
            max_side = max(w, h)
            if max_side > resize_max_side:
                scale = float(resize_max_side) / float(max_side)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
                if resized.mode != "RGB":
                    resized = resized.convert("RGB")
                resized.save(dst_abs)
            else:
                # Keep a local copy under resized root for a self-contained image tree.
                shutil.copy2(src_abs, dst_abs)

    rel_path = _as_repo_relative(dst_abs, repo_root=repo_root)
    with Image.open(dst_abs) as im2:
        size_cache[rel_path] = (int(im2.width), int(im2.height))
    path_cache[src_key] = rel_path
    return rel_path


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = args.instances_parquet
    if not src_path.exists():
        raise FileNotFoundError(f"Missing instances parquet: {src_path}")

    src = pd.read_parquet(src_path)
    if "topic" not in src.columns:
        raise ValueError("Expected 'topic' column in source parquet.")

    filt = src[(src["topic"] != "Animals") & (src["prompt_style"] == "original")].copy()
    if filt.empty:
        raise RuntimeError("Filtered source is empty (check input parquet/topics).")

    path_cache: Dict[str, str] = {}
    size_cache: Dict[str, tuple[int, int]] = {}
    dataset_root_resolved = args.dataset_root.resolve()
    resized_root_resolved = args.resized_image_root.resolve()

    records: List[Dict[str, object]] = []
    for _, row in filt.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        if not prompt.startswith("<image>"):
            prompt = f"<image>{prompt}"
        answer = str(row.get("ground_truth", "")).strip()
        if not answer:
            continue

        for image_variant in ("original", "flipped"):
            src_image_abs = _resolve_variant_path(
                row,
                image_variant=image_variant,
                dataset_root=args.dataset_root,
            )
            image_path = _resize_copy_if_needed(
                src_image_abs,
                resize_max_side=int(args.resize_max_side),
                dataset_root=dataset_root_resolved,
                resized_root=resized_root_resolved,
                repo_root=repo_root,
                path_cache=path_cache,
                size_cache=size_cache,
            )
            records.append(
                {
                    "images": np.array([image_path], dtype=object),
                    "problem": prompt,
                    "answer": answer,
                    "topic": str(row.get("topic", "")),
                    "sub_topic": str(row.get("sub_topic", "")),
                    "image_variant": image_variant,
                    "source_index": int(row.get("source_index", -1)),
                    "source_id": str(row.get("id", "")),
                    "expected_bias": str(row.get("expected_bias", "")),
                }
            )

    out_df = pd.DataFrame.from_records(records)
    if out_df.empty:
        raise RuntimeError("No rows were built for output dataset.")

    if not args.no_shuffle:
        out_df = out_df.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)

    widths = [wh[0] for wh in size_cache.values()] if size_cache else []
    heights = [wh[1] for wh in size_cache.values()] if size_cache else []
    max_side_seen = max(max(widths), max(heights)) if widths and heights else None
    min_side_seen = min(min(widths), min(heights)) if widths and heights else None

    summary: Dict[str, object] = {
        "source_parquet": str(src_path),
        "dataset_root": str(args.dataset_root),
        "resize_max_side": int(args.resize_max_side),
        "resized_image_root": str(args.resized_image_root) if int(args.resize_max_side) > 0 else None,
        "out_parquet": str(args.out_parquet),
        "rows_source_total": int(len(src)),
        "rows_source_filtered_noanimals_original_prompt": int(len(filt)),
        "rows_output_total": int(len(out_df)),
        "unique_image_paths_output": int(len(size_cache)),
        "image_size_stats": {
            "max_width": max(widths) if widths else None,
            "max_height": max(heights) if heights else None,
            "max_side": max_side_seen,
            "min_width": min(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "min_side": min_side_seen,
        },
        "topics_included": sorted(out_df["topic"].unique().tolist()),
        "image_variants": sorted(out_df["image_variant"].unique().tolist()),
        "counts_by_topic_variant": (
            out_df.groupby(["topic", "image_variant"], as_index=False).size().to_dict(orient="records")
        ),
        "counts_by_topic": out_df.groupby(["topic"], as_index=False).size().to_dict(orient="records"),
        "counts_by_variant": out_df.groupby(["image_variant"], as_index=False).size().to_dict(orient="records"),
        "seed": int(args.seed),
        "shuffled": bool(not args.no_shuffle),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = _parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
