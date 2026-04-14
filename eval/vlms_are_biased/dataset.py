from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from datasets import DownloadConfig, load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm

from .prompt_variants import PROMPT_STYLES, TOPICS, build_prompt_variants


HF_DATASET_NAME = "anvo25/vlms-are-biased"
PAPER_4SUBSET_TOPICS = TOPICS
PAPER_IMAGE_VARIANTS = ("original", "flipped")


def _safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_") or "item"


def _normalize_topics(topics: Iterable[str] | None) -> tuple[str, ...]:
    if topics is None:
        return tuple(TOPICS)
    cleaned = [str(t).strip() for t in topics if str(t).strip()]
    return tuple(cleaned)


def _normalize_prompt_styles(prompt_styles: Iterable[str] | None) -> tuple[str, ...]:
    if prompt_styles is None:
        return tuple(PROMPT_STYLES)
    cleaned = tuple(str(x).strip().lower() for x in prompt_styles if str(x).strip())
    bad = [x for x in cleaned if x not in PROMPT_STYLES]
    if bad:
        raise ValueError(f"Unsupported prompt styles: {bad}. Allowed: {PROMPT_STYLES}")
    return cleaned


def _load_hf_split(dataset_split: str):
    """Load split with online-first, local-cache fallback."""
    try:
        return load_dataset(HF_DATASET_NAME, split=dataset_split)
    except Exception as exc_online:
        try:
            return load_dataset(
                HF_DATASET_NAME,
                split=dataset_split,
                download_config=DownloadConfig(local_files_only=True),
            )
        except Exception as exc_local:
            raise RuntimeError(
                f"Failed to load {HF_DATASET_NAME}:{dataset_split}. "
                "Online download failed and no local cache was available."
            ) from exc_local

def _infer_template_variant(prompt: str) -> str:
    p = str(prompt).strip().lower()
    if p.startswith("count "):
        return "count"
    if p.startswith("how many "):
        return "how_many"
    return "other"


def _write_prompt_catalog_excel(out_df: pd.DataFrame, out_path: Path) -> None:
    if out_df.empty:
        return

    excel_path = out_path / "prompt_catalog_by_subset.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for topic, topic_df in out_df.groupby("topic", sort=True):
            tmp = topic_df.copy()
            tmp["template_variant"] = tmp["prompt"].apply(_infer_template_variant)
            sheet = (
                tmp.groupby(
                    ["prompt_style", "template_variant", "sub_topic", "prompt"],
                    as_index=False,
                    dropna=False,
                )
                .agg(n_instances=("index", "size"))
                .sort_values(["prompt_style", "template_variant", "sub_topic", "prompt"])
            )
            sheet_name = str(topic)[:31]
            sheet.to_excel(writer, sheet_name=sheet_name, index=False)


def _write_views(out_df: pd.DataFrame, out_path: Path, styles_norm: Sequence[str]) -> None:
    if out_df.empty:
        return

    # Global style-specific views.
    for style in styles_norm:
        out_df[out_df["prompt_style"] == style].to_parquet(out_path / f"instances_{style}.parquet", index=False)

    # Topic-level subset exports.
    subsets_dir = out_path / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    for topic, topic_df in out_df.groupby("topic", as_index=False):
        topic_safe = _safe_filename(str(topic).lower())
        topic_df.to_parquet(subsets_dir / f"{topic_safe}.parquet", index=False)
        for style in styles_norm:
            style_df = topic_df[topic_df["prompt_style"] == style]
            style_df.to_parquet(subsets_dir / f"{topic_safe}_{style}.parquet", index=False)

    counts = (
        out_df.groupby(["topic", "prompt_style"], as_index=False)
        .agg(n=("index", "size"))
        .sort_values(["topic", "prompt_style"])
    )
    counts.to_csv(out_path / "counts_by_topic_style.csv", index=False)
    _write_prompt_catalog_excel(out_df, out_path)


def _write_counts_by_topic(df: pd.DataFrame, out_path: Path, filename: str) -> pd.DataFrame:
    counts = (
        df.groupby("topic", as_index=False)
        .size()
        .sort_values("topic")
    )
    counts.to_csv(out_path / filename, index=False)
    return counts


def prepare_vlms_are_biased_dataset(
    out_dir: str | Path,
    dataset_split: str = "main",
    topics: Sequence[str] | None = None,
    prompt_styles: Sequence[str] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Prepare local parquet + images for VLMs-Are-Biased with Original/ITEM-Alias prompts."""
    out_path = Path(out_dir)
    images_dir = out_path / "images"
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    topics_norm = _normalize_topics(topics)
    styles_norm = _normalize_prompt_styles(prompt_styles)

    instances_path = out_path / "instances.parquet"
    if instances_path.exists() and not overwrite:
        out_df = pd.read_parquet(instances_path)
        _write_views(out_df, out_path, styles_norm)
        return out_df

    ds = _load_hf_split(dataset_split)

    rows: list[dict[str, object]] = []
    next_index = 1
    for src_idx in tqdm(range(len(ds)), desc="Preparing rows", unit="item"):
        item = ds[src_idx]
        topic = str(item.get("topic", "")).strip()
        if topic not in topics_norm:
            continue

        prompt_original = str(item.get("prompt", ""))

        item_id = str(item.get("ID", src_idx))
        image_filename = f"{_safe_filename(topic)}_{_safe_filename(item_id)}.png"
        image_path = images_dir / image_filename
        if not image_path.exists():
            item["image"].save(image_path)

        prompt_map = build_prompt_variants(topic=topic, original_prompt=prompt_original)
        for style in styles_norm:
            rows.append(
                {
                    "index": next_index,
                    "source_index": int(src_idx),
                    "dataset_split": str(dataset_split),
                    "topic": topic,
                    "sub_topic": str(item.get("sub_topic", "")),
                    "id": item_id,
                    "prompt_style": style,
                    "prompt": prompt_map[style],
                    "prompt_original": prompt_original,
                    "ground_truth": str(item.get("ground_truth", "")),
                    "expected_bias": str(item.get("expected_bias", "")),
                    "image_rel_path": str(Path("images") / image_filename),
                    "image_abs_path": str(image_path.resolve()),
                    "source_image_path": str(item.get("image_path", "")),
                }
            )
            next_index += 1

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(instances_path, index=False)
    _write_views(out_df, out_path, styles_norm)

    return out_df


def load_prepared_instances(
    dataset_path: str | Path,
    prompt_style: str = "both",
    topics: Sequence[str] | None = None,
    image_variant: str = "original",
) -> pd.DataFrame:
    path = Path(dataset_path)
    if path.is_dir():
        path = path / "instances.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    style = str(prompt_style).strip().lower()
    if style not in {"original", "item_alias", "both"}:
        raise ValueError("prompt_style must be one of: original, item_alias, both")
    if style != "both":
        df = df[df["prompt_style"] == style]

    if topics:
        topics_norm = {str(t).strip() for t in topics if str(t).strip()}
        df = df[df["topic"].isin(topics_norm)]

    variant = str(image_variant).strip().lower()
    if variant not in {"original", "flipped"}:
        raise ValueError("image_variant must be one of: original, flipped")

    df = df.reset_index(drop=True)
    if variant == "original":
        df["image_variant"] = "original"
        return df

    dataset_root = path.parent
    flipped_dir = dataset_root / "images_flipped"
    flipped_dir.mkdir(parents=True, exist_ok=True)

    unique_rel = sorted(df["image_rel_path"].astype(str).unique().tolist())
    for rel in unique_rel:
        rel_path = Path(rel)
        src_path = dataset_root / rel_path
        dst_path = flipped_dir / rel_path.name
        if dst_path.exists():
            try:
                with Image.open(dst_path) as chk:
                    chk.verify()
                continue
            except Exception:
                dst_path.unlink(missing_ok=True)
        if not src_path.exists():
            # Fallback to absolute path already materialized in parquet.
            src_alt = Path(
                df.loc[df["image_rel_path"].astype(str) == rel, "image_abs_path"].iloc[0]
            )
            src_path = src_alt
        tmp_dst = dst_path.with_name(dst_path.name + ".tmp.png")
        with Image.open(src_path) as img:
            ImageOps.flip(img).save(tmp_dst)
        tmp_dst.replace(dst_path)

    df["image_variant"] = "flipped"
    df["image_rel_path"] = df["image_rel_path"].astype(str).map(
        lambda p: str(Path("images_flipped") / Path(p).name)
    )
    df["image_abs_path"] = df["image_rel_path"].astype(str).map(
        lambda p: str((dataset_root / p).resolve())
    )
    return df


def materialize_variant_views(
    dataset_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    prompt_styles: Sequence[str] | None = None,
    image_variants: Sequence[str] | None = None,
    topics: Sequence[str] | None = None,
) -> dict[str, int]:
    dataset_root = Path(dataset_path)
    if dataset_root.is_file():
        dataset_root = dataset_root.parent
    variant_root = dataset_root if out_dir is None else Path(out_dir)
    variant_root.mkdir(parents=True, exist_ok=True)

    styles_norm = _normalize_prompt_styles(prompt_styles)
    variants_norm = tuple(str(v).strip().lower() for v in (image_variants or PAPER_IMAGE_VARIANTS))
    bad_variants = [v for v in variants_norm if v not in {"original", "flipped"}]
    if bad_variants:
        raise ValueError(f"Unsupported image variants: {bad_variants}")

    counts: dict[str, int] = {}
    for image_variant in variants_norm:
        for prompt_style in styles_norm:
            df = load_prepared_instances(
                dataset_path=dataset_root,
                prompt_style=prompt_style,
                topics=topics,
                image_variant=image_variant,
            )
            file_name = f"instances_img-{image_variant}_prompt-{prompt_style}.parquet"
            df.to_parquet(variant_root / file_name, index=False)
            counts[file_name] = int(len(df))
    return counts


def prepare_vlms_are_biased_paper_4subset(
    out_dir: str | Path,
    *,
    dataset_split: str = "original",
    overwrite: bool = False,
    materialize_flipped: bool = True,
    write_variant_parquets: bool = True,
) -> dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    full_df = prepare_vlms_are_biased_dataset(
        out_dir=out_path,
        dataset_split=dataset_split,
        topics=PAPER_4SUBSET_TOPICS,
        prompt_styles=PROMPT_STYLES,
        overwrite=overwrite,
    )

    # Preserve a backup of the full prompt-expanded cache before writing study views.
    full_df.to_parquet(out_path / "instances_with_identification.parquet", index=False)
    _write_counts_by_topic(full_df, out_path, "counts_by_topic.csv")

    original_df = full_df[full_df["prompt_style"] == "original"].reset_index(drop=True).copy()
    item_alias_df = full_df[full_df["prompt_style"] == "item_alias"].reset_index(drop=True).copy()

    original_df.to_parquet(out_path / "instances_original.parquet", index=False)
    item_alias_df.to_parquet(out_path / "instances_item_alias.parquet", index=False)

    # The study's counting-only view is the original-prompt slice over the 4-topic counting subset.
    counting_only_df = original_df.copy()
    counting_only_df.to_parquet(out_path / "instances_counting_only_original_prompt_only.parquet", index=False)
    counts_counting_only = _write_counts_by_topic(
        counting_only_df,
        out_path,
        "counts_by_topic_counting_only.csv",
    )

    variant_counts: dict[str, int] = {}
    if materialize_flipped:
        load_prepared_instances(
            dataset_path=out_path,
            prompt_style="both",
            topics=PAPER_4SUBSET_TOPICS,
            image_variant="flipped",
        )
    if write_variant_parquets:
        variant_counts = materialize_variant_views(
            dataset_path=out_path,
            prompt_styles=PROMPT_STYLES,
            image_variants=PAPER_IMAGE_VARIANTS,
            topics=PAPER_4SUBSET_TOPICS,
        )

    summary_counting_only = {
        "source_instances_backup": str(out_path / "instances_with_identification.parquet"),
        "filtered_instances": str(out_path / "instances.parquet"),
        "n_rows_before": int(len(full_df)),
        "n_rows_after": int(len(counting_only_df)),
        "removed_rows": int(len(full_df) - len(counting_only_df)),
        "counts_by_topic": {
            str(row["topic"]): int(row["size"])
            for _, row in counts_counting_only.iterrows()
        },
    }
    (out_path / "summary_counting_only.json").write_text(
        json.dumps(summary_counting_only, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset_split": str(dataset_split),
        "topics": list(PAPER_4SUBSET_TOPICS),
        "prompt_styles": list(PROMPT_STYLES),
        "image_variants": list(PAPER_IMAGE_VARIANTS),
        "n_rows_full": int(len(full_df)),
        "n_rows_original_prompt_only": int(len(counting_only_df)),
        "variant_counts": variant_counts,
        "out_dir": str(out_path),
    }
