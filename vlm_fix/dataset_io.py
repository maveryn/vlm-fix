from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from PIL import Image


DEFAULT_HF_CACHE_DIR = Path("data/hf_cache")


def _safe_slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(text).strip())
    value = value.strip("-")
    return value or "default"


def _ensure_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        img = value
    elif isinstance(value, bytes):
        img = Image.open(BytesIO(value))
    elif isinstance(value, str):
        img = Image.open(value)
    elif isinstance(value, dict) and "bytes" in value:
        img = Image.open(BytesIO(value["bytes"]))
    else:
        raise TypeError(f"Unsupported image value type: {type(value)}")
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _default_image_rel_path(record: dict[str, Any], row_idx: int) -> Path:
    game = _safe_slug(str(record.get("game", "unknown_game")))
    render_variant = _safe_slug(str(record.get("render_variant", "default")))
    state_id = str(record.get("state_id", row_idx + 1)).strip() or str(row_idx + 1)
    return Path("images") / game / render_variant / f"state_{state_id}.png"


def _normalize_image_rel_path(raw_path: str, record: dict[str, Any], row_idx: int) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return _default_image_rel_path(record=record, row_idx=row_idx)

    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return _default_image_rel_path(record=record, row_idx=row_idx)
    return path


def _materialize_hf_images(
    rows: list[dict[str, Any]],
    cache_root: Path,
    image_column: str,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(rows):
        if image_column not in item:
            raise KeyError(f"HF row is missing expected image column: {image_column}")

        record = {k: v for k, v in item.items() if k != image_column}
        rel_path = _normalize_image_rel_path(
            raw_path=str(record.get("image_path", "")),
            record=record,
            row_idx=row_idx,
        )
        out_path = cache_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            img = _ensure_pil(item[image_column])
            img.save(out_path)
        record["image_path"] = str(out_path.resolve())
        out_rows.append(record)
    return out_rows


def _materialize_hf_images_from_lookup(
    rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    cache_root: Path,
    image_column: str,
    image_key_column: str,
) -> list[dict[str, Any]]:
    image_path_by_key: dict[str, str] = {}
    for row_idx, item in enumerate(image_rows):
        if image_key_column not in item:
            raise KeyError(f"HF image row is missing expected key column: {image_key_column}")
        if image_column not in item:
            raise KeyError(f"HF image row is missing expected image column: {image_column}")

        rel_path = _normalize_image_rel_path(
            raw_path=str(item.get("image_rel_path") or item.get("image_path") or ""),
            record=item,
            row_idx=row_idx,
        )
        out_path = cache_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            img = _ensure_pil(item[image_column])
            img.save(out_path)
        image_path_by_key[str(item[image_key_column])] = str(out_path.resolve())

    out_rows: list[dict[str, Any]] = []
    for item in rows:
        key = str(item.get(image_key_column, "")).strip()
        if not key:
            raise KeyError(f"HF main row is missing expected image key: {image_key_column}")
        if key not in image_path_by_key:
            raise KeyError(f"No image row found for {image_key_column}={key}")
        record = dict(item)
        record["image_path"] = image_path_by_key[key]
        out_rows.append(record)
    return out_rows


def _load_hf_split(
    *,
    hf_repo: str,
    hf_config: str | None,
    hf_split: str,
    hf_revision: str | None,
):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HF dataset mode requires the `datasets` package. "
            "Install it with `pip install datasets` or use --dataset-source local."
        ) from exc

    try:
        return load_dataset(
            path=hf_repo,
            name=hf_config,
            split=hf_split,
            revision=hf_revision,
        )
    except Exception:
        return load_dataset(
            path=hf_repo,
            name=hf_config,
            split=hf_split,
            revision=hf_revision,
            download_config=DownloadConfig(local_files_only=True),
        )


def load_instances_frame(
    *,
    dataset_source: str,
    dataset_dir: str | Path | None,
    hf_repo: str | None,
    hf_config: str | None,
    hf_split: str,
    hf_revision: str | None,
    hf_cache_dir: str | Path = DEFAULT_HF_CACHE_DIR,
    image_column: str = "image",
    require_image_column: bool = False,
    hf_image_split: str = "images",
    image_key_column: str = "image_key",
) -> Tuple[pd.DataFrame, Path]:
    source = str(dataset_source).strip().lower()

    if source == "local":
        if dataset_dir is None:
            raise ValueError("dataset_dir is required when dataset_source=local")
        dataset_root = Path(dataset_dir)
        instances_path = dataset_root / "instances.parquet"
        if not instances_path.exists():
            raise FileNotFoundError(f"Missing dataset: {instances_path}")
        df = pd.read_parquet(instances_path).copy()
        if "index" not in df.columns:
            df.insert(0, "index", range(1, len(df) + 1))
        return df, dataset_root.resolve()

    if source != "hf":
        raise ValueError("dataset_source must be one of: local, hf")
    if not hf_repo:
        raise ValueError("hf_repo is required when dataset_source=hf")

    ds = _load_hf_split(
        hf_repo=str(hf_repo),
        hf_config=str(hf_config) if hf_config else None,
        hf_split=str(hf_split),
        hf_revision=str(hf_revision) if hf_revision else None,
    )

    cache_root = (
        Path(hf_cache_dir)
        / _safe_slug(str(hf_repo))
        / _safe_slug(str(hf_config or "default"))
        / _safe_slug(str(hf_split))
        / _safe_slug(str(hf_revision or "latest"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = [ds[i] for i in range(len(ds))]
    if require_image_column:
        if rows and image_column not in rows[0] and image_key_column in rows[0]:
            ds_images = _load_hf_split(
                hf_repo=str(hf_repo),
                hf_config=str(hf_config) if hf_config else None,
                hf_split=str(hf_image_split),
                hf_revision=str(hf_revision) if hf_revision else None,
            )
            image_rows = [ds_images[i] for i in range(len(ds_images))]
            records = _materialize_hf_images_from_lookup(
                rows=rows,
                image_rows=image_rows,
                cache_root=cache_root,
                image_column=image_column,
                image_key_column=image_key_column,
            )
        else:
            records = _materialize_hf_images(rows=rows, cache_root=cache_root, image_column=image_column)
    else:
        records = [{k: v for k, v in item.items() if k != image_column} for item in rows]

    df = pd.DataFrame(records)
    if "index" not in df.columns:
        df.insert(0, "index", range(1, len(df) + 1))
    return df, cache_root.resolve()
