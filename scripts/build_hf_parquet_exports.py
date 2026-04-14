#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import Dataset, Image as HFImage
from datasets import Sequence as HFSequence
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sft.scripts.build_finetune_parquet import build_all as build_vlm_fix_posttrain  # noqa: E402
from scripts.build_vlm_fix_text_only_dataset import build_text_only_dataset  # noqa: E402
from vlm_fix.generation import build_dataset as build_vlm_fix_dataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local HF-uploadable parquet exports for vlm_fix, vlm_fix_text_only, "
            "VLM-Fix post-training splits (D1/D2/D3), and synth_legs_train."
        )
    )
    parser.add_argument("--export-root", type=Path, default=Path("hf_export"))
    parser.add_argument("--work-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--states-per-game", type=int, default=300)
    parser.add_argument("--vlm-fix-seed", type=int, default=7)
    parser.add_argument("--posttrain-seed", type=int, default=20260317)
    parser.add_argument("--synth-legs-seed", type=int, default=20260322)
    parser.add_argument("--synth-legs-n-total", type=int, default=8192)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _encode_image_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _write_parquet_dataframe(df: pd.DataFrame, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return int(len(df))


def _write_dataset(ds: Dataset, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_path))
    return int(len(ds))


def _copy_dir_contents(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def _dataset_source_map(root: Path) -> dict[str, Path]:
    return {
        "vlm_fix_dir": root / "vlm_fix",
        "vlm_fix_text_only_dir": root / "vlm_fix_text_only",
        "posttrain_dir": root / "post_training_v2_disjoint",
        "synth_legs_parquet": root / "post_training" / "synth_legs_train_8192.parquet",
        "synth_legs_image_root": root / "post_training" / "synth_legs",
        "synth_legs_summary": root / "post_training" / "synth_legs_train_8192_summary.json",
    }


def _discover_prepared_dataset_sources() -> dict[str, Path] | None:
    candidate_roots: list[Path] = []
    direct = ROOT.parent / "dataset"
    if direct not in candidate_roots:
        candidate_roots.append(direct)

    for sibling in sorted(ROOT.parent.iterdir()):
        if not sibling.is_dir():
            continue
        candidate = sibling / "dataset"
        if candidate not in candidate_roots:
            candidate_roots.append(candidate)

    for root in candidate_roots:
        required = _dataset_source_map(root)
        if all(path.exists() for path in required.values()):
            return required
    return None


def _discover_existing_synth_legs_source() -> tuple[Path, Path, Path] | None:
    for root in [ROOT.parent / "dataset", *[p / "dataset" for p in sorted(ROOT.parent.iterdir()) if p.is_dir()]]:
        parquet_path = root / "post_training" / "synth_legs_train_8192.parquet"
        image_root = root / "post_training" / "synth_legs"
        summary_path = root / "post_training" / "synth_legs_train_8192_summary.json"
        if parquet_path.exists() and image_root.exists():
            return parquet_path, image_root, summary_path
    return None


def _stage_existing_synth_legs_source(
    *,
    src_parquet: Path,
    src_image_root: Path,
    src_summary: Path,
    dst_dir: Path,
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_image_root = dst_dir / "images"
    if dst_image_root.exists():
        shutil.rmtree(dst_image_root)
    shutil.copytree(src_image_root / "images", dst_image_root)

    df = pd.read_parquet(src_parquet).copy()
    old_prefix = "dataset/post_training/synth_legs/images/"
    new_prefix = "data/generated/synth_legs/images/"
    if "image_rel_path" in df.columns:
        df["image_rel_path"] = df["image_rel_path"].astype(str).str.replace(
            old_prefix, new_prefix, regex=False
        )
    if "images" in df.columns:
        df["images"] = df["images"].apply(
            lambda arr: [str(x).replace(old_prefix, new_prefix) for x in list(arr)]
        )

    out_parquet = dst_dir / "train_8192.parquet"
    df.to_parquet(out_parquet, index=False)

    if src_summary.exists():
        summary = json.loads(src_summary.read_text(encoding="utf-8"))
        summary["out_parquet"] = str(out_parquet)
        summary["out_image_dir"] = str(dst_image_root)
        (dst_dir / "train_8192_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return out_parquet


def _build_local_sources(args: argparse.Namespace) -> dict[str, Path]:
    work_root = (ROOT / args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    vlm_fix_dir = work_root / "vlm_fix"
    vlm_fix_text_only_dir = work_root / "vlm_fix_text_only"
    posttrain_dir = work_root / "post_training_v2_disjoint"
    synth_legs_dir = work_root / "synth_legs"

    staged_from_source = False
    source_paths = _discover_prepared_dataset_sources()
    if source_paths is not None:
        _copy_dir_contents(source_paths["vlm_fix_dir"], vlm_fix_dir)
        _copy_dir_contents(source_paths["vlm_fix_text_only_dir"], vlm_fix_text_only_dir)
        _copy_dir_contents(source_paths["posttrain_dir"], posttrain_dir)
        _stage_existing_synth_legs_source(
            src_parquet=source_paths["synth_legs_parquet"],
            src_image_root=source_paths["synth_legs_image_root"],
            src_summary=source_paths["synth_legs_summary"],
            dst_dir=synth_legs_dir,
        )
        staged_from_source = True

    if not staged_from_source:
        vlm_fix_instances = vlm_fix_dir / "instances.parquet"
        if args.force_rebuild or not vlm_fix_instances.exists():
            build_vlm_fix_dataset(
                out_dir=vlm_fix_dir,
                states_per_game=int(args.states_per_game),
                seed=int(args.vlm_fix_seed),
            )

        vlm_fix_text_only_instances = vlm_fix_text_only_dir / "instances.parquet"
        if args.force_rebuild or not vlm_fix_text_only_instances.exists():
            build_text_only_dataset(
                src_parquet=vlm_fix_instances,
                out_dir=vlm_fix_text_only_dir,
                games=["tictactoe", "reversi", "connect4", "dots_boxes"],
                source_image_text_order="image_first",
            )

        d1_train = posttrain_dir / "D1_train_standard_all4_8192.parquet"
        if args.force_rebuild or not d1_train.exists():
            build_vlm_fix_posttrain(
                instances_path=vlm_fix_instances,
                benchmark_root=vlm_fix_dir,
                out_dir=posttrain_dir,
                seed=int(args.posttrain_seed),
                exclude_benchmark_from_train=True,
            )

        synth_legs_train = synth_legs_dir / "train_8192.parquet"
        if args.force_rebuild or not synth_legs_train.exists():
            staged = False
            if not args.force_rebuild:
                existing = _discover_existing_synth_legs_source()
                if existing is not None:
                    src_parquet, src_image_root, src_summary = existing
                    _stage_existing_synth_legs_source(
                        src_parquet=src_parquet,
                        src_image_root=src_image_root,
                        src_summary=src_summary,
                        dst_dir=synth_legs_dir,
                    )
                    staged = True
            if not staged:
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "build_synth_legs_dataset.py"),
                    "--n-total",
                    str(int(args.synth_legs_n_total)),
                    "--seed",
                    str(int(args.synth_legs_seed)),
                    "--out-image-dir",
                    str(synth_legs_dir / "images"),
                    "--out-parquet",
                    str(synth_legs_train),
                    "--out-summary",
                    str(synth_legs_dir / "train_8192_summary.json"),
                    "--sample-xlsx",
                    str(synth_legs_dir / "synth_legs_20_samples.xlsx"),
                ]
                subprocess.run(cmd, check=True, cwd=str(ROOT))

    return {
        "vlm_fix_dir": vlm_fix_dir,
        "vlm_fix_text_only_dir": vlm_fix_text_only_dir,
        "posttrain_dir": posttrain_dir,
        "synth_legs_dir": synth_legs_dir,
    }


def _export_vlm_fix(vlm_fix_dir: Path, export_root: Path) -> dict[str, object]:
    out_dir = export_root / "vlm_fix"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_path = vlm_fix_dir / "instances.parquet"
    df = pd.read_parquet(instances_path).copy()
    if "index" not in df.columns:
        df.insert(0, "index", range(1, len(df) + 1))

    df["image_rel_path"] = df["image_path"].astype(str)
    df["image_key"] = (
        df["game"].astype(str)
        + "__"
        + df["render_variant"].astype(str)
        + "__"
        + df["state_id"].astype(int).astype(str).str.zfill(3)
    )

    main_cols = [
        "index",
        "game",
        "state_id",
        "board_state",
        "render_variant",
        "rule_variant",
        "image_text_order",
        "prompt_variant",
        "prompt_type",
        "question_target",
        "prompt",
        "answer",
        "valid_labels",
        "image_key",
        "image_rel_path",
        "render_meta",
        "standard_winner_idx",
        "inverse_winner_idx",
    ]
    main_df = df[main_cols].copy()
    main_rows = _write_parquet_dataframe(main_df, out_dir / "main.parquet")

    image_records = []
    image_df = (
        df[["image_key", "game", "state_id", "render_variant", "image_rel_path"]]
        .drop_duplicates()
        .sort_values(["game", "render_variant", "state_id"], kind="stable")
        .reset_index(drop=True)
    )
    for _, row in image_df.iterrows():
        rel_path = str(row["image_rel_path"])
        abs_path = vlm_fix_dir / rel_path
        image_records.append(
            {
                "image_key": str(row["image_key"]),
                "game": str(row["game"]),
                "state_id": int(row["state_id"]),
                "render_variant": str(row["render_variant"]),
                "image_rel_path": rel_path,
                "image": {"bytes": _encode_image_bytes(abs_path), "path": rel_path},
            }
        )
    image_ds = Dataset.from_list(image_records).cast_column("image", HFImage())
    image_rows = _write_dataset(image_ds, out_dir / "images.parquet")

    summary = {
        "config": "vlm_fix",
        "splits": {"main": main_rows, "images": image_rows},
        "source_instances": str(instances_path),
        "unique_images": int(len(image_df)),
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _export_vlm_fix_text_only(vlm_fix_text_only_dir: Path, export_root: Path) -> dict[str, object]:
    out_dir = export_root / "vlm_fix_text_only"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = vlm_fix_text_only_dir / "instances.parquet"
    df = pd.read_parquet(src).copy()
    rows = _write_parquet_dataframe(df, out_dir / "main.parquet")
    summary = {"config": "vlm_fix_text_only", "splits": {"main": rows}, "source_instances": str(src)}
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return int(pd.read_parquet(src).shape[0])


def _export_posttrain(posttrain_dir: Path, export_root: Path) -> list[dict[str, object]]:
    mapping = {
        "vlm_fix_posttrain_d1": {
            "train": posttrain_dir / "D1_train_standard_all4_8192.parquet",
            "test": posttrain_dir / "D1_test_inverse_core2400.parquet",
        },
        "vlm_fix_posttrain_d2": {
            "train": posttrain_dir / "D2_train_inverse_all4_8192.parquet",
            "test": posttrain_dir / "D2_test_standard_core2400.parquet",
        },
        "vlm_fix_posttrain_d3": {
            "train": posttrain_dir / "D3_train_ttt_reversi_both_rules_8192.parquet",
            "test": posttrain_dir / "D3_test_connect4_dots_both_rules_core2400.parquet",
        },
    }
    out: list[dict[str, object]] = []
    for config_name, split_map in mapping.items():
        out_dir = export_root / config_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_counts = {split: _copy(src, out_dir / f"{split}.parquet") for split, src in split_map.items()}
        summary = {
            "config": config_name,
            "splits": split_counts,
            "sources": {split: str(src) for split, src in split_map.items()},
        }
        (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        out.append(summary)
    return out


def _iter_synth_records(df: pd.DataFrame) -> Iterable[dict[str, object]]:
    for _, row in df.iterrows():
        rel_path = str(row["image_rel_path"])
        abs_path = ROOT / rel_path
        yield {
            "images": [{"bytes": _encode_image_bytes(abs_path), "path": rel_path}],
            "problem": str(row["problem"]),
            "prompt": str(row["prompt"]),
            "answer": str(row["answer"]),
            "subset": str(row["subset"]),
            "species": str(row["species"]),
            "style": str(row["style"]),
            "legs": int(row["legs"]),
            "image_rel_path": rel_path,
        }


def _export_synth_legs(synth_legs_dir: Path, export_root: Path) -> dict[str, object]:
    out_dir = export_root / "synth_legs_train"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = synth_legs_dir / "train_8192.parquet"
    df = pd.read_parquet(src).copy()
    ds = Dataset.from_list(list(_iter_synth_records(df)))
    ds = ds.cast_column("images", HFSequence(HFImage()))
    rows = _write_dataset(ds, out_dir / "train.parquet")

    summary = {"config": "synth_legs_train", "splits": {"train": rows}, "source_instances": str(src)}
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_upload_readme(export_root: Path, summaries: list[dict[str, object]]) -> None:
    yaml_lines = ["---", "configs:"]
    for item in summaries:
        yaml_lines.append(f"- config_name: {item['config']}")
        yaml_lines.append("  data_files:")
        for split in item["splits"]:
            yaml_lines.append(f"  - split: {split}")
            yaml_lines.append(f"    path: {item['config']}/{split}.parquet")
    yaml_lines.append("---")
    yaml_lines.append("")
    yaml_lines.append("# vlm-fix HF Upload")
    yaml_lines.append("")
    yaml_lines.append("This folder contains local parquet exports prepared for Hugging Face upload.")
    yaml_lines.append("")
    yaml_lines.append("Included configs:")
    for item in summaries:
        splits = ", ".join(item["splits"].keys())
        yaml_lines.append(f"- `{item['config']}`: {splits}")
    yaml_lines.append("")
    yaml_lines.append("Not included:")
    yaml_lines.append("- `VLMBias` training data")
    yaml_lines.append("")
    yaml_lines.append("VLMBias is derived from `anvo25/vlms-are-biased` using the existing prep/build code.")
    (export_root / "README.md").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    export_root = (ROOT / args.export_root).resolve()
    if args.force_export or export_root.exists():
        _ensure_clean_dir(export_root)
    else:
        export_root.mkdir(parents=True, exist_ok=True)

    sources = _build_local_sources(args)

    summaries: list[dict[str, object]] = []
    summaries.append(_export_vlm_fix(sources["vlm_fix_dir"], export_root))
    summaries.append(_export_vlm_fix_text_only(sources["vlm_fix_text_only_dir"], export_root))
    summaries.extend(_export_posttrain(sources["posttrain_dir"], export_root))
    summaries.append(_export_synth_legs(sources["synth_legs_dir"], export_root))

    _write_upload_readme(export_root, summaries)
    manifest = {
        "export_root": str(export_root),
        "configs": summaries,
        "notes": {
            "vlm_fix": "Normalized into main + images splits to avoid repeating image bytes in every benchmark row.",
            "vlmbias": "Excluded from HF export by request; derive from anvo25/vlms-are-biased instead.",
        },
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
