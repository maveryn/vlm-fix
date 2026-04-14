#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

GAMES = ["tictactoe", "reversi", "connect4", "dots_boxes"]
GAME_DISPLAY = {
    "tictactoe": "Tic-Tac-Toe",
    "reversi": "Reversi",
    "connect4": "Connect Four",
    "dots_boxes": "Dots and Boxes",
}
PROMPT_VARIANTS = ["standard", "tag", "tag_sem"]
PROMPT_DISPLAY = {
    "standard": "Base",
    "tag": "Alias",
    "tag_sem": "SemAlias",
}
MODEL_SPECS = [
    ("GPT-5.2", "gpt-5.2__prompt-direct-reduced300.jsonl"),
    ("Molmo2 4B", "allenai_Molmo2-4B__prompt-direct.jsonl"),
    ("Molmo2 8B", "allenai_Molmo2-8B__prompt-direct.jsonl"),
    ("Qwen3-VL 32B", "Qwen_Qwen3-VL-32B-Instruct__prompt-direct.jsonl"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the published VLM-Fix interactive demo subset from prepared benchmark "
            "artifacts and cached model runs."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "demo")
    parser.add_argument("--states-per-game", type=int, default=300)
    return parser.parse_args()


def _candidate_dirs(relative_paths: list[str]) -> list[Path]:
    candidates: list[Path] = []
    bases = [ROOT, ROOT.parent]
    for sibling in sorted(ROOT.parent.iterdir()):
        if sibling.is_dir():
            bases.append(sibling)
    seen: set[Path] = set()
    for base in bases:
        for rel in relative_paths:
            candidate = (base / rel).resolve()
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _discover_dataset_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        dataset_dir = explicit.resolve()
        if (dataset_dir / "instances.parquet").is_file() and (dataset_dir / "images").is_dir():
            return dataset_dir
        raise FileNotFoundError(f"Dataset dir is missing instances.parquet or images/: {dataset_dir}")

    for candidate in _candidate_dirs(["data/generated/vlm_fix", "dataset/vlm_fix"]):
        if (candidate / "instances.parquet").is_file() and (candidate / "images").is_dir():
            return candidate
    raise FileNotFoundError("Could not discover a prepared VLM-Fix dataset directory.")


def _discover_runs_root(explicit: Path | None) -> Path:
    if explicit is not None:
        runs_root = explicit.resolve()
        if runs_root.is_dir():
            return runs_root
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    for candidate in _candidate_dirs(["runs/vlm_fix"]):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not discover cached VLM-Fix run files.")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _load_instances(dataset_dir: Path, states_per_game: int) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    df = pd.read_parquet(dataset_dir / "instances.parquet").copy()
    mask = (
        (df["render_variant"] == "canonical")
        & (df["prompt_type"] == "direct")
        & (df["image_text_order"] == "image_first")
        & (df["question_target"] == "winner")
        & (df["prompt_variant"].isin(PROMPT_VARIANTS))
        & (df["game"].isin(GAMES))
    )
    df = df[mask].copy()

    selected_states: dict[str, list[int]] = {}
    parts: list[pd.DataFrame] = []
    for game in GAMES:
        game_df = df[df["game"] == game].copy()
        states = sorted(int(state) for state in game_df["state_id"].unique())[:states_per_game]
        selected_states[game] = states
        parts.append(game_df[game_df["state_id"].isin(states)].copy())

    filtered = pd.concat(parts, ignore_index=True)
    return filtered, selected_states


def _load_run_lookup(
    run_path: Path,
    *,
    game: str,
    selected_states: list[int],
) -> dict[tuple[str, int, str, str], dict]:
    df = pd.read_json(run_path, lines=True).copy()
    mask = (
        (df["game"] == game)
        & (df["render_variant"] == "canonical")
        & (df["prompt_type"] == "direct")
        & (df["image_text_order"] == "image_first")
        & (df["question_target"] == "winner")
        & (df["prompt_variant"].isin(PROMPT_VARIANTS))
        & (df["rule_variant"].isin(["standard", "inverse"]))
        & (df["state_id"].isin(selected_states))
    )
    df = df[mask].copy()
    lookup: dict[tuple[str, int, str, str], dict] = {}
    for row in df.to_dict(orient="records"):
        key = (
            str(row["game"]),
            int(row["state_id"]),
            str(row["prompt_variant"]),
            str(row["rule_variant"]),
        )
        lookup[key] = row
    return lookup


def _copy_demo_image(src: Path, dst: Path, out_dir: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst.relative_to(out_dir).as_posix()


def _build_examples(
    *,
    dataset_dir: Path,
    runs_root: Path,
    out_dir: Path,
    instances: pd.DataFrame,
    selected_states: dict[str, list[int]],
) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    instance_lookup: dict[tuple[str, int, str, str], dict] = {}
    for row in instances.to_dict(orient="records"):
        key = (
            str(row["game"]),
            int(row["state_id"]),
            str(row["prompt_variant"]),
            str(row["rule_variant"]),
        )
        instance_lookup[key] = row

    model_lookups: dict[str, dict[tuple[str, int, str, str], dict]] = {}
    for model_name, filename in MODEL_SPECS:
        lookup: dict[tuple[str, int, str, str], dict] = {}
        for game in GAMES:
            run_path = runs_root / game / filename
            if not run_path.is_file():
                raise FileNotFoundError(f"Missing cached run file: {run_path}")
            lookup.update(_load_run_lookup(run_path, game=game, selected_states=selected_states[game]))
        model_lookups[model_name] = lookup

    examples: list[dict] = []
    model_stats: dict[str, dict[str, float]] = {
        model_name: {"correct": 0, "total": 0} for model_name, _ in MODEL_SPECS
    }
    breakdown_stats: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        PROMPT_DISPLAY[prompt_variant]: {
            rule_variant: {
                model_name: {"correct": 0, "total": 0} for model_name, _ in MODEL_SPECS
            }
            for rule_variant in ("standard", "inverse")
        }
        for prompt_variant in PROMPT_VARIANTS
    }
    task_counts: dict[str, int] = defaultdict(int)

    for game in GAMES:
        game_display = GAME_DISPLAY[game]
        for state_id in selected_states[game]:
            anchor_std = instance_lookup[(game, state_id, "standard", "standard")]
            anchor_inv = instance_lookup[(game, state_id, "standard", "inverse")]

            source_image = dataset_dir / str(anchor_std["image_path"])
            target_image = out_dir / "images" / game / f"state_{state_id:03d}.png"
            image_rel = _copy_demo_image(source_image, target_image, out_dir)

            compare_groups = []
            for prompt_variant in PROMPT_VARIANTS:
                prompt_display = PROMPT_DISPLAY[prompt_variant]
                key_std = (game, state_id, prompt_variant, "standard")
                key_inv = (game, state_id, prompt_variant, "inverse")
                std_row = instance_lookup[key_std]
                inv_row = instance_lookup[key_inv]

                group_predictions = []
                for model_name, _ in MODEL_SPECS:
                    std_pred = model_lookups[model_name][key_std]
                    inv_pred = model_lookups[model_name][key_inv]
                    for item in (std_pred, inv_pred):
                        flag = item.get("correct")
                        if isinstance(flag, bool):
                            model_stats[model_name]["total"] += 1
                            model_stats[model_name]["correct"] += int(flag)
                    for rule_variant, item in (("standard", std_pred), ("inverse", inv_pred)):
                        flag = item.get("correct")
                        if isinstance(flag, bool):
                            bucket = breakdown_stats[prompt_display][rule_variant][model_name]
                            bucket["total"] += 1
                            bucket["correct"] += int(flag)
                    group_predictions.append(
                        {
                            "model": model_name,
                            "variants": [
                                {
                                    "label": "Standard rule",
                                    "answer": _clean_text(std_pred.get("prediction"))
                                    or _clean_text(std_pred.get("extracted_answer")),
                                    "correct": bool(std_pred["correct"]) if pd.notna(std_pred.get("correct")) else None,
                                },
                                {
                                    "label": "Inverse rule",
                                    "answer": _clean_text(inv_pred.get("prediction"))
                                    or _clean_text(inv_pred.get("extracted_answer")),
                                    "correct": bool(inv_pred["correct"]) if pd.notna(inv_pred.get("correct")) else None,
                                },
                            ],
                        }
                    )

                compare_groups.append(
                    {
                        "label": prompt_display,
                        "prompt_panels": [
                            {"label": "Standard rule", "text": _clean_text(std_row["prompt"])},
                            {"label": "Inverse rule", "text": _clean_text(inv_row["prompt"])},
                        ],
                        "predictions": group_predictions,
                    }
                )

            example = {
                "id": f"{game}__state_{state_id:03d}",
                "title": f"{game_display} · State {state_id:03d}",
                "task": game_display,
                "mode": "image_qa",
                "prompt": (
                    f"The same {game_display.lower()} terminal board is evaluated under three prompt families "
                    "with paired standard and inverse rule descriptions."
                ),
                "media": [
                    {
                        "type": "image",
                        "src": image_rel,
                        "alt": f"{game_display} state {state_id:03d}",
                        "caption": f"{game_display} state {state_id:03d}",
                    }
                ],
                "answer": {
                    "explanation": "The board stays fixed while only the rule and prompt wording change.",
                    "variants": [
                        {"label": "Standard rule", "answer": _clean_text(anchor_std["answer"])},
                        {"label": "Inverse rule", "answer": _clean_text(anchor_inv["answer"])},
                    ],
                },
                "predictions": [],
                "meta": {
                    "game": game_display,
                    "game_slug": game,
                    "state_id": state_id,
                    "compare_groups": compare_groups,
                },
            }
            examples.append(example)
            task_counts[game_display] += 1

    summary = {
        "total_examples": len(examples),
        "tasks": dict(task_counts),
        "prompt_families": [PROMPT_DISPLAY[key] for key in PROMPT_VARIANTS],
        "accuracy_breakdown": [
            {
                "prompt_family": PROMPT_DISPLAY[prompt_variant],
                "rule_variant": rule_variant,
                "rule_label": "Standard" if rule_variant == "standard" else "Inverse",
                "models": {
                    model_name: {
                        "correct": int(stats["correct"]),
                        "total": int(stats["total"]),
                        "accuracy": float(stats["correct"] / stats["total"]) if stats["total"] else 0.0,
                    }
                    for model_name, stats in breakdown_stats[PROMPT_DISPLAY[prompt_variant]][rule_variant].items()
                },
            }
            for prompt_variant in PROMPT_VARIANTS
            for rule_variant in ("standard", "inverse")
        ],
        "models": [model_name for model_name, _ in MODEL_SPECS],
        "model_summary": {
            model_name: {
                "correct": int(stats["correct"]),
                "total": int(stats["total"]),
                "accuracy": float(stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }
            for model_name, stats in model_stats.items()
        },
        "default_compare_models": [model_name for model_name, _ in MODEL_SPECS],
    }
    return examples, summary


def _write_examples(out_dir: Path, examples: list[dict], summary: dict) -> None:
    examples_path = out_dir / "examples.jsonl"
    summary_path = out_dir / "summary.json"

    with examples_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    dataset_dir = _discover_dataset_dir(args.dataset_dir)
    runs_root = _discover_runs_root(args.runs_root)
    out_dir = args.out_dir.resolve()

    if out_dir.exists():
        for name in ("examples.jsonl", "summary.json"):
            target = out_dir / name
            if target.exists():
                target.unlink()

    instances, selected_states = _load_instances(dataset_dir, args.states_per_game)
    examples, summary = _build_examples(
        dataset_dir=dataset_dir,
        runs_root=runs_root,
        out_dir=out_dir,
        instances=instances,
        selected_states=selected_states,
    )
    _write_examples(out_dir, examples, summary)
    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "runs_root": str(runs_root),
                "out_dir": str(out_dir),
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
