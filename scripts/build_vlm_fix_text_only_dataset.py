#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _token_for_label(label: str) -> str:
    s = str(label).strip()
    if not s:
        return "?"
    known = {
        "Black": "B",
        "White": "W",
        "Red": "R",
        "Yellow": "Y",
    }
    if s in known:
        return known[s]
    if len(s) == 1:
        return s
    return s[0].upper()


def _token_for_cell(v: int, labels: Sequence[str]) -> str:
    if int(v) == 1:
        return _token_for_label(labels[0])
    if int(v) == 2:
        return _token_for_label(labels[1])
    return " "


def _render_tictactoe(board: Sequence[int], labels: Sequence[str]) -> str:
    if len(board) != 9:
        raise ValueError(f"tictactoe board must have len=9, got {len(board)}")
    rows = [board[0:3], board[3:6], board[6:9]]
    sep = "───┼───┼───"
    lines = []
    for i, row in enumerate(rows):
        a, b, c = (_token_for_cell(x, labels) for x in row)
        lines.append(f" {a} │ {b} │ {c} ")
        if i < 2:
            lines.append(sep)
    return "\n".join(lines)


def _render_open_grid(board: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    n_rows = len(board)
    n_cols = len(board[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("grid board cannot be empty")
    sep = "───" + ("┼───" * (n_cols - 1))
    lines = []
    for r, row in enumerate(board):
        if len(row) != n_cols:
            raise ValueError("inconsistent row lengths")
        tokens = [_token_for_cell(x, labels) for x in row]
        lines.append(" " + " │ ".join(tokens) + " ")
        if r < n_rows - 1:
            lines.append(sep)
    return "\n".join(lines)


def _render_boxed_grid(board: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    n_rows = len(board)
    n_cols = len(board[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("boxed board cannot be empty")
    top = "┌" + "┬".join(["───"] * n_cols) + "┐"
    mid = "├" + "┼".join(["───"] * n_cols) + "┤"
    bot = "└" + "┴".join(["───"] * n_cols) + "┘"
    lines = [top]
    for r, row in enumerate(board):
        if len(row) != n_cols:
            raise ValueError("inconsistent row lengths")
        tokens = [_token_for_cell(x, labels) for x in row]
        lines.append("│ " + " │ ".join(tokens) + " │")
        if r < n_rows - 1:
            lines.append(mid)
    lines.append(bot)
    return "\n".join(lines)


def _render_dots_boxes(board: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    n_rows = len(board)
    n_cols = len(board[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("dots_boxes board cannot be empty")
    edge = "•" + ("───•" * n_cols)
    lines = []
    for row in board:
        if len(row) != n_cols:
            raise ValueError("inconsistent row lengths")
        lines.append(edge)
        tokens = [_token_for_cell(x, labels) for x in row]
        lines.append("│ " + " │ ".join(tokens) + " │")
    lines.append(edge)
    return "\n".join(lines)


def _board_text_for_row(row: pd.Series) -> str:
    game = str(row["game"])
    labels = [x.strip() for x in str(row["valid_labels"]).split("|")]
    board = json.loads(str(row["board_state"]))

    if game == "tictactoe":
        return _render_tictactoe(board, labels)
    if game == "reversi":
        return _render_boxed_grid(board, labels)
    if game == "connect4":
        return _render_open_grid(board, labels)
    if game == "dots_boxes":
        return _render_dots_boxes(board, labels)
    raise ValueError(f"Unsupported game: {game}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_games(games: Iterable[str]) -> list[str]:
    out: list[str] = []
    for g in games:
        x = str(g).strip()
        if x:
            out.append(x)
    if not out:
        raise ValueError("games cannot be empty")
    return out


def build_text_only_dataset(
    src_parquet: Path,
    out_dir: Path,
    games: Sequence[str],
    source_image_text_order: str = "image_first",
) -> Path:
    _ensure_dir(out_dir)
    df = pd.read_parquet(src_parquet)

    filt = df[
        (df["game"].isin(list(games)))
        & (df["render_variant"] == "canonical")
        & (df["prompt_variant"] == "standard")
        & (df["prompt_type"] == "direct")
        & (df["image_text_order"] == source_image_text_order)
    ].copy()

    keep_cols = [
        "index",
        "game",
        "state_id",
        "board_state",
        "rule_variant",
        "question_target",
        "prompt",
        "answer",
        "valid_labels",
        "standard_winner_idx",
        "inverse_winner_idx",
    ]
    filt = filt[keep_cols].sort_values(["game", "state_id", "rule_variant", "question_target", "index"]).reset_index(drop=True)

    filt["modality"] = "text_only"
    filt["render_variant"] = "canonical_text"
    filt["prompt_variant"] = "standard"
    filt["prompt_type"] = "direct"
    filt["source_image_text_order"] = source_image_text_order
    filt["image_text_order"] = "text_first"
    filt["board_text"] = filt.apply(_board_text_for_row, axis=1)
    filt["input_text"] = filt["prompt"].astype(str) + "\n\nBoard:\n" + filt["board_text"].astype(str)

    # Keep a stable contiguous index for this dataset.
    filt["index"] = range(1, len(filt) + 1)

    out_parquet = out_dir / "instances.parquet"
    filt.to_parquet(out_parquet, index=False)

    counts = (
        filt.groupby(["game", "rule_variant", "question_target"])
        .size()
        .reset_index(name="n")
        .sort_values(["game", "rule_variant", "question_target"])
    )
    counts.to_csv(out_dir / "counts_by_game_rule_target.csv", index=False)

    summary = {
        "dataset": "vlm_fix_text_only",
        "source_dataset": "vlm_fix",
        "source_parquet": str(src_parquet),
        "source_filter": {
            "render_variant": "canonical",
            "prompt_variant": "standard",
            "prompt_type": "direct",
            "image_text_order": source_image_text_order,
        },
        "games": sorted(filt["game"].unique().tolist()),
        "rule_variants": sorted(filt["rule_variant"].unique().tolist()),
        "question_targets": sorted(filt["question_target"].unique().tolist()),
        "total_instances": int(len(filt)),
        "instances_per_game": {k: int(v) for k, v in filt.groupby("game").size().to_dict().items()},
        "instances_per_game_rule_target": {
            "|".join(map(str, k)): int(v)
            for k, v in filt.groupby(["game", "rule_variant", "question_target"]).size().to_dict().items()
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_parquet


def main() -> int:
    p = argparse.ArgumentParser(description="Build paired vlm_fix text-only dataset from canonical direct rows.")
    p.add_argument("--src-parquet", type=Path, default=Path("data/generated/vlm_fix/instances.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("data/generated/vlm_fix_text_only"))
    p.add_argument("--games", nargs="+", default=["tictactoe", "reversi", "connect4", "dots_boxes"])
    p.add_argument("--source-image-text-order", type=str, default="image_first")
    args = p.parse_args()

    out = build_text_only_dataset(
        src_parquet=args.src_parquet,
        out_dir=args.out_dir,
        games=_normalize_games(args.games),
        source_image_text_order=str(args.source_image_text_order),
    )
    print(f"[ok] wrote: {out}")
    print(f"[ok] summary: {args.out_dir / 'summary.json'}")
    print(f"[ok] counts: {args.out_dir / 'counts_by_game_rule_target.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
