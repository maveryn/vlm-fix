#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import pandas as pd
from datasets import Dataset, Sequence as HFSequence
from datasets import Image as HFImage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_fix.games import connect4, dots_boxes, reversi, tictactoe
from vlm_fix.prompts import prompt_for
from vlm_fix.render import connect4 as connect4_render
from vlm_fix.render import dots_boxes as dots_boxes_render
from vlm_fix.render import reversi as reversi_render
from vlm_fix.render import tictactoe as tictactoe_render


QUESTION_TARGETS = ("winner", "loser")
TOPUP_PREFIX = "Inspect the final board carefully."

LABELS_BY_GAME = {
    "tictactoe": ("X", "O"),
    "reversi": ("Black", "White"),
    "connect4": ("Red", "Yellow"),
    "dots_boxes": ("A", "B"),
}

IMAGE_SIZES = {
    "tictactoe": 384,
    "reversi": 480,
    "connect4": 448,
    "dots_boxes": 480,
}


@dataclass(frozen=True)
class StateSlot:
    board: Any
    source_state_id: int
    is_ttt_topup: bool = False


def _board_key(board: Any) -> str:
    return json.dumps(board, separators=(",", ":"), sort_keys=False)


def _board_key_from_json(text: str) -> str:
    return _board_key(json.loads(str(text)))


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
        raise TypeError(f"Unsupported image type: {type(value)}")
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _encode_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _format_problem(prompt: str) -> str:
    text = str(prompt or "").replace("<image>", "").strip()
    return f"<image>{text}"


def _winner_idx(game: str, board: Any, rule_variant: str) -> int:
    if game == "tictactoe":
        if rule_variant == "standard":
            lbl = tictactoe.canonical_label(board)
        else:
            lbl = tictactoe.misere_label(board)
        if lbl == "X":
            return 1
        if lbl == "O":
            return 2
        raise ValueError("tictactoe board has no unique winner")
    if game == "reversi":
        idx = reversi.canonical_winner_idx(board) if rule_variant == "standard" else reversi.misere_winner_idx(board)
        if idx not in (1, 2):
            raise ValueError("reversi board has no unique winner")
        return idx
    if game == "connect4":
        idx = connect4.canonical_winner_idx(board) if rule_variant == "standard" else connect4.inverse_winner_idx(board)
        if idx not in (1, 2):
            raise ValueError("connect4 board has no unique winner")
        return idx
    if game == "dots_boxes":
        idx = dots_boxes.canonical_winner_idx(board) if rule_variant == "standard" else dots_boxes.inverse_winner_idx(board)
        if idx not in (1, 2):
            raise ValueError("dots_boxes board has no unique winner")
        return idx
    raise ValueError(f"Unsupported game: {game}")


def _render_canonical(game: str, board: Any, labels: Tuple[str, str]) -> Image.Image:
    if game == "tictactoe":
        board_rows = tictactoe.board_to_rows(board)
        return tictactoe_render.render(board_rows, style="canonical", size=IMAGE_SIZES[game], render_scale=3)
    if game == "reversi":
        return reversi_render.render(board, style="canonical", size_px=IMAGE_SIZES[game], render_scale=3)
    if game == "connect4":
        return connect4_render.render(board, style="canonical", size=IMAGE_SIZES[game], render_scale=3)
    if game == "dots_boxes":
        return dots_boxes_render.render(board, style="canonical", size_px=IMAGE_SIZES[game], render_scale=3)
    raise ValueError(f"Unsupported game: {game}")


def sample_tictactoe_slots(seed: int, excluded_keys: set[str] | None = None) -> List[StateSlot]:
    excluded_keys = excluded_keys or set()
    all_states = tictactoe.enumerate_terminal_winner_states(limit=10000)
    exclusive: List[Tuple[int, Tuple[int, ...]]] = []
    for i, board in enumerate(all_states):
        cats = tictactoe.winning_categories(board)
        lines = tictactoe.winning_line_names(board)
        if len(cats) == 1 and len(lines) == 1:
            exclusive.append((i + 1, board))
    if len(exclusive) != 920:
        raise RuntimeError(f"Expected 920 exclusive tictactoe states, got {len(exclusive)}")

    rng = random.Random(seed ^ 0xA1A1)
    base = [(source_state_id, board) for source_state_id, board in exclusive if _board_key(board) not in excluded_keys]
    if not base:
        raise RuntimeError("No tic-tac-toe train states remain after applying exclusions")
    rng.shuffle(base)

    slots = [StateSlot(board=board, source_state_id=source_state_id, is_ttt_topup=False) for source_state_id, board in base]
    topup_needed = 1024 - len(base)
    if topup_needed < 0:
        raise RuntimeError(f"Too many tic-tac-toe train states after exclusions: {len(base)} > 1024")
    if topup_needed <= len(base):
        topup_indices = rng.sample(range(len(base)), topup_needed)
    else:
        topup_indices = [rng.randrange(len(base)) for _ in range(topup_needed)]
    for idx in topup_indices:
        source_state_id, board = base[idx]
        slots.append(StateSlot(board=board, source_state_id=source_state_id, is_ttt_topup=True))

    rng.shuffle(slots)
    if len(slots) != 1024:
        raise RuntimeError(f"Expected 1024 tictactoe state slots, got {len(slots)}")
    return slots


def _sample_balanced_slots_excluding(
    *,
    total_states: int,
    excluded_keys: set[str],
    seed: int,
    seed_offset: int,
    labels: Sequence[Any],
    sampler_fn,
    label_fn,
    rounds_limit: int = 64,
) -> List[StateSlot]:
    if total_states <= 0 or total_states % 2 != 0:
        raise ValueError("total_states must be a positive even number")

    target_each = total_states // 2
    buckets: Dict[Any, Dict[str, Any]] = {label: {} for label in labels}
    rng = random.Random(seed ^ seed_offset)
    candidate_total = max(1324, total_states + len(excluded_keys))
    if candidate_total % 2 != 0:
        candidate_total += 1

    rounds = 0
    while min(len(buckets[label]) for label in labels) < target_each and rounds < rounds_limit:
        rounds += 1
        round_seed = rng.randint(0, 2**31 - 1)
        candidates = sampler_fn(total_states=candidate_total, seed=round_seed)
        for board in candidates:
            key = _board_key(board)
            if key in excluded_keys:
                continue
            label = label_fn(board)
            if label not in buckets:
                continue
            if len(buckets[label]) < target_each and key not in buckets[label]:
                buckets[label][key] = board
            if min(len(buckets[l]) for l in labels) >= target_each:
                break

    if min(len(buckets[label]) for label in labels) < target_each:
        have = {str(label): len(buckets[label]) for label in labels}
        raise RuntimeError(f"Failed to sample disjoint balanced states. have={have}, need_each={target_each}")

    boards: List[Any] = []
    for label in labels:
        keys = sorted(buckets[label].keys())[:target_each]
        boards.extend(buckets[label][k] for k in keys)
    rng.shuffle(boards)
    return [StateSlot(board=b, source_state_id=i + 1, is_ttt_topup=False) for i, b in enumerate(boards)]


def sample_reversi_slots(seed: int, excluded_keys: set[str] | None = None) -> List[StateSlot]:
    return _sample_balanced_slots_excluding(
        total_states=1024,
        excluded_keys=excluded_keys or set(),
        seed=seed,
        seed_offset=0xB2B2,
        labels=("Black", "White"),
        sampler_fn=reversi.sample_balanced_terminal_states,
        label_fn=reversi.canonical_label,
    )


def sample_dots_boxes_slots(seed: int, excluded_keys: set[str] | None = None) -> List[StateSlot]:
    return _sample_balanced_slots_excluding(
        total_states=1024,
        excluded_keys=excluded_keys or set(),
        seed=seed,
        seed_offset=0xD4D4,
        labels=(1, 2),
        sampler_fn=dots_boxes.sample_balanced_terminal_states,
        label_fn=dots_boxes.canonical_winner_idx,
    )


def sample_connect4_slots(
    seed: int,
    excluded_keys: set[str] | None = None,
    max_attempts: int = 5_000_000,
) -> List[StateSlot]:
    excluded_keys = excluded_keys or set()
    targets = {"horizontal": 341, "vertical": 341, "main_diagonal": 171, "anti_diagonal": 171}
    buckets: Dict[str, Dict[str, Tuple[Tuple[int, ...], ...]]] = {k: {} for k in targets}
    rng = random.Random(seed ^ 0xC3C3)
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        if all(len(buckets[k]) >= targets[k] for k in targets):
            break

        board = connect4.empty_board()
        player = 1
        terminal_state = None
        for _ in range(connect4.ROWS * connect4.COLS):
            cols = connect4.legal_columns(board)
            if not cols:
                break
            col = rng.choice(cols)
            connect4.drop_piece(board, col, player)
            w = connect4.canonical_winner_idx(board)
            if w == -1:
                terminal_state = None
                break
            if w in {1, 2}:
                terminal_state = connect4.board_to_tuple(board)
                break
            player = 2 if player == 1 else 1

        if terminal_state is None:
            continue

        cats = connect4.winning_categories(terminal_state)
        if len(cats) != 1:
            continue
        cat = cats[0]
        if cat not in targets:
            continue
        if len(buckets[cat]) >= targets[cat]:
            continue

        key = _board_key(terminal_state)
        if key in excluded_keys:
            continue
        buckets[cat][key] = terminal_state

    if not all(len(buckets[k]) >= targets[k] for k in targets):
        have = {k: len(v) for k, v in buckets.items()}
        raise RuntimeError(f"Failed to sample balanced connect4 states. have={have}, need={targets}")

    boards: List[Tuple[Tuple[int, ...], ...]] = []
    for cat in ("horizontal", "vertical", "main_diagonal", "anti_diagonal"):
        keys = sorted(buckets[cat].keys())[: targets[cat]]
        boards.extend(buckets[cat][k] for k in keys)

    rng.shuffle(boards)
    if len(boards) != 1024:
        raise RuntimeError(f"Expected 1024 connect4 state slots, got {len(boards)}")
    return [StateSlot(board=b, source_state_id=i + 1, is_ttt_topup=False) for i, b in enumerate(boards)]


def _benchmark_base_state_keys(instances_df: pd.DataFrame) -> Dict[str, set[str]]:
    required = {"game", "state_id", "board_state"}
    missing = sorted(required - set(instances_df.columns))
    if missing:
        raise RuntimeError(f"instances parquet missing columns for benchmark-key extraction: {missing}")

    base = instances_df.sort_values(["game", "state_id"], kind="mergesort").drop_duplicates(subset=["game", "state_id"])
    out: Dict[str, set[str]] = {}
    for game, group in base.groupby("game"):
        out[str(game)] = {_board_key_from_json(text) for text in group["board_state"].tolist()}
    return out


def _iter_train_rows(
    game_rule_slices: Sequence[Tuple[str, str]],
    state_pools: Dict[str, List[StateSlot]],
) -> Iterator[dict[str, Any]]:
    for game, rule_variant in game_rule_slices:
        labels = LABELS_BY_GAME[game]
        for slot in state_pools[game]:
            winner_idx = _winner_idx(game=game, board=slot.board, rule_variant=rule_variant)
            img = _render_canonical(game=game, board=slot.board, labels=labels)
            image_bytes = _encode_png_bytes(img)
            for question_target in QUESTION_TARGETS:
                prompt = prompt_for(
                    game=game,
                    rule_variant=rule_variant,
                    labels=labels,
                    question_target=question_target,
                    prompt_type="direct",
                    prompt_variant="standard",
                )
                if slot.is_ttt_topup:
                    prompt = f"{TOPUP_PREFIX} {prompt}"
                ans_idx = winner_idx if question_target == "winner" else (2 if winner_idx == 1 else 1)
                yield {
                    "images": [{"bytes": image_bytes}],
                    "problem": _format_problem(prompt),
                    "answer": labels[ans_idx - 1],
                }


def _iter_test_rows(
    instances_df: pd.DataFrame,
    benchmark_root: Path,
    games: Sequence[str],
    rule_variants: Sequence[str],
) -> Iterator[dict[str, Any]]:
    keep_games = set(games)
    keep_rules = set(rule_variants)
    filtered = instances_df[
        (instances_df["game"].isin(keep_games))
        & (instances_df["rule_variant"].isin(keep_rules))
        & (instances_df["prompt_variant"] == "standard")
        & (instances_df["render_variant"] == "canonical")
        & (instances_df["prompt_type"] == "direct")
        & (instances_df["image_text_order"] == "image_first")
        & (instances_df["question_target"].isin(["winner", "loser"]))
    ].copy()
    filtered = filtered.sort_values(["game", "rule_variant", "state_id", "question_target"], kind="mergesort")

    for row in filtered.itertuples(index=False):
        img_path = benchmark_root / str(row.image_path)
        img = _ensure_pil(str(img_path))
        image_bytes = _encode_png_bytes(img)
        yield {
            "images": [{"bytes": image_bytes}],
            "problem": _format_problem(str(row.prompt)),
            "answer": str(row.answer),
        }


def _generate_train_rows_for_dataset(game_rule_slices: Sequence[Tuple[str, str]], state_pools: Dict[str, List[StateSlot]]):
    yield from _iter_train_rows(game_rule_slices=game_rule_slices, state_pools=state_pools)


def _generate_test_rows_for_dataset(
    instances_df: pd.DataFrame,
    benchmark_root: Path,
    games: Sequence[str],
    rule_variants: Sequence[str],
):
    yield from _iter_test_rows(
        instances_df=instances_df,
        benchmark_root=benchmark_root,
        games=games,
        rule_variants=rule_variants,
    )


def _write_parquet(generator_fn, out_path: Path, gen_kwargs: Dict[str, Any]) -> int:
    rows = list(generator_fn(**gen_kwargs))
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("images", HFSequence(HFImage()))
    ds.to_parquet(str(out_path))
    return int(len(ds))


def _count_test_rows(instances_df: pd.DataFrame, games: Sequence[str], rule_variants: Sequence[str]) -> int:
    keep_games = set(games)
    keep_rules = set(rule_variants)
    df = instances_df[
        (instances_df["game"].isin(keep_games))
        & (instances_df["rule_variant"].isin(keep_rules))
        & (instances_df["prompt_variant"] == "standard")
        & (instances_df["render_variant"] == "canonical")
        & (instances_df["prompt_type"] == "direct")
        & (instances_df["image_text_order"] == "image_first")
        & (instances_df["question_target"].isin(["winner", "loser"]))
    ]
    return int(len(df))


def build_all(
    instances_path: Path,
    benchmark_root: Path,
    out_dir: Path,
    seed: int,
    exclude_benchmark_from_train: bool = False,
) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_df = pd.read_parquet(instances_path)
    required_cols = {
        "game",
        "state_id",
        "rule_variant",
        "prompt_variant",
        "render_variant",
        "prompt_type",
        "image_text_order",
        "question_target",
        "prompt",
        "answer",
        "image_path",
        "board_state",
    }
    missing = sorted(required_cols - set(instances_df.columns))
    if missing:
        raise RuntimeError(f"instances parquet missing columns: {missing}")

    benchmark_keys = _benchmark_base_state_keys(instances_df) if exclude_benchmark_from_train else {}
    excluded = {game: set(benchmark_keys.get(game, set())) for game in LABELS_BY_GAME}

    state_pools: Dict[str, List[StateSlot]] = {
        "tictactoe": sample_tictactoe_slots(seed=seed, excluded_keys=excluded["tictactoe"]),
        "reversi": sample_reversi_slots(seed=seed, excluded_keys=excluded["reversi"]),
        "connect4": sample_connect4_slots(seed=seed, excluded_keys=excluded["connect4"]),
        "dots_boxes": sample_dots_boxes_slots(seed=seed, excluded_keys=excluded["dots_boxes"]),
    }

    counts: Dict[str, int] = {}

    d1_train = out_dir / "D1_train_standard_all4_8192.parquet"
    d2_train = out_dir / "D2_train_inverse_all4_8192.parquet"
    d3_train = out_dir / "D3_train_ttt_reversi_both_rules_8192.parquet"
    d1_test = out_dir / "D1_test_inverse_core2400.parquet"
    d2_test = out_dir / "D2_test_standard_core2400.parquet"
    d3_test = out_dir / "D3_test_connect4_dots_both_rules_core2400.parquet"

    counts[d1_train.name] = _write_parquet(
        _generate_train_rows_for_dataset,
        d1_train,
        gen_kwargs={
            "game_rule_slices": [
                ("tictactoe", "standard"),
                ("reversi", "standard"),
                ("connect4", "standard"),
                ("dots_boxes", "standard"),
            ],
            "state_pools": state_pools,
        },
    )
    counts[d2_train.name] = _write_parquet(
        _generate_train_rows_for_dataset,
        d2_train,
        gen_kwargs={
            "game_rule_slices": [
                ("tictactoe", "inverse"),
                ("reversi", "inverse"),
                ("connect4", "inverse"),
                ("dots_boxes", "inverse"),
            ],
            "state_pools": state_pools,
        },
    )
    counts[d3_train.name] = _write_parquet(
        _generate_train_rows_for_dataset,
        d3_train,
        gen_kwargs={
            "game_rule_slices": [
                ("tictactoe", "standard"),
                ("tictactoe", "inverse"),
                ("reversi", "standard"),
                ("reversi", "inverse"),
            ],
            "state_pools": state_pools,
        },
    )

    counts[d1_test.name] = _write_parquet(
        _generate_test_rows_for_dataset,
        d1_test,
        gen_kwargs={
            "instances_df": instances_df,
            "benchmark_root": benchmark_root,
            "games": ["tictactoe", "reversi", "connect4", "dots_boxes"],
            "rule_variants": ["inverse"],
        },
    )
    counts[d2_test.name] = _write_parquet(
        _generate_test_rows_for_dataset,
        d2_test,
        gen_kwargs={
            "instances_df": instances_df,
            "benchmark_root": benchmark_root,
            "games": ["tictactoe", "reversi", "connect4", "dots_boxes"],
            "rule_variants": ["standard"],
        },
    )
    counts[d3_test.name] = _write_parquet(
        _generate_test_rows_for_dataset,
        d3_test,
        gen_kwargs={
            "instances_df": instances_df,
            "benchmark_root": benchmark_root,
            "games": ["connect4", "dots_boxes"],
            "rule_variants": ["standard", "inverse"],
        },
    )

    # Validation checks
    expected = {
        d1_train.name: 8192,
        d2_train.name: 8192,
        d3_train.name: 8192,
        d1_test.name: _count_test_rows(instances_df, ["tictactoe", "reversi", "connect4", "dots_boxes"], ["inverse"]),
        d2_test.name: _count_test_rows(instances_df, ["tictactoe", "reversi", "connect4", "dots_boxes"], ["standard"]),
        d3_test.name: _count_test_rows(instances_df, ["connect4", "dots_boxes"], ["standard", "inverse"]),
    }
    for name, exp in expected.items():
        got = counts.get(name, -1)
        if got != exp:
            raise RuntimeError(f"Row-count mismatch for {name}: got {got}, expected {exp}")

    train_unique_keys = {game: {_board_key(slot.board) for slot in slots} for game, slots in state_pools.items()}
    train_overlap_counts = {
        game: int(len(train_unique_keys[game] & excluded.get(game, set())))
        for game in LABELS_BY_GAME
    }

    diagnostics = {
        "seed": int(seed),
        "instances_source": str(instances_path),
        "benchmark_root": str(benchmark_root),
        "exclude_benchmark_from_train": bool(exclude_benchmark_from_train),
        "counts": counts,
        "expected_counts": expected,
        "ttt_topup_slots": int(sum(1 for s in state_pools["tictactoe"] if s.is_ttt_topup)),
        "state_slots_per_game": {k: int(len(v)) for k, v in state_pools.items()},
        "train_unique_states_per_game": {k: int(len(v)) for k, v in train_unique_keys.items()},
        "benchmark_base_states_per_game": {k: int(len(v)) for k, v in benchmark_keys.items()},
        "train_overlap_with_benchmark_per_game": train_overlap_counts,
    }

    if exclude_benchmark_from_train and any(v != 0 for v in train_overlap_counts.values()):
        raise RuntimeError(f"Benchmark overlap remains in train pools: {train_overlap_counts}")
    (out_dir / "summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build finetune train/test parquets for D1/D2/D3.")
    parser.add_argument(
        "--instances",
        "--instances-path",
        type=Path,
        default=Path("dataset/vlm_fix/instances.parquet"),
        help="Path to benchmark instances.parquet",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("dataset/vlm_fix"),
        help="Root folder for benchmark image files referenced by instances.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/post_training"),
        help="Output directory for D1/D2/D3 parquet files",
    )
    parser.add_argument(
        "--exclude-benchmark-from-train",
        action="store_true",
        help="Exclude current benchmark base states from D1/D2/D3 train sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260317, help="Global random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instances_path = args.instances if args.instances.is_absolute() else (PROJECT_ROOT / args.instances).resolve()
    benchmark_root = (
        args.benchmark_root if args.benchmark_root.is_absolute() else (PROJECT_ROOT / args.benchmark_root).resolve()
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir).resolve()
    counts = build_all(
        instances_path=instances_path,
        benchmark_root=benchmark_root,
        out_dir=out_dir,
        seed=int(args.seed),
        exclude_benchmark_from_train=bool(args.exclude_benchmark_from_train),
    )
    print("[done] finetune datasets created")
    for name, n in counts.items():
        print(f"  - {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
