from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence, Tuple

SIZE = 6
Board = Tuple[Tuple[int, ...], ...]  # 1 player1, 2 player2 (fully filled)


def board_to_tuple(board: Sequence[Sequence[int]]) -> Board:
    return tuple(tuple(int(v) for v in row) for row in board)


def counts(board: Sequence[Sequence[int]]) -> Tuple[int, int]:
    p1 = sum(1 for r in range(SIZE) for c in range(SIZE) if int(board[r][c]) == 1)
    p2 = sum(1 for r in range(SIZE) for c in range(SIZE) if int(board[r][c]) == 2)
    return p1, p2


def canonical_winner_idx(board: Sequence[Sequence[int]]) -> int:
    p1, p2 = counts(board)
    if p1 > p2:
        return 1
    if p2 > p1:
        return 2
    return 0


def inverse_winner_idx(board: Sequence[Sequence[int]]) -> int:
    p1, p2 = counts(board)
    if p1 < p2:
        return 1
    if p2 < p1:
        return 2
    return 0


def sample_balanced_terminal_states(total_states: int = 300, seed: int = 0, max_attempts: int = 2_000_000) -> List[Board]:
    if total_states <= 0 or total_states % 2 != 0:
        raise ValueError("total_states must be a positive even number")

    target_each = total_states // 2
    total_cells = SIZE * SIZE
    margins = [2, 4, 6, 8, 10, 12]
    rng = random.Random(seed)
    attempts = 0
    buckets: Dict[int, Dict[str, Board]] = {1: {}, 2: {}}

    while min(len(buckets[1]), len(buckets[2])) < target_each and attempts < max_attempts:
        attempts += 1
        want = 1 if len(buckets[1]) < len(buckets[2]) else 2
        margin = rng.choice(margins)

        if want == 1:
            count_p1 = (total_cells + margin) // 2
        else:
            count_p1 = (total_cells - margin) // 2
        count_p2 = total_cells - count_p1
        if count_p1 == count_p2:
            continue

        cells = [1] * count_p1 + [2] * count_p2
        rng.shuffle(cells)
        board = tuple(tuple(cells[i * SIZE : (i + 1) * SIZE]) for i in range(SIZE))
        w = canonical_winner_idx(board)
        if w not in {1, 2}:
            continue
        key = json.dumps(board, separators=(",", ":"))
        if len(buckets[w]) < target_each and key not in buckets[w]:
            buckets[w][key] = board

    if min(len(buckets[1]), len(buckets[2])) < target_each:
        raise RuntimeError(
            f"Could not sample balanced dots_boxes states. Have p1={len(buckets[1])}, p2={len(buckets[2])}, need {target_each} each."
        )

    p1_states = [buckets[1][k] for k in sorted(buckets[1].keys())[:target_each]]
    p2_states = [buckets[2][k] for k in sorted(buckets[2].keys())[:target_each]]
    out = p1_states + p2_states
    rng.shuffle(out)
    return out
