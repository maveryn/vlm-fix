from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence, Tuple

ROWS = 4
COLS = 4
Board = Tuple[Tuple[int, ...], ...]  # 0 empty, 1 player1, 2 player2


def empty_board() -> List[List[int]]:
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]


def board_to_tuple(board: Sequence[Sequence[int]]) -> Board:
    return tuple(tuple(int(v) for v in row) for row in board)


def legal_columns(board: Sequence[Sequence[int]]) -> List[int]:
    return [c for c in range(COLS) if int(board[0][c]) == 0]


def drop_piece(board: List[List[int]], col: int, player: int) -> int:
    for r in range(ROWS - 1, -1, -1):
        if int(board[r][col]) == 0:
            board[r][col] = int(player)
            return r
    raise ValueError(f"Column {col} is full")


def _winning_lines() -> List[Tuple[Tuple[int, int], ...]]:
    lines: List[Tuple[Tuple[int, int], ...]] = []
    # Horizontal
    for r in range(ROWS):
        lines.append(tuple((r, c) for c in range(COLS)))
    # Vertical
    for c in range(COLS):
        lines.append(tuple((r, c) for r in range(ROWS)))
    # Main diagonal
    lines.append(tuple((i, i) for i in range(ROWS)))
    # Anti diagonal
    lines.append(tuple((i, COLS - 1 - i) for i in range(ROWS)))
    return lines


_LINES = _winning_lines()


def canonical_winner_idx(board: Sequence[Sequence[int]]) -> int:
    winners = set()
    for line in _LINES:
        r0, c0 = line[0]
        mark = int(board[r0][c0])
        if mark == 0:
            continue
        ok = True
        for rr, cc in line[1:]:
            if int(board[rr][cc]) != mark:
                ok = False
                break
        if ok:
            winners.add(mark)
    if len(winners) == 1:
        return next(iter(winners))
    if len(winners) > 1:
        return -1
    return 0


def inverse_winner_idx(board: Sequence[Sequence[int]]) -> int:
    w = canonical_winner_idx(board)
    if w == 1:
        return 2
    if w == 2:
        return 1
    return 0


def winning_categories(board: Sequence[Sequence[int]]) -> List[str]:
    w = canonical_winner_idx(board)
    if w not in {1, 2}:
        return []
    out: List[str] = []
    # Horizontal
    for r in range(ROWS):
        if all(int(board[r][c]) == w for c in range(COLS)):
            out.append("horizontal")
    # Vertical
    for c in range(COLS):
        if all(int(board[r][c]) == w for r in range(ROWS)):
            out.append("vertical")
    if all(int(board[i][i]) == w for i in range(ROWS)):
        out.append("main_diagonal")
    if all(int(board[i][COLS - 1 - i]) == w for i in range(ROWS)):
        out.append("anti_diagonal")
    return sorted(set(out))


def sample_balanced_terminal_states(total_states: int = 300, seed: int = 0, max_attempts: int = 2_000_000) -> List[Board]:
    if total_states <= 0 or total_states % 2 != 0:
        raise ValueError("total_states must be a positive even number")

    target_each = total_states // 2
    buckets: Dict[int, Dict[str, Board]] = {1: {}, 2: {}}
    rng = random.Random(seed)
    attempts = 0

    while min(len(buckets[1]), len(buckets[2])) < target_each and attempts < max_attempts:
        attempts += 1
        board = empty_board()
        player = 1
        terminal_state: Board | None = None

        for _ in range(ROWS * COLS):
            cols = legal_columns(board)
            if not cols:
                break
            col = rng.choice(cols)
            _ = drop_piece(board, col, player)
            w = canonical_winner_idx(board)
            if w == -1:
                terminal_state = None
                break
            if w in {1, 2}:
                terminal_state = board_to_tuple(board)
                break
            player = 2 if player == 1 else 1

        if terminal_state is None:
            continue
        w = canonical_winner_idx(terminal_state)
        if w not in {1, 2}:
            continue
        key = json.dumps(terminal_state, separators=(",", ":"))
        if len(buckets[w]) < target_each and key not in buckets[w]:
            buckets[w][key] = terminal_state

    if min(len(buckets[1]), len(buckets[2])) < target_each:
        raise RuntimeError(
            f"Could not sample balanced connect4 states. Have p1={len(buckets[1])}, p2={len(buckets[2])}, need {target_each} each."
        )

    p1_states = [buckets[1][k] for k in sorted(buckets[1].keys())[:target_each]]
    p2_states = [buckets[2][k] for k in sorted(buckets[2].keys())[:target_each]]
    out = p1_states + p2_states
    rng.shuffle(out)
    return out
