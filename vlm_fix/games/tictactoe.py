from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence, Tuple

Board = Tuple[int, ...]  # len=9, 0 empty, 1 X, 2 O

_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

_LINES_DETAILED: Tuple[Tuple[int, int, int, str, str], ...] = (
    (0, 1, 2, "horizontal", "row_top"),
    (3, 4, 5, "horizontal", "row_mid"),
    (6, 7, 8, "horizontal", "row_bottom"),
    (0, 3, 6, "vertical", "col_left"),
    (1, 4, 7, "vertical", "col_mid"),
    (2, 5, 8, "vertical", "col_right"),
    (0, 4, 8, "main_diagonal", "diag_main"),
    (2, 4, 6, "anti_diagonal", "diag_anti"),
)

_CATEGORY_TO_LINES = {
    "horizontal": ["row_top", "row_mid", "row_bottom"],
    "vertical": ["col_left", "col_mid", "col_right"],
    "main_diagonal": ["diag_main"],
    "anti_diagonal": ["diag_anti"],
}


def board_to_rows(board: Sequence[int]) -> List[List[int]]:
    return [list(board[0:3]), list(board[3:6]), list(board[6:9])]


def check_winner(board: Sequence[int]) -> int:
    winners = set()
    for a, b, c in _LINES:
        mark = int(board[a])
        if mark != 0 and mark == int(board[b]) == int(board[c]):
            winners.add(mark)
    if len(winners) == 1:
        return next(iter(winners))
    if len(winners) > 1:
        return -1
    return 0


def canonical_label(board: Sequence[int]) -> str:
    w = check_winner(board)
    if w == 1:
        return "X"
    if w == 2:
        return "O"
    return "Draw"


def misere_label(board: Sequence[int]) -> str:
    w = check_winner(board)
    if w == 1:
        return "O"
    if w == 2:
        return "X"
    return "Draw"


def winning_categories(board: Sequence[int]) -> List[str]:
    w = check_winner(board)
    if w not in {1, 2}:
        return []
    out: List[str] = []
    for a, b, c, cat, _line_name in _LINES_DETAILED:
        if int(board[a]) == w and int(board[b]) == w and int(board[c]) == w:
            out.append(cat)
    return sorted(set(out))


def winning_line_names(board: Sequence[int]) -> List[str]:
    w = check_winner(board)
    if w not in {1, 2}:
        return []
    out: List[str] = []
    for a, b, c, _cat, line_name in _LINES_DETAILED:
        if int(board[a]) == w and int(board[b]) == w and int(board[c]) == w:
            out.append(line_name)
    return sorted(set(out))


def enumerate_terminal_winner_states(limit: int = 10000) -> List[Board]:
    """Enumerate legal terminal boards with exactly one winner (no draws)."""
    results = set()

    def dfs(board: List[int], player: int) -> None:
        w = check_winner(board)
        if w != 0 or all(cell != 0 for cell in board):
            if w in {1, 2}:
                x_count = sum(1 for x in board if x == 1)
                o_count = sum(1 for x in board if x == 2)
                # Reachable count constraints for alternating play from empty board.
                if x_count == o_count or x_count == o_count + 1:
                    results.add(tuple(board))
            return

        for idx in range(9):
            if board[idx] == 0:
                board[idx] = player
                dfs(board, 2 if player == 1 else 1)
                board[idx] = 0

    dfs([0] * 9, 1)
    out = sorted(results, key=lambda s: json.dumps(s))
    return out[:limit]


def _state_key(board: Board) -> str:
    return json.dumps(board, separators=(",", ":"), sort_keys=False)


def _allocate_line_targets(total: int, line_names: List[str], capacities: Dict[str, int]) -> Dict[str, int]:
    if total <= 0:
        return {name: 0 for name in line_names}
    n = len(line_names)
    base = total // n
    targets = {name: min(base, capacities.get(name, 0)) for name in line_names}
    assigned = sum(targets.values())
    while assigned < total:
        progressed = False
        for name in line_names:
            if targets[name] < capacities.get(name, 0):
                targets[name] += 1
                assigned += 1
                progressed = True
                if assigned >= total:
                    break
        if not progressed:
            break
    return targets


def _allocate_xo_for_lines(
    line_targets: Dict[str, int],
    x_caps: Dict[str, int],
    o_caps: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    x_targets = {line: int(v) // 2 for line, v in line_targets.items()}
    x_total_target = sum(int(v) for v in line_targets.values()) // 2
    need_x = x_total_target - sum(x_targets.values())

    if need_x > 0:
        odd_lines = [line for line, v in line_targets.items() if int(v) % 2 == 1]
        for line in odd_lines:
            if need_x <= 0:
                break
            px = x_targets[line] + 1
            po = int(line_targets[line]) - px
            if px <= int(x_caps.get(line, 0)) and po <= int(o_caps.get(line, 0)):
                x_targets[line] = px
                need_x -= 1

    o_targets = {line: int(line_targets[line]) - int(x_targets[line]) for line in line_targets}
    return x_targets, o_targets


def collect_exclusive_balanced_states(
    total_states: int,
    seed: int,
    category_targets: Dict[str, int] | None = None,
) -> Tuple[List[Board], Dict[str, int]]:
    """Collect states with exclusive winning category.

    Default target: horizontal=100, vertical=100, main_diagonal=50, anti_diagonal=50.
    """
    if total_states <= 0:
        raise ValueError("total_states must be > 0")

    targets = category_targets or {
        "horizontal": 100,
        "vertical": 100,
        "main_diagonal": 50,
        "anti_diagonal": 50,
    }
    if sum(int(v) for v in targets.values()) != total_states:
        raise ValueError(f"category targets must sum to {total_states}, got {targets}")

    all_states = enumerate_terminal_winner_states(limit=10000)

    # line_buckets[line_name][winner_label] -> boards
    line_buckets: Dict[str, Dict[str, List[Board]]] = {}
    for line_group in _CATEGORY_TO_LINES.values():
        for line_name in line_group:
            line_buckets[line_name] = {"X": [], "O": []}

    for board in all_states:
        cats = winning_categories(board)
        if len(cats) != 1:
            continue
        lines = winning_line_names(board)
        if len(lines) != 1:
            continue
        line = lines[0]
        winner = canonical_label(board)
        if winner in {"X", "O"}:
            line_buckets[line][winner].append(board)

    # deterministic shuffle pools
    for line_name in sorted(line_buckets.keys()):
        for winner in ("X", "O"):
            pool = sorted(line_buckets[line_name][winner], key=_state_key)
            random.Random(seed ^ hash((line_name, winner)) ^ 0x5A5A5A).shuffle(pool)
            line_buckets[line_name][winner] = pool

    selected: List[Board] = []
    selected_keys = set()

    for category in ["horizontal", "vertical", "main_diagonal", "anti_diagonal"]:
        cat_target = int(targets.get(category, 0))
        lines = _CATEGORY_TO_LINES[category]
        line_caps = {ln: len(line_buckets[ln]["X"]) + len(line_buckets[ln]["O"]) for ln in lines}
        line_targets = _allocate_line_targets(cat_target, lines, line_caps)

        x_caps = {ln: len(line_buckets[ln]["X"]) for ln in lines}
        o_caps = {ln: len(line_buckets[ln]["O"]) for ln in lines}
        x_targets, o_targets = _allocate_xo_for_lines(line_targets, x_caps, o_caps)

        for ln in lines:
            want_x = min(int(x_targets[ln]), len(line_buckets[ln]["X"]))
            want_o = min(int(o_targets[ln]), len(line_buckets[ln]["O"]))
            for board in line_buckets[ln]["X"][:want_x] + line_buckets[ln]["O"][:want_o]:
                key = _state_key(board)
                if key not in selected_keys:
                    selected.append(board)
                    selected_keys.add(key)

    if len(selected) < total_states:
        # backfill from any exclusive pool
        residual: List[Board] = []
        for category in ["horizontal", "vertical", "main_diagonal", "anti_diagonal"]:
            for ln in _CATEGORY_TO_LINES[category]:
                for winner in ("X", "O"):
                    for board in line_buckets[ln][winner]:
                        key = _state_key(board)
                        if key not in selected_keys:
                            residual.append(board)
                            selected_keys.add(key)
        residual = sorted(residual, key=_state_key)
        random.Random(seed ^ 0x777777).shuffle(residual)
        selected.extend(residual[: total_states - len(selected)])

    selected = selected[:total_states]
    random.Random(seed ^ 0xBEEFBEEF).shuffle(selected)

    winner_dist = {"X": 0, "O": 0}
    for board in selected:
        winner_dist[canonical_label(board)] += 1

    return selected, winner_dist
