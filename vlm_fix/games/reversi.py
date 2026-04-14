from __future__ import annotations

import json
import random
from typing import List, Sequence, Tuple

SIZE = 5
Board = Tuple[Tuple[int, ...], ...]  # 0 empty, 1 player1, 2 player2
_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def initial_board() -> List[List[int]]:
    board = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
    c1 = SIZE // 2 - 1
    c2 = SIZE // 2
    board[c1][c1] = 2
    board[c1][c2] = 1
    board[c2][c1] = 1
    board[c2][c2] = 2
    return board


def board_to_tuple(board: Sequence[Sequence[int]]) -> Board:
    return tuple(tuple(int(x) for x in row) for row in board)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < SIZE and 0 <= c < SIZE


def _flips_for_move(board: Sequence[Sequence[int]], r: int, c: int, player: int) -> List[Tuple[int, int]]:
    if board[r][c] != 0:
        return []
    other = 2 if player == 1 else 1
    flips: List[Tuple[int, int]] = []
    for dr, dc in _DIRS:
        rr, cc = r + dr, c + dc
        path: List[Tuple[int, int]] = []
        while _in_bounds(rr, cc) and int(board[rr][cc]) == other:
            path.append((rr, cc))
            rr += dr
            cc += dc
        if path and _in_bounds(rr, cc) and int(board[rr][cc]) == player:
            flips.extend(path)
    return flips


def legal_moves(board: Sequence[Sequence[int]], player: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for r in range(SIZE):
        for c in range(SIZE):
            if _flips_for_move(board, r, c, player):
                out.append((r, c))
    return out


def apply_move(board: List[List[int]], r: int, c: int, player: int) -> bool:
    flips = _flips_for_move(board, r, c, player)
    if not flips:
        return False
    board[r][c] = player
    for rr, cc in flips:
        board[rr][cc] = player
    return True


def is_terminal(board: Sequence[Sequence[int]]) -> bool:
    if all(int(board[r][c]) != 0 for r in range(SIZE) for c in range(SIZE)):
        return True
    return not legal_moves(board, 1) and not legal_moves(board, 2)


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


def misere_winner_idx(board: Sequence[Sequence[int]]) -> int:
    p1, p2 = counts(board)
    if p1 < p2:
        return 1
    if p2 < p1:
        return 2
    return 0


def canonical_label(board: Sequence[Sequence[int]]) -> str:
    w = canonical_winner_idx(board)
    if w == 1:
        return "Black"
    if w == 2:
        return "White"
    return "Draw"


def misere_label(board: Sequence[Sequence[int]]) -> str:
    w = misere_winner_idx(board)
    if w == 1:
        return "Black"
    if w == 2:
        return "White"
    return "Draw"


def sample_terminal_states(max_states: int = 1000, seed: int = 0, max_attempts: int = 800000) -> List[Board]:
    rng = random.Random(seed)
    states = set()
    attempts = 0

    while len(states) < max_states and attempts < max_attempts:
        attempts += 1
        board = initial_board()
        player = 1
        passes = 0

        while passes < 2:
            moves = legal_moves(board, player)
            if not moves:
                passes += 1
                player = 2 if player == 1 else 1
                continue
            passes = 0
            r, c = rng.choice(moves)
            apply_move(board, r, c, player)
            player = 2 if player == 1 else 1
            if all(board[rr][cc] != 0 for rr in range(SIZE) for cc in range(SIZE)):
                break

        if is_terminal(board):
            w = canonical_winner_idx(board)
            if w in (1, 2):
                states.add(board_to_tuple(board))

    out = sorted(states, key=lambda s: json.dumps(s, separators=(",", ":")))
    return out


def sample_balanced_terminal_states(total_states: int = 300, seed: int = 0) -> List[Board]:
    if total_states <= 0 or total_states % 2 != 0:
        raise ValueError("total_states must be a positive even number")

    target_each = total_states // 2
    buckets: dict[str, dict[str, Board]] = {"Black": {}, "White": {}}
    rng = random.Random(seed)

    # Aggregate from multiple random seeds to get enough unique terminal states per winner.
    rounds = 0
    while min(len(buckets["Black"]), len(buckets["White"])) < target_each and rounds < 300:
        rounds += 1
        round_seed = rng.randint(0, 2**31 - 1)
        candidates = sample_terminal_states(max_states=2000, seed=round_seed, max_attempts=120000)
        for board in candidates:
            lbl = canonical_label(board)
            if lbl not in {"Black", "White"}:
                continue
            key = json.dumps(board, separators=(",", ":"))
            if len(buckets[lbl]) < target_each and key not in buckets[lbl]:
                buckets[lbl][key] = board
            if min(len(buckets["Black"]), len(buckets["White"])) >= target_each:
                break

    if min(len(buckets["Black"]), len(buckets["White"])) < target_each:
        raise RuntimeError(
            f"Could not sample balanced reversi states: have Black={len(buckets['Black'])}, White={len(buckets['White'])}, need {target_each} each"
        )

    black_states = [buckets["Black"][k] for k in sorted(buckets["Black"].keys())[:target_each]]
    white_states = [buckets["White"][k] for k in sorted(buckets["White"].keys())[:target_each]]

    out = black_states + white_states
    rng.shuffle(out)
    return out
