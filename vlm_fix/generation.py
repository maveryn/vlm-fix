from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .games import connect4, dots_boxes, reversi, tictactoe
from .prompts import prompt_for
from .render import connect4 as connect4_render
from .render import dots_boxes as dots_boxes_render
from .render import reversi as reversi_render
from .render import tictactoe as tictactoe_render

GAMES = ["tictactoe", "reversi", "connect4", "dots_boxes"]
RENDER_VARIANTS = ["canonical", "checkerboard", "glyph"]
RULE_VARIANTS = ["standard", "inverse"]
IMAGE_TEXT_ORDERS = ["image_first", "text_first"]
PROMPT_VARIANTS = ["standard", "tag", "tag_sem", "desc"]
PROMPT_TYPES = ["direct", "cot"]
QUESTION_TARGETS = ["winner", "loser"]

BANNED_GLYPHS = {"X", "O", "A", "B"}
TAG_POOL = ["KAP", "TOV", "NEX", "RIL", "POM"]


@dataclass(frozen=True)
class StateSpec:
    game: str
    state_id: int
    board_json: str
    image_rel_path: str
    render_variant: str
    labels: Tuple[str, str]
    meta_json: str
    standard_winner_idx: int
    inverse_winner_idx: int


def _safe_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _other_idx(idx: int) -> int:
    if idx == 1:
        return 2
    if idx == 2:
        return 1
    return 0


def _stable_key_int(text: str) -> int:
    out = 0
    for i, ch in enumerate(text):
        out += (i + 1) * ord(ch)
    return out


def _choose_glyph_pair(game: str, state_id: int, seed: int) -> Tuple[str, str]:
    letters = [chr(ord("A") + i) for i in range(26)]
    pool = [x for x in letters if x not in BANNED_GLYPHS]
    rng = random.Random(seed ^ (state_id * 1315423911) ^ _stable_key_int(game))
    g1, g2 = rng.sample(pool, 2)
    return g1, g2


def _pick_tag_pair(
    seed: int,
    game: str,
    state_id: int,
    rule_variant: str,
    question_target: str,
    image_text_order: str,
    prompt_type: str,
) -> Tuple[str, str]:
    key = "|".join(
        [
            game,
            str(state_id),
            str(rule_variant),
            str(question_target),
            str(image_text_order),
            str(prompt_type),
        ]
    )
    rng = random.Random(seed ^ _stable_key_int(key))
    t1, t2 = rng.sample(TAG_POOL, 2)
    return str(t1), str(t2)


def _prompt_allowed(render_variant: str, prompt_variant: str) -> bool:
    if prompt_variant == "standard":
        return render_variant in {"canonical", "checkerboard", "glyph"}
    return render_variant == "canonical"


def _ttt_boards(states_per_game: int, seed: int) -> List[Tuple[int, ...]]:
    boards, _dist = tictactoe.collect_exclusive_balanced_states(
        total_states=states_per_game,
        seed=seed,
        category_targets={
            "horizontal": 100,
            "vertical": 100,
            "main_diagonal": 50,
            "anti_diagonal": 50,
        },
    )
    return boards


def _reversi_boards(states_per_game: int, seed: int) -> List[Tuple[Tuple[int, ...], ...]]:
    return reversi.sample_balanced_terminal_states(total_states=states_per_game, seed=seed)


def _connect4_boards(states_per_game: int, seed: int) -> List[Tuple[Tuple[int, ...], ...]]:
    return connect4.sample_balanced_terminal_states(total_states=states_per_game, seed=seed)


def _dots_boxes_boards(states_per_game: int, seed: int) -> List[Tuple[Tuple[int, ...], ...]]:
    return dots_boxes.sample_balanced_terminal_states(total_states=states_per_game, seed=seed)


def _ttt_rows(board: Sequence[int]) -> List[List[int]]:
    return [list(board[0:3]), list(board[3:6]), list(board[6:9])]


def _render_image(
    game: str,
    board,
    render_variant: str,
    labels: Tuple[str, str],
    image_size: int,
    meta: Dict[str, object] | None = None,
):
    meta = meta or {}
    if game == "tictactoe":
        board_rows = _ttt_rows(board)
        if render_variant == "canonical":
            return tictactoe_render.render(board_rows, style="canonical", size=image_size, render_scale=3)
        if render_variant == "checkerboard":
            return tictactoe_render.render(board_rows, style="checkerboard", size=image_size, render_scale=3)
        if render_variant == "glyph":
            return tictactoe_render.render(
                board_rows,
                style="glyph_random",
                size=image_size,
                meta={"glyph_p1": labels[0], "glyph_p2": labels[1]},
                render_scale=3,
            )

    if game == "reversi":
        if render_variant == "canonical":
            return reversi_render.render(board, style="canonical", size_px=image_size, render_scale=3)
        if render_variant == "checkerboard":
            return reversi_render.render(board, style="checkerboard", size_px=image_size, render_scale=3)
        if render_variant == "glyph":
            return reversi_render.render(
                board,
                style="glyph_random",
                size_px=image_size,
                meta={"glyph_p1": labels[0], "glyph_p2": labels[1]},
                render_scale=3,
            )

    if game == "connect4":
        if render_variant == "canonical":
            return connect4_render.render(board, style="canonical", size=image_size, render_scale=3)
        if render_variant == "checkerboard":
            return connect4_render.render(board, style="checkerboard", size=image_size, render_scale=3)
        if render_variant == "glyph":
            return connect4_render.render(
                board,
                style="glyphs",
                size=image_size,
                meta={"glyph_p1": labels[0], "glyph_p2": labels[1]},
                render_scale=3,
            )

    if game == "dots_boxes":
        if render_variant == "canonical":
            return dots_boxes_render.render(board, style="canonical", size_px=image_size, render_scale=3)
        if render_variant == "checkerboard":
            return dots_boxes_render.render(board, style="checkerboard", size_px=image_size, render_scale=3)
        if render_variant == "glyph":
            return dots_boxes_render.render(
                board,
                style="glyphs",
                size_px=image_size,
                meta={"glyph_p1": labels[0], "glyph_p2": labels[1]},
                render_scale=3,
            )

    raise ValueError(f"Unsupported render request: game={game}, render_variant={render_variant}")


def _winner_indices(game: str, board) -> Tuple[int, int]:
    if game == "tictactoe":
        c = tictactoe.check_winner(board)
        if c not in {1, 2}:
            raise ValueError("tictactoe board must have a unique winner")
        return c, _other_idx(c)
    if game == "reversi":
        c = reversi.canonical_winner_idx(board)
        m = reversi.misere_winner_idx(board)
        if c not in {1, 2} or m not in {1, 2}:
            raise ValueError("reversi board must have non-draw winners for both rules")
        return c, m
    if game == "connect4":
        c = connect4.canonical_winner_idx(board)
        m = connect4.inverse_winner_idx(board)
        if c not in {1, 2} or m not in {1, 2}:
            raise ValueError("connect4 board must have non-draw winners for both rules")
        return c, m
    if game == "dots_boxes":
        c = dots_boxes.canonical_winner_idx(board)
        m = dots_boxes.inverse_winner_idx(board)
        if c not in {1, 2} or m not in {1, 2}:
            raise ValueError("dots_boxes board must have non-draw winners for both rules")
        return c, m
    raise ValueError(game)


def _canonical_labels(game: str) -> Tuple[str, str]:
    if game == "tictactoe":
        return ("X", "O")
    if game == "reversi":
        return ("Black", "White")
    if game == "connect4":
        return ("Red", "Yellow")
    if game == "dots_boxes":
        return ("A", "B")
    raise ValueError(game)


def build_dataset(
    out_dir: str | Path,
    states_per_game: int = 300,
    seed: int = 7,
    image_size_ttt: int = 384,
    image_size_reversi: int = 480,
    image_size_connect4: int = 448,
    image_size_dots_boxes: int = 480,
    games: Sequence[str] | None = None,
    render_variants: Sequence[str] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "images")

    if states_per_game != 300:
        raise ValueError("This benchmark is configured for 300 states per game.")

    use_games = [str(g).strip() for g in (games or GAMES) if str(g).strip()]
    use_render_variants = [str(v).strip() for v in (render_variants or RENDER_VARIANTS) if str(v).strip()]
    if not use_games:
        raise ValueError("games cannot be empty")
    if not use_render_variants:
        raise ValueError("render_variants cannot be empty")

    base_states = {}
    for game in use_games:
        if game == "tictactoe":
            base_states[game] = _ttt_boards(states_per_game=states_per_game, seed=seed ^ 0x1111)
        elif game == "reversi":
            base_states[game] = _reversi_boards(states_per_game=states_per_game, seed=seed ^ 0x2222)
        elif game == "connect4":
            base_states[game] = _connect4_boards(states_per_game=states_per_game, seed=seed ^ 0x3333)
        elif game == "dots_boxes":
            base_states[game] = _dots_boxes_boards(states_per_game=states_per_game, seed=seed ^ 0x4444)
        else:
            raise ValueError(f"Unsupported game: {game}")

    image_sizes = {
        "tictactoe": image_size_ttt,
        "reversi": image_size_reversi,
        "connect4": image_size_connect4,
        "dots_boxes": image_size_dots_boxes,
    }

    state_specs: List[StateSpec] = []
    for game in use_games:
        for render_variant in use_render_variants:
            for i, board in enumerate(base_states[game]):
                state_id = i + 1
                if render_variant in {"canonical", "checkerboard"}:
                    labels = _canonical_labels(game)
                    meta: Dict[str, object] = {}
                elif render_variant == "glyph":
                    g1, g2 = _choose_glyph_pair(game=game, state_id=state_id, seed=seed)
                    labels = (g1, g2)
                    meta = {"glyph_p1": g1, "glyph_p2": g2}
                else:
                    raise ValueError(f"Unsupported render_variant: {render_variant}")

                standard_winner_idx, inverse_winner_idx = _winner_indices(game=game, board=board)
                img_rel = Path("images") / game / render_variant / f"state_{state_id:03d}.png"
                img_abs = out_dir / img_rel
                _ensure_dir(img_abs.parent)
                img = _render_image(
                    game=game,
                    board=board,
                    render_variant=render_variant,
                    labels=labels,
                    image_size=int(image_sizes[game]),
                    meta=meta,
                )
                img.save(img_abs)

                state_specs.append(
                    StateSpec(
                        game=game,
                        state_id=state_id,
                        board_json=_safe_json(board),
                        image_rel_path=str(img_rel),
                        render_variant=render_variant,
                        labels=labels,
                        meta_json=_safe_json(meta),
                        standard_winner_idx=standard_winner_idx,
                        inverse_winner_idx=inverse_winner_idx,
                    )
                )

    rows: List[Dict[str, object]] = []
    idx = 1
    for spec in state_specs:
        p1, p2 = spec.labels
        valid_labels = f"{p1}|{p2}"

        for rule_variant in RULE_VARIANTS:
            rule_winner = spec.standard_winner_idx if rule_variant == "standard" else spec.inverse_winner_idx
            for image_text_order in IMAGE_TEXT_ORDERS:
                for prompt_variant in PROMPT_VARIANTS:
                    if not _prompt_allowed(render_variant=spec.render_variant, prompt_variant=prompt_variant):
                        continue
                    for prompt_type in PROMPT_TYPES:
                        for question_target in QUESTION_TARGETS:
                            answer_idx = rule_winner if question_target == "winner" else _other_idx(rule_winner)
                            answer = p1 if answer_idx == 1 else p2

                            tag_pair = None
                            if prompt_variant in {"tag", "tag_sem"}:
                                tag_pair = _pick_tag_pair(
                                    seed=seed,
                                    game=spec.game,
                                    state_id=spec.state_id,
                                    rule_variant=rule_variant,
                                    question_target=question_target,
                                    image_text_order=image_text_order,
                                    prompt_type=prompt_type,
                                )

                            prompt = prompt_for(
                                game=spec.game,
                                rule_variant=rule_variant,
                                labels=spec.labels,
                                question_target=question_target,
                                prompt_type=prompt_type,
                                prompt_variant=prompt_variant,
                                tag_pair=tag_pair,
                            )

                            rows.append(
                                {
                                    "index": idx,
                                    "game": spec.game,
                                    "state_id": spec.state_id,
                                    "board_state": spec.board_json,
                                    "render_variant": spec.render_variant,
                                    "rule_variant": rule_variant,
                                    "image_text_order": image_text_order,
                                    "prompt_variant": prompt_variant,
                                    "prompt_type": prompt_type,
                                    "question_target": question_target,
                                    "prompt": prompt,
                                    "answer": answer,
                                    "valid_labels": valid_labels,
                                    "image_path": spec.image_rel_path,
                                    "render_meta": spec.meta_json,
                                    "standard_winner_idx": spec.standard_winner_idx,
                                    "inverse_winner_idx": spec.inverse_winner_idx,
                                }
                            )
                            idx += 1

    df = pd.DataFrame(rows)
    out_parquet = out_dir / "instances.parquet"
    df.to_parquet(out_parquet, index=False)

    summary = {
        "dataset": "vlm_fix",
        "states_per_game": states_per_game,
        "games": sorted(df["game"].unique().tolist()),
        "render_variants": sorted(df["render_variant"].unique().tolist()),
        "rule_variants": sorted(df["rule_variant"].unique().tolist()),
        "image_text_orders": sorted(df["image_text_order"].unique().tolist()),
        "prompt_variants": sorted(df["prompt_variant"].unique().tolist()),
        "prompt_types": sorted(df["prompt_type"].unique().tolist()),
        "question_targets": sorted(df["question_target"].unique().tolist()),
        "total_instances": int(len(df)),
        "instances_per_game": {k: int(v) for k, v in df.groupby("game").size().to_dict().items()},
        "instances_per_game_render_rule": {
            "|".join(map(str, k)): int(v)
            for k, v in df.groupby(["game", "render_variant", "rule_variant"]).size().to_dict().items()
        },
        "answer_distribution": {
            "|".join(map(str, k)): int(v) for k, v in df.groupby(["game", "answer"]).size().to_dict().items()
        },
    }

    # Additional useful diagnostics.
    ttt_base = df[df["game"] == "tictactoe"].drop_duplicates(subset=["state_id"])
    if not ttt_base.empty:
        line_dist = {"horizontal": 0, "vertical": 0, "main_diagonal": 0, "anti_diagonal": 0}
        for _, row in ttt_base.iterrows():
            board = json.loads(str(row["board_state"]))
            cats = tictactoe.winning_categories(board)
            if len(cats) == 1 and cats[0] in line_dist:
                line_dist[cats[0]] += 1
        summary["tictactoe_state_win_category_distribution"] = line_dist

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    df.groupby(["game", "render_variant", "rule_variant"]).size().reset_index(name="n").to_csv(
        out_dir / "counts_by_game_render_rule.csv", index=False
    )

    return out_parquet
