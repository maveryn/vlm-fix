from __future__ import annotations

from typing import Sequence


def _normalize_rule(rule_variant: str) -> str:
    rv = str(rule_variant).strip().lower()
    if rv in {"standard", "canonical"}:
        return "standard"
    if rv in {"inverse", "misere", "inverted_score"}:
        return "inverse"
    raise ValueError(f"Unsupported rule_variant: {rule_variant}")


def _normalize_prompt_variant(prompt_variant: str) -> str:
    pv = str(prompt_variant).strip().lower()
    aliases = {
        "original": "standard",
        "standard": "standard",
        "tag_alias": "tag",
        "tag": "tag",
        "tag_alias_semantic": "tag_sem",
        "tag_sem": "tag_sem",
        "desc": "desc",
        "desc_rule": "desc",
        "descriptive_rule": "desc",
        "descriptive_neutral": "desc",
    }
    out = aliases.get(pv)
    if out is None:
        raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")
    return out


def _ask_positive(rule_variant: str, question_target: str) -> bool:
    return (rule_variant == "standard" and question_target == "winner") or (
        rule_variant == "inverse" and question_target == "loser"
    )


def prompt_for(
    game: str,
    rule_variant: str,
    labels: Sequence[str],
    question_target: str,
    prompt_type: str,
    prompt_variant: str = "standard",
    tag_pair: tuple[str, str] | None = None,
) -> str:
    if len(labels) != 2:
        raise ValueError(f"Expected two labels, got: {labels}")
    p1, p2 = str(labels[0]), str(labels[1])

    rv = _normalize_rule(rule_variant)
    pv = _normalize_prompt_variant(prompt_variant)

    if question_target not in {"winner", "loser"}:
        raise ValueError(f"Unsupported question_target: {question_target}")
    if prompt_type not in {"direct", "cot"}:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    if prompt_type == "direct":
        ending = f"Answer with only {p1} or {p2}. Do not add any other text."
    else:
        ending = f"Reason step by step. After that, give the answer ({p1} or {p2}) inside \\boxed{{ }}."

    # Descriptive prompts intentionally avoid explicit win/lose wording.
    if pv == "desc":
        if game == "tictactoe":
            if _ask_positive(rv, question_target):
                q = "Which player forms a 3-in-a-row line?"
            else:
                q = "Which player does not form a 3-in-a-row line?"
            return (
                f"You are given a 3x3 grid from a two-player game. Players are {p1} and {p2}. "
                f"One player forms a 3-in-a-row line (horizontal, vertical, or diagonal) and the other does not. "
                f"{q} {ending}"
            )
        if game == "connect4":
            if _ask_positive(rv, question_target):
                q = "Which player forms a 4-in-a-row line?"
            else:
                q = "Which player does not form a 4-in-a-row line?"
            return (
                f"You are given a 4x4 vertical grid from a two-player game. Players are {p1} and {p2}. "
                f"One player forms a 4-in-a-row line (horizontal, vertical, or diagonal) and the other does not. "
                f"{q} {ending}"
            )
        if game == "reversi":
            if _ask_positive(rv, question_target):
                q = "Which player has more pieces on the grid?"
            else:
                q = "Which player has fewer pieces on the grid?"
            return f"You are given a 5x5 grid from a two-player game. Players are {p1} and {p2}. {q} {ending}"
        if game == "dots_boxes":
            if _ask_positive(rv, question_target):
                q = "Which player has more claimed cells?"
            else:
                q = "Which player has fewer claimed cells?"
            return (
                f"You are given a 6x6 dot grid from a two-player game. Players are {p1} and {p2}. "
                f"{q} {ending}"
            )
        raise ValueError(f"Unsupported game: {game}")

    # Standard and tag/tag_sem variants share game-rule framing.
    if pv == "standard":
        win_term = "wins"
        lose_term = "loses"
        outcome_tags_line = ""
        question_line = "Who is the winner?" if question_target == "winner" else "Who is the loser?"
    else:
        if tag_pair is None or len(tag_pair) != 2:
            raise ValueError("tag/tag_sem prompt variants require tag_pair=(win_tag, lose_tag)")
        win_term = str(tag_pair[0]).strip()
        lose_term = str(tag_pair[1]).strip()
        if not win_term or not lose_term or win_term.lower() == lose_term.lower():
            raise ValueError(f"Invalid tag_pair: {tag_pair}")
        if pv == "tag_sem":
            outcome_tags_line = (
                f"Outcome tags: {win_term} and {lose_term}. "
                f"{win_term} means favorable outcome; {lose_term} means unfavorable outcome."
            )
        else:
            outcome_tags_line = f"Outcome tags: {win_term} and {lose_term}."
        question_line = f"Who is the {win_term}?" if question_target == "winner" else f"Who is the {lose_term}?"
        win_term = f"is {win_term}"
        lose_term = f"is {lose_term}"

    if game == "tictactoe":
        rule = (
            f"If a player has 3 in a row (horizontal, vertical, or diagonal), that player {win_term}, "
            f"and the other player {lose_term}."
            if rv == "standard"
            else f"If a player has 3 in a row (horizontal, vertical, or diagonal), that player {lose_term}, "
            f"and the other player {win_term}."
        )
        parts = [f"You are given a 3x3 grid for a two-player game.", f"Players are {p1} and {p2}."]
        if outcome_tags_line:
            parts.append(outcome_tags_line)
        parts.extend([rule, "The game has ended.", question_line, ending])
        return " ".join(parts)

    if game == "connect4":
        rule = (
            f"If a player has 4 in a row (horizontal, vertical, or diagonal), that player {win_term}, "
            f"and the other player {lose_term}."
            if rv == "standard"
            else f"If a player has 4 in a row (horizontal, vertical, or diagonal), that player {lose_term}, "
            f"and the other player {win_term}."
        )
        parts = [f"You are given a 4x4 vertical grid for a two-player game.", f"Players are {p1} and {p2}."]
        if outcome_tags_line:
            parts.append(outcome_tags_line)
        parts.extend([rule, "The game has ended.", question_line, ending])
        return " ".join(parts)

    if game == "reversi":
        rule = (
            f"When the game ends, if a player has more pieces on the grid than the other player, that player {win_term}, "
            f"and the other player {lose_term}."
            if rv == "standard"
            else f"When the game ends, if a player has fewer pieces on the grid than the other player, that player {win_term}, "
            f"and the other player {lose_term}."
        )
        parts = [f"You are given a 5x5 grid for a two-player game.", f"Players are {p1} and {p2}."]
        if outcome_tags_line:
            parts.append(outcome_tags_line)
        parts.extend([rule, "The game has ended.", question_line, ending])
        return " ".join(parts)

    if game == "dots_boxes":
        rule = (
            f"When the game ends, if a player has claimed more boxes than the other player, that player {win_term}, "
            f"and the other player {lose_term}."
            if rv == "standard"
            else f"When the game ends, if a player has claimed fewer boxes than the other player, that player {win_term}, "
            f"and the other player {lose_term}."
        )
        parts = [
            f"You are given a 6x6 dot grid for a two-player game.",
            f"Players are {p1} and {p2}.",
        ]
        if outcome_tags_line:
            parts.append(outcome_tags_line)
        parts.extend([rule, "The game has ended.", question_line, ending])
        return " ".join(parts)

    raise ValueError(f"Unsupported game: {game}")
