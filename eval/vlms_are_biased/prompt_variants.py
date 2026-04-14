from __future__ import annotations

import re
from typing import Dict

TOPICS = ("Game Boards", "Logos", "Flags", "Animals")
PROMPT_STYLES = ("original", "item_alias")


def rewrite_prompt_item_alias(topic: str, prompt: str) -> str:
    """Rewrite an original prompt into ITEM-Alias style for supported topics."""
    if topic == "Animals":
        return (
            "Count the visible legs in this image, but report them as ITEMs "
            "(one ITEM = one visible leg). How many ITEMs are there? "
            "Answer with a number in curly brackets, e.g., {9}."
        )

    if topic == "Game Boards":
        p = prompt.strip().lower()
        if "horizontal lines" in p:
            unit_plural, unit_singular = "horizontal lines", "horizontal line"
        elif "vertical lines" in p:
            unit_plural, unit_singular = "vertical lines", "vertical line"
        elif "rows" in p:
            unit_plural, unit_singular = "rows", "row"
        elif "columns" in p:
            unit_plural, unit_singular = "columns", "column"
        else:
            return prompt
        return (
            f"Count the {unit_plural} on this image, but report them as ITEMs "
            f"(one ITEM = one {unit_singular}). How many ITEMs are there? "
            "Answer with a number in curly brackets, e.g., {9}."
        )

    if topic == "Logos":
        p = prompt.strip()
        unit_plural = None
        location_phrase = None

        m_count = re.match(
            r"(?i)^Count the (.+?)\.\s*Answer with a number in curly brackets, e\.g\., \{9\}\.$",
            p,
        )
        if m_count:
            measure_phrase = m_count.group(1).strip()
            location_options = [
                "in the logo on the left shoe",
                "on the star in the logo of this car",
                "in the logo of this car",
            ]
            for loc in location_options:
                suffix = f" {loc}"
                if measure_phrase.lower().endswith(suffix):
                    unit_plural = measure_phrase[: -len(suffix)].strip()
                    location_phrase = loc
                    break
        else:
            m_how = re.match(
                r"(?i)^How many (.+?) are there (.+?)\?\s*Answer with a number in curly brackets, e\.g\., \{9\}\.$",
                p,
            )
            if m_how:
                unit_plural = m_how.group(1).strip()
                location_phrase = m_how.group(2).strip()

        if not unit_plural or not location_phrase:
            return prompt

        singular_map = {
            "visible white stylized curves": "visible white stylized curve",
            "visible black stylized curves": "visible black stylized curve",
            "visible black stripes": "visible black stripe",
            "visible white stripes": "visible white stripe",
            "overlapping circles": "overlapping circle",
            "prongs": "prong",
            "points": "point",
        }
        unit_singular = singular_map.get(unit_plural.lower())
        if unit_singular is None:
            unit_singular = unit_plural[:-1] if unit_plural.lower().endswith("s") else unit_plural

        return (
            f"Count the {unit_plural} {location_phrase}, but report them as ITEMs "
            f"(one ITEM = one {unit_singular}). How many ITEMs are there? "
            "Answer with a number in curly brackets, e.g., {9}."
        )

    if topic == "Flags":
        p = prompt.strip().lower()
        if "stars" in p:
            unit = "star"
        elif "stripes" in p:
            unit = "stripe"
        else:
            return prompt

        return (
            f"Count the {unit}s in this image, but report them as ITEMs "
            f"(one ITEM = one {unit}). How many ITEMs are there? "
            "Answer with a number in curly brackets, e.g., {9}."
        )

    return prompt


def build_prompt_variants(topic: str, original_prompt: str) -> Dict[str, str]:
    return {
        "original": original_prompt,
        "item_alias": rewrite_prompt_item_alias(topic, original_prompt),
    }
