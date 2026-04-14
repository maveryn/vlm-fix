from __future__ import annotations

import math
from typing import Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def _lanczos_resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _load_font(px: int):
    size = max(8, int(px))
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, glyph: str, max_side: float, start_size: int):
    size = max(8, int(start_size))
    while size >= 8:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), glyph, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= max_side and h <= max_side:
            return font
        size -= 2
    return _load_font(8)


def _draw_centered_text(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, font, fill):
    try:
        draw.text((cx, cy), text, font=font, fill=fill, anchor="mm")
    except TypeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((cx - w / 2, cy - h / 2), text, font=font, fill=fill)


def _draw_star_outline(draw: ImageDraw.ImageDraw, cx: float, cy: float, r_outer: float, outline, width: int):
    r_inner = r_outer * 0.45
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * (math.pi / 5)
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    for i in range(10):
        draw.line([pts[i], pts[(i + 1) % 10]], fill=outline, width=width)


def _draw_square_outline(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, outline, width: int):
    draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=outline, width=width)


def _draw_star_filled(draw: ImageDraw.ImageDraw, cx: float, cy: float, r_outer: float, fill, outline, width: int):
    r_inner = r_outer * 0.45
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * (math.pi / 5)
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    draw.polygon(pts, fill=fill, outline=outline)
    for i in range(10):
        draw.line([pts[i], pts[(i + 1) % 10]], fill=outline, width=width)


def _draw_triangle_filled(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int):
    pts = [
        (cx, cy - r),
        (cx - r * 0.92, cy + r * 0.82),
        (cx + r * 0.92, cy + r * 0.82),
    ]
    draw.polygon(pts, fill=fill, outline=outline)
    for i in range(3):
        draw.line([pts[i], pts[(i + 1) % 3]], fill=outline, width=width)


def _draw_square_filled(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int):
    draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def _draw_circle_filled(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def _draw_crescent_filled(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    r: float,
    fill,
    outline,
    width: int,
    bg,
):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)
    cut_r = r * 0.78
    shift = r * 0.35
    draw.ellipse(
        (cx - cut_r + shift, cy - cut_r, cx + cut_r + shift, cy + cut_r),
        fill=bg,
        outline=bg,
        width=max(1, width // 2),
    )
    draw.arc((cx - r, cy - r, cx + r, cy + r), start=70, end=290, fill=outline, width=width)


def _to_rgb(v, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(v, (tuple, list)) and len(v) == 3:
        return (int(v[0]), int(v[1]), int(v[2]))
    return default


def _draw_filled_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    cx: float,
    cy: float,
    r: float,
    fill,
    outline,
    width: int,
    bg,
):
    s = str(shape).strip().lower()
    if s == "triangle":
        _draw_triangle_filled(draw, cx, cy, r, fill=fill, outline=outline, width=width)
        return
    if s == "square":
        _draw_square_filled(draw, cx, cy, r * 0.95, fill=fill, outline=outline, width=width)
        return
    if s == "star":
        _draw_star_filled(draw, cx, cy, r, fill=fill, outline=outline, width=width)
        return
    if s == "circle":
        _draw_circle_filled(draw, cx, cy, r * 0.98, fill=fill, outline=outline, width=width)
        return
    if s in {"moon", "crescent"}:
        _draw_crescent_filled(draw, cx, cy, r * 0.98, fill=fill, outline=outline, width=width, bg=bg)
        return
    _draw_circle_filled(draw, cx, cy, r * 0.98, fill=fill, outline=outline, width=width)


def _render_grid_base(size: int, scale: int):
    size_hi = size * scale
    bg = (255, 255, 255)
    line = (20, 20, 20)
    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)

    margin = int(size_hi * 0.12)
    board_px = size_hi - 2 * margin
    cell = board_px / 3.0
    lw = max(2 * scale, size_hi // 80)

    for i in (1, 2):
        x = int(round(margin + i * cell))
        y = int(round(margin + i * cell))
        draw.line([(x, margin), (x, size_hi - margin)], fill=line, width=lw)
        draw.line([(margin, y), (size_hi - margin, y)], fill=line, width=lw)

    return img, draw, margin, cell, size_hi


def _render_checkerboard_base(size: int, scale: int):
    size_hi = size * scale
    bg = (246, 239, 229)
    frame = (84, 62, 44)
    light = (240, 217, 181)
    dark = (181, 136, 99)

    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)

    margin = int(size_hi * 0.11)
    board_px = size_hi - 2 * margin
    cell = board_px / 3.0
    frame_w = max(2 * scale, size_hi // 96)

    draw.rectangle(
        (margin - frame_w, margin - frame_w, size_hi - margin + frame_w, size_hi - margin + frame_w),
        fill=frame,
        outline=frame,
    )

    for r in range(3):
        for c in range(3):
            x0 = int(round(margin + c * cell))
            y0 = int(round(margin + r * cell))
            x1 = int(round(margin + (c + 1) * cell))
            y1 = int(round(margin + (r + 1) * cell))
            sq = light if ((r + c) % 2 == 0) else dark
            draw.rectangle((x0, y0, x1, y1), fill=sq, outline=sq)

    return img, draw, margin, cell, size_hi, light, dark


def _render_canonical_glyph(board: Sequence[Sequence[int]], size: int, glyph_x: str, glyph_o: str, render_scale: int = 3):
    size = max(32, int(size))
    scale = max(1, int(render_scale))
    img, draw, margin, cell, _size_hi = _render_grid_base(size, scale)

    pad = int(cell * 0.22)
    max_side = cell - 2 * pad
    font_start = int(max_side * 1.25)
    font_x = _fit_font(draw, glyph_x, max_side=max_side, start_size=font_start)
    font_o = _fit_font(draw, glyph_o, max_side=max_side, start_size=font_start)

    for r in range(3):
        for c in range(3):
            mark = int(board[r][c])
            if mark == 0:
                continue
            x0 = int(round(margin + c * cell)) + pad
            y0 = int(round(margin + r * cell)) + pad
            x1 = int(round(margin + (c + 1) * cell)) - pad
            y1 = int(round(margin + (r + 1) * cell)) - pad
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if mark == 1:
                _draw_centered_text(draw, cx, cy, glyph_x, font_x, (0, 0, 0))
            elif mark == 2:
                _draw_centered_text(draw, cx, cy, glyph_o, font_o, (0, 0, 0))

    if scale == 1:
        return img
    return img.resize((size, size), resample=_lanczos_resample())


def _render_checkerboard_glyph(
    board: Sequence[Sequence[int]],
    size: int,
    glyph_x: str,
    glyph_o: str,
    render_scale: int = 3,
):
    size = max(32, int(size))
    scale = max(1, int(render_scale))
    img, draw, margin, cell, _size_hi, _light_sq, _dark_sq = _render_checkerboard_base(size, scale)

    pad = int(cell * 0.20)
    max_side = cell - 2 * pad
    font_start = int(max_side * 1.25)
    font_x = _fit_font(draw, glyph_x, max_side=max_side, start_size=font_start)
    font_o = _fit_font(draw, glyph_o, max_side=max_side, start_size=font_start)

    for r in range(3):
        for c in range(3):
            mark = int(board[r][c])
            if mark == 0:
                continue
            x0 = int(round(margin + c * cell)) + pad
            y0 = int(round(margin + r * cell)) + pad
            x1 = int(round(margin + (c + 1) * cell)) - pad
            y1 = int(round(margin + (r + 1) * cell)) - pad
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            # Ensure strong contrast on both light/dark squares.
            fill = (24, 24, 24) if ((r + c) % 2 == 0) else (246, 246, 246)
            if mark == 1:
                _draw_centered_text(draw, cx, cy, glyph_x, font_x, fill)
            elif mark == 2:
                _draw_centered_text(draw, cx, cy, glyph_o, font_o, fill)

    if scale == 1:
        return img
    return img.resize((size, size), resample=_lanczos_resample())


def render(
    board: Sequence[Sequence[int]],
    style: str = "canonical",
    size: int = 384,
    meta: dict | None = None,
    render_scale: int = 3,
):
    meta = meta or {}
    if style == "canonical":
        return _render_canonical_glyph(board, size=size, glyph_x="X", glyph_o="O", render_scale=render_scale)
    if style == "checkerboard":
        return _render_checkerboard_glyph(board, size=size, glyph_x="X", glyph_o="O", render_scale=render_scale)
    if style == "glyph_random":
        gx = str(meta.get("glyph_p1", "A")).strip().upper()[:1] or "A"
        go = str(meta.get("glyph_p2", "B")).strip().upper()[:1] or "B"
        if gx == go:
            gx, go = "A", "B"
        return _render_canonical_glyph(board, size=size, glyph_x=gx, glyph_o=go, render_scale=render_scale)
    raise ValueError(f"Unsupported tictactoe style: {style}")
