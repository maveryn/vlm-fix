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


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_side: float, start_size: int):
    size = max(8, int(start_size))
    while size >= 8:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
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
        draw.text((cx - w / 2.0, cy - h / 2.0), text, font=font, fill=fill)


def _to_rgb(v, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(v, (tuple, list)) and len(v) == 3:
        return (int(v[0]), int(v[1]), int(v[2]))
    return default


def _draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, r_outer: float, fill, outline, width: int):
    r_inner = r_outer * 0.45
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * (math.pi / 5)
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    draw.polygon(pts, fill=fill, outline=outline)
    for i in range(10):
        draw.line([pts[i], pts[(i + 1) % 10]], fill=outline, width=width)


def _draw_triangle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int):
    pts = [
        (cx, cy - r),
        (cx - r * 0.92, cy + r * 0.82),
        (cx + r * 0.92, cy + r * 0.82),
    ]
    draw.polygon(pts, fill=fill, outline=outline)
    for i in range(3):
        draw.line([pts[i], pts[(i + 1) % 3]], fill=outline, width=width)


def _draw_square(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int):
    draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def _draw_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def _draw_crescent(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill, outline, width: int, bg):
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


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: float, cy: float, r: float, fill, outline, width: int, bg):
    s = str(shape).strip().lower()
    if s == "triangle":
        _draw_triangle(draw, cx, cy, r, fill=fill, outline=outline, width=width)
        return
    if s == "star":
        _draw_star(draw, cx, cy, r, fill=fill, outline=outline, width=width)
        return
    if s == "circle":
        _draw_circle(draw, cx, cy, r * 0.98, fill=fill, outline=outline, width=width)
        return
    if s == "square":
        _draw_square(draw, cx, cy, r * 0.95, fill=fill, outline=outline, width=width)
        return
    if s in {"crescent", "moon"}:
        _draw_crescent(draw, cx, cy, r * 0.98, fill=fill, outline=outline, width=width, bg=bg)
        return
    _draw_circle(draw, cx, cy, r * 0.98, fill=fill, outline=outline, width=width)


def _base(board: Sequence[Sequence[int]], size_px: int, render_scale: int = 3):
    n = len(board)
    if n <= 0 or any(len(row) != n for row in board):
        raise ValueError("dots_boxes renderer expects a non-empty square board")

    size_px = max(96, int(size_px))
    scale = max(1, int(render_scale))
    size_hi = size_px * scale

    bg = (255, 255, 255)
    grid_line = (145, 145, 145)
    dot_color = (18, 18, 18)

    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)

    margin = int(size_hi * 0.12)
    board_px = size_hi - 2 * margin
    cell = board_px / n
    lw = max(3, size_hi // 230)

    for i in range(n + 1):
        x = int(round(margin + i * cell))
        y = int(round(margin + i * cell))
        draw.line([(x, margin), (x, size_hi - margin)], fill=grid_line, width=lw)
        draw.line([(margin, y), (size_hi - margin, y)], fill=grid_line, width=lw)

    dot_r = max(2, int(round(cell * 0.07)))
    for r in range(n + 1):
        for c in range(n + 1):
            x = margin + c * cell
            y = margin + r * cell
            draw.ellipse((x - dot_r, y - dot_r, x + dot_r, y + dot_r), fill=dot_color)

    return img, draw, n, margin, cell, scale


def _base_checkerboard(board: Sequence[Sequence[int]], size_px: int, render_scale: int = 3):
    n = len(board)
    if n <= 0 or any(len(row) != n for row in board):
        raise ValueError("dots_boxes renderer expects a non-empty square board")

    size_px = max(96, int(size_px))
    scale = max(1, int(render_scale))
    size_hi = size_px * scale

    bg = (246, 239, 229)
    frame = (84, 62, 44)
    light = (240, 217, 181)
    dark = (181, 136, 99)
    grid_line = (92, 72, 52)
    dot_color = (18, 18, 18)

    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)

    margin = int(size_hi * 0.12)
    board_px = size_hi - 2 * margin
    cell = board_px / n
    frame_w = max(2 * scale, size_hi // 200)
    draw.rectangle(
        (margin - frame_w, margin - frame_w, size_hi - margin + frame_w, size_hi - margin + frame_w),
        fill=frame,
        outline=frame,
    )

    for r in range(n):
        for c in range(n):
            x0 = int(round(margin + c * cell))
            y0 = int(round(margin + r * cell))
            x1 = int(round(margin + (c + 1) * cell))
            y1 = int(round(margin + (r + 1) * cell))
            sq = light if ((r + c) % 2 == 0) else dark
            draw.rectangle((x0, y0, x1, y1), fill=sq, outline=sq)

    lw = max(3, size_hi // 230)
    for i in range(n + 1):
        x = int(round(margin + i * cell))
        y = int(round(margin + i * cell))
        draw.line([(x, margin), (x, size_hi - margin)], fill=grid_line, width=lw)
        draw.line([(margin, y), (size_hi - margin, y)], fill=grid_line, width=lw)

    dot_r = max(2, int(round(cell * 0.07)))
    for r in range(n + 1):
        for c in range(n + 1):
            x = margin + c * cell
            y = margin + r * cell
            draw.ellipse((x - dot_r, y - dot_r, x + dot_r, y + dot_r), fill=dot_color)

    return img, draw, n, margin, cell, scale


def _render_letters(board: Sequence[Sequence[int]], text_p1: str, text_p2: str, size_px: int, render_scale: int = 3):
    img, draw, n, margin, cell, scale = _base(board, size_px=size_px, render_scale=render_scale)

    max_side = cell * 0.46
    font = _fit_font(draw, text_p1, max_side=max_side, start_size=int(max_side * 1.6))
    for r in range(n):
        for c in range(n):
            mark = int(board[r][c])
            text = text_p1 if mark == 1 else text_p2
            cx = margin + (c + 0.5) * cell
            cy = margin + (r + 0.5) * cell
            _draw_centered_text(draw, cx, cy, text, font, (0, 0, 0))

    if scale == 1:
        return img
    return img.resize((size_px, size_px), resample=_lanczos_resample())


def _render_letters_checkerboard(board: Sequence[Sequence[int]], text_p1: str, text_p2: str, size_px: int, render_scale: int = 3):
    img, draw, n, margin, cell, scale = _base_checkerboard(board, size_px=size_px, render_scale=render_scale)

    max_side = cell * 0.46
    font = _fit_font(draw, text_p1, max_side=max_side, start_size=int(max_side * 1.6))
    for r in range(n):
        for c in range(n):
            mark = int(board[r][c])
            text = text_p1 if mark == 1 else text_p2
            cx = margin + (c + 0.5) * cell
            cy = margin + (r + 0.5) * cell
            _draw_centered_text(draw, cx, cy, text, font, (0, 0, 0))

    if scale == 1:
        return img
    return img.resize((size_px, size_px), resample=_lanczos_resample())


def render(board: Sequence[Sequence[int]], style: str = "canonical", size_px: int = 480, meta: dict | None = None, render_scale: int = 3):
    meta = meta or {}
    if style == "canonical":
        return _render_letters(board, text_p1="A", text_p2="B", size_px=size_px, render_scale=render_scale)
    if style == "checkerboard":
        return _render_letters_checkerboard(board, text_p1="A", text_p2="B", size_px=size_px, render_scale=render_scale)
    if style == "glyphs":
        g1 = str(meta.get("glyph_p1", "C")).strip().upper()[:1] or "C"
        g2 = str(meta.get("glyph_p2", "D")).strip().upper()[:1] or "D"
        if g1 == g2:
            g1, g2 = "C", "D"
        return _render_letters(board, text_p1=g1, text_p2=g2, size_px=size_px, render_scale=render_scale)
    raise ValueError(f"Unsupported dots_boxes style: {style}")
