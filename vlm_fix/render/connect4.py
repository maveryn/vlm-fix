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


def _dims(board: Sequence[Sequence[int]]) -> Tuple[int, int]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    if rows <= 0 or cols <= 0 or any(len(row) != cols for row in board):
        raise ValueError("connect4 renderer expects a non-empty rectangular board")
    return rows, cols


def _render_canonical(board: Sequence[Sequence[int]], size: int, render_scale: int = 3):
    rows, cols = _dims(board)
    size = max(96, int(size))
    scale = max(1, int(render_scale))
    size_hi = size * scale

    bg = (245, 247, 255)
    board_color = (44, 90, 201)
    empty_color = (235, 239, 255)
    p1_color = (218, 45, 54)
    p2_color = (245, 212, 65)
    edge = (42, 62, 122)

    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)
    margin = int(size_hi * 0.08)
    board_w = size_hi - 2 * margin
    board_h = int(round(board_w * rows / cols))
    top = (size_hi - board_h) // 2
    left = margin
    draw.rounded_rectangle(
        (left, top, left + board_w, top + board_h),
        radius=max(scale, size_hi // 24),
        fill=board_color,
    )

    cell_w = board_w / cols
    cell_h = board_h / rows
    pad = min(cell_w, cell_h) * 0.14
    disc_outline = max(1, scale)

    for r in range(rows):
        for c in range(cols):
            x0 = left + c * cell_w + pad
            y0 = top + r * cell_h + pad
            x1 = left + (c + 1) * cell_w - pad
            y1 = top + (r + 1) * cell_h - pad
            v = int(board[r][c])
            color = empty_color
            if v == 1:
                color = p1_color
            elif v == 2:
                color = p2_color
            draw.ellipse((x0, y0, x1, y1), fill=color, outline=edge, width=disc_outline)

    if scale == 1:
        return img
    return img.resize((size, size), resample=_lanczos_resample())


def _render_checkerboard(board: Sequence[Sequence[int]], size: int, render_scale: int = 3):
    rows, cols = _dims(board)
    size = max(96, int(size))
    scale = max(1, int(render_scale))
    size_hi = size * scale

    bg = (246, 239, 229)
    frame = (84, 62, 44)
    light = (240, 217, 181)
    dark = (181, 136, 99)
    p1_color = (218, 45, 54)
    p2_color = (245, 212, 65)
    empty_color = (242, 242, 242)
    disc_edge = (52, 52, 52)

    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)

    margin = int(size_hi * 0.08)
    board_w = size_hi - 2 * margin
    board_h = int(round(board_w * rows / cols))
    top = (size_hi - board_h) // 2
    left = margin
    frame_w = max(2 * scale, size_hi // 120)
    draw.rectangle(
        (left - frame_w, top - frame_w, left + board_w + frame_w, top + board_h + frame_w),
        fill=frame,
        outline=frame,
    )

    cell_w = board_w / cols
    cell_h = board_h / rows
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(left + c * cell_w))
            y0 = int(round(top + r * cell_h))
            x1 = int(round(left + (c + 1) * cell_w))
            y1 = int(round(top + (r + 1) * cell_h))
            sq = light if ((r + c) % 2 == 0) else dark
            draw.rectangle((x0, y0, x1, y1), fill=sq, outline=sq)

    pad = min(cell_w, cell_h) * 0.14
    disc_outline = max(1, scale)
    for r in range(rows):
        for c in range(cols):
            x0 = left + c * cell_w + pad
            y0 = top + r * cell_h + pad
            x1 = left + (c + 1) * cell_w - pad
            y1 = top + (r + 1) * cell_h - pad
            v = int(board[r][c])
            if v == 1:
                color = p1_color
            elif v == 2:
                color = p2_color
            else:
                color = empty_color
            draw.ellipse((x0, y0, x1, y1), fill=color, outline=disc_edge, width=disc_outline)

    if scale == 1:
        return img
    return img.resize((size, size), resample=_lanczos_resample())


def _render_grid_base(board: Sequence[Sequence[int]], size: int, render_scale: int = 3):
    rows, cols = _dims(board)
    size = max(96, int(size))
    scale = max(1, int(render_scale))
    size_hi = size * scale

    bg = (255, 255, 255)
    line = (28, 28, 28)
    img = Image.new("RGB", (size_hi, size_hi), bg)
    draw = ImageDraw.Draw(img)

    margin = int(size_hi * 0.10)
    board_w = size_hi - 2 * margin
    board_h = int(round(board_w * rows / cols))
    top = (size_hi - board_h) // 2
    left = margin
    draw.rectangle((left, top, left + board_w, top + board_h), fill=(255, 255, 255), outline=line, width=max(2, size_hi // 180))

    cell_w = board_w / cols
    cell_h = board_h / rows
    lw = max(2 * scale, size_hi // 220)
    for i in range(1, cols):
        x = int(round(left + i * cell_w))
        draw.line([(x, top), (x, top + board_h)], fill=line, width=lw)
    for i in range(1, rows):
        y = int(round(top + i * cell_h))
        draw.line([(left, y), (left + board_w, y)], fill=line, width=lw)

    return img, draw, left, top, board_w, board_h, cell_w, cell_h, scale


def _render_glyphs(board: Sequence[Sequence[int]], glyph_p1: str, glyph_p2: str, size: int, render_scale: int = 3):
    img, draw, left, top, _bw, _bh, cell_w, cell_h, scale = _render_grid_base(board, size=size, render_scale=render_scale)
    pad = min(cell_w, cell_h) * 0.20
    max_side = min(cell_w, cell_h) - 2 * pad
    font_start = int(max_side * 1.20)
    f1 = _fit_font(draw, glyph_p1, max_side=max_side, start_size=font_start)
    f2 = _fit_font(draw, glyph_p2, max_side=max_side, start_size=font_start)

    for r in range(len(board)):
        for c in range(len(board[0])):
            v = int(board[r][c])
            if v == 0:
                continue
            cx = left + (c + 0.5) * cell_w
            cy = top + (r + 0.5) * cell_h
            if v == 1:
                _draw_centered_text(draw, cx, cy, glyph_p1, f1, (0, 0, 0))
            else:
                _draw_centered_text(draw, cx, cy, glyph_p2, f2, (0, 0, 0))

    if scale == 1:
        return img
    return img.resize((size, size), resample=_lanczos_resample())


def render(board: Sequence[Sequence[int]], style: str = "canonical", size: int = 448, meta: dict | None = None, render_scale: int = 3):
    meta = meta or {}
    if style == "canonical":
        return _render_canonical(board, size=size, render_scale=render_scale)
    if style == "checkerboard":
        return _render_checkerboard(board, size=size, render_scale=render_scale)
    if style == "glyphs":
        g1 = str(meta.get("glyph_p1", "C")).strip().upper()[:1] or "C"
        g2 = str(meta.get("glyph_p2", "D")).strip().upper()[:1] or "D"
        if g1 == g2:
            g1, g2 = "C", "D"
        return _render_glyphs(board, glyph_p1=g1, glyph_p2=g2, size=size, render_scale=render_scale)
    raise ValueError(f"Unsupported connect4 style: {style}")
