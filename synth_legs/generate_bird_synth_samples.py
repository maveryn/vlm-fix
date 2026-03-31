#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "bird_synth_samples"


def _rand_color(palette: List[str], lo: float = 0.85, hi: float = 1.0) -> Tuple[int, int, int]:
    r, g, b = ImageColor.getrgb(random.choice(palette))
    s = random.uniform(lo, hi)
    return (int(r * s), int(g * s), int(b * s))


def _line(draw: ImageDraw.ImageDraw, pts, color, width):
    draw.line(pts, fill=color, width=width, joint="curve")


def _draw_leg(draw: ImageDraw.ImageDraw, hip: Tuple[float, float], scale: float, color, width: int, style: str):
    if style == "heron":
        a1 = random.uniform(math.radians(68), math.radians(95))
        l1 = random.uniform(62, 96) * scale
        a2 = a1 + random.uniform(math.radians(-24), math.radians(20))
        l2 = random.uniform(58, 90) * scale
    elif style == "owl":
        a1 = random.uniform(math.radians(78), math.radians(110))
        l1 = random.uniform(34, 56) * scale
        a2 = a1 + random.uniform(math.radians(-18), math.radians(28))
        l2 = random.uniform(28, 46) * scale
    else:
        a1 = random.uniform(math.radians(72), math.radians(106))
        l1 = random.uniform(44, 74) * scale
        a2 = a1 + random.uniform(math.radians(-20), math.radians(30))
        l2 = random.uniform(36, 64) * scale

    knee = (hip[0] + math.cos(a1) * l1, hip[1] + math.sin(a1) * l1)
    foot = (knee[0] + math.cos(a2) * l2, knee[1] + math.sin(a2) * l2)
    _line(draw, [hip, knee], color, width)
    _line(draw, [knee, foot], color, width)

    toe_len = random.uniform(8, 16) * scale
    for t in (-0.7, 0.0, 0.7):
        toe_a = a2 + t
        toe = (foot[0] + math.cos(toe_a) * toe_len, foot[1] + math.sin(toe_a) * toe_len)
        _line(draw, [foot, toe], color, max(1, width - 1))


def _draw_bird(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    scale: float,
    body_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    leg_count: int,
    style: str,
):
    # style presets provide variety comparable to quadruped generator.
    if style == "owl":
        body_w = random.uniform(175, 235) * scale
        body_h = random.uniform(150, 220) * scale
        head_r = random.uniform(44, 66) * scale
        beak_len = random.uniform(28, 48) * scale
        tail_len = random.uniform(38, 68) * scale
    elif style == "heron":
        body_w = random.uniform(190, 260) * scale
        body_h = random.uniform(95, 140) * scale
        head_r = random.uniform(34, 52) * scale
        beak_len = random.uniform(74, 120) * scale
        tail_len = random.uniform(60, 108) * scale
    elif style == "duck":
        body_w = random.uniform(205, 285) * scale
        body_h = random.uniform(105, 158) * scale
        head_r = random.uniform(36, 54) * scale
        beak_len = random.uniform(56, 92) * scale
        tail_len = random.uniform(48, 80) * scale
    else:  # passerine
        body_w = random.uniform(170, 250) * scale
        body_h = random.uniform(110, 175) * scale
        head_r = random.uniform(36, 56) * scale
        beak_len = random.uniform(42, 84) * scale
        tail_len = random.uniform(52, 94) * scale

    # Body
    body_bbox = [cx - body_w * 0.5, cy - body_h * 0.5, cx + body_w * 0.5, cy + body_h * 0.5]
    draw.ellipse(
        body_bbox,
        fill=body_color + (255,),
        outline=(24, 24, 24, 235),
        width=max(2, int(1.8 * scale)),
    )

    # Optional coat pattern (similar idea as quadruped samples).
    pattern = random.choice(["none", "spot", "stripe"])
    if pattern == "spot":
        for _ in range(random.randint(3, 7)):
            rx = random.uniform(body_bbox[0] + 0.08 * body_w, body_bbox[2] - 0.08 * body_w)
            ry = random.uniform(body_bbox[1] + 0.10 * body_h, body_bbox[3] - 0.10 * body_h)
            rrw = random.uniform(10, 24) * scale
            rrh = rrw * random.uniform(0.6, 1.2)
            c = tuple(max(0, x - random.randint(20, 52)) for x in body_color) + (190,)
            draw.ellipse([rx - rrw, ry - rrh, rx + rrw, ry + rrh], fill=c)
    elif pattern == "stripe":
        for _ in range(random.randint(3, 6)):
            sx = random.uniform(body_bbox[0] + 0.10 * body_w, body_bbox[2] - 0.16 * body_w)
            sy = random.uniform(body_bbox[1] + 0.14 * body_h, body_bbox[3] - 0.14 * body_h)
            ex = sx + random.uniform(14, 44) * scale
            ey = sy + random.uniform(-10, 10) * scale
            c = tuple(max(0, x - random.randint(16, 40)) for x in body_color) + (170,)
            _line(draw, [(sx, sy), (ex, ey)], c, max(1, int(1.8 * scale)))

    # Wing (vary pose)
    wing_color = tuple(max(0, c - random.randint(14, 46)) for c in body_color)
    wx = cx + random.uniform(-0.10, 0.16) * body_w
    wy = cy + random.uniform(-0.08, 0.12) * body_h
    ww = body_w * random.uniform(0.38, 0.62)
    wh = body_h * random.uniform(0.34, 0.58)
    wing_mode = random.choice(["folded", "raised", "low"])
    if wing_mode == "raised":
        wy -= 0.12 * body_h
    elif wing_mode == "low":
        wy += 0.10 * body_h
    draw.ellipse([wx - ww * 0.5, wy - wh * 0.5, wx + ww * 0.5, wy + wh * 0.5], fill=wing_color + (240,))

    # Head + beak + eye
    hx = cx + random.uniform(0.30, 0.52) * body_w
    hy = cy - random.uniform(0.10, 0.30) * body_h
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=body_color + (255,), outline=(24, 24, 24, 235), width=max(2, int(1.6 * scale)))

    # Crest variety.
    if random.random() < 0.45:
        crest_h = random.uniform(16, 34) * scale
        crest_w = random.uniform(8, 18) * scale
        cx0 = hx + random.uniform(-0.10, 0.08) * head_r
        cy0 = hy - head_r * random.uniform(0.70, 0.90)
        draw.polygon(
            [(cx0 - crest_w, cy0), (cx0, cy0 - crest_h), (cx0 + crest_w, cy0)],
            fill=tuple(max(0, c - random.randint(12, 40)) for c in body_color) + (230,),
        )

    beak_h = random.uniform(16, 34) * scale
    bx = hx + head_r
    by = hy + random.uniform(-4, 7) * scale
    beak = [(bx, by - beak_h * 0.5), (bx + beak_len, by), (bx, by + beak_h * 0.5)]
    draw.polygon(beak, fill=accent_color + (255,), outline=(40, 40, 40, 220))

    eye_r = random.uniform(4, 9) * scale
    ex = hx + random.uniform(0.10, 0.28) * head_r
    ey = hy - random.uniform(0.08, 0.24) * head_r
    draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(22, 22, 22, 255))

    # Tail
    tx = cx - body_w * 0.48
    ty = cy - random.uniform(0.02, 0.14) * body_h
    t2 = (tx - tail_len, ty - random.uniform(16, 44) * scale)
    t3 = (tx - tail_len, ty + random.uniform(16, 44) * scale)
    draw.polygon([(tx, ty), t2, t3], fill=wing_color + (230,), outline=(24, 24, 24, 175))

    # Legs
    leg_color = (90, 70, 45, 255)
    leg_width = int(random.uniform(1.6, 2.8) * scale)
    hip_y = cy + body_h * random.uniform(0.24, 0.36)
    if leg_count > 0:
        span = body_w * random.uniform(0.14, 0.40)
        for i in range(leg_count):
            t = (i + 0.5) / leg_count
            hip_x = cx - span + 2 * span * t + random.uniform(-6, 6) * scale
            _draw_leg(draw, (hip_x, hip_y), scale, leg_color, leg_width, style=style)


def make_one(idx: int, size: int = 640, aa: int = 3) -> dict:
    W = H = size * aa
    img = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(img, "RGBA")

    sky_pal = ["#f6f7fb", "#e9f2ff", "#f5efe4", "#eef7ef", "#f8f1fa", "#e8eef8", "#f3efe9"]
    ground_pal = ["#dde7d5", "#d8dfcf", "#e7d9c7", "#d9d9d9", "#d4e2d8", "#d7dfd4", "#d9d4c8"]
    bird_pal = ["#355c7d", "#6c5b7b", "#2a9d8f", "#264653", "#7f5539", "#6a994e", "#8d99ae", "#5e548e"]
    accent_pal = ["#f4a261", "#e76f51", "#ffcc66", "#ffd166", "#f7b267", "#ee6c4d"]

    sky = _rand_color(sky_pal)
    ground = _rand_color(ground_pal, 0.92, 1.04)
    draw.rectangle([0, 0, W, H], fill=sky)
    horizon = int(random.uniform(0.58, 0.72) * H)
    draw.rectangle([0, horizon, W, H], fill=ground)

    # Background consistency with animal generator: clean, minimal clutter.
    # Add only a soft shadow.
    style = random.choice(["passerine", "owl", "heron", "duck"])
    # User-requested distribution: birds sampled uniformly from 1..3 legs.
    leg_count = random.randint(1, 3)

    scale = random.uniform(0.88, 1.20) * aa
    cx = random.uniform(0.30, 0.70) * W
    cy = random.uniform(0.44, 0.60) * H
    shadow_w = random.uniform(160, 280) * scale
    shadow_h = random.uniform(14, 38) * scale
    draw.ellipse([cx - shadow_w * 0.5, horizon - shadow_h * 0.45, cx + shadow_w * 0.5, horizon + shadow_h * 0.55], fill=(20, 20, 20, random.randint(18, 40)))

    body_color = _rand_color(bird_pal, 0.86, 1.06)
    accent_color = _rand_color(accent_pal, 0.90, 1.05)
    _draw_bird(draw, cx, cy, scale, body_color, accent_color, leg_count, style=style)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.45 * aa))
    out = img.resize((size, size), Image.Resampling.LANCZOS)

    strip_h = 28
    labeled = Image.new("RGB", (size, size + strip_h), (248, 248, 248))
    labeled.paste(out, (0, 0))
    d2 = ImageDraw.Draw(labeled)
    label = f"bird:{style} | legs={leg_count}"
    d2.text((8, size + 6), label, fill=(40, 40, 40), font=ImageFont.load_default())

    fname = f"bird_{idx:03d}_{style}_legs{leg_count}.png"
    out_path = OUT_DIR / fname
    labeled.save(out_path, quality=95)
    return {"file": fname, "style": style, "legs": leg_count}


def make_contact_sheet(paths: List[Path], out_path: Path, cols: int = 6):
    if not paths:
        return
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = ims[0].size
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(ims):
        r = i // cols
        c = i % cols
        sheet.paste(im, (c * w, r * h))
    sheet.save(out_path, quality=95)


def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old files in this sample folder before regenerating.
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()

    n = 36
    meta = []
    out_paths: List[Path] = []
    for i in range(n):
        row = make_one(i)
        meta.append(row)
        out_paths.append(OUT_DIR / row["file"])

    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    make_contact_sheet(out_paths, OUT_DIR / "contact_sheet.png", cols=6)
    print(f"[ok] wrote {n} bird samples to {OUT_DIR}")
    print(f"[ok] contact sheet: {OUT_DIR / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
