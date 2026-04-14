#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "animals_synth_samples"


def _rand_color(palette: List[str], lo: float = 0.85, hi: float = 1.0) -> Tuple[int, int, int]:
    r, g, b = ImageColor.getrgb(random.choice(palette))
    s = random.uniform(lo, hi)
    return (int(r * s), int(g * s), int(b * s))


def _line(draw: ImageDraw.ImageDraw, pts, color, width):
    draw.line(pts, fill=color, width=width, joint="curve")


def _draw_leg(
    draw: ImageDraw.ImageDraw,
    hip: Tuple[float, float],
    scale: float,
    color: Tuple[int, int, int, int],
    width: int,
    style: str = "normal",
):
    # style: normal/bird/insect
    if style == "bird":
        a1 = random.uniform(math.radians(74), math.radians(110))
        l1 = random.uniform(40, 66) * scale
        a2 = a1 + random.uniform(math.radians(-18), math.radians(24))
        l2 = random.uniform(34, 58) * scale
    elif style == "insect":
        a1 = random.uniform(math.radians(55), math.radians(120))
        l1 = random.uniform(30, 54) * scale
        a2 = a1 + random.uniform(math.radians(-40), math.radians(40))
        l2 = random.uniform(28, 50) * scale
    else:
        a1 = random.uniform(math.radians(78), math.radians(104))
        l1 = random.uniform(46, 78) * scale
        a2 = a1 + random.uniform(math.radians(-22), math.radians(20))
        l2 = random.uniform(36, 64) * scale

    knee = (hip[0] + math.cos(a1) * l1, hip[1] + math.sin(a1) * l1)
    foot = (knee[0] + math.cos(a2) * l2, knee[1] + math.sin(a2) * l2)
    _line(draw, [hip, knee], color, width)
    _line(draw, [knee, foot], color, width)

    toe_len = random.uniform(8, 16) * scale
    for t in (-0.65, 0.0, 0.65):
        toe_a = a2 + t
        toe = (foot[0] + math.cos(toe_a) * toe_len, foot[1] + math.sin(toe_a) * toe_len)
        _line(draw, [foot, toe], color, max(1, width - 1))


def _draw_shadow(draw: ImageDraw.ImageDraw, cx: float, ground_y: int, w: float, h: float):
    draw.ellipse(
        [cx - w * 0.5, ground_y - h * 0.45, cx + w * 0.5, ground_y + h * 0.55],
        fill=(20, 20, 20, random.randint(18, 40)),
    )


def _draw_bird(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    scale: float,
    body_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    leg_count: int,
):
    body_w = random.uniform(160, 240) * scale
    body_h = random.uniform(110, 170) * scale
    wing_color = tuple(max(0, c - random.randint(14, 40)) for c in body_color)

    draw.ellipse(
        [cx - body_w * 0.5, cy - body_h * 0.5, cx + body_w * 0.5, cy + body_h * 0.5],
        fill=body_color + (255,),
        outline=(24, 24, 24, 240),
        width=max(2, int(2.0 * scale)),
    )
    draw.ellipse(
        [cx - body_w * 0.05, cy - body_h * 0.08, cx + body_w * 0.42, cy + body_h * 0.34],
        fill=wing_color + (245,),
    )

    hr = random.uniform(34, 54) * scale
    hx = cx + random.uniform(0.30, 0.50) * body_w
    hy = cy - random.uniform(0.10, 0.24) * body_h
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=body_color + (255,), outline=(30, 30, 30, 230), width=max(2, int(1.6 * scale)))
    beak = [(hx + hr, hy - hr * 0.22), (hx + hr + random.uniform(38, 72) * scale, hy), (hx + hr, hy + hr * 0.22)]
    draw.polygon(beak, fill=accent_color + (255,), outline=(40, 40, 40, 220))

    leg_color = (95, 72, 48, 255)
    hip_y = cy + body_h * random.uniform(0.24, 0.34)
    span = body_w * random.uniform(0.15, 0.36)
    if leg_count > 0:
        for i in range(leg_count):
            t = (i + 0.5) / leg_count
            hip_x = cx - span + 2 * span * t + random.uniform(-6, 6) * scale
            _draw_leg(draw, (hip_x, hip_y), scale, leg_color, max(2, int(2.3 * scale)), style="bird")


def _draw_quadruped(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    scale: float,
    body_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    leg_count: int,
) -> str:
    # Quadruped archetype variety.
    qstyle = random.choice(["canine", "feline", "deer", "boar"])

    if qstyle == "canine":
        body_w = random.uniform(220, 300) * scale
        body_h = random.uniform(105, 145) * scale
        head_scale = random.uniform(0.80, 0.95)
        muzzle_scale = random.uniform(1.00, 1.25)
    elif qstyle == "feline":
        body_w = random.uniform(210, 285) * scale
        body_h = random.uniform(95, 135) * scale
        head_scale = random.uniform(0.78, 0.90)
        muzzle_scale = random.uniform(0.80, 1.00)
    elif qstyle == "deer":
        body_w = random.uniform(230, 315) * scale
        body_h = random.uniform(98, 138) * scale
        head_scale = random.uniform(0.74, 0.86)
        muzzle_scale = random.uniform(1.10, 1.35)
    else:  # boar
        body_w = random.uniform(210, 300) * scale
        body_h = random.uniform(125, 170) * scale
        head_scale = random.uniform(0.88, 1.02)
        muzzle_scale = random.uniform(0.92, 1.10)

    body_bbox = [cx - body_w * 0.5, cy - body_h * 0.45, cx + body_w * 0.5, cy + body_h * 0.45]
    draw.rounded_rectangle(
        body_bbox,
        radius=int(random.uniform(22, 48) * scale),
        fill=body_color + (255,),
        outline=(28, 28, 28, 230),
        width=max(2, int(1.8 * scale)),
    )

    # Optional coat pattern.
    pattern = random.choice(["none", "spot", "stripe"])
    if pattern == "spot":
        for _ in range(random.randint(3, 8)):
            rx = random.uniform(body_bbox[0] + 0.08 * body_w, body_bbox[2] - 0.08 * body_w)
            ry = random.uniform(body_bbox[1] + 0.10 * body_h, body_bbox[3] - 0.10 * body_h)
            rrw = random.uniform(12, 28) * scale
            rrh = rrw * random.uniform(0.65, 1.15)
            c = tuple(max(0, x - random.randint(26, 55)) for x in body_color) + (200,)
            draw.ellipse([rx - rrw, ry - rrh, rx + rrw, ry + rrh], fill=c)
    elif pattern == "stripe":
        for _ in range(random.randint(3, 6)):
            sx = random.uniform(body_bbox[0] + 0.08 * body_w, body_bbox[2] - 0.18 * body_w)
            sy = random.uniform(body_bbox[1] + 0.08 * body_h, body_bbox[3] - 0.08 * body_h)
            ex = sx + random.uniform(12, 48) * scale
            ey = sy + random.uniform(-8, 8) * scale
            c = tuple(max(0, x - random.randint(22, 45)) for x in body_color) + (180,)
            _line(draw, [(sx, sy), (ex, ey)], c, max(1, int(2.0 * scale)))

    hr = random.uniform(38, 64) * scale * head_scale
    hx = cx + body_w * random.uniform(0.40, 0.57)
    hy = cy - body_h * random.uniform(0.06, 0.18)
    draw.ellipse(
        [hx - hr, hy - hr, hx + hr, hy + hr],
        fill=body_color + (255,),
        outline=(30, 30, 30, 220),
        width=max(2, int(1.4 * scale)),
    )

    # Ears / antlers by archetype.
    if qstyle in {"canine", "feline"}:
        ear_h = random.uniform(20, 42) * scale
        for s in (-1, 1):
            ex = hx + s * random.uniform(8, 22) * scale
            draw.polygon(
                [
                    (ex, hy - hr * 0.78),
                    (ex + s * random.uniform(10, 16) * scale, hy - hr - ear_h),
                    (ex + s * random.uniform(20, 30) * scale, hy - hr * 0.60),
                ],
                fill=body_color + (255,),
            )
    elif qstyle == "deer" and random.random() < 0.7:
        horn_color = (95, 80, 58, 235)
        for s in (-1, 1):
            bx = hx + s * random.uniform(8, 16) * scale
            by = hy - hr * random.uniform(0.78, 0.90)
            p1 = (bx + s * random.uniform(8, 14) * scale, by - random.uniform(24, 40) * scale)
            p2 = (p1[0] + s * random.uniform(10, 20) * scale, p1[1] - random.uniform(16, 34) * scale)
            _line(draw, [(bx, by), p1, p2], horn_color, max(1, int(1.8 * scale)))
            _line(
                draw,
                [p1, (p1[0] + s * random.uniform(10, 18) * scale, p1[1] + random.uniform(-2, 10) * scale)],
                horn_color,
                max(1, int(1.4 * scale)),
            )
    else:  # boar ears small
        for s in (-1, 1):
            ex = hx + s * random.uniform(10, 16) * scale
            ey = hy - hr * random.uniform(0.70, 0.82)
            draw.ellipse(
                [ex - random.uniform(8, 14) * scale, ey - random.uniform(10, 16) * scale,
                 ex + random.uniform(8, 14) * scale, ey + random.uniform(10, 16) * scale],
                fill=body_color + (250,),
            )

    # Muzzle / snout
    nose_w = hr * random.uniform(0.22, 0.32) * muzzle_scale
    nose_h = hr * random.uniform(0.18, 0.28)
    nose = [hx + hr * 0.70, hy - nose_h * 0.5, hx + hr * 0.70 + nose_w, hy + nose_h * 0.5]
    draw.ellipse(nose, fill=accent_color + (245,))

    if qstyle == "boar" and random.random() < 0.6:
        tusk_col = (240, 230, 210, 240)
        for t in (-1, 1):
            tx = nose[0] + random.uniform(1, 5) * scale
            ty = hy + t * random.uniform(4, 8) * scale
            _line(
                draw,
                [(tx, ty), (tx + random.uniform(8, 16) * scale, ty + t * random.uniform(3, 8) * scale)],
                tusk_col,
                max(1, int(1.5 * scale)),
            )

    # Tail with style differences.
    tail_base = (cx - body_w * 0.48, cy - body_h * 0.05)
    tail_tip = (tail_base[0] - random.uniform(64, 120) * scale, tail_base[1] - random.uniform(8, 46) * scale)
    tail_w = max(2, int((1.6 if qstyle in {"deer", "feline"} else 2.4) * scale))
    _line(draw, [tail_base, tail_tip], (65, 52, 40, 230), tail_w)
    if qstyle in {"feline", "deer"} and random.random() < 0.7:
        tuft_r = random.uniform(6, 12) * scale
        draw.ellipse([tail_tip[0] - tuft_r, tail_tip[1] - tuft_r, tail_tip[0] + tuft_r, tail_tip[1] + tuft_r], fill=(55, 42, 32, 220))

    leg_color = (90, 70, 46, 255)
    hip_y = cy + body_h * random.uniform(0.27, 0.38)
    span = body_w * random.uniform(0.24, 0.43)
    if leg_count > 0:
        for i in range(leg_count):
            t = (i + 0.5) / leg_count
            hip_x = cx - span + 2 * span * t + random.uniform(-6, 6) * scale
            _draw_leg(draw, (hip_x, hip_y), scale, leg_color, max(2, int(2.6 * scale)), style="normal")

    return qstyle


def _draw_lizard(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    scale: float,
    body_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    leg_count: int,
):
    body_w = random.uniform(200, 300) * scale
    body_h = random.uniform(72, 112) * scale
    draw.ellipse(
        [cx - body_w * 0.5, cy - body_h * 0.5, cx + body_w * 0.5, cy + body_h * 0.5],
        fill=body_color + (255,),
        outline=(25, 25, 25, 220),
        width=max(2, int(1.6 * scale)),
    )
    hr = random.uniform(26, 42) * scale
    hx = cx + body_w * random.uniform(0.47, 0.60)
    hy = cy - body_h * random.uniform(0.0, 0.10)
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=body_color + (255,), outline=(24, 24, 24, 220), width=max(2, int(1.3 * scale)))
    tongue = [(hx + hr, hy), (hx + hr + random.uniform(36, 62) * scale, hy - random.uniform(4, 10) * scale), (hx + hr + random.uniform(30, 56) * scale, hy + random.uniform(4, 10) * scale)]
    draw.polygon(tongue, fill=accent_color + (230,))

    # Long curved tail
    p0 = (cx - body_w * 0.45, cy - body_h * 0.02)
    p1 = (p0[0] - random.uniform(70, 110) * scale, p0[1] - random.uniform(24, 46) * scale)
    p2 = (p1[0] - random.uniform(90, 145) * scale, p1[1] + random.uniform(12, 40) * scale)
    _line(draw, [p0, p1, p2], (70, 55, 40, 230), max(2, int(2.0 * scale)))

    leg_color = (92, 74, 51, 255)
    hip_y = cy + body_h * random.uniform(0.16, 0.30)
    span = body_w * random.uniform(0.20, 0.42)
    if leg_count > 0:
        for i in range(leg_count):
            t = (i + 0.5) / leg_count
            hip_x = cx - span + 2 * span * t + random.uniform(-5, 5) * scale
            _draw_leg(draw, (hip_x, hip_y), scale, leg_color, max(2, int(2.0 * scale)), style="normal")


def _draw_insect_or_spider(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    scale: float,
    body_color: Tuple[int, int, int],
    species: str,
    leg_count: int,
):
    # Insect: 2-segment body, Spider: 2-round body with thicker front segment.
    if species == "spider":
        r1 = random.uniform(44, 64) * scale
        r2 = random.uniform(32, 48) * scale
        draw.ellipse([cx - r1, cy - r1 * 0.75, cx + r1, cy + r1 * 0.75], fill=body_color + (255,), outline=(20, 20, 20, 220), width=max(2, int(1.5 * scale)))
        draw.ellipse([cx + r1 * 0.55 - r2, cy - r2 * 0.72, cx + r1 * 0.55 + r2, cy + r2 * 0.72], fill=body_color + (255,), outline=(20, 20, 20, 220), width=max(2, int(1.4 * scale)))
        body_w = r1 * 2.0
    else:
        rw = random.uniform(180, 250) * scale
        rh = random.uniform(72, 104) * scale
        draw.ellipse([cx - rw * 0.50, cy - rh * 0.50, cx + rw * 0.10, cy + rh * 0.50], fill=body_color + (255,), outline=(20, 20, 20, 220), width=max(2, int(1.4 * scale)))
        draw.ellipse([cx - rw * 0.05, cy - rh * 0.38, cx + rw * 0.40, cy + rh * 0.38], fill=body_color + (255,), outline=(20, 20, 20, 220), width=max(2, int(1.4 * scale)))
        body_w = rw * 0.95

    leg_color = (70, 56, 40, 255)
    hip_y = cy + random.uniform(-6, 8) * scale
    span = body_w * random.uniform(0.30, 0.52)
    if leg_count > 0:
        for i in range(leg_count):
            t = (i + 0.5) / leg_count
            hip_x = cx - span + 2 * span * t + random.uniform(-4, 4) * scale
            _draw_leg(draw, (hip_x, hip_y), scale, leg_color, max(2, int(1.9 * scale)), style="insect")


def make_one(idx: int, size: int = 640, aa: int = 3) -> dict:
    W = H = size * aa
    img = Image.new("RGB", (W, H), (244, 244, 244))
    draw = ImageDraw.Draw(img, "RGBA")

    sky_pal = ["#f6f7fb", "#e9f2ff", "#f5efe4", "#eef7ef", "#f8f1fa"]
    ground_pal = ["#dde7d5", "#d8dfcf", "#e7d9c7", "#d9d9d9", "#d4e2d8"]
    animal_pal = ["#355c7d", "#6c5b7b", "#2a9d8f", "#264653", "#7f5539", "#6a994e", "#8d99ae"]
    accent_pal = ["#f4a261", "#e76f51", "#ffcc66", "#ffd166", "#f7b267", "#ee6c4d"]

    species = "quadruped"
    # User-requested distribution: animals sampled uniformly from 3..5 legs.
    leg_count = random.randint(3, 5)

    sky = _rand_color(sky_pal)
    ground = _rand_color(ground_pal, 0.92, 1.04)
    draw.rectangle([0, 0, W, H], fill=sky)
    horizon = int(random.uniform(0.58, 0.72) * H)
    draw.rectangle([0, horizon, W, H], fill=ground)

    body_color = _rand_color(animal_pal, 0.86, 1.06)
    accent = _rand_color(accent_pal, 0.90, 1.05)
    scale = random.uniform(0.86, 1.16) * aa
    cx = random.uniform(0.30, 0.70) * W
    cy = random.uniform(0.44, 0.60) * H

    _draw_shadow(draw, cx, horizon, random.uniform(160, 280) * scale, random.uniform(16, 44) * scale)

    qstyle = _draw_quadruped(draw, cx, cy, scale, body_color, accent, leg_count)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.45 * aa))
    out = img.resize((size, size), Image.Resampling.LANCZOS)

    strip_h = 28
    labeled = Image.new("RGB", (size, size + strip_h), (248, 248, 248))
    labeled.paste(out, (0, 0))
    d2 = ImageDraw.Draw(labeled)
    label = f"{species}:{qstyle} | legs={leg_count}"
    d2.text((8, size + 6), label, fill=(40, 40, 40), font=ImageFont.load_default())

    fname = f"animal_{idx:03d}_{qstyle}_legs{leg_count}.png"
    out_path = OUT_DIR / fname
    labeled.save(out_path, quality=95)
    return {"file": fname, "species": species, "style": qstyle, "legs": leg_count}


def make_contact_sheet(paths: List[Path], out_path: Path, cols: int = 5):
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
    random.seed(123)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old sample outputs in this folder before regenerating.
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()

    n = 36
    rows = []
    paths = []
    for i in range(n):
        row = make_one(i)
        rows.append(row)
        paths.append(OUT_DIR / row["file"])

    (OUT_DIR / "metadata.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    make_contact_sheet(paths, OUT_DIR / "contact_sheet.png", cols=6)
    print(f"[ok] wrote {n} synthetic animal samples to {OUT_DIR}")
    print(f"[ok] contact sheet: {OUT_DIR / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
