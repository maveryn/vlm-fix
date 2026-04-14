#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_fix.generation import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build vlm_fix dataset (4 games, 6 benchmark variants).")
    parser.add_argument("--out-dir", type=str, default="data/generated/vlm_fix")
    parser.add_argument("--states-per-game", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--render-variants", nargs="+", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = build_dataset(
        out_dir=out_dir,
        states_per_game=args.states_per_game,
        seed=args.seed,
        games=args.games,
        render_variants=args.render_variants,
    )
    print(f"[ok] wrote: {out_parquet}")
    print(f"[ok] summary: {out_dir / 'summary.json'}")
    print(f"[ok] counts: {out_dir / 'counts_by_game_render_rule.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
