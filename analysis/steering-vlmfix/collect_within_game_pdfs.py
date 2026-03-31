#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import re

ROOT = Path('analysis/steering-vlmfix/outputs')
DEST = ROOT / 'within_game_pdf_gallery_16'
DEST.mkdir(parents=True, exist_ok=True)

# Remove existing PDFs in gallery so reruns stay clean.
for p in DEST.glob('*.pdf'):
    p.unlink()

copied = []

run_dirs = sorted(
    [d for d in ROOT.glob('within_game_*') if d.is_dir()]
    + [d for d in (ROOT / 'transfer_matrix_cached').glob('within_game_*') if d.is_dir()]
)
for rd in run_dirs:
    # Example: within_game_qwen25vl7b_spg100_rep3_last12_alpha1
    model_part = rd.name.replace('within_game_', '')
    # Keep only model tag before first known suffix segment.
    # This preserves names like qwen25vl7b, qwen25vl3b, molmo2_8b, internvl35_8b.
    model_tag = re.split(r'_spg\d+_|_states\d+_|_rep\d+_', model_part)[0]

    for game in ['tictactoe', 'reversi', 'connect4', 'dots_boxes']:
        src = rd / f'within_{game}.pdf'
        if not src.exists():
            continue
        dst = DEST / f'{model_tag}__{game}.pdf'
        shutil.copy2(src, dst)
        copied.append(dst)

print(f'gallery: {DEST}')
print(f'copied: {len(copied)}')
for p in sorted(copied):
    print(p)
