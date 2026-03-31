#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_vlm_fix_text_only_open10.sh
# Optional env:
#   NO_SKIP_EXISTING=1   # rerun even if jsonl exists
#   BATCH_SIZE=128       # for non-32B models
#   BATCH_SIZE_32B=32    # for Qwen3-VL-32B
#   TP_SIZE_32B=2        # tensor parallel size for Qwen3-VL-32B (set 4 if needed)
#   MAX_NEW_TOKENS=1024

if [[ "${NO_SKIP_EXISTING:-0}" == "1" ]]; then
  SKIP_FLAG="--no-skip-existing"
else
  SKIP_FLAG="--skip-existing"
fi

# 1) Run 9 open models (excluding 32B) on single GPU by default.
python scripts/run_vlm_fix_text_only_matrix.py \
  --dataset-dir data/generated/vlm_fix_text_only \
  --runs-dir runs/vlm_fix_text_only \
  --results-dir results/vlm_fix_text_only \
  --models \
    Qwen/Qwen2.5-VL-3B-Instruct \
    Qwen/Qwen2.5-VL-7B-Instruct \
    Qwen/Qwen3-VL-4B-Instruct \
    Qwen/Qwen3-VL-8B-Instruct \
    OpenGVLab/InternVL3_5-4B \
    OpenGVLab/InternVL3_5-8B \
    OpenGVLab/InternVL3_5-14B \
    allenai/Molmo2-4B \
    allenai/Molmo2-8B \
  --games tictactoe reversi connect4 dots_boxes \
  --batch-size "${BATCH_SIZE:-128}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --run-tag prompt-direct \
  ${SKIP_FLAG}

# 2) Run 32B separately with tensor parallel (multi-GPU) to avoid OOM.
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
FIXATION_TP_SIZE="${TP_SIZE_32B:-2}" \
python scripts/run_vlm_fix_text_only_matrix.py \
  --dataset-dir data/generated/vlm_fix_text_only \
  --runs-dir runs/vlm_fix_text_only \
  --results-dir results/vlm_fix_text_only \
  --models Qwen/Qwen3-VL-32B-Instruct \
  --games tictactoe reversi connect4 dots_boxes \
  --batch-size "${BATCH_SIZE_32B:-32}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --run-tag prompt-direct \
  ${SKIP_FLAG}

python scripts/build_vlm_fix_text_only_reports.py \
  --runs-dir runs/vlm_fix_text_only \
  --results-dir results/vlm_fix_text_only \
  --run-tag prompt-direct
