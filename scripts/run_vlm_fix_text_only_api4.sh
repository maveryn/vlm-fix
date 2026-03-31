#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_vlm_fix_text_only_api4.sh
# Optional env:
#   NO_SKIP_EXISTING=1   # rerun even if jsonl exists
#   BATCH_SIZE=8         # API concurrency (not true vLLM batching)
#   MAX_NEW_TOKENS=1024

if [[ "${NO_SKIP_EXISTING:-0}" == "1" ]]; then
  SKIP_FLAG="--no-skip-existing"
else
  SKIP_FLAG="--skip-existing"
fi

python scripts/run_vlm_fix_text_only_matrix.py \
  --dataset-dir data/generated/vlm_fix_text_only \
  --runs-dir runs/vlm_fix_text_only \
  --results-dir results/vlm_fix_text_only \
  --models \
    gpt-4.1 \
    gpt-5.2 \
    claude-sonnet-4-0 \
    claude-sonnet-4-5 \
  --games tictactoe reversi connect4 dots_boxes \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --run-tag prompt-direct \
  ${SKIP_FLAG}

python scripts/build_vlm_fix_text_only_reports.py \
  --runs-dir runs/vlm_fix_text_only \
  --results-dir results/vlm_fix_text_only \
  --run-tag prompt-direct

