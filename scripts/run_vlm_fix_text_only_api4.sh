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

DATASET_SOURCE="${DATASET_SOURCE:-local}"
DATASET_DIR="${DATASET_DIR:-data/generated/vlm_fix_text_only}"
HF_DATASET_REPO="${HF_DATASET_REPO:-}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-vlm_fix_text_only}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-main}"
HF_CACHE_DIR="${HF_CACHE_DIR:-data/hf_cache}"

DATASET_ARGS=(--dataset-source "${DATASET_SOURCE}")
if [[ "${DATASET_SOURCE}" == "hf" ]]; then
  if [[ -z "${HF_DATASET_REPO}" ]]; then
    echo "[error] HF_DATASET_REPO is required when DATASET_SOURCE=hf" >&2
    exit 1
  fi
  DATASET_ARGS+=(
    --hf-repo "${HF_DATASET_REPO}"
    --hf-config "${HF_DATASET_CONFIG}"
    --hf-split "${HF_DATASET_SPLIT}"
    --hf-cache-dir "${HF_CACHE_DIR}"
  )
  if [[ -n "${HF_DATASET_REVISION:-}" ]]; then
    DATASET_ARGS+=(--hf-revision "${HF_DATASET_REVISION}")
  fi
else
  DATASET_ARGS+=(--dataset-dir "${DATASET_DIR}")
fi

python scripts/run_vlm_fix_text_only_matrix.py \
  "${DATASET_ARGS[@]}" \
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
