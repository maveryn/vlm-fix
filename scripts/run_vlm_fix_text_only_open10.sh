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

# 1) Run 9 open models (excluding 32B) on single GPU by default.
python scripts/run_vlm_fix_text_only_matrix.py \
  "${DATASET_ARGS[@]}" \
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
  "${DATASET_ARGS[@]}" \
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
