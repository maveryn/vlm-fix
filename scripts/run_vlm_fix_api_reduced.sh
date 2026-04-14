#!/usr/bin/env bash
set -euo pipefail

# Run reduced VLM-Fix direct benchmark:
# - Keep only the 6 variant families
# - Keep only image_first + winner (drops order/target expansions)
# - Keeps both rules (standard/inverse)
# => 14,400 total questions across 4 games (3,600/game)

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("gpt-4.1")
fi

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DATASET_SOURCE="${DATASET_SOURCE:-local}"
DATASET_DIR="${DATASET_DIR:-data/generated/vlm_fix}"
HF_DATASET_REPO="${HF_DATASET_REPO:-}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-vlm_fix}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-main}"
HF_CACHE_DIR="${HF_CACHE_DIR:-data/hf_cache}"
RUNS_DIR="${RUNS_DIR:-runs/vlm_fix}"
RESULTS_DIR="${RESULTS_DIR:-results/vlm_fix}"
RUN_TAG="${RUN_TAG:-prompt-direct-reduced300}"

if [[ "${NO_SKIP_EXISTING:-0}" == "1" ]]; then
  SKIP_ARG="--no-skip-existing"
else
  SKIP_ARG="--skip-existing"
fi

if [[ "${NO_QUIET:-0}" == "1" ]]; then
  QUIET_ARG="--no-quiet"
else
  QUIET_ARG="--quiet"
fi

echo "[run] models: ${MODELS[*]}"
echo "[run] dataset_source: ${DATASET_SOURCE}"
echo "[run] run_tag: ${RUN_TAG}"
echo "[run] batch_size=${BATCH_SIZE} max_new_tokens=${MAX_NEW_TOKENS}"

DATASET_ARGS=(--dataset-source "${DATASET_SOURCE}")
if [[ "${DATASET_SOURCE}" == "hf" ]]; then
  if [[ -z "${HF_DATASET_REPO}" ]]; then
    echo "[error] HF_DATASET_REPO is required when DATASET_SOURCE=hf" >&2
    exit 1
  fi
  echo "[run] hf_repo: ${HF_DATASET_REPO}"
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
  echo "[run] dataset_dir: ${DATASET_DIR}"
  DATASET_ARGS+=(--dataset-dir "${DATASET_DIR}")
fi

python scripts/run_vlm_fix_matrix.py \
  "${DATASET_ARGS[@]}" \
  --models "${MODELS[@]}" \
  --prompt-types direct \
  --render-variants canonical checkerboard glyph \
  --prompt-variants standard tag tag_sem desc \
  --rule-variants standard inverse \
  --image-text-orders image_first \
  --question-targets winner \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --runs-dir "${RUNS_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --run-tag "${RUN_TAG}" \
  ${SKIP_ARG} \
  ${QUIET_ARG}

echo "[done] outputs in ${RESULTS_DIR} and ${RUNS_DIR}"
