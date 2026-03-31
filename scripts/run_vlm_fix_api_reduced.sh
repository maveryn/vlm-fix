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
DATASET_DIR="${DATASET_DIR:-data/generated/vlm_fix}"
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
echo "[run] dataset: ${DATASET_DIR}"
echo "[run] run_tag: ${RUN_TAG}"
echo "[run] batch_size=${BATCH_SIZE} max_new_tokens=${MAX_NEW_TOKENS}"

python scripts/run_vlm_fix_matrix.py \
  --dataset-dir "${DATASET_DIR}" \
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
