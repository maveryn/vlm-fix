#!/usr/bin/env bash
set -euo pipefail

# Reduced VLM-Fix CoT for API models:
# - canonical render only
# - prompt variants: standard, tag, tag_sem
# - image_first + winner only
# - matched to the existing reduced 300-state-per-rule API seed set
# => 600 questions per game per variant, 1,800/game, 7,200/model

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("gpt-4.1")
fi

DATASET_SOURCE="${DATASET_SOURCE:-local}"
DATASET_DIR="${DATASET_DIR:-data/generated/vlm_fix}"
HF_DATASET_REPO="${HF_DATASET_REPO:-}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-vlm_fix}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-main}"
HF_CACHE_DIR="${HF_CACHE_DIR:-data/hf_cache}"

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

echo "[run] reduced CoT models: ${MODELS[*]}"
echo "[run] dataset_source=${DATASET_SOURCE}"
echo "[run] run_tag=${RUN_TAG:-prompt-cot-reduced300}"
echo "[run] batch_size=${BATCH_SIZE:-8} max_new_tokens=${MAX_NEW_TOKENS:-1024}"

DATASET_ARGS=(--dataset-source "${DATASET_SOURCE}")
if [[ "${DATASET_SOURCE}" == "hf" ]]; then
  if [[ -z "${HF_DATASET_REPO}" ]]; then
    echo "[error] HF_DATASET_REPO is required when DATASET_SOURCE=hf" >&2
    exit 1
  fi
  echo "[run] hf_repo=${HF_DATASET_REPO}"
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
  echo "[run] dataset_dir=${DATASET_DIR}"
  DATASET_ARGS+=(--dataset-dir "${DATASET_DIR}")
fi

python scripts/run_vlm_fix_api_cot_reduced.py \
  "${DATASET_ARGS[@]}" \
  --runs-dir "${RUNS_DIR:-runs/vlm_fix}" \
  --results-dir "${RESULTS_DIR:-results/vlm_fix}" \
  --models "${MODELS[@]}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --run-tag "${RUN_TAG:-prompt-cot-reduced300}" \
  --seed-reference-model "${SEED_REFERENCE_MODEL:-gpt-4.1}" \
  --seed-reference-run-tag "${SEED_REFERENCE_RUN_TAG:-prompt-direct-reduced300}" \
  ${SKIP_ARG} \
  ${QUIET_ARG}

echo "[done] reduced CoT outputs in ${RESULTS_DIR:-results/vlm_fix} and ${RUNS_DIR:-runs/vlm_fix}"
