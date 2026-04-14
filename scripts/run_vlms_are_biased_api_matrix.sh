#!/usr/bin/env bash
set -euo pipefail

# Run VLMs-Are-Biased on 4 topics x 4 variants:
# - topics: Game Boards, Logos, Flags, Animals
# - prompt styles: original, item_alias
# - image variants: original, flipped
#
# Usage:
#   bash scripts/run_vlms_are_biased_api_matrix.sh gpt-4.1 gpt-5.2 claude-sonnet-4-0
#   bash scripts/run_vlms_are_biased_api_matrix.sh   # defaults to the 3 API models above
#
# Optional env overrides:
#   DATASET_DIR=dataset/vlms_are_biased_prepared
#   RUNS_DIR=runs/vlms_are_biased
#   RESULTS_DIR=results/vlms_are_biased_api
#   RESULTS_XLSX=vlms_are_biased_summary_api3.xlsx
#   BATCH_SIZE=128
#   MAX_NEW_TOKENS=1024
#   DATASET_SPLIT=main
#   PREPARE_IF_MISSING=1   # set to 0 to disable auto-prepare

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("claude-sonnet-4-0" "gpt-4.1" "gpt-5.2")
fi

TOPICS=("Game Boards" "Logos" "Flags" "Animals")
PROMPT_STYLES=("original" "item_alias")
IMAGE_VARIANTS=("original" "flipped")

DATASET_DIR="${DATASET_DIR:-dataset/vlms_are_biased_prepared}"
RUNS_DIR="${RUNS_DIR:-runs/vlms_are_biased}"
RESULTS_DIR="${RESULTS_DIR:-results/vlms_are_biased_api}"
RESULTS_XLSX="${RESULTS_XLSX:-vlms_are_biased_summary_api3.xlsx}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DATASET_SPLIT="${DATASET_SPLIT:-main}"
PREPARE_IF_MISSING="${PREPARE_IF_MISSING:-1}"

if [[ "${PREPARE_IF_MISSING}" == "1" ]]; then
  PREPARE_ARG="--prepare-if-missing"
else
  PREPARE_ARG=""
fi

echo "[run] models: ${MODELS[*]}"
echo "[run] topics: ${TOPICS[*]}"
echo "[run] prompt_styles: ${PROMPT_STYLES[*]}"
echo "[run] image_variants: ${IMAGE_VARIANTS[*]}"
echo "[run] dataset_dir=${DATASET_DIR}"
echo "[run] runs_dir=${RUNS_DIR}"
echo "[run] results_dir=${RESULTS_DIR}"
echo "[run] results_xlsx=${RESULTS_XLSX}"
echo "[run] batch_size=${BATCH_SIZE} max_new_tokens=${MAX_NEW_TOKENS}"

python eval/run_vlms_are_biased_matrix.py \
  --models "${MODELS[@]}" \
  --topics "${TOPICS[@]}" \
  --prompt-styles "${PROMPT_STYLES[@]}" \
  --image-variants "${IMAGE_VARIANTS[@]}" \
  --dataset-dir "${DATASET_DIR}" \
  ${PREPARE_ARG} \
  --dataset-split "${DATASET_SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --runs-dir "${RUNS_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --results-xlsx "${RESULTS_XLSX}"

echo "[done] matrix run complete."
echo "[done] runs: ${RUNS_DIR}"
echo "[done] results: ${RESULTS_DIR}/${RESULTS_XLSX}"
