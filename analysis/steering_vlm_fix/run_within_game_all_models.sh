#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner:
# - runs within-game steering matrix (4 games) per model
# - plots within-game PDFs with current style
# - copies within PDFs to each run dir root
# - refreshes single gallery folder with all available PDFs

ROOT="analysis/steering_vlm_fix"
OUT_BASE="$ROOT/outputs"
RUN_BASE="$OUT_BASE/transfer_matrix_cached"
SETUP_IDS="within_tictactoe,within_reversi,within_connect4,within_dots_boxes"

STATES_PER_GAME="${STATES_PER_GAME:-100}"
REPEATS="${REPEATS:-3}"
SEED_BASE="${SEED_BASE:-1}"
TEST_SIZE="${TEST_SIZE:-0.30}"
LAST_N_LAYERS="${LAST_N_LAYERS:-12}"
ALPHA="${ALPHA:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CENTROID_POLICY="${CENTROID_POLICY:-source_correct}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Default model list (short tags -> HF ids)
declare -A MODEL_MAP
MODEL_MAP["qwen25vl7b"]="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_MAP["qwen25vl3b"]="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_MAP["molmo2_8b"]="allenai/Molmo2-8B"
MODEL_MAP["internvl35_8b"]="OpenGVLab/InternVL3_5-8B"

# Optional CLI filter: comma-separated short tags
# Example:
#   bash analysis/steering_vlm_fix/run_within_game_all_models.sh qwen25vl3b,molmo2_8b
MODEL_FILTER="${1:-}"

if [[ -n "$MODEL_FILTER" ]]; then
  IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_FILTER"
else
  MODEL_TAGS=("qwen25vl7b" "qwen25vl3b" "molmo2_8b" "internvl35_8b")
fi

mkdir -p "$RUN_BASE"

resolve_run_dir() {
  local base_dir="$1"
  if [[ -f "$base_dir/run_meta.json" ]]; then
    echo "$base_dir"
    return 0
  fi
  local parent
  local stem
  parent="$(dirname "$base_dir")"
  stem="$(basename "$base_dir")"
  local cand
  cand="$(
    find "$parent" -maxdepth 1 -mindepth 1 -type d -name "${stem}*" -printf '%T@ %p\n' \
      | sort -nr \
      | awk '{print $2}' \
      | while read -r d; do
          if [[ -f "$d/run_meta.json" ]]; then
            echo "$d"
            break
          fi
        done
  )"
  if [[ -z "$cand" ]]; then
    return 1
  fi
  echo "$cand"
}

total="${#MODEL_TAGS[@]}"
idx=0
for tag in "${MODEL_TAGS[@]}"; do
  idx=$((idx + 1))
  model="${MODEL_MAP[$tag]:-}"
  if [[ -z "$model" ]]; then
    echo "[warn] unknown model tag: $tag (skip)"
    continue
  fi

  run_name="within_game_${tag}_spg${STATES_PER_GAME}_rep${REPEATS}_last${LAST_N_LAYERS}_alpha${ALPHA}"
  run_dir="$RUN_BASE/$run_name"

  echo "[$idx/$total] model=$model"
  echo "          run_dir=$run_dir"

  if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "$run_dir/within_tictactoe.pdf" ]] && [[ -f "$run_dir/within_reversi.pdf" ]] && [[ -f "$run_dir/within_connect4.pdf" ]] && [[ -f "$run_dir/within_dots_boxes.pdf" ]]; then
    echo "          skip existing (all 4 within PDFs present)"
    continue
  fi

  python "$ROOT/run_transfer_matrix_cached.py" \
    --model "$model" \
    --setup-ids "$SETUP_IDS" \
    --states-per-game "$STATES_PER_GAME" \
    --repeats "$REPEATS" \
    --seed-base "$SEED_BASE" \
    --test-size "$TEST_SIZE" \
    --last-n-layers "$LAST_N_LAYERS" \
    --alpha "$ALPHA" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --centroid-train-policy "$CENTROID_POLICY" \
    --out-dir "$RUN_BASE" \
    --run-name "$run_name"

  run_dir="$(resolve_run_dir "$run_dir")"
  echo "          resolved_run_dir=$run_dir"

  python "$ROOT/plot_transfer_matrix_layerwise.py" \
    --run-dir "$run_dir"

  # Promote within PDFs to run root for easy collection.
  for g in tictactoe reversi connect4 dots_boxes; do
    src="$run_dir/plots_layerwise/within_${g}.pdf"
    if [[ -f "$src" ]]; then
      cp "$src" "$run_dir/within_${g}.pdf"
    fi
    src_png="$run_dir/plots_layerwise/within_${g}.png"
    if [[ -f "$src_png" ]]; then
      cp "$src_png" "$run_dir/within_${g}.png"
    fi
  done

done

# Refresh global gallery folder (PDF only)
python "$ROOT/collect_within_game_pdfs.py"

echo "[done] within-game runs complete"
echo "Gallery: $OUT_BASE/within_game_pdf_gallery_16"
