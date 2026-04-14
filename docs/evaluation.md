# Evaluation

This document covers the evaluation layer that is currently present in
`vlm-fix`.

## Current Scope

The repo currently includes:

- VLM-Fix image-based evaluation
- VLM-Fix text-only evaluation
- `VLMs-Are-Biased` image-based evaluation and matrix reporting
- shared local/HF dataset loading for the VLM-Fix runners
- reduced direct and CoT wrappers for the paper's API models
- activation-steering analysis scripts for VLM-Fix

## Model Scope

The tracked public evaluation stack currently targets:

- 10 open-weight VLMs
- 4 API-backed VLMs

Open-weight models:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-32B-Instruct`
- `OpenGVLab/InternVL3_5-4B`
- `OpenGVLab/InternVL3_5-8B`
- `OpenGVLab/InternVL3_5-14B`
- `allenai/Molmo2-4B`
- `allenai/Molmo2-8B`

API-backed models:

- `gpt-4.1`
- `gpt-5.2`
- `claude-sonnet-4-0`
- `claude-sonnet-4-5`

These same model keys are used by the `VLMs-Are-Biased` runners through the
shared `eval.model_registry` layer.

## Current Image Evaluation Path

Primary entrypoints:

- [`scripts/run_vlm_fix_matrix.py`](../scripts/run_vlm_fix_matrix.py)
- [`eval/run_vlm_fix_eval.py`](../eval/run_vlm_fix_eval.py)

Current input formats:

- local dataset directory containing `instances.parquet`
- HF dataset config loaded via `--dataset-source hf`

Typical command:

```bash
python scripts/run_vlm_fix_matrix.py \
  --dataset-dir data/generated/vlm_fix \
  --models Qwen/Qwen2.5-VL-7B-Instruct allenai/Molmo2-4B
```

HF-backed command shape:

```bash
python scripts/run_vlm_fix_matrix.py \
  --dataset-source hf \
  --hf-repo maveryn/vlm-fix \
  --hf-config vlm_fix \
  --hf-split main \
  --models Qwen/Qwen2.5-VL-7B-Instruct
```

Typical outputs:

- `runs/vlm_fix/<game>/*.jsonl`
- `results/vlm_fix/*.csv`
- `results/vlm_fix/*.parquet`

Useful wrappers:

- [`scripts/run_vlm_fix_api_reduced.sh`](../scripts/run_vlm_fix_api_reduced.sh)
- [`scripts/run_vlm_fix_api_cot_reduced.py`](../scripts/run_vlm_fix_api_cot_reduced.py)
- [`scripts/run_vlm_fix_api_cot_reduced_openai.sh`](../scripts/run_vlm_fix_api_cot_reduced_openai.sh)
- [`scripts/run_vlm_fix_api_cot_reduced_anthropic.sh`](../scripts/run_vlm_fix_api_cot_reduced_anthropic.sh)

## Current Text-Only Evaluation Path

Primary entrypoints:

- [`scripts/run_vlm_fix_text_only_matrix.py`](../scripts/run_vlm_fix_text_only_matrix.py)
- [`scripts/build_vlm_fix_text_only_reports.py`](../scripts/build_vlm_fix_text_only_reports.py)

Current input formats:

- local dataset directory containing `instances.parquet`
- HF dataset config loaded via `--dataset-source hf`

Typical command:

```bash
python scripts/run_vlm_fix_text_only_matrix.py \
  --dataset-dir data/generated/vlm_fix_text_only \
  --models Qwen/Qwen2.5-VL-7B-Instruct
```

HF-backed command shape:

```bash
python scripts/run_vlm_fix_text_only_matrix.py \
  --dataset-source hf \
  --hf-repo maveryn/vlm-fix \
  --hf-config vlm_fix_text_only \
  --hf-split main \
  --models Qwen/Qwen2.5-VL-7B-Instruct
```

Typical outputs:

- `runs/vlm_fix_text_only/<game>/*.jsonl`
- `results/vlm_fix_text_only/*.csv`
- `results/vlm_fix_text_only/*.parquet`

## Current VLMs-Are-Biased Evaluation Path

Primary entrypoints:

- [`eval/prepare_vlms_are_biased_paper_4subset.py`](../eval/prepare_vlms_are_biased_paper_4subset.py)
- [`eval/run_vlms_are_biased_eval.py`](../eval/run_vlms_are_biased_eval.py)
- [`eval/run_vlms_are_biased_matrix.py`](../eval/run_vlms_are_biased_matrix.py)
- [`scripts/run_vlms_are_biased_api_matrix.sh`](../scripts/run_vlms_are_biased_api_matrix.sh)

Study subset preparation:

```bash
python eval/prepare_vlms_are_biased_paper_4subset.py \
  --out-dir data/generated/vlms_are_biased_hf_original_4subset_322
```

That prep flow materializes:

- the 4 study topics: `Game Boards`, `Logos`, `Flags`, `Animals`
- both prompt styles: `original`, `item_alias`
- both image variants: `original`, `flipped`
- explicit 2x2 parquet views for the matrix runs

Typical matrix command:

```bash
python eval/run_vlms_are_biased_matrix.py \
  --dataset-dir data/generated/vlms_are_biased_hf_original_4subset_322 \
  --topics "Game Boards" "Logos" "Flags" "Animals" \
  --prompt-styles original item_alias \
  --image-variants original flipped \
  --models Qwen/Qwen2.5-VL-7B-Instruct
```

Typical API wrapper:

```bash
bash scripts/run_vlms_are_biased_api_matrix.sh gpt-4.1 gpt-5.2 claude-sonnet-4-5
```

Typical outputs:

- `runs/vlms_are_biased/<topic>/*.jsonl`
- `results/vlms_are_biased/*.csv`
- `results/vlms_are_biased/*.xlsx`
- `results/vlms_are_biased/*.tex`

## Steering Analysis

Current entrypoints:

- [`analysis/steering_vlm_fix/run_transfer_matrix_cached.py`](../analysis/steering_vlm_fix/run_transfer_matrix_cached.py)
- [`analysis/steering_vlm_fix/plot_transfer_matrix_layerwise.py`](../analysis/steering_vlm_fix/plot_transfer_matrix_layerwise.py)

These scripts still assume a local benchmark dataset directory.

## HF Runner Flags

The main evaluation change is now in place for:

- `scripts/run_vlm_fix_matrix.py`
- `eval/run_vlm_fix_eval.py`
- `scripts/run_vlm_fix_api_cot_reduced.py`
- `scripts/run_vlm_fix_text_only_matrix.py`

Expected new flags:

- `--dataset-source local|hf`
- `--hf-repo`
- `--hf-config`
- `--hf-split`
- `--hf-revision`
- `--hf-cache-dir`

The shell wrappers for reduced API and text-only runs also accept
`DATASET_SOURCE=hf` plus the corresponding `HF_DATASET_*` environment
variables.

## Note On Data Source

Unlike the VLM-Fix benchmark, `VLMs-Are-Biased` is not published as a config
inside the shared `maveryn/vlm-fix` HF dataset repo. The public workflow here
is:

1. prepare the paper subset locally from `anvo25/vlms-are-biased`
2. run the matrix or single-eval entrypoints against that local prepared cache
