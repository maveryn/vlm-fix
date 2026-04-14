# Hugging Face Datasets

This document describes the public dataset layout for `vlm-fix`.

## Goal

The shared Hugging Face dataset repo holds the public datasets used by the
VLM-Fix benchmark and post-training dataset workflows.

It replaces the old pattern of committing small local payloads such as
`data/vlm_fix_base/` and `data/synth_legs_100/`.

## Config Layout

Config names:

- `vlm_fix`
- `vlm_fix_text_only`
- `vlm_fix_posttrain_d1`
- `vlm_fix_posttrain_d2`
- `vlm_fix_posttrain_d3`
- `synth_legs_train`

Split layout:

- `vlm_fix`: `main`
- `vlm_fix_text_only`: `main`
- `vlm_fix_posttrain_d1`: `train`, `test`
- `vlm_fix_posttrain_d2`: `train`, `test`
- `vlm_fix_posttrain_d3`: `train`, `test`
- `synth_legs_train`: `train`

## Schema Guidance

### `vlm_fix`

Keep the current benchmark metadata columns where possible:

- `index`
- `game`
- `state_id`
- `board_state`
- `render_variant`
- `rule_variant`
- `image_text_order`
- `prompt_variant`
- `prompt_type`
- `question_target`
- `prompt`
- `answer`
- `valid_labels`
- `render_meta`
- `standard_winner_idx`
- `inverse_winner_idx`

Add a native HF `image` column for the rendered board.

### `vlm_fix_text_only`

Keep the current text-only columns:

- `index`
- `game`
- `state_id`
- `board_state`
- `rule_variant`
- `question_target`
- `prompt`
- `answer`
- `valid_labels`
- `board_text`
- `input_text`

No image column is needed here.

### Post-Training Configs

The post-training configs preserve the parquet schema used by the dataset
builders so downstream consumers can read them without an extra translation
layer.

## Migration Order

1. Publish `vlm_fix`.
2. Publish `vlm_fix_text_only`.
3. Validate the existing HF runner path against the published configs.
4. Publish `vlm_fix_posttrain_d1`, `vlm_fix_posttrain_d2`,
   `vlm_fix_posttrain_d3`, and `synth_legs_train`.

## Loader Implications

The current evaluation code expects local file paths for images.

The HF-backed evaluation path uses one of these approaches:

- materialize HF images to a local cache directory before model execution
- update model wrappers to accept in-memory PIL images where supported

The first approach is the lower-risk default for public runs.

## Explicit Exclusion

`VLMs-Are-Biased` and the derived VLMBias RLVR training set are not part of
the `maveryn/vlm-fix` dataset repo. They are derived locally from the
upstream dataset `anvo25/vlms-are-biased`.
