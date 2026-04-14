# Datasets

This repository tracks dataset-generation code. It does not track the public
benchmark payloads themselves.

## Current In-Repo Builders

### 1. VLM-Fix Benchmark

Code:

- [`vlm_fix/`](../vlm_fix)
- [`scripts/build_vlm_fix_dataset.py`](../scripts/build_vlm_fix_dataset.py)

Build locally:

```bash
python scripts/build_vlm_fix_dataset.py \
  --out-dir data/generated/vlm_fix \
  --states-per-game 300 \
  --seed 7
```

Local outputs:

- `data/generated/vlm_fix/instances.parquet`
- `data/generated/vlm_fix/summary.json`
- `data/generated/vlm_fix/counts_by_game_render_rule.csv`
- `data/generated/vlm_fix/images/...`

### 2. VLM-Fix Text-Only

Code:

- [`scripts/build_vlm_fix_text_only_dataset.py`](../scripts/build_vlm_fix_text_only_dataset.py)

Build locally:

```bash
python scripts/build_vlm_fix_text_only_dataset.py \
  --src-parquet data/generated/vlm_fix/instances.parquet \
  --out-dir data/generated/vlm_fix_text_only
```

Local outputs:

- `data/generated/vlm_fix_text_only/instances.parquet`
- `data/generated/vlm_fix_text_only/summary.json`
- `data/generated/vlm_fix_text_only/counts_by_game_rule_target.csv`

### 3. Synth-Legs

Code:

- [`synth_legs/`](../synth_legs)
- [`scripts/build_synth_legs_dataset.py`](../scripts/build_synth_legs_dataset.py)

Build locally:

```bash
python scripts/build_synth_legs_dataset.py \
  --n-total 8192 \
  --out-image-dir data/generated/synth_legs/images \
  --out-parquet data/generated/synth_legs/train_8192.parquet \
  --out-summary data/generated/synth_legs/train_8192_summary.json
```

### 4. VLM-Fix Post-Training Splits (`D1` / `D2` / `D3`)

Code:

- [`sft/scripts/build_finetune_parquet.py`](../sft/scripts/build_finetune_parquet.py)

Build locally:

```bash
python sft/scripts/build_finetune_parquet.py \
  --instances-path data/generated/vlm_fix/instances.parquet \
  --benchmark-root data/generated/vlm_fix \
  --out-dir data/generated/post_training_v2_disjoint \
  --seed 20260317 \
  --exclude-benchmark-from-train
```

### 5. VLMs-Are-Biased Generic Prepared Cache

Code:

- [`eval/prepare_vlms_are_biased.py`](../eval/prepare_vlms_are_biased.py)
- [`eval/vlms_are_biased/`](../eval/vlms_are_biased)

Build locally:

```bash
python eval/prepare_vlms_are_biased.py \
  --out-dir data/generated/vlms_are_biased_prepared \
  --dataset-split main
```

Local outputs:

- `data/generated/vlms_are_biased_prepared/instances.parquet`
- `data/generated/vlms_are_biased_prepared/instances_original.parquet`
- `data/generated/vlms_are_biased_prepared/instances_item_alias.parquet`
- `data/generated/vlms_are_biased_prepared/counts_by_topic_style.csv`
- `data/generated/vlms_are_biased_prepared/images/...`

### 6. VLMs-Are-Biased Paper 4-Topic Counting Subset

This is the paper-specific slice derived from the original HF dataset
`anvo25/vlms-are-biased`. It covers the 4 counting topics used in the study:
`Game Boards`, `Logos`, `Flags`, and `Animals`.

Code:

- [`eval/prepare_vlms_are_biased_paper_4subset.py`](../eval/prepare_vlms_are_biased_paper_4subset.py)
- [`eval/vlms_are_biased/`](../eval/vlms_are_biased)

Build locally:

```bash
python eval/prepare_vlms_are_biased_paper_4subset.py \
  --out-dir data/generated/vlms_are_biased_hf_original_4subset_322
```

Local outputs:

- `data/generated/vlms_are_biased_hf_original_4subset_322/instances.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_original.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_item_alias.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_counting_only_original_prompt_only.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_img-original_prompt-original.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_img-original_prompt-item_alias.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_img-flipped_prompt-original.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/instances_img-flipped_prompt-item_alias.parquet`
- `data/generated/vlms_are_biased_hf_original_4subset_322/counts_by_topic.csv`
- `data/generated/vlms_are_biased_hf_original_4subset_322/counts_by_topic_counting_only.csv`
- `data/generated/vlms_are_biased_hf_original_4subset_322/counts_by_topic_style.csv`
- `data/generated/vlms_are_biased_hf_original_4subset_322/images/...`
- `data/generated/vlms_are_biased_hf_original_4subset_322/images_flipped/...`

### 7. VLMBias No-Animals RLVR Set

Code:

- [`scripts/build_post_training_vlms_bias_noanimals_origflip.py`](../scripts/build_post_training_vlms_bias_noanimals_origflip.py)

Build locally:

```bash
python scripts/build_post_training_vlms_bias_noanimals_origflip.py
```

## Public Data Policy

This repo should not ship:

- `data/` benchmark subsets
- `dataset/` payload snapshots
- local image trees for public release slices
- heavyweight parquet payloads in git

Instead, all public datasets should live in one Hugging Face dataset repo.
The exception is `VLMs-Are-Biased`: that benchmark remains anchored to the
original upstream HF dataset `anvo25/vlms-are-biased`, and the paper subset is
prepared locally from it rather than republished inside `maveryn/vlm-fix`.

## Target HF Dataset Layout

Recommended configs:

- `vlm_fix`
- `vlm_fix_text_only`
- `vlm_fix_posttrain_d1`
- `vlm_fix_posttrain_d2`
- `vlm_fix_posttrain_d3`
- `synth_legs_train`

The exact config and split plan is documented in
[hf_datasets.md](hf_datasets.md).

## Migration Order

1. Publish `vlm_fix` and `vlm_fix_text_only`.
2. Validate the already-wired `--dataset-source hf` runner path against the
   published configs.
3. Publish `D1` / `D2` / `D3` and synth-legs.
4. Keep `VLMs-Are-Biased` derived locally from `anvo25/vlms-are-biased`.
## Notes

- Current scripts still build and consume local directories under
  `data/generated/`.
- The main VLM-Fix runners now also accept `--dataset-source hf` plus
  `--hf-repo`, `--hf-config`, `--hf-split`, and `--hf-revision`.
- The `VLMs-Are-Biased` paper subset is currently a local prep flow, not an HF
  config in `maveryn/vlm-fix`.
- The local-build path remains useful for development even after the HF-backed
  public release is in place.
