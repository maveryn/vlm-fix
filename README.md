# vlm-fix

`vlm-fix` contains the public benchmark, dataset builders, evaluation runners,
interactive demo assets, and analysis utilities for VLM-Fix and the related
dataset workflows released alongside it.

<p align="center">
  <a href="https://maveryn.github.io/vlm-fix/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-14324a?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2604.12119"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2604.12119-b31b1b?style=for-the-badge"></a>
  <a href="https://huggingface.co/datasets/maveryn/vlm-fix"><img alt="Dataset" src="https://img.shields.io/badge/HuggingFace-Dataset-f59e0b?style=for-the-badge"></a>
  <a href="https://maveryn.github.io/vlm-fix/demo/"><img alt="Demo" src="https://img.shields.io/badge/Interactive-Demo-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="docs/assets/figures/vlm_fix_four_games.png" alt="VLM-Fix benchmark games" width="92%">
</p>

## Resources

| Resource | Link |
| --- | --- |
| Project page | https://maveryn.github.io/vlm-fix/ |
| Interactive demo | https://maveryn.github.io/vlm-fix/demo/ |
| Paper | https://arxiv.org/abs/2604.12119 |
| Dataset | https://huggingface.co/datasets/maveryn/vlm-fix |

## Repository Layout

- benchmark generation code in [`vlm_fix/`](vlm_fix)
- image and text-only evaluation runners in [`eval/`](eval) and [`scripts/`](scripts)
- `VLMs-Are-Biased` subset preparation and evaluation in [`eval/vlms_are_biased/`](eval/vlms_are_biased)
- post-training dataset builders in [`sft/scripts/`](sft/scripts) and [`scripts/`](scripts)
- mechanistic analysis in [`analysis/steering_vlm_fix/`](analysis/steering_vlm_fix)
- published interactive demo source in [`demo/`](demo)
- generated GitHub Pages site in [`docs/`](docs)

## Dataset Release

Public VLM-Fix datasets are hosted on Hugging Face:

- [maveryn/vlm-fix](https://huggingface.co/datasets/maveryn/vlm-fix)

Current dataset configs:

- `vlm_fix`
- `vlm_fix_text_only`
- `vlm_fix_posttrain_d1`
- `vlm_fix_posttrain_d2`
- `vlm_fix_posttrain_d3`
- `synth_legs_train`

`VLMs-Are-Biased` is not mirrored into that dataset repo. The subset used by
the evaluation scripts here is derived locally from the original upstream dataset
`anvo25/vlms-are-biased`.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main VLM-Fix benchmark from Hugging Face:

```bash
python scripts/run_vlm_fix_matrix.py \
  --dataset-source hf \
  --hf-repo maveryn/vlm-fix \
  --hf-config vlm_fix \
  --hf-split main \
  --models Qwen/Qwen2.5-VL-7B-Instruct allenai/Molmo2-4B
```

Run the text-only benchmark:

```bash
python scripts/run_vlm_fix_text_only_matrix.py \
  --dataset-source hf \
  --hf-repo maveryn/vlm-fix \
  --hf-config vlm_fix_text_only \
  --hf-split main \
  --models Qwen/Qwen2.5-VL-7B-Instruct
```

## VLMs-Are-Biased Subset

This repo includes a 4-topic counting subset with:

- topics: `Game Boards`, `Logos`, `Flags`, `Animals`
- prompt styles: `original`, `item_alias`
- image variants: `original`, `flipped`

Prepare that subset locally:

```bash
python eval/prepare_vlms_are_biased_paper_4subset.py \
  --out-dir data/generated/vlms_are_biased_hf_original_4subset_322
```

Run the evaluation matrix:

```bash
python eval/run_vlms_are_biased_matrix.py \
  --dataset-dir data/generated/vlms_are_biased_hf_original_4subset_322 \
  --topics "Game Boards" "Logos" "Flags" "Animals" \
  --prompt-styles original item_alias \
  --image-variants original flipped \
  --models Qwen/Qwen2.5-VL-7B-Instruct
```

## Local Dataset Builds

Build the VLM-Fix benchmark locally:

```bash
python scripts/build_vlm_fix_dataset.py --out-dir data/generated/vlm_fix
python scripts/build_vlm_fix_text_only_dataset.py \
  --src-parquet data/generated/vlm_fix/instances.parquet \
  --out-dir data/generated/vlm_fix_text_only
```

Export local parquet bundles for HF upload:

```bash
python scripts/build_hf_parquet_exports.py
```

Large generated outputs are not tracked in git. Benchmark caches, parquet
exports, runs, and result payloads are expected to live under local output
directories such as `data/generated/`, `hf_export/`, `runs/`, and `results/`.

## Documentation

- [docs/README.md](docs/README.md)
- [docs/datasets.md](docs/datasets.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/hf_datasets.md](docs/hf_datasets.md)
- [docs/posttraining_repro.md](docs/posttraining_repro.md)

## Notes

- The main evaluation entrypoints support both `--dataset-source local` and
  `--dataset-source hf`.
- HF-backed runs materialize images into a local cache before model execution.
