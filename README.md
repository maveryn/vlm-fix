# VLM-Fix Release Package

This repository contains the minimal code and packaged data needed to reproduce the VLM-Fix benchmark generation, the released base-slice dataset payload, the synth-legs generator plus a 100-example release subset, the evaluation scripts used for the 10 open + 4 API models in the paper, and the VLM-Fix activation-steering code.

## Repository Layout

```text
vlm-fix/
├── README.md
├── requirements.txt
├── vlm_fix/
│   ├── games/
│   ├── render/
│   ├── generation.py
│   └── prompts.py
├── synth_legs/
│   ├── generate_bird_synth_samples.py
│   └── generate_animals_synth_samples.py
├── eval/
│   ├── model_registry.py
│   ├── run_vlm_fix_eval.py
│   └── models/
├── scripts/
│   ├── build_vlm_fix_dataset.py
│   ├── build_vlm_fix_text_only_dataset.py
│   ├── build_synth_legs_dataset.py
│   ├── run_vlm_fix_matrix.py
│   ├── run_vlm_fix_api_reduced.sh
│   ├── run_vlm_fix_api_cot_reduced.py
│   ├── run_vlm_fix_api_cot_reduced*.sh
│   ├── run_vlm_fix_text_only_matrix.py
│   ├── run_vlm_fix_text_only_open10.sh
│   ├── run_vlm_fix_text_only_api4.sh
│   └── build_vlm_fix_text_only_reports.py
├── analysis/
│   └── steering-vlmfix/
│       ├── run_transfer_matrix_cached.py
│       ├── plot_transfer_matrix_layerwise.py
│       ├── collect_within_game_pdfs.py
│       └── run_within_game_all_models.sh
└── data/
    ├── vlm_fix_base/
    │   ├── annotations.jsonl
    │   ├── summary.json
    │   └── images/
    └── synth_legs_100/
        ├── annotations.jsonl
        ├── summary.json
        └── images/
```

## Included Data

- `data/vlm_fix_base/`: the released core VLM-Fix slice using the canonical image rendering and base prompt family.
  - Images are included for all 4 games.
  - `annotations.jsonl` contains the direct-prompt evaluation rows for both rules, both query targets, and both input orders.
- `data/synth_legs_100/`: a 100-example release subset of the synth-legs dataset with relative image paths and metadata.

## Regenerating VLM-Fix

```bash
python scripts/build_vlm_fix_dataset.py --out-dir data/generated/vlm_fix
python scripts/build_vlm_fix_text_only_dataset.py \
  --src-parquet data/generated/vlm_fix/instances.parquet \
  --out-dir data/generated/vlm_fix_text_only
```

## Running Evaluation

Generic image-based evaluation:

```bash
python scripts/run_vlm_fix_matrix.py \
  --dataset-dir data/generated/vlm_fix \
  --models Qwen/Qwen2.5-VL-7B-Instruct allenai/Molmo2-4B
```

Reduced API direct slice used in the paper:

```bash
bash scripts/run_vlm_fix_api_reduced.sh gpt-4.1 gpt-5.2 claude-sonnet-4-0 claude-sonnet-4-5
```

Reduced API CoT slice used in the paper:

```bash
bash scripts/run_vlm_fix_api_cot_reduced_openai.sh
bash scripts/run_vlm_fix_api_cot_reduced_anthropic.sh
```

Text-only evaluation:

```bash
bash scripts/run_vlm_fix_text_only_open10.sh
bash scripts/run_vlm_fix_text_only_api4.sh
```

## Running VLM-Fix Activation Steering

```bash
python analysis/steering-vlmfix/run_transfer_matrix_cached.py \
  --dataset data/generated/vlm_fix/instances.parquet \
  --dataset-root data/generated/vlm_fix \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --setup-ids within_tictactoe,within_reversi,within_connect4,within_dots_boxes
```

## Notes

- All packaged paths are relative. There are no repository-specific absolute paths in the tracked files.
- The model registry is intentionally limited to the 10 open models and 4 API models used in the paper.
- Generated outputs default to `data/generated/`, `runs/`, and `results/`, which are gitignored.
