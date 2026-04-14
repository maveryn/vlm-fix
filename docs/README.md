# Docs Index

This folder documents the public `vlm-fix` repository.

## Current Phase

The repo currently contains:

- the core `vlm-fix` benchmark package
- local benchmark builders and evaluation scripts
- HF-capable dataset loading for the VLM-Fix runners
- the `VLMs-Are-Biased` prep/eval package, including the paper's 4-topic
  counting subset builder and the 2x2 prompt/image study variants
- the post-training dataset builders for `D1` / `D2` / `D3` and the VLMBias
  no-Animals RLVR set
- synth-legs generation code
- VLM-Fix activation-steering utilities

The repo does not yet contain:

- tracked dataset payloads
- the full paper manuscript and provenance layer

## Start Here

- [datasets.md](datasets.md)
  - what dataset-generation code is already in `vlm-fix`
  - how to prepare the `VLMs-Are-Biased` paper subset locally
  - how the public VLM-Fix data should move to Hugging Face

- [evaluation.md](evaluation.md)
  - current benchmark evaluation entrypoints
  - model scope for the public release
  - current VLM-Fix and `VLMs-Are-Biased` evaluation entrypoints

- [hf_datasets.md](hf_datasets.md)
  - proposed single-repo HF dataset layout
  - config names, split names, and schema expectations for VLM-Fix datasets
  - migration order for publishing and loader refactors

- [posttraining_repro.md](posttraining_repro.md)
  - post-training dataset builders and dataset coverage

## Repository Scope

`vlm-fix` is the public code repository:

- code and documentation live here
- generated datasets live in a shared HF dataset repo
- heavyweight local runs, checkpoints, and result payloads stay out of git
