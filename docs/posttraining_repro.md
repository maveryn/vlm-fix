# Post-Training Dataset Builders

This page documents the dataset-building layer that is already present in
`vlm-fix`.

It covers dataset builders and dataset layout only. It does not document a
full training runtime.

## Present In Repo

- `sft/scripts/build_finetune_parquet.py`
- `scripts/build_post_training_vlms_bias_noanimals_origflip.py`
- `scripts/build_hf_parquet_exports.py`

## Dataset Coverage

The current builders cover these training datasets:

- `vlm_fix_posttrain_d1`
- `vlm_fix_posttrain_d2`
- `vlm_fix_posttrain_d3`
- `synth_legs_train`
- `vlmbias_posttrain_noanimals_origflip`

Only the VLM-Fix and synth-legs datasets are in the shared HF repo.
The VLMBias training set is derived locally from the prepared
`VLMs-Are-Biased` paper subset, which is built from
`anvo25/vlms-are-biased`.

## Tracked Here

- dataset builders
- compact documentation

## Not Tracked Here

- checkpoints
- merged model exports
- local training caches
- large run payloads
