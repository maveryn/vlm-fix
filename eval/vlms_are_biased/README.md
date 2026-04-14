# VLMs-Are-Biased Eval

This subpackage contains:

- prompt rewriting for `Original` and `ITEM-Alias`
- dataset preparation (HF download + local parquet/image cache)
- paper-specific 4-topic counting subset preparation
- model inference runner
- scoring and reporting

Entrypoints:

- `python eval/prepare_vlms_are_biased.py ...`
- `python eval/prepare_vlms_are_biased_paper_4subset.py ...`
- `python eval/run_vlms_are_biased_matrix.py ...`
- `python eval/run_vlms_are_biased_eval.py ...`
