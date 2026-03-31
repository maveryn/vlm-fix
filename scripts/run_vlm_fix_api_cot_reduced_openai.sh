#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_vlm_fix_api_cot_reduced.sh \
  gpt-4.1 \
  gpt-5.2
