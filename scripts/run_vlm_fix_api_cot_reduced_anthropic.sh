#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_vlm_fix_api_cot_reduced.sh \
  claude-sonnet-4-0 \
  claude-sonnet-4-5
