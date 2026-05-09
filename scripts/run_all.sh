#!/usr/bin/env bash
# End-to-end pipeline. Pass --fast for a smoke-test run (< 10 min on a single GPU).
# Requires `uv` (https://docs.astral.sh/uv/). The first invocation will resolve
# and install the project's dependencies into .venv automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

FAST=""
if [[ "${1:-}" == "--fast" ]]; then
    FAST="--fast"
fi

uv run scripts/01_download_data.py
uv run scripts/02_extract_features.py
uv run scripts/03_train_traditional.py    $FAST
uv run scripts/04_train_resnet.py         $FAST
uv run scripts/08_explain_traditional.py  $FAST
uv run scripts/09_explain_cnn.py          $FAST
uv run scripts/06_evaluate_all.py

echo "Done. Figures and metrics are under results/."
