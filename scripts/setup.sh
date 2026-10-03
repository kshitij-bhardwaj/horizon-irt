#!/usr/bin/env bash
# Fetch METR's public data at the pinned commit and create a Python environment.
# METR's data is not redistributed in this repository; this script downloads it from METR's own repo.
set -euo pipefail
cd "$(dirname "$0")/.."
METR_SHA=52cb829c7a2efb2d659285c4b1768d191d97f8d2
if [ ! -d eval-analysis-public/.git ]; then
  git clone --filter=blob:none https://github.com/METR/eval-analysis-public.git
fi
git -C eval-analysis-public fetch -q origin "$METR_SHA" || true
git -C eval-analysis-public checkout -q "$METR_SHA"
PYTHON="${PYTHON:-python3.12}"
command -v "$PYTHON" >/dev/null || PYTHON=python3
"$PYTHON" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "Ready. Use .venv/bin/python (METR data at eval-analysis-public @ ${METR_SHA:0:7})."
