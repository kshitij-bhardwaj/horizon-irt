#!/usr/bin/env bash
# Run METR's own headline stages (bootstrap -> logistic fits -> doubling-time CIs) without DVC.
# Uses the dvc/cairosvg shims in replication/dvc_shim and the venv created by scripts/setup.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/../eval-analysis-public"
PY="${PY:-$HERE/../.venv/bin/python}"
export PYTHONPATH="$REPO/src:$HERE/dvc_shim"

WINDOWS=(
  "from2019 2019-01-01 2030-01-01"
  "from2023 2023-01-01 2030-01-01"
  "from2024 2024-01-01 2030-01-01"
  "paper_era 2019-01-01 2025-02-25"
)

for R in "${@:-time-horizon-1-1 time-horizon-1-0}"; do
  for report in $R; do
    cd "$REPO/reports/$report"
    mkdir -p metrics/trendline_ci metrics/logistic_fits
    if [[ ! -f data/wrangled/bootstrap/headline.csv ]]; then
      "$PY" -m horizon.wrangle.bootstrap --fig-name headline --runs-file data/raw/runs.jsonl \
        --output-bootstrap-horizons-file data/wrangled/bootstrap/headline.csv --n-bootstrap 1000 >/dev/null 2>&1
    fi
    "$PY" -m horizon.wrangle.logistic --fig-name headline --runs-file data/raw/runs.jsonl \
      --output-logistic-fits-file data/wrangled/logistic_fits/headline.csv \
      --release-dates ../../data/external/release_dates.yaml --bootstrap-file data/wrangled/bootstrap/headline.csv \
      --output-metrics-file metrics/logistic_fits/headline.yaml >/dev/null 2>&1
    for w in "${WINDOWS[@]}"; do
      read -r name after before <<<"$w"
      log=$("$PY" -m horizon.compute_trendline_ci --input-file data/wrangled/bootstrap/headline.csv \
        --agent-summaries-file data/wrangled/logistic_fits/headline.csv \
        --release-dates ../../data/external/release_dates.yaml \
        --output-metrics-file "metrics/trendline_ci/$name.yaml" \
        --after-date "$after" --before-date "$before" 2>&1)
      point=$(grep -A6 doubling_time_days "metrics/trendline_ci/$name.yaml" | grep point_estimate | awk '{print $2}')
      echo "$report $name: point=${point} $(grep -o '95% CI.*' <<<"$log")"
    done
  done
done
