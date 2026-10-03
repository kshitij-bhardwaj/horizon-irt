# IRT study of METR time horizons

- Write-up: [RESULTS.md](RESULTS.md)
- Design decisions: [DECISIONS.md](DECISIONS.md)
- Validation of METR's original numbers: [../replication/VALIDATION.md](../replication/VALIDATION.md)

## Reproduce (Apple M1, 8 GB, CPU only; about 1 h in total)

```bash
PY=../.venv/bin/python    # created by scripts/setup.sh (Python 3.12, versions pinned to METR's uv.lock)
$PY -m tests.test_models                           # T1–T6 correctness checks (~5 min)
$PY exp2_heterogeneity.py                          # E2: tau, LRTs, variance decomposition (~2 min)
$PY exp1_predictive.py                             # E1: protocols A, B, C (~6 min)
$PY exp3_horizons.py --n-boot 200                  # E3: horizons, p80, paired bootstrap trends (~45 min)
$PY exp3b_bias_sim.py                              # E3b: bias of METR's estimator under heterogeneity (~5 min)
$PY exp3c_cross_suite.py                           # E3c: TH1.0 vs TH1.1 stability (~1 min)
$PY exp4_stability.py --n-boot 300                 # E4: ridge, weights, priors, failure case (~2 min)
$PY exp2_heterogeneity.py --report time-horizon-1-0 && $PY exp1_predictive.py --report time-horizon-1-0 --protocols AB
```

## Layout

| Path | Contents |
|---|---|
| `irt/` | `data.py` (cells, weights), `models.py` (MAP and MML estimators, horizons), `metrics.py`, `bootstrap.py`, `trend.py`, `simulate.py`, `plotstyle.py` |
| `results/` | Every table as CSV/JSON (`*_th11` is TH1.1, `*_th10` is TH1.0) |
| `figures/` | PDF (for LaTeX) and PNG |
| `logs/` | Full stdout of every run |
