# HorizonCAT

Adaptive, cost-aware measurement of AI-agent 50% time horizons on METR's task suite, with saturation detection and a task-suite audit. It is built on the IRT model of the companion study (`../study`), and on BRIDGE's finding that IRT difficulty is linear in log human time.

- Design decisions, novelty and positioning (BRIDGE, ATLAS, Ndzomga, …): [DECISIONS_PROTOTYPE.md](DECISIONS_PROTOTYPE.md)
- Interactive page: source `web/index.html`; the public site in `../docs/` is built by `build_pages.py` (METR's run data is fetched from METR at view time, not re-hosted)

## Reproduce

```bash
PY=../.venv/bin/python    # created by scripts/setup.sh
$PY -m tests.test_engine          # E1–E4: EVR algebra, Fisher matrix, quantile bank, interval calibration
$PY backtest.py                   # replay back-test, 20 held-out agents x 6 policies x 20 replays (~20 min)
$PY frontier.py --resim           # measurement range, suite design, hypothetical frontier agents
$PY export.py                     # writes web/data.json (full export, git-ignored)
$PY build_pages.py <github-owner>  # writes ../docs/ for GitHub Pages
```

## Layout

| Path | Contents |
|---|---|
| `horizoncat/bank.py` | Leave-one-agent-out item bank from study model M: task posteriors compressed to 10 quantile points; token costs |
| `horizoncat/engine.py` | (h, β) grid posterior, expected-variance-reduction-per-cost selection, stopping and saturation, baseline policies |
| `horizoncat/design.py` | Fisher information over (h, β), slope-aware SE, tasks needed to extend the suite |
| `backtest.py`, `frontier.py`, `export.py` | Experiments and data export |
| `results/`, `logs/` | Outputs |
