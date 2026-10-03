# Validation: METR "Measuring AI Ability to Complete Long Tasks" (+ Time Horizon 1.1)

- Paper: Kwa et al., 2025, arXiv:2503.14499. TH1.1 update: metr.org/blog/2026-1-29-time-horizon-1-1
- Code/data: github.com/METR/eval-analysis-public (checked out at `52cb829`, 2026-03-05)
- Hardware: Apple M1, 8 GB RAM, CPU only. The full headline pipeline runs in about 1 minute.

## What was run

| Path | What it is |
|---|---|
| `replication/validate_horizon.py` | Independent re-implementation, written from the paper: Newton-Raphson weighted logistic fit, own 3-level hierarchical bootstrap (family, then task, then run), SOTA filter, OLS doubling time. It doesn't import METR's package. |
| `replication/run_official.sh` | METR's own stages (`wrangle.bootstrap`, `wrangle.logistic`, `compute_trendline_ci`), run without DVC. |
| `replication/dvc_shim/` | Stubs for `dvc.api.params_show` (merges `params.yaml` with `fig_params/figs.yaml`) and `cairosvg` (only used for a plot logo). |
| `.venv` (from `scripts/setup.sh`) | Python 3.12 with the versions pinned in `uv.lock`: pandas 2.3.3, numpy 2.4.1, scikit-learn 1.8.0, scipy 1.17.0. pandas 3.x breaks `wrangle.logistic`. |

Rerun:

```bash
.venv/bin/python replication/validate_horizon.py --report time-horizon-1-1
replication/run_official.sh
```

## 1. Independent implementation vs METR's pipeline (same data, λ = 1e-5)

Doubling-time point estimates in days (95% CI):

| Report / window | METR pipeline | Independent |
|---|---|---|
| TH1.1, 2023+ | 128.74 [105, 157] | 128.7 [104, 157] |
| TH1.1, 2024+ | 102.18 [82, 126] | 102.2 [79, 127] |
| TH1.0, 2019+ | 201.15 [176, 225] | 201.1 [177, 227] |
| TH1.0, 2023+ | 175.6 [138, 229] | 175.6 [139, 230] |
| TH1.0, 2024+ | 131.32 [104, 171] | 131.3 [104, 172] |
| TH1.0, 2019 to 2025-02-25 | 216.17 [184, 247] | 216.2 [187, 248] |

Per-model p50 horizons agree for all 20 TH1.1 agents to within about 2% (bootstrap medians and CI bounds). Only the RNG streams differ.

## 2. Against published numbers (each at its own data snapshot and hyperparameters)

| Claim | Published | Reproduced | Snapshot / λ |
|---|---|---|---|
| Claude 3.7 Sonnet p50 | ~59 min | 59.0 min [31, 103] | `4f79669` (2025-03-18), λ = 0.1 |
| Doubling time 2019–2025 | ~7 months | 212 d [169, 245] | same |
| TH1 doubling time 2024+ (quoted in TH1.1 blog) | 108.9 d | 108.5 d [81, 162] | same |
| Opus 4.5 p50 (TH1.1) | 320 [170, 729] | 320.4 [170, 799] | `8d68adb` (2026-01-30), λ = 0.1 |
| GPT-5 p50 (TH1.1) | 214 [117, 480] | 214.0 [120, 454] | same |
| o3 p50 (TH1.1) | 121 [74, 201] | 120.7 [74, 204] | same |
| TH1.1 doubling time 2023+ | 130.8 d | 130.8 d [108, 162] | same |
| TH1.1 doubling time 2024+ | 88.6 d | 94.1 d [75, 120] | same. **Not exact.** The published value is inside our CI, but the exact window or model set is unclear. |
| TH1.1 doubling time 2019+ (hybrid TH1 + TH1.1) | 196.5 d | not attempted | needs METR's stitching of GPT-2, davinci-002 and gpt-3.5 from TH1.0 |

## 3. Finding: results are sensitive to one undocumented hyperparameter change

Between the TH1.1 release and the current repo, `regularization` in `fig_params/figs.yaml` changed from **0.1 to 1e-5**. The per-agent sample weights sum to 1, so λ = 0.1 is a strong L2 penalty on the logistic slope. It flattens each curve and moves p50 away from the centre of the data. On identical data (2026-01-30 snapshot):

| Quantity | λ = 0.1 (blog) | λ = 1e-5 (current) | Δ |
|---|---|---|---|
| Opus 4.5 p50 | 320 min | 293 min | −9% |
| GPT-5 p50 | 214 min | 203 min | −5% |
| TH1.1 doubling time 2023+ | 130.8 d | 136.6 d | +4% |

The paper reports that the fit is "not sensitive" to regularization. That holds for the trend within CIs, but the frontier point estimates that get quoted publicly move by about 10%.

## 4. Other observations worth following up

- **Saturation at the frontier.** On current data, Claude Opus 4.6's p50 is 719 min with a 95% CI of [304, 4264]. The upper bound is about 6× the point estimate because few tasks exceed 8 h, and TH1.1 notes that only 5 of 31 long tasks have measured human times. The logistic fit is extrapolating past its data here.
- **Humans are fit like an agent.** The "human" alias gets a p50 of about 108 min [56, 210]. Comparing human and model logistic slopes is a natural analysis that neither the paper nor the blog reports.
