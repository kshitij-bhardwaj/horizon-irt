"""Independent re-implementation of METR's time-horizon estimator.

Written from the paper (Kwa et al. 2025, arXiv:2503.14499) and TH1.1 notes, not by
calling METR's `horizon` package, so agreement with their pipeline is a real check.

Model, per agent a:
    P(success | task t) = sigmoid(b0_a + b1_a * log2(human_minutes_t))
    fitted by weighted maximum likelihood (weights = invsqrt_task_weight),
    with a tiny L2 penalty on the slope only (lambda = 1e-5, matching params.yaml).
    p-horizon: log2 h_p = (logit(p) - b0) / b1.

Trend: OLS of log2(p50) on release date over SOTA agents -> doubling time.
Uncertainty: 3-level hierarchical bootstrap (task family -> task -> run).
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1] / "eval-analysis-public"


def fit_logistic(x, y, w, lam=1e-5, iters=100):
    """Weighted logistic regression by Newton-Raphson; L2 penalty on slope only."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    P = np.diag([0.0, lam])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        grad = X.T @ (w * (p - y)) + P @ beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + P
        step = np.linalg.solve(H + 1e-12 * np.eye(2), grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def horizon(beta, p):
    return 2 ** ((np.log(p / (1 - p)) - beta[0]) / beta[1])


def fit_all(df, lam, ps=(0.5, 0.8)):
    out = {}
    for agent, g in df.groupby("alias"):
        if g.score_binarized.nunique() < 2:
            continue
        b = fit_logistic(np.log2(g.human_minutes.values), g.score_binarized.values.astype(float),
                         g.invsqrt_task_weight.values, lam)
        out[agent] = {f"p{int(p * 100)}": horizon(b, p) for p in ps} | {"b0": b[0], "b1": b[1]}
    return pd.DataFrame(out).T


class HierBootstrap:
    """Resample families, then tasks within each drawn family, then runs within (task, agent)."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.fam_tasks = {f: g.task_id.unique() for f, g in self.df.groupby("task_family")}
        self.task_rows = {}
        for t, g in self.df.groupby("task_id"):
            g = g.sort_values("alias")
            codes, starts, counts = np.unique(g.alias.values, return_index=True, return_counts=True)
            per_row_start = np.repeat(starts, counts)
            per_row_count = np.repeat(counts, counts)
            self.task_rows[t] = (g.index.values, per_row_start, per_row_count)
        self.fams = np.array(list(self.fam_tasks))

    def sample(self, rng):
        idx = []
        for f in rng.choice(self.fams, len(self.fams)):
            tasks = self.fam_tasks[f]
            for t in rng.choice(tasks, len(tasks)):
                rows, start, count = self.task_rows[t]
                pick = start + (rng.random(len(rows)) * count).astype(int)
                idx.append(rows[pick])
        return self.df.iloc[np.concatenate(idx)]


def sota_agents(fits, dates, after, before):
    d = fits.assign(date=[pd.Timestamp(dates[a]) for a in fits.index])
    d = d[(d.date >= after) & (d.date < before)].sort_values("date")
    best, keep = -np.inf, []
    for day, g in d.groupby("date", sort=True):
        best = max(best, g.p50.max())
        keep += list(g.index[g.p50 >= best])
    return keep


def doubling_days(p50s, dates):
    t = np.array([(pd.Timestamp(d) - pd.Timestamp("2019-01-01")).days for d in dates], float)
    slope = np.polyfit(t, np.log2(np.asarray(p50s, float)), 1)[0]
    return 1 / slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="time-horizon-1-1")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "out"))
    ap.add_argument("--root", default=str(ROOT), help="eval-analysis-public checkout (e.g. an older commit)")
    a = ap.parse_args()

    root = pathlib.Path(a.root)
    runs = pd.read_json(root / f"reports/{a.report}/data/raw/runs.jsonl", lines=True)
    runs = runs[runs.alias != "human"]
    dates = yaml.safe_load(open(root / "data/external/release_dates.yaml"))["date"]

    fits = fit_all(runs, a.lam)
    boot = HierBootstrap(runs)
    t0 = time.time()
    samples = []
    for i in range(a.n_boot):
        rng = np.random.default_rng(1000 + i)
        samples.append(fit_all(boot.sample(rng), a.lam)["p50"])
    B = pd.DataFrame(samples)
    print(f"bootstrap {a.n_boot} x {len(fits)} agents in {time.time() - t0:.0f}s")

    fits["p50_lo"], fits["p50_hi"] = B.quantile(0.025), B.quantile(0.975)
    fits["release"] = [dates.get(x) for x in fits.index]
    print(fits.sort_values("p50", ascending=False)[["release", "p50", "p50_lo", "p50_hi", "p80"]]
          .round(1).to_string())

    trends = {}
    windows = {"2019+": ("2019-01-01", "2030-01-01"), "2023+": ("2023-01-01", "2030-01-01"),
               "2024+": ("2024-01-01", "2030-01-01"),
               "paper-era 2019..2025-02-25": ("2019-01-01", "2025-02-25")}
    dated = fits[fits.release.notna()]
    for name, (lo, hi) in windows.items():
        s = sota_agents(dated, dates, pd.Timestamp(lo), pd.Timestamp(hi))
        if len(s) < 2:
            continue
        point = doubling_days(dated.loc[s, "p50"], [dates[x] for x in s])
        bd = []
        for _, row in B[s].iterrows():
            ok = row.notna() & np.isfinite(row) & (row > 1e-3)
            if ok.sum() >= 2:
                dt = doubling_days(row[ok], [dates[x] for x in row[ok].index])
                if dt > 0:
                    bd.append(dt)
        trends[name] = {"sota": s, "point_days": point, "median_days": float(np.median(bd)),
                        "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
        print(f"{name}: doubling {point:.1f} d (boot median {np.median(bd):.1f}, "
              f"95% CI {np.percentile(bd, 2.5):.0f}-{np.percentile(bd, 97.5):.0f}); SOTA={s}")

    out = pathlib.Path(a.out) / a.report
    out.mkdir(parents=True, exist_ok=True)
    fits.to_csv(out / "fits.csv")
    B.to_csv(out / "bootstrap_p50.csv", index=False)
    json.dump(trends, open(out / "trends.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
