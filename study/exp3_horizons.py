"""E3: what does modelling task heterogeneity do to the reported horizons and the doubling trend?

Theory (tested in T5, checked here on real fits):
  * With u ~ N(0, tau^2) symmetric, E_u s(eta0 + u) = 1/2 exactly where eta0 = 0, so the 50% horizon of the
    conditional model (typical task, u = 0) equals its population-averaged (marginal) 50% horizon.
  * Away from 1/2 the marginal curve is flatter by ~ sqrt(1 + c^2 tau^2), c = 16 sqrt(3) / (15 pi)
    (Zeger, Liang & Albert 1988), so the gap between p50 and p80 horizons is ~ that factor larger, in log units,
    for the marginal curve than for a typical task.  METR's p80 is a marginal quantity.
Both estimators are refitted on the same hierarchical bootstrap samples (paired), D12.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from irt import data, models, plotstyle, trend
from irt.bootstrap import HierBootstrap
from irt.plotstyle import MUTED, SLOTS, plt

OUT = pathlib.Path(__file__).parent
C_ZLA = 16 * np.sqrt(3) / (15 * np.pi)
WINDOWS = {"2023+": "2023-01-01", "2024+": "2024-01-01"}


def estimate(C, init_b1=None, init_m=None):
    b1 = models.fit_map(C, "agent", weights="metr", lam=1e-5)
    b1c = models.fit_map(C, "agent", weights="count", init=init_b1)
    m = models.fit_mml(C, "agent", weights="count", init=init_m or b1c)
    out = {}
    for i, ag in enumerate(C.agents):
        out[ag] = {"B1_p50": b1.horizon(i, .5), "B1_p80": b1.horizon(i, .8),
                   "M_p50": m.horizon(i, .5), "M_p80_cond": m.horizon(i, .8, "conditional"),
                   "M_p80_marg": m.horizon(i, .8, "marginal")}
    return pd.DataFrame(out).T, b1c, m


def trends(H, dates, cols):
    res = {}
    for col in cols:
        h = H[col].to_dict()
        for w, after in WINDOWS.items():
            s = trend.sota_agents(H["B1_p50"].to_dict(), dates, after)   # same agent set for all columns (D13)
            res[f"{col} {w}"] = trend.doubling_days(h, dates, s)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="time-horizon-1-1")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--from-saved", action="store_true", help="summarise saved bootstrap samples, skip refitting")
    a = ap.parse_args()
    tag = "th11" if a.report.endswith("1-1") else "th10"
    runs, dates = data.load_runs(a.report)
    C = data.aggregate(runs, dates)
    H, b1c, m = estimate(C)
    cols = ["B1_p50", "M_p50", "B1_p80", "M_p80_cond", "M_p80_marg"]
    point = trends(H, dates, cols)
    H["ratio50_80_B1"] = H.B1_p50 / H.B1_p80
    H["ratio50_80_M_cond"] = H.M_p50 / H.M_p80_cond
    H["ratio50_80_M_marg"] = H.M_p50 / H.M_p80_marg
    s = np.sqrt(1 + C_ZLA**2 * m.tau**2)
    H["ZLA_pred_marg_ratio"] = H.ratio50_80_M_cond ** s
    print(f"tau (count-weighted pseudo-MML) = {m.tau:.3f}; ZLA attenuation factor s = {s:.3f}")
    print(H.sort_values("B1_p50", ascending=False).round(2).to_string())
    print(json.dumps(point, indent=1))

    if a.from_saved:
        B = pd.read_csv(OUT / f"results/e3_boot_horizons_{tag}.csv")
        T = pd.read_csv(OUT / f"results/e3_boot_trends_{tag}.csv")
        return summarise(H, B, T, m, s, point, cols, tag)
    boot = HierBootstrap(runs)
    rows, tr = [], []
    t0 = time.time()
    for b in range(a.n_boot):
        rng = np.random.default_rng(10_000 + b)
        Cb = data.aggregate(boot.sample(rng), dates)
        Hb, _, _ = estimate(Cb, init_b1=b1c, init_m=m)
        Hb["boot"] = b
        rows.append(Hb.reset_index(names="agent"))
        tr.append(trends(Hb, dates, cols))
        if b % 20 == 0:
            print(f"boot {b} ({time.time() - t0:.0f}s)", flush=True)
    B = pd.concat(rows)
    T = pd.DataFrame(tr)
    B.to_csv(OUT / f"results/e3_boot_horizons_{tag}.csv", index=False)
    T.to_csv(OUT / f"results/e3_boot_trends_{tag}.csv", index=False)
    summarise(H, B, T, m, s, point, cols, tag)


def summarise(H, B, T, m, s, point, cols, tag):
    ci = B.groupby("agent")[cols].quantile([0.025, 0.975]).unstack()
    flat = ci.copy()
    flat.columns = [f"{c}_q{q}" for c, q in ci.columns]
    H.join(flat).to_csv(OUT / f"results/e3_horizons_{tag}.csv")
    trend_tab = pd.DataFrame({"point": point, "lo": T.quantile(0.025), "hi": T.quantile(0.975),
                              "median": T.median()})
    # Paired differences of doubling times (same bootstrap samples).
    paired = {f"M_p50 - B1_p50 {w}": T[f"M_p50 {w}"] - T[f"B1_p50 {w}"] for w in WINDOWS}
    paired_tab = pd.DataFrame({k: {"median": v.median(), "lo": v.quantile(.025), "hi": v.quantile(.975)}
                               for k, v in paired.items()}).T
    trend_tab.to_csv(OUT / f"results/e3_trends_{tag}.csv")
    print(trend_tab.round(1).to_string())
    print(paired_tab.round(2).to_string())
    json.dump({"tau_count_weighted": m.tau, "zla_factor": s, "point": point,
               "paired_doubling_diff": paired_tab.to_dict("index")},
              open(OUT / f"results/e3_{tag}.json", "w"), indent=2, default=float)
    figure(H, ci, tag, s)


def figure(H, ci, tag, s):
    plotstyle.setup()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    order = H.sort_values("B1_p50").index
    y = np.arange(len(order))
    for col, c, mk, lab, dy in [("B1_p80", SLOTS[1], "s", "METR p80 (marginal)", -0.15),
                                ("M_p80_cond", SLOTS[0], "o", "M p80, typical task (u=0)", 0.15)]:
        lo, hi = ci[(col, 0.025)].reindex(order), ci[(col, 0.975)].reindex(order)
        ax[0].hlines(y + dy, lo, hi, color=c, lw=1.2, alpha=0.6)
        ax[0].plot(H.loc[order, col], y + dy, mk, color=c, ms=4, mec="white", mew=0.6, label=lab)
    ax[0].plot(H.loc[order, "B1_p50"], y, "|", color=MUTED, ms=8, mew=1.5, label="METR p50")
    ax[0].set_xscale("log")
    ax[0].set_yticks(y, [a.replace(" (Inspect)", "") for a in order], fontsize=6.5)
    ax[0].set_xlabel("horizon (human minutes, log scale)")
    ax[0].set_title("80% horizons: marginal vs typical task", loc="left")
    ax[0].legend(loc="lower right", fontsize=7)
    ax[0].grid(axis="y", visible=False)
    ax[1].scatter(np.log2(H.ratio50_80_M_cond), np.log2(H.ratio50_80_M_marg), s=22, color=SLOTS[0],
                  edgecolor="white", linewidth=0.6, zorder=3, label="M marginal (tests theory)")
    ax[1].scatter(np.log2(H.ratio50_80_M_cond), np.log2(H.ratio50_80_B1), s=18, color=SLOTS[1], marker="s",
                  edgecolor="white", linewidth=0.6, zorder=3, label="METR estimator")
    xx = np.linspace(0, np.log2(H.ratio50_80_M_cond).max() * 1.1, 10)
    ax[1].plot(xx, s * xx, color=SLOTS[1], lw=1.5, label=f"theory: slope s = {s:.2f}")
    ax[1].plot(xx, xx, color=MUTED, lw=1, ls="--", label="no attenuation")
    ax[1].set_xlabel("log$_2$(p50 / p80), typical task (M)")
    ax[1].set_ylabel("log$_2$(p50 / p80), population-averaged")
    ax[1].set_title("p50→p80 gap is stretched by s", loc="left")
    ax[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"figures/e3_p80_{tag}.pdf"); fig.savefig(OUT / f"figures/e3_p80_{tag}.png")


if __name__ == "__main__":
    main()
