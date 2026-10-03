"""E3b: is METR's 50% horizon biased when task difficulty varies beyond length?

Simulate from the fitted model M on the real design (same agents, tasks, run counts, weights) with fresh task
effects u_t ~ N(0, tau^2) each replicate, so the true p50 of every agent is known exactly:
    log2 H50_true = xbar - alpha_a / beta_a        (conditional = marginal at p = 1/2, T5)
Then apply METR's estimator (per-agent weighted logistic, lam = 1e-5) and M's own estimator.
Bias(METR) != 0 with Bias(M) ~ 0 means the offset seen in E3 is structural, not a quirk of the realised tasks.
A second arm fixes u_t at its posterior means from the real data ("realised tasks") to quantify that component.
"""

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.special import expit

from irt import data, models, plotstyle, trend
from irt.data import Cells
from irt.plotstyle import INK2, MUTED, SLOTS, plt

OUT = pathlib.Path(__file__).parent


def sim_cells(C, alpha, beta, u, rng):
    df = C.df.copy()
    eta = alpha[df.a.values] + beta[df.a.values] * (df.x.values - C.xbar) + u[df.t.values]
    df["k"] = rng.binomial(df.n.values, expit(eta))
    return Cells(df, C.agents, C.tasks, C.xbar, C.dates)


def main(reps=60, fit_m_every=3):
    C = data.load()
    b1c = models.fit_map(C, "agent", weights="count")
    m = models.fit_mml(C, "agent", weights="count", init=b1c)
    true = C.xbar - m.alpha / m.beta
    rows = []
    for arm in ["fresh u", "realised u (posterior mean)"]:
        for r in range(reps):
            rng = np.random.default_rng(500 + r)
            u = m.tau * rng.standard_normal(C.T) if arm == "fresh u" else m.u
            S = sim_cells(C, m.alpha, m.beta, u, rng)
            f = models.fit_map(S, "agent", weights="metr", lam=1e-5)
            est = {"METR": np.log2(f.horizons(C.agents, ps=(0.5,))["cond_p50"])}
            if arm == "fresh u" and r % fit_m_every == 0:
                fm = models.fit_mml(S, "agent", weights="count", init=m)
                est["M"] = np.log2(fm.horizons(C.agents, ps=(0.5,))["cond_p50"])
            for name, h in est.items():
                rows += [{"arm": arm, "rep": r, "est": name, "agent": ag, "err_log2": h[i] - true[i],
                          "true_log2": true[i]} for i, ag in enumerate(C.agents)]
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "results/e3b_bias_sim_th11.csv", index=False)
    S = R.groupby(["arm", "est", "agent"]).err_log2.agg(["mean", "std", "count"]).reset_index()
    S["se"] = S["std"] / np.sqrt(S["count"])
    S["true_log2"] = S.agent.map(dict(zip(C.agents, true)))
    print(S.sort_values(["arm", "est", "true_log2"]).round(3).to_string(index=False))

    # Effect on the doubling time: METR estimates under 'fresh u' vs truth, same SOTA set (D13).
    sota_ref = dict(zip(C.agents, 2 ** true))
    full_metr = models.fit_map(C, "agent", weights="metr", lam=1e-5)
    sota_ref = dict(zip(C.agents, full_metr.horizons(C.agents, ps=(0.5,))["cond_p50"]))
    td = {}
    for w, after in {"2023+": "2023-01-01", "2024+": "2024-01-01"}.items():
        s = trend.sota_agents(sota_ref, C.dates, after)
        truth = trend.doubling_days(dict(zip(C.agents, 2 ** true)), C.dates, s)
        sims = []
        for r, g in R[(R.arm == "fresh u") & (R.est == "METR")].groupby("rep"):
            h = dict(zip(g.agent, 2 ** (g.err_log2 + g.true_log2)))
            sims.append(trend.doubling_days(h, C.dates, s))
        td[w] = {"truth": truth, "metr_mean": float(np.mean(sims)), "metr_sd": float(np.std(sims)),
                 "metr_q": [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))]}
    print(json.dumps(td, indent=1))
    corr = {arm: float(np.corrcoef(*S[(S.arm == arm) & (S.est == "METR")][["true_log2", "mean"]].values.T)[0, 1])
            for arm in S.arm.unique()}
    json.dump({"tau": m.tau, "doubling": td, "corr_bias_vs_true_log2": corr},
              open(OUT / "results/e3b_th11.json", "w"), indent=2)
    print("corr(bias, true log2 p50):", corr)

    plotstyle.setup()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (arm, est), c, mk in zip([("fresh u", "METR"), ("realised u (posterior mean)", "METR"), ("fresh u", "M")],
                                 [SLOTS[1], SLOTS[2], SLOTS[0]], ["s", "D", "o"]):
        g = S[(S.arm == arm) & (S.est == est)].sort_values("true_log2")
        ax.errorbar(g.true_log2, g["mean"], yerr=1.96 * g.se, fmt=mk, color=c, ms=4, elinewidth=0.8,
                    mec="white", mew=0.5, label=f"{est} estimator, {arm}")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("true log$_2$ p50 (minutes)")
    ax.set_ylabel("bias of estimated log$_2$ p50")
    ax.set_title("METR's p50 under task heterogeneity (simulated from M)", loc="left", fontsize=9)
    ax.legend(loc="best", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "figures/e3b_bias_th11.pdf"); fig.savefig(OUT / "figures/e3b_bias_th11.png")


if __name__ == "__main__":
    main()
