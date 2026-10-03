"""E4: stability of METR's estimator (direction 2).

(a) Ridge sweep. Per agent, METR minimises J = sum w l(alpha + beta x) + lam/2 beta^2, sum w = 1. To first order the
    penalty rotates the fitted curve about the agent's information-weighted pivot
        xv_a = sum w p(1-p) x / sum w p(1-p),     I_a = sum w p(1-p) (x - xv_a)^2   (profile Fisher info of beta)
    with beta_lam ~= beta_0 * I_a / (I_a + lam), hence
        log2 H50(lam) - xv_a ~= (log2 H50(0) - xv_a) * (1 + lam / I_a).
    Horizons are pushed *away* from the pivot: up for strong agents, down for weak ones -> trend steepens.
(b) Weighting sweep: invsqrt (METR), equal-task, per-run (none).
(c) Slope prior: ridge-to-zero vs hierarchical (shrink to the population slope), judged by unseen-family NLL (CV)
    and by bootstrap CI width of the horizons.
(d) Failure case: the frontier agent whose 50% point lies where tasks are scarce (Claude Opus 4.6).
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.special import expit

from irt import data, metrics, models, plotstyle, trend
from irt.bootstrap import HierBootstrap
from irt.plotstyle import INK2, MUTED, SLOTS, plt

OUT = pathlib.Path(__file__).parent
LAMS = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
KAPPAS = [1, 10, 30, 100, 300, 1000, 3000, 10000]   # kappa -> inf is the common-slope model B2


def b1(C, lam=1e-5, weights="metr", kappa=None):
    if kappa is not None:
        return models.fit_map(C, "agent", weights=weights, hier_kappa=kappa)
    return models.fit_map(C, "agent", weights=weights, lam=lam)


def p50s(C, fit):
    return dict(zip(C.agents, fit.horizons(C.agents, ps=(0.5,))["cond_p50"]))


def doubling(h, dates, sota_ref):
    out = {}
    for w, after in {"2023+": "2023-01-01", "2024+": "2024-01-01"}.items():
        s = trend.sota_agents(sota_ref, dates, after)
        out[w] = trend.doubling_days(h, dates, s)
    return out


def pivot_info(C, fit, weights="metr"):
    """Information-weighted pivot xv_a and profile Fisher information I_a of beta at the fitted curve."""
    df = C.df
    w = C.weights(weights) * df.n.values
    p = expit(fit.eta0(df.a.values, df.x.values))
    v = w * p * (1 - p)
    xv = np.bincount(df.a, v * df.x, minlength=C.A) / np.bincount(df.a, v, minlength=C.A)
    I = np.bincount(df.a, v * (df.x - xv[df.a]) ** 2, minlength=C.A)
    return xv, I


def ridge_sweep(C):
    f0 = b1(C, 0.0)
    xv, I = pivot_info(C, f0)
    h0 = np.log2(f0.horizons(C.agents, ps=(0.5,))["cond_p50"])
    rows, trends = [], []
    ref = p50s(C, b1(C, 1e-5))
    for lam in LAMS:
        f = b1(C, lam)
        h = np.log2(f.horizons(C.agents, ps=(0.5,))["cond_p50"])
        pred = xv + (h0 - xv) * (1 + lam / I)
        for i, ag in enumerate(C.agents):
            rows.append({"lam": lam, "agent": ag, "log2_p50": h[i], "pred_first_order": pred[i],
                         "pivot": xv[i], "I": I[i], "beta": f.beta[i]})
        trends.append({"lam": lam, **doubling(dict(zip(C.agents, 2 ** h)), C.dates, ref)})
    return pd.DataFrame(rows), pd.DataFrame(trends)


def weighting_sweep(C):
    ref = p50s(C, b1(C))
    rows = []
    for wmode in ["metr", "equal", "none"]:
        f = b1(C, 1e-5 if wmode != "none" else 1e-5 * 1e3, weights=wmode)   # keep lam/(sum of weights) comparable
        h = p50s(C, f)
        rows.append({"weights": wmode, **doubling(h, C.dates, ref),
                     **{f"p50 {a}": h[a] for a in C.agents}})
    return pd.DataFrame(rows)


def prior_cv(C, folds=5, seed=0):
    """Unseen-family NLL for ridge (lam) vs hierarchical (kappa) slope priors, count-scale weights for both."""
    rng = np.random.default_rng(seed)
    fams = C.df.family.unique()
    fold = C.df.family.map(dict(zip(fams, rng.permutation(np.arange(len(fams)) % folds)))).values
    configs = [("ridge", l) for l in [1e-5, 1e-2, 0.1, 1, 10, 100, 1000]] + [("hier", k) for k in KAPPAS]
    preds = {c: np.zeros(len(C.df)) for c in configs}
    for f in range(folds):
        tr = C.subset(fold != f)
        te = C.df[fold == f]
        for kind, v in configs:
            fit = models.fit_map(tr, "agent", weights="count", **({"lam": v} if kind == "ridge" else {"hier_kappa": v}))
            preds[(kind, v)][fold == f] = fit.predict(te, "fixed")
    rows = []
    base = preds[("ridge", 1e-5)]
    for (kind, v), p in preds.items():
        s = metrics.summarise(C.df, p, weight_col="w_metr")
        d = metrics.paired_family_bootstrap(C.df, p, base, weight_col="w_metr")
        rows.append({"prior": kind, "strength": v, "nll_w": s["nll"], "dNLL_w": d["diff"], "lo": d["lo"], "hi": d["hi"]})
    return pd.DataFrame(rows)


def bootstrap_priors(runs, dates, kappa, n_boot=300):
    """Paired bootstrap of horizons and doubling times: METR (count-scale ridge ~0) vs hierarchical kappa."""
    boot = HierBootstrap(runs)
    rows, tr = [], []
    C = data.aggregate(runs, dates)
    ref = p50s(C, b1(C))
    for b in range(n_boot):
        Cb = data.aggregate(boot.sample(np.random.default_rng(20_000 + b)), dates)
        for name, fit in [("METR", b1(Cb, 1e-5)), ("hier", b1(Cb, weights="count", kappa=kappa))]:
            h = p50s(Cb, fit)
            rows += [{"boot": b, "est": name, "agent": a, "p50": v} for a, v in h.items()]
            tr.append({"boot": b, "est": name, **doubling(h, dates, ref)})
    return pd.DataFrame(rows), pd.DataFrame(tr)


def influence(C, agent):
    """Leave-one-family-out jackknife of log2 p50 for one agent (METR estimator)."""
    i = C.agents.index(agent)
    full = np.log2(b1(C).horizon(i, 0.5))
    rows = []
    for fam in C.df.family.unique():
        sub = C.subset(C.df.family.values != fam)
        h = np.log2(b1(sub).horizon(i, 0.5))
        g = C.df[(C.df.family == fam) & (C.df.a == i)]
        rows.append({"family": fam, "delta_log2_p50": h - full, "max_minutes": g.minutes.max(),
                     "agent_success": g.k.sum() / max(g.n.sum(), 1)})
    J = pd.DataFrame(rows)
    nf = len(J)
    se = np.sqrt((nf - 1) / nf * np.sum((J.delta_log2_p50 - J.delta_log2_p50.mean()) ** 2))
    return J.sort_values("delta_log2_p50", key=abs, ascending=False), full, se


def support_table(C):
    """For each agent: share of its (METR-weighted) task mass above its own 50% horizon, and #tasks above it."""
    f = b1(C)
    rows = []
    for i, ag in enumerate(C.agents):
        g = C.df[C.df.a == i]
        h = np.log2(f.horizon(i, 0.5))
        above = g.x > h
        rows.append({"agent": ag, "log2_p50": h, "tasks_above": int(above.sum()),
                     "weight_above": float((g.w_metr * g.n)[above].sum() / (g.w_metr * g.n).sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="time-horizon-1-1")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--failure-agent", default="Claude Opus 4.6 (Inspect)")
    a = ap.parse_args()
    tag = "th11" if a.report.endswith("1-1") else "th10"
    runs, dates = data.load_runs(a.report)
    C = data.aggregate(runs, dates)
    res = {}

    R, Tr = ridge_sweep(C)
    R.to_csv(OUT / f"results/e4a_ridge_{tag}.csv", index=False); Tr.to_csv(OUT / f"results/e4a_ridge_trend_{tag}.csv", index=False)
    print(Tr.round(1).to_string(index=False))
    err = (R.log2_p50 - R.pred_first_order).abs()
    res["ridge_first_order_max_abs_err_log2"] = R.assign(err=err).groupby("lam").err.max().to_dict()
    print("first-order prediction max |err| (log2) by lam:", res["ridge_first_order_max_abs_err_log2"])

    W = weighting_sweep(C)
    W.to_csv(OUT / f"results/e4b_weights_{tag}.csv", index=False)
    print(W[["weights", "2023+", "2024+"]].round(1).to_string(index=False))

    P = prior_cv(C)
    P.to_csv(OUT / f"results/e4c_prior_cv_{tag}.csv", index=False)
    print(P.round(4).to_string(index=False))
    kappa = float(P[P.prior == "hier"].sort_values("nll_w").strength.iloc[0])
    res["kappa_selected"] = kappa

    HB, TB = bootstrap_priors(runs, dates, kappa, a.n_boot)
    HB.to_csv(OUT / f"results/e4c_boot_{tag}.csv", index=False); TB.to_csv(OUT / f"results/e4c_boot_trend_{tag}.csv", index=False)
    full = {"METR": p50s(C, b1(C)), "hier": p50s(C, b1(C, weights="count", kappa=kappa))}
    ci = HB.groupby(["est", "agent"]).p50.quantile([0.025, 0.975]).unstack()
    ci["width_log2"] = np.log2(ci[0.975] / ci[0.025])
    ci["point"] = [full[e][ag] for e, ag in ci.index]
    ci.to_csv(OUT / f"results/e4c_ci_{tag}.csv")
    print(ci.round(1).to_string())
    tq = TB.groupby("est")[["2023+", "2024+"]].quantile([0.025, 0.5, 0.975])
    print(tq.round(1).to_string())
    res["trend_ci_by_prior"] = {f"{e}|{q}": v for (e, q), v in tq.to_dict("index").items()}
    res["doubling_point_by_prior"] = {e: doubling(full[e], dates, full["METR"]) for e in full}

    J, full_h, se = influence(C, a.failure_agent)
    J.to_csv(OUT / f"results/e4d_influence_{tag}.csv", index=False)
    S = support_table(C)
    S.to_csv(OUT / f"results/e4d_support_{tag}.csv", index=False)
    res["failure"] = {"agent": a.failure_agent, "log2_p50": full_h, "jackknife_se_log2": se,
                      "top_influence": J.head(8).to_dict("records")}
    print(J.head(10).round(3).to_string(index=False))
    print(f"{a.failure_agent}: log2 p50 {full_h:.2f} ({2 ** full_h:.0f} min), jackknife SE {se:.2f} log2")
    print(S.sort_values("log2_p50").round(3).to_string(index=False))
    json.dump(res, open(OUT / f"results/e4_{tag}.json", "w"), indent=2, default=float)
    figures(C, R, ci, S, J, a.failure_agent, kappa, tag)


def figures(C, R, ci, S, J, fail, kappa, tag):
    plotstyle.setup()
    # E4a: predicted vs actual shift.
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    base = R[R.lam == 1e-5].set_index("agent").log2_p50
    lab = {0.01: SLOTS[2], 0.1: SLOTS[0], 0.3: SLOTS[1]}
    for lam, c in lab.items():
        g = R[R.lam == lam].set_index("agent")
        ax[0].scatter(g.pred_first_order - base, g.log2_p50 - base, s=16, color=c, edgecolor="white",
                      linewidth=0.5, label=f"λ = {lam}", zorder=3)
    lim = np.abs(R.log2_p50 - R.agent.map(base)).max() * 1.1
    ax[0].plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--")
    ax[0].set_xlabel("predicted Δlog$_2$ p50 (first-order)"); ax[0].set_ylabel("actual Δlog$_2$ p50")
    ax[0].set_title("Ridge rotates curves about a pivot", loc="left"); ax[0].legend(loc="upper left")
    g = R[R.lam == 0.1].set_index("agent")
    ax[1].scatter(base - g["pivot"], g.log2_p50 - base, s=16, color=SLOTS[0], edgecolor="white", linewidth=0.5, zorder=3)
    ax[1].axhline(0, color=MUTED, lw=1); ax[1].axvline(0, color=MUTED, lw=1)
    for ag in g.index:
        if ag in (fail, "GPT-4 0314", "Claude Opus 4.5 (Inspect)"):
            ax[1].annotate(ag.replace(" (Inspect)", ""), (base[ag] - g["pivot"][ag], g.log2_p50[ag] - base[ag]),
                           fontsize=7, color=INK2, xytext=(4, -10), textcoords="offset points")
    ax[1].set_xlabel("log$_2$ p50 − pivot $\\bar x_{v,a}$"); ax[1].set_ylabel("Δlog$_2$ p50 at λ = 0.1")
    ax[1].set_title("Strong agents up, weak agents down", loc="left")
    fig.tight_layout(); fig.savefig(OUT / f"figures/e4a_ridge_{tag}.pdf"); fig.savefig(OUT / f"figures/e4a_ridge_{tag}.png")

    # E4c/d: CI width vs support above horizon.
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    m = ci.loc["METR"].join(S.set_index("agent"))
    hh = ci.loc["hier"]
    for ag in m.index:
        ax[0].plot([m.weight_above[ag]] * 2, [m.width_log2[ag], hh.width_log2[ag]], color=MUTED, lw=0.8, zorder=1)
    ax[0].scatter(m.weight_above, m.width_log2, s=18, color=SLOTS[1], marker="s", edgecolor="white", linewidth=0.5,
                  label="METR (ridge≈0)", zorder=3)
    ax[0].scatter(m.weight_above, hh.reindex(m.index).width_log2, s=18, color=SLOTS[0], edgecolor="white",
                  linewidth=0.5, label=f"hierarchical slope (κ={kappa:g})", zorder=3)
    ax[0].annotate(fail.replace(" (Inspect)", ""), (m.weight_above[fail], m.width_log2[fail]), fontsize=7,
                   color=INK2, xytext=(5, -3), textcoords="offset points")
    ax[0].set_xlabel("share of task weight longer than the agent's p50")
    ax[0].set_ylabel("95% CI width of p50 (log$_2$ units)")
    ax[0].set_title("Uncertainty grows as support vanishes", loc="left"); ax[0].legend(loc="upper right", fontsize=7)
    i = C.agents.index(fail)
    g = C.df[C.df.a == i]
    edges = np.arange(np.floor(g.x.min()), np.ceil(g.x.max()) + 1.5, 1.5)
    b = np.digitize(g.x, edges)
    xs = [g.x[b == k].mean() for k in np.unique(b)]
    ys = [g.k[b == k].sum() / g.n[b == k].sum() for k in np.unique(b)]
    ns = [g.n[b == k].sum() for k in np.unique(b)]
    ax[1].scatter(xs, ys, s=np.array(ns) * 0.8, color=MUTED, alpha=0.5, label="binned success (area ∝ runs)")
    xx = np.linspace(g.x.min(), g.x.max() + 3, 200)
    for lam, c in [(1e-5, SLOTS[1]), (0.1, SLOTS[0])]:
        f = b1(C, lam)  # noqa: B007
        ax[1].plot(xx, expit(f.alpha[i] + f.beta[i] * (xx - C.xbar)), color=c, label=f"METR fit, λ={lam:g}")
    fh = b1(C, weights="count", kappa=kappa)
    ax[1].plot(xx, expit(fh.alpha[i] + fh.beta[i] * (xx - C.xbar)), color=SLOTS[2], ls="--",
               label=f"hierarchical slope, κ={kappa:g}")
    ax[1].axhline(0.5, color=MUTED, lw=1, ls="--")
    ax[1].set_xlabel("log$_2$ human minutes"); ax[1].set_ylabel("P(success)")
    ax[1].set_title(f"Failure case: {fail.replace(' (Inspect)', '')}", loc="left"); ax[1].legend(loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"figures/e4cd_uncertainty_{tag}.pdf"); fig.savefig(OUT / f"figures/e4cd_uncertainty_{tag}.png")


if __name__ == "__main__":
    main()
