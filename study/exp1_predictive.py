"""E1: out-of-sample comparison of METR's per-agent logistic (B1) against IRT models.

Protocol A  (known task, unseen cell)   5-fold CV over (agent, task) cells.
Protocol B  (unseen task family)        5-fold CV over task families; IRT must predict with u integrated out.
Protocol C  (new agent, m tasks seen)   leave-one-agent-out; reveal m tasks, estimate the agent, predict the rest
                                        and recover METR's full-data 50% horizon.
All fits use the proper (unweighted) likelihood except where stated (D4, D11).
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

from irt import data, metrics, models, plotstyle
from irt.plotstyle import INK2, MUTED, SLOTS, plt

OUT = pathlib.Path(__file__).parent
RHOS = (1.0, 3.0, 10.0, 30.0)
RANKS = (1, 2, 3)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
def fit_suite(Ctr, with_factors=True, seed=0):
    """All models on one training set. Returns {name: (fit, how)} where `how` is the predict mode."""
    b0 = models.fit_map(Ctr, "none")
    b2 = models.fit_map(Ctr, "common")
    b1 = models.fit_map(Ctr, "agent")
    b1p = models.fit_map(Ctr, "agent", weights="metr", lam=1e-5)       # METR's exact estimator
    r = models.fit_mml(Ctr, "none", init=b0, tau0=3.0)
    l = models.fit_mml(Ctr, "common", init=b2)
    m = models.fit_mml(Ctr, "agent", init=b1)
    suite = {"B0 agent-only": b0, "B2 common slope": b2, "B1 METR (MLE)": b1, "B1 METR (paper wts)": b1p,
             "R Rasch": r, "L LLTM-R": l, "M agent-slope LLTM-R": m}
    if with_factors:
        f0 = models.fit_map(Ctr, "agent", tau=m.tau, init=m)
        suite["F0 = M as MAP plug-in"] = f0
        rng = np.random.default_rng(seed)
        inner = rng.random(len(Ctr.df)) < 0.1                        # inner validation for (K, rho)
        Cin, Cval = Ctr.subset(~inner), Ctr.subset(inner)
        for K in RANKS:
            scores = {}
            for rho in RHOS:
                f = models.fit_map(Cin, "agent", tau=m.tau, rank=K, rho=rho, init=m, seed=seed)
                p = f.predict(Cval.df)
                scores[rho] = metrics.summarise(Cval.df, p)["nll"]
            rho = min(scores, key=scores.get)
            suite[f"F{K} rank-{K} factors"] = models.fit_map(Ctr, "agent", tau=m.tau, rank=K, rho=rho, init=m,
                                                             seed=seed)
            suite[f"F{K} rank-{K} factors"].info["rho"] = rho
            suite[f"F{K} rank-{K} factors"].info["inner_nll"] = scores
    return suite


def predict(fit, df, how):
    return fit.predict(df, how=how)


# ------------------------------------------------------------------------------------------------
def protocol_A(C, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(np.arange(len(C.df)) % folds)
    preds = {}
    meta = []
    for f in range(folds):
        log(f"A fold {f}")
        tr, te = fold != f, fold == f
        S = fit_suite(C.subset(tr), with_factors=True, seed=seed + f)
        te_df = C.df[te]
        for name, fit in S.items():
            preds.setdefault(name, np.zeros(len(C.df)))[te] = fit.predict(te_df, "posterior")
            meta.append({"fold": f, "model": name, "tau": fit.tau, "rho": fit.info.get("rho"),
                         "converged": fit.info["converged"]})
    return preds, pd.DataFrame(meta)


def protocol_B(C, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fams = C.df.family.unique()
    fam_fold = dict(zip(fams, rng.permutation(np.arange(len(fams)) % folds)))
    fold = C.df.family.map(fam_fold).values
    preds = {}
    for f in range(folds):
        log(f"B fold {f}")
        tr, te = fold != f, fold == f
        S = fit_suite(C.subset(tr), with_factors=False)
        te_df = C.df[te]
        for name, fit in S.items():
            if fit.log_post is not None:
                preds.setdefault(name + " [marginal]", np.zeros(len(C.df)))[te] = fit.predict(te_df, "marginal")
                if name.startswith("M"):
                    preds.setdefault(name + " [plug-in u=0]", np.zeros(len(C.df)))[te] = fit.predict(te_df, "fixed")
            else:
                preds.setdefault(name, np.zeros(len(C.df)))[te] = fit.predict(te_df, "fixed")
    return preds


def score(C, preds, ref):
    rows = []
    for name, p in preds.items():
        s = metrics.summarise(C.df, p)
        sw = metrics.summarise(C.df, p, weight_col="w_metr")
        d = metrics.paired_family_bootstrap(C.df, p, preds[ref])
        rows.append({"model": name, **s, "nll_w": sw["nll"], "auroc_w": sw["auroc"],
                     "dNLL_vs_ref": d["diff"], "dNLL_lo": d["lo"], "dNLL_hi": d["hi"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# Protocol C helpers: fit one new agent's (alpha, beta) with a Gaussian prior learned from the other agents.
def _gauss_prior(alpha, beta):
    X = np.column_stack([alpha, beta])
    mu = X.mean(0)
    S = np.cov(X.T) + 1e-3 * np.eye(2)
    return mu, np.linalg.inv(S)


def fit_new_agent(df, w, xbar, prior, log_post=None, tau=0.0):
    """MAP for (alpha, beta). With log_post (T,Q): likelihood integrates each revealed task's u over its
    posterior from the other agents (IRT); otherwise plain logistic (METR)."""
    n, k, xc = df.n.values.astype(float), df.k.values.astype(float), df.x.values - xbar
    mu, Pm = prior if prior is not None else (np.zeros(2), np.zeros((2, 2)))
    lp = log_post[df.t.values] if log_post is not None else None
    uq = tau * models.Z_GRID

    def fun(th):
        eta = th[0] + th[1] * xc
        if lp is None:
            ll = w * (k * log_expit(eta) + (n - k) * log_expit(-eta))
            g = w * (k - n * expit(eta))
            obj = -ll.sum()
        else:
            E = eta[:, None] + uq[None, :]
            L = lp + w[:, None] * (k[:, None] * log_expit(E) + (n - k)[:, None] * log_expit(-E))
            lse = logsumexp(L, axis=1)
            rq = np.exp(L - lse[:, None])
            g = np.sum(rq * (w[:, None] * (k[:, None] - n[:, None] * expit(E))), axis=1)
            obj = -lse.sum()
        d = th - mu
        obj += 0.5 * d @ Pm @ d
        grad = -np.array([g.sum(), (g * xc).sum()]) + Pm @ d
        return obj, grad

    return minimize(fun, mu.copy(), jac=True, method="L-BFGS-B").x


def protocol_C(C, ms=(8, 16, 32, 64), reps=10, seed=0):
    rows = []
    full = models.fit_map(C, "agent", weights="metr", lam=1e-5)
    target = np.log2(full.horizons(C.agents)["cond_p50"])
    # Each method's own full-data estimate, to separate sampling error from between-estimator offset (D14).
    m_full = models.fit_mml(C, "agent", weights="count", init=models.fit_map(C, "agent", weights="count"))
    own = {"B1": target, "M": np.log2(m_full.horizons(C.agents)["cond_p50"])}
    for ai, agent in enumerate(C.agents):
        log(f"C agent {ai} {agent}")
        others = C.subset(C.df.a.values != ai)
        b1o = models.fit_map(others, "agent")
        mo = models.fit_mml(others, "agent", init=b1o)
        keep = np.arange(C.A) != ai
        prior_b1 = _gauss_prior(b1o.alpha[keep], b1o.beta[keep])
        prior_m = _gauss_prior(mo.alpha[keep], mo.beta[keep])
        mine = C.df[C.df.a == ai]
        rng = np.random.default_rng(seed + ai)
        for m in ms:
            for r in range(reps):
                rev = np.zeros(len(mine), bool)
                rev[rng.choice(len(mine), size=m, replace=False)] = True
                dr, dh = mine[rev], mine[~rev]
                wr = dr.w_count.values
                est = {
                    "B1 METR, flat prior": fit_new_agent(dr, wr, C.xbar, None),
                    "B1 METR, population prior": fit_new_agent(dr, wr, C.xbar, prior_b1),
                    "M IRT, population prior": fit_new_agent(dr, wr, C.xbar, prior_m, mo.log_post, mo.tau),
                }
                for name, (al, be) in est.items():
                    eta = al + be * (dh.x.values - C.xbar)
                    if name.startswith("M"):
                        p = np.exp(logsumexp(mo.log_post[dh.t.values] + log_expit(eta[:, None] + mo.tau * models.Z_GRID[None, :]), axis=1))
                    else:
                        p = expit(eta)
                    h = C.xbar - al / be if be < 0 else np.nan     # log2 p50 (conditional = marginal, T5)
                    rows.append({"agent": agent, "m": m, "rep": r, "method": name,
                                 "nll_hidden": metrics.summarise(dh, p)["nll"],
                                 "log2_p50": h, "abs_err_log2_p50": abs(h - target[ai]) if np.isfinite(h) else np.nan,
                                 "abs_err_own_target": abs(h - own[name[0:2].strip()][ai]) if np.isfinite(h) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
def figures(tag, tabA, tabB, dfC, predsA, C):
    plotstyle.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, tab, title in [(axes[0], tabA, "A: known task, unseen cell"), (axes[1], tabB, "B: unseen task family")]:
        t = tab.sort_values("dNLL_vs_ref", ascending=False).reset_index(drop=True)
        y = np.arange(len(t))
        for i, r in t.iterrows():
            c = SLOTS[0] if r.dNLL_hi < 0 else (SLOTS[7] if r.dNLL_lo > 0 else MUTED)
            ax.plot([r.dNLL_lo, r.dNLL_hi], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.plot(r.dNLL_vs_ref, i, "o", color=c, ms=5, mec="white", mew=0.8)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y, t.model)
        ax.set_xlabel("ΔNLL per run vs METR (paper wts)\n← better        worse →")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / f"figures/e1_nll_{tag}.pdf"); fig.savefig(OUT / f"figures/e1_nll_{tag}.png")

    # Reliability diagram for protocol A.
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    for name, c, mk in [("B1 METR (paper wts)", SLOTS[1], "s"), ("M agent-slope LLTM-R", SLOTS[0], "o")]:
        p = predsA[name]
        bins = np.minimum((p * 10).astype(int), 9)
        n, k = C.df.n.values, C.df.k.values
        xs = [np.sum(n[bins == b] * p[bins == b]) / n[bins == b].sum() for b in range(10) if (bins == b).any()]
        ys = [k[bins == b].sum() / n[bins == b].sum() for b in range(10) if (bins == b).any()]
        ax.plot(xs, ys, marker=mk, color=c, label=name, ms=5, mec="white", mew=0.8)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("predicted P(success)"); ax.set_ylabel("observed success rate")
    ax.set_title("Calibration, protocol A", loc="left"); ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"figures/e1_calibration_{tag}.pdf"); fig.savefig(OUT / f"figures/e1_calibration_{tag}.png")

    # Protocol C.
    g = dfC.groupby(["method", "m"]).agg(err=("abs_err_log2_p50", "median"),
                                         q25=("abs_err_log2_p50", lambda v: np.nanpercentile(v, 25)),
                                         q75=("abs_err_log2_p50", lambda v: np.nanpercentile(v, 75)),
                                         own=("abs_err_own_target", "median"),
                                         nll=("nll_hidden", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for (meth, gg), c, mk in zip(g.groupby("method", sort=False), [SLOTS[1], SLOTS[2], SLOTS[0]], ["^", "s", "o"]):
        axes[0].plot(gg.m, gg.err, marker=mk, color=c, label=meth, mec="white", mew=0.8)
        axes[0].fill_between(gg.m, gg.q25, gg.q75, color=c, alpha=0.12, lw=0)
        axes[1].plot(gg.m, gg.nll, marker=mk, color=c, label=meth, mec="white", mew=0.8)
        if meth.startswith("M"):
            axes[0].plot(gg.m, gg.own, ls="--", lw=1.5, color=c, label="M IRT vs its own full-data p50")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.set_xlabel("tasks revealed for the new agent ($m$)")
    axes[0].set_ylabel("|log$_2$ error| of 50% horizon\n(median, IQR band)")
    axes[0].set_title("Horizon recovery (target: METR full-data p50)", loc="left", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=6.5)
    axes[1].set_ylabel("NLL per run on hidden tasks"); axes[1].set_title("Prediction of hidden tasks", loc="left")
    axes[1].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / f"figures/e1_new_agent_{tag}.pdf"); fig.savefig(OUT / f"figures/e1_new_agent_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="time-horizon-1-1")
    ap.add_argument("--protocols", default="ABC")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    tag = "th11" if a.report.endswith("1-1") else "th10"
    C = data.load(a.report)
    ref = "B1 METR (paper wts)"
    res = {}
    if "A" in a.protocols:
        predsA, metaA = protocol_A(C)
        tabA = score(C, predsA, ref)
        tabA.to_csv(OUT / f"results/e1_A_{tag}.csv", index=False)
        metaA.to_csv(OUT / f"results/e1_A_meta_{tag}.csv", index=False)
        np.savez(OUT / f"results/e1_A_preds_{tag}.npz", **{k.replace(" ", "_"): v for k, v in predsA.items()})
        print(tabA.round(4).to_string(index=False))
    if "B" in a.protocols:
        predsB = protocol_B(C)
        tabB = score(C, predsB, ref)
        tabB.to_csv(OUT / f"results/e1_B_{tag}.csv", index=False)
        print(tabB.round(4).to_string(index=False))
    if "C" in a.protocols:
        dfC = protocol_C(C, reps=a.reps)
        dfC.to_csv(OUT / f"results/e1_C_{tag}.csv", index=False)
        print(dfC.groupby(["m", "method"])[["abs_err_log2_p50", "abs_err_own_target", "nll_hidden"]].median().round(3).to_string())
    if "A" not in a.protocols:
        tabA = pd.read_csv(OUT / f"results/e1_A_{tag}.csv")
        z = np.load(OUT / f"results/e1_A_preds_{tag}.npz")
        predsA = {k.replace("_", " "): z[k] for k in z.files}
    if "B" not in a.protocols:
        tabB = pd.read_csv(OUT / f"results/e1_B_{tag}.csv")
    if "C" not in a.protocols:
        if not (OUT / f"results/e1_C_{tag}.csv").exists():
            return                                   # replication runs (A, B only) skip the combined figures
        dfC = pd.read_csv(OUT / f"results/e1_C_{tag}.csv")
    figures(tag, tabA, tabB, dfC, predsA, C)


if __name__ == "__main__":
    main()
