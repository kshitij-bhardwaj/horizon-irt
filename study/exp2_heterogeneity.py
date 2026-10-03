"""E2: how much of task difficulty is *not* explained by human task length?

Fits the nested family (D5) on all data with the proper (unweighted) likelihood (D4) and reports:
  (a) log-likelihood / AIC / BIC per model and likelihood-ratio tests between nested pairs;
  (b) tau-hat with a profile-likelihood 95% CI;
  (c) a latent-scale variance decomposition of task difficulty into length vs residual;
  (d) the tasks that deviate most from what their length predicts (posterior means of u_t);
  (e) a dispersion check: observed vs model-expected share of cells with mixed outcomes.
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from irt import data, models, plotstyle
from irt.plotstyle import INK2, MUTED, SOURCE_COLOR, plt

OUT = pathlib.Path(__file__).parent


def fit_family(C):
    b0 = models.fit_map(C, "none", name="B0 agent-only")
    b2 = models.fit_map(C, "common", name="B2 common slope")
    b1 = models.fit_map(C, "agent", name="B1 METR (agent slopes)")
    r = models.fit_mml(C, "none", init=b0, tau0=3.0, name="R Rasch")
    l = models.fit_mml(C, "common", init=b2, name="L LLTM-R")
    m = models.fit_mml(C, "agent", init=b1, name="M agent-slope LLTM-R")
    return {"B0": b0, "B2": b2, "B1": b1, "R": r, "L": l, "M": m}


def lrt(f_small, f_big, df, boundary=False):
    stat = max(0.0, 2 * (f_big.loglik - f_small.loglik))
    p = stats.chi2.sf(stat, df)
    if boundary:   # tau = 0 is on the boundary: null is 0.5 chi2_0 + 0.5 chi2_1 (Self & Liang 1987)
        p = 0.5 * stats.chi2.sf(stat, 1)
    return stat, p


def profile_tau(C, m, grid):
    ll = []
    for tau in grid:
        f = models.fit_mml(C, "agent", init=m, tau_fixed=tau)
        ll.append(f.loglik)
    ll = np.array(ll)
    inside = grid[ll >= m.loglik - 0.5 * stats.chi2.ppf(0.95, 1)]
    return ll, (float(inside.min()), float(inside.max()))


def mixed_cells(C, fit):
    """Observed vs expected fraction of multi-run cells whose runs disagree.
    Expected uses the task posterior: E_u[1 - s^n - (1-s)^n]."""
    df = C.df[C.df.n >= 2]
    e0 = fit.eta0(df.a.values, df.x.values)
    S = expit(e0[:, None] + fit.tau * models.Z_GRID[None, :])
    n = df.n.values[:, None]
    pm = 1 - S**n - (1 - S) ** n
    exp_mixed = np.einsum("cq,cq->c", np.exp(fit.log_post[df.t.values]), pm)
    obs = ((df.k > 0) & (df.k < df.n)).values
    z = (obs.sum() - exp_mixed.sum()) / np.sqrt(np.sum(exp_mixed * (1 - exp_mixed)))
    return {"cells": int(len(df)), "observed": float(obs.mean()), "expected": float(exp_mixed.mean()), "z": float(z)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="time-horizon-1-1")
    a = ap.parse_args()
    tag = "th11" if a.report.endswith("1-1") else "th10"
    C = data.load(a.report)
    F = fit_family(C)
    n_runs = int(C.df.n.sum())

    rows = []
    for key, f in F.items():
        rows.append({"model": f.model, "k_params": f.n_params, "loglik": f.loglik,
                     "AIC": 2 * f.n_params - 2 * f.loglik, "BIC": np.log(n_runs) * f.n_params - 2 * f.loglik,
                     "tau": f.tau, "converged": f.info["converged"]})
    table = pd.DataFrame(rows)
    A = C.A
    tests = {
        "B2 vs B1 (agent slopes; df=A-1)": lrt(F["B2"], F["B1"], A - 1),
        "B1 vs M (tau>0; boundary)": lrt(F["B1"], F["M"], 1, boundary=True),
        "B2 vs L (tau>0; boundary)": lrt(F["B2"], F["L"], 1, boundary=True),
        "L vs M (agent slopes given tau; df=A-1)": lrt(F["L"], F["M"], A - 1),
        "R vs L (length given tau; df=1)": lrt(F["R"], F["L"], 1),
    }

    m, l, r = F["M"], F["L"], F["R"]
    grid = np.exp(np.linspace(np.log(m.tau * 0.6), np.log(m.tau * 1.6), 21))
    prof_ll, tau_ci = profile_tau(C, m, grid)

    # Latent-scale decomposition of task difficulty under L: d_t = beta (x_t - xbar) + u_t.
    beta_L = float(l.beta[0])
    var_len = beta_L**2 * np.var(C.tasks.x.values)
    decomp = {"beta_L": beta_L, "var_length": var_len, "tau2_L": l.tau**2,
              "R2_length_latent": var_len / (var_len + l.tau**2),
              "R2_length_vs_rasch": 1 - l.tau**2 / r.tau**2,
              "icc_task_given_length": l.tau**2 / (l.tau**2 + np.pi**2 / 3)}

    # Per-task posterior summaries from M.
    post = np.exp(m.log_post)
    uq = m.tau * models.Z_GRID
    sd = np.sqrt(post @ uq**2 - m.u**2)
    solved = C.df.groupby("t").apply(lambda g: g.k.sum() / g.n.sum(), include_groups=False)
    tasks = C.tasks.assign(u_mean=m.u, u_sd=sd, solve_rate=solved.reindex(range(C.T)).values)
    tasks.to_csv(OUT / f"results/e2_task_effects_{tag}.csv", index=False)
    by_source = tasks.groupby("source").agg(n=("u_mean", "size"), mean_u=("u_mean", "mean"),
                                            se=("u_mean", lambda v: v.std(ddof=1) / np.sqrt(len(v))))

    disp = {k: mixed_cells(C, F[k]) for k in ["M"]}
    disp["B1 (tau=0)"] = mixed_cells(C, models.Fit(**{**F["B1"].__dict__, "tau": 1e-9,
                                                       "log_post": np.tile(models.LOG_V, (C.T, 1))}))

    res = {"report": a.report, "n_runs": n_runs, "n_cells": len(C.df), "A": C.A, "T": C.T,
           "models": table.to_dict("records"),
           "lrt": {k: {"stat": s, "p": p} for k, (s, p) in tests.items()},
           "tau_M": m.tau, "tau_M_profile_ci95": tau_ci, "tau_L": l.tau, "tau_R": r.tau,
           "decomposition": decomp, "u_by_source": by_source.reset_index().to_dict("records"),
           "mixed_cells": disp,
           "slopes_B1": dict(zip(C.agents, F["B1"].beta)), "slopes_M": dict(zip(C.agents, m.beta))}
    json.dump(res, open(OUT / f"results/e2_{tag}.json", "w"), indent=2, default=float)

    print(table.round(2).to_string(index=False))
    for k, (s, p) in tests.items():
        print(f"LRT {k}: stat={s:.1f} p={p:.2e}")
    print(f"tau_M={m.tau:.3f} CI95 {tau_ci}; tau_L={l.tau:.3f}; tau_R={r.tau:.3f}")
    print(json.dumps(decomp, indent=1))
    print(by_source.round(3))
    print("mixed cells:", disp)
    print("most harder-than-length:\n", tasks.nsmallest(8, "u_mean")[["task_id", "source", "minutes", "solve_rate", "u_mean", "u_sd"]].round(2).to_string(index=False))
    print("most easier-than-length:\n", tasks.nlargest(8, "u_mean")[["task_id", "source", "minutes", "solve_rate", "u_mean", "u_sd"]].round(2).to_string(index=False))

    figures(C, F, tasks, grid, prof_ll, tau_ci, tag)


def figures(C, F, tasks, grid, prof_ll, tau_ci, tag):
    plotstyle.setup()
    m, l = F["M"], F["L"]
    # Fig E2a: effective difficulty vs length. Effective log-difficulty in "length units" = x_t - u_t / |beta_L|.
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
    markers = {"HCAST": "o", "SWAA": "s", "RE-Bench": "D"}
    for src, g in tasks.groupby("source"):
        ax[0].errorbar(g.x, g.u_mean, yerr=1.96 * g.u_sd, fmt=markers[src], ms=3.5, color=SOURCE_COLOR[src],
                       ecolor=SOURCE_COLOR[src], elinewidth=0.5, alpha=0.85, label=f"{src} (n={len(g)})")
    ax[0].axhline(0, color=MUTED, lw=1)
    ax[0].set_xlabel("log$_2$ human minutes  $x_t$")
    ax[0].set_ylabel("task effect $\\hat u_t$ (logit; + = easier)")
    ax[0].set_title(f"Residual difficulty beyond length (model M, $\\hat\\tau$={m.tau:.2f})", loc="left")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.2, columnspacing=1.0)
    ax[1].plot(grid, prof_ll - prof_ll.max(), color=plotstyle.SLOTS[0], marker="o", ms=3)
    ax[1].axhline(-0.5 * stats.chi2.ppf(0.95, 1), color=MUTED, lw=1, ls="--")
    ax[1].axvspan(*tau_ci, color=plotstyle.SLOTS[0], alpha=0.08)
    ax[1].text(np.mean(tau_ci), 0.85 * prof_ll.min() + 0.15 * prof_ll.max() - prof_ll.max(), "95% CI", color=INK2, fontsize=8, ha="center")
    ax[1].set_xlabel("$\\tau$")
    ax[1].set_ylabel("profile log-lik. (relative)")
    ax[1].set_title("Profile likelihood of $\\tau$", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"figures/e2_task_effects_{tag}.pdf")
    fig.savefig(OUT / f"figures/e2_task_effects_{tag}.png")

    # Fig E2b: per-agent slope estimates, B1 vs M (conditional slopes are steeper: attenuation).
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    b1, bm = F["B1"].beta, m.beta
    ax.scatter(-b1, -bm, s=22, color=plotstyle.SLOTS[0], edgecolor="white", linewidth=0.8, zorder=3)
    lim = [0, (-b1).max() * 1.15]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="equal")
    c = 16 * np.sqrt(3) / (15 * np.pi)
    ax.plot(lim, np.array(lim) * np.sqrt(1 + c**2 * m.tau**2), color=plotstyle.SLOTS[1], lw=1.5,
            label=f"$\\sqrt{{1+c^2\\tau^2}}$ = {np.sqrt(1 + c**2 * m.tau**2):.2f}")
    ax.set_xlim(lim); ax.set_ylim(0, (-bm).max() * 1.15)
    ax.set_xlabel("$-\\beta_a$  METR (marginal)")
    ax.set_ylabel("$-\\beta_a$  model M (conditional)")
    ax.set_title("Slope attenuation", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"figures/e2_slopes_{tag}.pdf")
    fig.savefig(OUT / f"figures/e2_slopes_{tag}.png")


if __name__ == "__main__":
    main()
