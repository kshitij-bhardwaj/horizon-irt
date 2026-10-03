"""Frontier failure cases and suite design (P10, P12).

1. Measurement range: SE(h) of the current suite (one run per task) across horizons, from the test-information
   function. Marks Opus 4.6 (public runs) and Claude Mythos Preview (METR headline only, 1045 min).
2. Hypothetical agents at 12 h, 24 h, 48 h, 1 week: run the cost-aware CAT on responses simulated from the bank
   model -> does it saturate?  Then add N new tasks of length L (difficulty unknown: u ~ N(0, tau^2)) and repeat.
3. Design table: new tasks needed at each length L for SE(h*) <= 0.5, and their token cost (extrapolated, flagged).
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from horizoncat import bank as hb, design, engine
from horizoncat.bank import sdata

OUT = pathlib.Path(__file__).parent / "results"
LOG2 = lambda minutes: float(np.log2(minutes))
_cal = OUT / "calibrate_sigma_v.json"
SIGMA_V = json.load(open(_cal))["best_sigma_v"] if _cal.exists() else 0.0
HYPOTHETICAL = {"12 h": LOG2(720), "24 h": LOG2(1440), "48 h": LOG2(2880), "1 week": LOG2(10080)}
NEW_LENGTHS = {"8 h": LOG2(480), "16 h": LOG2(960), "32 h": LOG2(1920), "64 h": LOG2(3840), "1 week": LOG2(10080)}


def extend_bank(bank, L, n, tau):
    """Append n hypothetical tasks of log2-length L whose difficulty is unknown: u ~ N(0, tau^2), represented by
    the K equal-mass prior quantile means (same compression as P3)."""
    K = bank.u.shape[1]
    edges = norm.ppf(np.linspace(0, 1, K + 1))
    # E[Z | Z in (a, b)] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
    zq = (norm.pdf(edges[:-1]) - norm.pdf(edges[1:])) / (1.0 / K)
    new_u = np.tile(tau * zq, (n, 1))
    cost_fn, _ = design.token_cost_model(bank.x[bank.x >= 4], bank.cost[bank.x >= 4])
    b = hb.Bank(task_ids=bank.task_ids + [f"new_{i}" for i in range(n)], x=np.r_[bank.x, np.full(n, L)],
                u=np.vstack([bank.u, new_u]), cost=np.r_[bank.cost, np.full(n, cost_fn(L))],
                pass_rate=np.r_[bank.pass_rate, np.full(n, np.nan)], source=bank.source + ["new"] * n,
                family=bank.family + ["new"] * n, mu_b=bank.mu_b, sd_b=bank.sd_b, tau=bank.tau, agents=bank.agents)
    return b


def simulate_cat(bank, h_true, reps=30, seed=0, max_tasks=None, gamma=1.0, sigma_v=0.0):
    grid = engine.make_grid(bank.mu_b, bank.sd_b)
    Pm = engine.predictive_matrix(bank.x, bank.u, grid, sigma_v=sigma_v)
    out = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        beta = bank.mu_b + bank.sd_b * rng.standard_normal()
        u_true = bank.u[np.arange(len(bank.x)), rng.integers(0, bank.u.shape[1], len(bank.x))]
        u_true = u_true + sigma_v * rng.standard_normal(len(bank.x))      # agent-by-task interaction (P18)
        respond = lambda t: int(rng.random() < expit(beta * (bank.x[t] - h_true) + u_true[t]))
        tr, stop, _ = engine.run(bank, grid, Pm, engine.InfoPolicy(bank, rng, gamma), respond, target_sd=0.5,
                                 max_tasks=max_tasks or len(bank.x))
        last = tr[-1]
        out.append({"stop": stop, "sd": last["h_sd"], "tokens": last["tokens"], "tasks": last["step"],
                    "covered": last["h_lo"] <= h_true <= last["h_hi"], "p_beyond": last["p_beyond_suite"],
                    "abs_err": abs(last["h_med"] - h_true)})
    return pd.DataFrame(out)


def main():
    C = sdata.load()
    runs, _ = sdata.load_runs()
    bank, m = hb.build(C, runs)
    beta = bank.mu_b
    hs = np.arange(0, 14.01, 0.05)
    sd_beta = 1.5 * bank.sd_b                                   # same slope prior as the CAT grid (P5)
    I_bank = design.info_matrix_bank(hs, beta, bank.x, bank.u)
    se = design.se_h(I_bank, sd_beta)                           # slope estimated (P16)
    se_known_slope = 1 / np.sqrt(I_bank[:, 0, 0])
    rng_measurable = hs[se <= 0.5]
    res = {"mu_b": beta, "tau": bank.tau, "longest_task_log2": float(bank.x.max()),
           "measurable_range_log2": [float(rng_measurable.min()), float(rng_measurable.max())],
           "measurable_range_minutes": [float(2 ** rng_measurable.min()), float(2 ** rng_measurable.max())]}
    curve = pd.DataFrame({"h": hs, "se_one_run_per_task": se, "se_known_slope": se_known_slope})

    # Design table.
    cost_fn, cm = design.token_cost_model(bank.x[bank.x >= 4], bank.cost[bank.x >= 4])
    rows = []
    for hn, h in HYPOTHETICAL.items():
        for ln, L in NEW_LENGTHS.items():
            n = design.tasks_needed(h, beta, sd_beta, bank.x, bank.u, L, bank.tau)
            rows.append({"horizon": hn, "new_task_length": ln, "tasks_needed": n,
                         "tokens_per_run_extrap": cost_fn(L), "total_tokens_extrap": n * cost_fn(L),
                         "extrapolated_cost": L > bank.x.max()})
    D = pd.DataFrame(rows)
    res["cost_model"] = cm

    # Curves for an extended suite: +20 tasks at 32 h, +20 at 1 week.
    for label, L, n in [("+20 × 32 h", NEW_LENGTHS["32 h"], 20), ("+20 × 1 week", NEW_LENGTHS["1 week"], 20)]:
        curve[f"se {label}"] = design.se_h(I_bank + n * design.info_matrix_new(hs, beta, L, bank.tau), sd_beta)

    # CAT on hypothetical agents, current vs extended suite (reuses saved sims unless --resim).
    sims = []
    cached = OUT / "frontier_cat_sims.csv"
    for hn, h in ({} if (cached.exists() and "--resim" not in sys.argv) else HYPOTHETICAL).items():
        for label, b in [("current suite", bank), ("+20 tasks at 1.5× horizon", extend_bank(bank, h + LOG2(1.5), 20, bank.tau))]:
          for gamma in (1.0, 0.0):
            S = simulate_cat(b, h, gamma=gamma, sigma_v=SIGMA_V)
            sims.append({"horizon": hn, "suite": label, "gamma": gamma, "precision_reached": float((S.stop == "precision").mean()),
                         "saturated": float((S.stop == "saturated").mean()), "median_sd": float(S.sd.median()),
                         "coverage95": float(S.covered.mean()), "median_abs_err": float(S.abs_err.median()),
                         "median_tokens_M": float(S.tokens.median() / 1e6)})
    Sims = pd.DataFrame(sims) if sims else pd.read_csv(cached)

    # Opus 4.6 (real runs, leave-one-out bank): replay the cost-aware CAT.
    ai = C.agents.index("Claude Opus 4.6 (Inspect)")
    res["opus46_se_design"] = float(design.se_h(design.info_matrix_bank(np.array([10.04]), beta, bank.x, bank.u), sd_beta)[0])
    b46, _ = hb.build(C, runs, exclude_agent=C.agents[ai])
    grid = engine.make_grid(b46.mu_b, b46.sd_b)
    Pm = engine.predictive_matrix(b46.x, b46.u, grid, sigma_v=SIGMA_V)
    cells = C.df[C.df.a == ai].set_index("t")
    rate = (cells.k / cells.n).to_dict()
    o46 = []
    for r in range(30):
        rng = np.random.default_rng(900 + r)
        tr, stop, _ = engine.run(b46, grid, Pm, engine.InfoPolicy(b46, rng, 1.0),
                                 lambda t: int(rng.random() < rate[t]), target_sd=0.5, max_tasks=120,
                                 allowed=sorted(rate))
        o46.append({"stop": stop, "sd": tr[-1]["h_sd"], "lo": tr[-1]["h_lo"], "hi": tr[-1]["h_hi"],
                    "p_beyond": tr[-1]["p_beyond_suite"], "tokens": tr[-1]["tokens"]})
    O = pd.DataFrame(o46)
    res["opus46_replay"] = {"stops": O.stop.value_counts().to_dict(), "median_sd": float(O.sd.median()),
                            "median_ci_minutes": [float(2 ** O.lo.median()), float(2 ** O.hi.median())],
                            "median_p_beyond": float(O.p_beyond.median())}

    curve.to_csv(OUT / "frontier_se_curve.csv", index=False)
    D.to_csv(OUT / "frontier_design.csv", index=False)
    Sims.to_csv(OUT / "frontier_cat_sims.csv", index=False)
    json.dump(res, open(OUT / "frontier.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
    print(D.pivot(index="horizon", columns="new_task_length", values="tasks_needed").to_string())
    print(Sims.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
