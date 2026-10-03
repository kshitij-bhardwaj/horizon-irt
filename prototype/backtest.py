"""Replay back-test of HorizonCAT on METR TH1.1 (P11).

For each agent a: build the bank WITHOUT a (leave-one-agent-out), then replay a's real runs: when a policy asks
for task t, the outcome is one of a's actual runs on t drawn uniformly (= Bernoulli(k_at / n_at)).
All policies share the same Bayesian estimator; only task selection differs (P7).
Reference values: the agent's h from model M fitted on ALL data (h_ref_M; the estimand), and METR's p50.
"""

import json
import pathlib
import time

import numpy as np
import pandas as pd

from horizoncat import bank as hb, engine
from horizoncat.bank import sdata, smodels

OUT = pathlib.Path(__file__).parent / "results"
POLICIES = [("random", lambda b, r: engine.RandomPolicy(b, r)),
            ("stratified", lambda b, r: engine.StratifiedPolicy(b, r)),
            ("midpass", lambda b, r: engine.MidPassFilterPolicy(b, r)),
            ("info", lambda b, r: engine.InfoPolicy(b, r, gamma=0.0)),
            ("cost05", lambda b, r: engine.InfoPolicy(b, r, gamma=0.5)),
            ("cost1", lambda b, r: engine.InfoPolicy(b, r, gamma=1.0))]


def main(reps=20, max_tasks=80, sigma_v=0.0):
    OUT.mkdir(exist_ok=True)
    C = sdata.load()
    runs, _ = sdata.load_runs()
    full_m = smodels.fit_mml(C, "agent", init=smodels.fit_map(C, "agent"))
    metr = smodels.fit_map(C, "agent", weights="metr", lam=1e-5)
    rows, full_rows = [], []
    t0 = time.time()
    for ai, agent in enumerate(C.agents):
        bank, _ = hb.build(C, runs, exclude_agent=agent)
        grid = engine.make_grid(bank.mu_b, bank.sd_b)
        Pm = engine.predictive_matrix(bank.x, bank.u, grid, sigma_v=sigma_v)
        cells = C.df[C.df.a == ai].set_index("t")
        rate = (cells.k / cells.n).to_dict()
        allowed = sorted(rate)
        h_ref = float(C.xbar - full_m.alpha[ai] / full_m.beta[ai])
        h_metr = float(np.log2(metr.horizon(ai, 0.5)))
        # Full suite once: every available task, one run each (order irrelevant for the posterior).
        for r in range(reps):
            rng = np.random.default_rng(10_000 * ai + r)
            st = engine.State(grid.log_prior.copy())
            for t in allowed:
                engine.update(st, Pm, t, int(rng.random() < rate[t]))
            w = st.post()
            m, v = engine.h_moments(w, grid)
            lo, _, hi = engine.h_quantiles(w, grid)
            full_rows.append({"agent": agent, "rep": r, "tokens": float(bank.cost[allowed].sum()),
                              "tasks": len(allowed), "h_mean": m, "h_sd": np.sqrt(v), "h_lo": lo, "h_hi": hi,
                              "h_ref": h_ref, "h_metr": h_metr})
        for pname, mk in POLICIES:
            for r in range(reps):
                rng = np.random.default_rng(10_000 * ai + 100 * r + 7)
                respond = lambda t: int(rng.random() < rate[t])
                tr, stop, _ = engine.run(bank, grid, Pm, mk(bank, rng), respond, target_sd=0.0,
                                         max_tasks=max_tasks, allowed=allowed)
                for row in tr:
                    rows.append({"agent": agent, "policy": pname, "rep": r, "stop": stop, "h_ref": h_ref,
                                 "h_metr": h_metr, **row})
        print(f"[{time.time() - t0:.0f}s] {ai + 1}/{C.A} {agent}  h_ref={h_ref:.2f}", flush=True)
    T = pd.DataFrame(rows)
    F = pd.DataFrame(full_rows)
    T.to_csv(OUT / "backtest_traces.csv.gz", index=False)
    F.to_csv(OUT / "backtest_fullsuite.csv", index=False)
    summarise(T, F)


def summarise(T, F, target=0.5):
    first = (T[T.h_sd <= target].sort_values("step").groupby(["agent", "policy", "rep"]).head(1)
             .set_index(["agent", "policy", "rep"]))
    runs = T.groupby(["agent", "policy", "rep"]).tail(1).set_index(["agent", "policy", "rep"])
    reached = first.reindex(runs.index)
    full_tok = F.groupby("agent").tokens.first()
    S = pd.DataFrame({
        "reached": reached.h_sd.notna(),
        "tokens": reached.tokens, "tasks": reached.step,
        "abs_err_ref": (reached.h_mean - reached.h_ref).abs(),
        "abs_err_metr": (reached.h_mean - reached.h_metr).abs(),
        # Coverage is only defined where the target was reached (P20: the first version counted unreached runs as
        # misses, which reported reach_rate x coverage).
        "covers_ref": ((reached.h_lo <= reached.h_ref) & (reached.h_ref <= reached.h_hi)).where(reached.h_sd.notna()),
        "final_sd": runs.h_sd, "final_stop": runs.stop,
        "final_covers_ref": (runs.h_lo <= runs.h_ref) & (runs.h_ref <= runs.h_hi),
    }).reset_index()
    S["frac_full_tokens"] = S.tokens / S.agent.map(full_tok)
    S.to_csv(OUT / "backtest_per_run.csv", index=False)
    g = S.groupby("policy")
    tab = pd.DataFrame({
        "reach_rate": g.reached.mean(),
        "median_tasks": g.tasks.median(),
        "median_tokens_M": g.tokens.median() / 1e6,
        "median_frac_full_tokens": g.frac_full_tokens.median(),
        "median_abs_err_ref": g.abs_err_ref.median(),
        "coverage95_ref": g.covers_ref.mean(),          # among runs that reached the target (NaN skipped)
        "final_coverage95_ref": g.final_covers_ref.mean(),  # every run, at its last step
    })
    Fs = F.assign(abs_err=(F.h_mean - F.h_ref).abs(), covers=(F.h_lo <= F.h_ref) & (F.h_ref <= F.h_hi))
    full = {"median_tokens_M": float(Fs.tokens.median() / 1e6), "median_sd": float(Fs.h_sd.median()),
            "median_abs_err_ref": float(Fs.abs_err.median()), "coverage95_ref": float(Fs.covers.mean())}
    tab.to_csv(OUT / "backtest_summary.csv")
    json.dump({"full_suite": full}, open(OUT / "backtest_fullsuite_summary.json", "w"), indent=2)
    print(tab.round(3).to_string())
    print("full suite:", full)
    print(S[S.policy == "cost1"].groupby("agent").agg(reach=("reached", "mean"), frac=("frac_full_tokens", "median"),
          stop=("final_stop", lambda s: s.mode()[0])).round(3).to_string())


if __name__ == "__main__":
    import sys
    if "--sigma-v" in sys.argv:
        sv = float(sys.argv[sys.argv.index("--sigma-v") + 1])
    else:
        sv = json.load(open(OUT / "calibrate_sigma_v.json"))["best_sigma_v"] if (OUT / "calibrate_sigma_v.json").exists() else 0.0
    print("sigma_v =", sv, flush=True)
    if "--summarise" in sys.argv:
        summarise(pd.read_csv(OUT / "backtest_traces.csv.gz"), pd.read_csv(OUT / "backtest_fullsuite.csv"))
    else:
        main(sigma_v=sv)
