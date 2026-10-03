"""Export everything the interactive page needs into one JSON (P13).

Contents: shared task metadata; one leave-one-agent-out bank per agent (u points, slope prior, tau); a bank on all
agents (hypothetical mode, audit); replay rates k/n per (agent, task); reference horizons; back-test curves;
frontier/design results; task-audit table with flags (P14); a Python reference trace for JS parity (P15).
"""

import json
import pathlib

import numpy as np
import pandas as pd

from horizoncat import bank as hb, design, engine
from horizoncat.bank import sdata, smodels

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"


def audit(C, bank_all, m_all):
    """Per-task audit (P14). Flags are rules on the bank posterior and solve rates, explained in the page."""
    post = np.exp(m_all.log_post)
    uq = m_all.tau * smodels.Z_GRID
    u_mean = post @ uq
    u_sd = np.sqrt(np.maximum(post @ uq**2 - u_mean**2, 0))
    solve = C.df.groupby("t").apply(lambda g: g.k.sum() / g.n.sum(), include_groups=False).reindex(range(C.T)).values
    n_agents_solved = C.df.assign(s=C.df.k > 0).groupby("t").s.sum().reindex(range(C.T)).values
    frontier_h = float(np.log2(720))                 # information for a 12-hour-horizon agent
    I_front = np.array([design.info_bank(np.array([frontier_h]), bank_all.mu_b, bank_all.x[[t]], bank_all.u[[t]])[0]
                        for t in range(C.T)])
    tau = m_all.tau
    rows = []
    for t in range(C.T):
        flags = []
        mins = float(C.tasks.minutes[t])
        if solve[t] == 0 and mins <= 60:
            flags.append("unsolved-short")
        if u_mean[t] < -tau and solve[t] > 0:
            flags.append("harder-than-length")
        if u_mean[t] > tau:
            flags.append("easier-than-length")
        if solve[t] in (0.0, 1.0):
            flags.append("no-variation")
        rows.append({"id": C.tasks.task_id[t], "family": C.tasks.family[t], "source": C.tasks.source[t],
                     "minutes": round(mins, 3), "solve": round(float(solve[t]), 3),
                     "agents_solved": int(n_agents_solved[t]), "u": round(float(u_mean[t]), 2),
                     "u_sd": round(float(u_sd[t]), 2), "info12h": round(float(I_front[t]), 4),
                     "tokens": int(bank_all.cost[t]), "flags": flags})
    return rows


def backtest_curves(T, F):
    """Median SD(h) and median |h - h_ref| as a function of tokens spent, per policy, relative to full suite."""
    full = F.groupby("agent").tokens.first()
    T = T.assign(frac=T.tokens / T.agent.map(full))
    grid = np.geomspace(1e-4, 1.0, 40)
    out = {}
    for pol, g in T.groupby("policy"):
        sd_med, err_med = [], []
        for f in grid:
            last = g[g.frac <= f].groupby(["agent", "rep"]).tail(1)
            n_runs = g.groupby(["agent", "rep"]).ngroups
            if len(last) < 0.5 * n_runs:
                sd_med.append(None); err_med.append(None); continue
            sd_med.append(round(float(last.h_sd.median()), 4))
            err_med.append(round(float((last.h_mean - last.h_ref).abs().median()), 4))
        out[pol] = {"frac": grid.round(6).tolist(), "sd": sd_med, "err": err_med}
    return out


def main():
    C = sdata.load()
    runs, _ = sdata.load_runs()
    bank_all, m_all = hb.build(C, runs)
    banks = {}
    for ag in C.agents:
        b, _ = hb.build(C, runs, exclude_agent=ag)
        banks[ag] = {"u": np.round(b.u, 3).tolist(), "mu_b": round(b.mu_b, 4), "sd_b": round(b.sd_b, 4),
                     "tau": round(b.tau, 4)}
        print("bank", ag, flush=True)
    full_m = smodels.fit_mml(C, "agent", init=smodels.fit_map(C, "agent"))
    metr = smodels.fit_map(C, "agent", weights="metr", lam=1e-5)
    rates = {}
    for ai, ag in enumerate(C.agents):
        g = C.df[C.df.a == ai]
        rates[ag] = {int(t): [int(k), int(n)] for t, k, n in zip(g.t, g.k, g.n)}
    agents = [{"name": ag, "release": C.dates.get(ag), "h_ref": round(float(C.xbar - full_m.alpha[i] / full_m.beta[i]), 3),
               "h_metr": round(float(np.log2(metr.horizon(i, 0.5))), 3)} for i, ag in enumerate(C.agents)]

    T = pd.read_csv(RES / "backtest_traces.csv.gz")
    F = pd.read_csv(RES / "backtest_fullsuite.csv")
    summary = pd.read_csv(RES / "backtest_summary.csv").to_dict("records")
    v1 = pd.read_csv(RES / "v1" / "backtest_summary.csv").set_index("policy")
    v1["coverage95_ref"] = pd.read_csv(RES / "v1" / "coverage_corrected.csv").set_index("policy").coverage_reached  # P20
    summary_v1 = v1.reset_index().to_dict("records")
    full_v1 = json.load(open(RES / "v1" / "backtest_fullsuite_summary.json"))["full_suite"]
    full_summary = json.load(open(RES / "backtest_fullsuite_summary.json"))

    # Python reference trace for JS parity (P15): all-agent bank, cost-aware gamma=1, fixed outcome rule.
    sigma_v = json.load(open(RES / "calibrate_sigma_v.json"))["best_sigma_v"]
    grid = engine.make_grid(bank_all.mu_b, bank_all.sd_b)
    Pm = engine.predictive_matrix(bank_all.x, bank_all.u, grid, sigma_v=sigma_v)
    h_true = 7.0
    respond = lambda t: int(bank_all.x[t] < h_true)          # deterministic: succeed iff task shorter than 2^7 min
    tr, stop, _ = engine.run(bank_all, grid, Pm, engine.InfoPolicy(bank_all, np.random.default_rng(0), 1.0),
                             respond, target_sd=0.5, max_tasks=40)
    parity = {"rule": "y = 1 iff x_t < 7.0; policy gamma=1; target_sd=0.5", "tasks": [r["task"] for r in tr],
              "h_mean": [round(r["h_mean"], 6) for r in tr], "h_sd": [round(r["h_sd"], 6) for r in tr], "stop": stop}

    cost_fn, cm = design.token_cost_model(bank_all.x[bank_all.x >= 4], bank_all.cost[bank_all.x >= 4])
    payload = {
        "meta": {"source": "METR eval-analysis-public TH1.1 runs.jsonl @52cb829", "n_runs": int(C.df.n.sum()),
                 "xbar": C.xbar, "grid": {"h_min": engine.H_MIN, "h_max": engine.H_MAX, "h_step": engine.H_STEP,
                                          "n_beta": engine.N_BETA, "beta_inflate": 1.5, "beta_cap": -0.05},
                 "sigma_v": sigma_v, "gh_v": engine.GH_V.tolist(), "gh_w": engine.GH_W.tolist()},
        "tasks": {"id": bank_all.task_ids, "x": np.round(bank_all.x, 4).tolist(), "cost": bank_all.cost.round(0).tolist(),
                  "pass_rate": np.round(bank_all.pass_rate, 3).tolist(), "source": bank_all.source,
                  "family": bank_all.family},
        "bank_all": {"u": np.round(bank_all.u, 3).tolist(), "mu_b": round(bank_all.mu_b, 4),
                     "sd_b": round(bank_all.sd_b, 4), "tau": round(bank_all.tau, 4)},
        "banks_loo": banks, "rates": rates, "agents": agents,
        "public_headlines": {"Claude Mythos Preview (early)": {"p50_min": 1044.8, "ci": [508.9, 3304.3],
                                                               "release": "2026-04-07"},
                             "GPT-5.4": {"p50_min": 341.7, "ci": [186.6, 768.8], "release": "2026-03-05"},
                             "Gemini 3.1 Pro": {"p50_min": 384.1, "ci": [233.5, 694.8], "release": "2026-02-19"}},
        "backtest": {"summary": summary, "full_suite": full_summary["full_suite"], "curves": backtest_curves(T, F),
                     "summary_v1": summary_v1, "full_suite_v1": full_v1,
                     "calibration": json.load(open(RES / "calibrate_sigma_v.json"))},
        "frontier": {**json.load(open(RES / "frontier.json")),
                     "se_curve": pd.read_csv(RES / "frontier_se_curve.csv").round(4).to_dict("list"),
                     "design": pd.read_csv(RES / "frontier_design.csv").to_dict("records"),
                     "sims": pd.read_csv(RES / "frontier_cat_sims.csv").to_dict("records"), "cost_model": cm},
        "audit": audit(C, bank_all, m_all),
        "parity": parity,
    }
    out = HERE / "web" / "data.json"
    out.parent.mkdir(exist_ok=True)
    enc = lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)
    json.dump(payload, open(out, "w"), separators=(",", ":"), default=enc)
    print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB; parity stop", stop, "steps", len(tr))


if __name__ == "__main__":
    main()
