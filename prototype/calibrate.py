"""Choose the agent-by-task interaction scale sigma_v by held-out predictive likelihood (P19).

For each agent (leave-one-agent-out bank) and 3 random half-splits of its tasks:
  * posterior over (h, beta) from ONE replayed run of each task in half A (what an evaluation would see);
  * score every real run of the agent on half-B tasks under the posterior predictive  sum_g w_g p_tg(sigma_v).
Sum the held-out log-likelihood over agents and splits; pick the best sigma_v on a grid.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from horizoncat import bank as hb, engine
from horizoncat.bank import sdata

OUT = pathlib.Path(__file__).parent / "results"
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def main(splits=3):
    C = sdata.load()
    runs, _ = sdata.load_runs()
    rows = []
    for ai, agent in enumerate(C.agents):
        bank, _ = hb.build(C, runs, exclude_agent=agent)
        grid = engine.make_grid(bank.mu_b, bank.sd_b)
        cells = C.df[C.df.a == ai]
        t_all, k_all, n_all = cells.t.values, cells.k.values, cells.n.values
        for sv in SIGMAS:
            Pm = engine.predictive_matrix(bank.x, bank.u, grid, sigma_v=sv)
            for sp in range(splits):
                rng = np.random.default_rng(1000 * ai + sp)
                inA = rng.random(len(t_all)) < 0.5
                logp = grid.log_prior.copy()
                for t, k, n in zip(t_all[inA], k_all[inA], n_all[inA]):
                    y = rng.random() < k / n                         # one replayed run per task in A
                    p = np.clip(Pm[t], 1e-12, 1 - 1e-12)
                    logp += np.log(p) if y else np.log1p(-p)
                w = np.exp(logp - logp.max()); w /= w.sum()
                pred = np.clip(Pm[t_all[~inA]] @ w, 1e-12, 1 - 1e-12)
                kB, nB = k_all[~inA], n_all[~inA]
                ll = float(np.sum(kB * np.log(pred) + (nB - kB) * np.log1p(-pred)))
                rows.append({"agent": agent, "sigma_v": sv, "split": sp, "heldout_ll": ll, "runs": int(nB.sum())})
        print(agent, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "calibrate_sigma_v.csv", index=False)
    tot = R.groupby("sigma_v").apply(lambda g: g.heldout_ll.sum() / g.runs.sum(), include_groups=False)
    best = float(tot.idxmax())
    # paired uncertainty: per-agent differences vs sigma_v = 0, bootstrap over agents
    per = R.groupby(["agent", "sigma_v"]).heldout_ll.sum().unstack()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per), (2000, len(per)))
    diff = (per[best] - per[0.0]).values
    ci = [float(np.percentile(diff[idx].sum(1), 2.5)), float(np.percentile(diff[idx].sum(1), 97.5))]
    json.dump({"per_run_heldout_ll": tot.to_dict(), "best_sigma_v": best,
               "total_ll_gain_vs_0": float(diff.sum()), "ci95_gain_bootstrap_agents": ci},
              open(OUT / "calibrate_sigma_v.json", "w"), indent=2)
    print(tot.round(5).to_string()); print("best", best, "gain", diff.sum(), ci)


if __name__ == "__main__":
    main()
