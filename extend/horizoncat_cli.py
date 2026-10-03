"""HorizonCAT command line: estimate a NEW agent's 50% time horizon from its runs on METR tasks, and plan which tasks
to run next. Intended for researchers who can run newer models (via their APIs) on METR's task environments.

The item bank is built from all agents in METR's public TH1.1 data (study model M, see prototype/DECISIONS_PROTOTYPE.md).
Your agent's runs are never sent anywhere; everything runs locally.

Input: a JSONL file with one run per line, using METR's run schema (only these fields are needed):
    {"task_id": "hackthebox/babyencryption", "score_binarized": 1}
Optional fields: "run_id", "tokens_count". Only the first run of each task is used (P7: one run per task).

Examples:
    python extend/horizoncat_cli.py estimate --runs my_model_runs.jsonl
    python extend/horizoncat_cli.py plan --runs my_model_runs.jsonl --next 5 --gamma 0
    python extend/horizoncat_cli.py plan --next 5            # no runs yet: where to start
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from horizoncat import bank as hb, engine  # noqa: E402
from horizoncat.bank import sdata  # noqa: E402

SIGMA_V = 1.0   # agent-by-task scale selected by held-out likelihood (P19)


def fmt_minutes(m):
    if m < 1:
        return f"{m * 60:.0f} s"
    if m < 60:
        return f"{m:.0f} min"
    if m < 60 * 24:
        return f"{m / 60:.1f} h"
    return f"{m / 1440:.1f} days"


def load_runs(path):
    runs, seen, ignored = [], set(), 0
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        if "task_id" not in r or "score_binarized" not in r:
            sys.exit(f"line {i + 1}: needs 'task_id' and 'score_binarized'")
        if r["task_id"] in seen:
            ignored += 1
            continue
        seen.add(r["task_id"])
        runs.append((r["task_id"], int(bool(r["score_binarized"]))))
    return runs, ignored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["estimate", "plan"])
    ap.add_argument("--runs", type=pathlib.Path, help="JSONL of your agent's runs (task_id, score_binarized)")
    ap.add_argument("--next", type=int, default=5, help="plan: number of tasks to recommend")
    ap.add_argument("--gamma", type=float, default=0.0, help="plan: cost exponent (0 = information only, 1 = per token)")
    ap.add_argument("--target-sd", type=float, default=0.5, help="precision target for SD of log2 horizon")
    a = ap.parse_args()

    C = sdata.load()
    runs_all, _ = sdata.load_runs()
    bank, _ = hb.build(C, runs_all)
    grid = engine.make_grid(bank.mu_b, bank.sd_b)
    Pm = engine.predictive_matrix(bank.x, bank.u, grid, sigma_v=SIGMA_V)
    tid = {t: i for i, t in enumerate(bank.task_ids)}

    state = engine.State(grid.log_prior.copy())
    if a.runs:
        runs, ignored = load_runs(a.runs)
        unknown = [t for t, _ in runs if t not in tid]
        for t, y in runs:
            if t in tid:
                engine.update(state, Pm, tid[t], y)
        print(f"Used {len(state.asked)} runs ({ignored} repeat runs ignored; {len(unknown)} unknown task ids"
              + (f", e.g. {unknown[:3]}" if unknown else "") + ").")
    w = state.post()
    m, v = engine.h_moments(w, grid)
    lo, med, hi = engine.h_quantiles(w, grid)
    beyond = float(w[grid.h > bank.x.max()].sum())
    available = np.array([i for i in range(len(bank.x)) if i not in set(state.asked)])
    sat, _ = engine.saturated(state, Pm, grid, available, bank)
    status = ("precise" if np.sqrt(v) <= a.target_sd else "SATURATED: the suite cannot pin this horizon; see "
              "prototype/frontier.py for how many longer tasks are needed" if sat else "more tasks needed")
    if state.asked:
        print(f"50% horizon: {fmt_minutes(2 ** med)}  (95% interval {fmt_minutes(2 ** lo)} to {fmt_minutes(2 ** hi)}; "
              f"SD {np.sqrt(v):.2f} doublings; {100 * beyond:.0f}% of posterior beyond the longest task)")
        print(f"Status: {status}")
    if a.command == "plan":
        evr = engine.expected_var_reduction(w, Pm[available], grid)
        score = evr / bank.cost[available] ** a.gamma
        order = available[np.argsort(-score)][: a.next]
        print(f"\nNext {len(order)} tasks to run (gamma={a.gamma:g}; run each once, then re-run this command):")
        for i in order:
            j = int(np.where(available == i)[0][0])
            print(f"  {bank.task_ids[i]:55s} human {fmt_minutes(2 ** bank.x[i]):>9s}   "
                  f"~{bank.cost[i] / 1e3:,.0f}k tokens   expected variance reduction {evr[j]:.3f}")


if __name__ == "__main__":
    main()
