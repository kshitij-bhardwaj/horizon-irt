"""HorizonCAT correctness tests. Run from prototype/:  python -m tests.test_engine

E1 expected variance reduction is in [0, Var(h)] and matches a brute-force two-outcome computation
E2 test information equals the variance of the score, checked by finite differences (P10)
E3 quantile compression of task posteriors preserves the posterior mean and spread (P3)
E4 under the model, 95% credible intervals for h cover the truth about 95% of the time (calibration, P6)
"""

import sys
import time

import numpy as np
from scipy.special import expit

from horizoncat import bank as hb, design, engine
from horizoncat.bank import sdata, smodels

CELLS = sdata.load()
RUNS, _ = sdata.load_runs()
BANK, MFIT = hb.build(CELLS, RUNS)
GRID = engine.make_grid(BANK.mu_b, BANK.sd_b)
PM = engine.predictive_matrix(BANK.x, BANK.u, GRID)


def test_E1_evr():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(GRID.h.size))
    evr = engine.expected_var_reduction(w, PM, GRID)
    _, v = engine.h_moments(w, GRID)
    assert np.all(evr >= -1e-9) and np.all(evr <= v + 1e-9)
    for t in rng.choice(len(BANK.x), 5, replace=False):
        p1 = w @ PM[t]
        w1 = w * PM[t] / p1
        w0 = w * (1 - PM[t]) / (1 - p1)
        brute = v - (p1 * engine.h_moments(w1, GRID)[1] + (1 - p1) * engine.h_moments(w0, GRID)[1])
        assert abs(brute - evr[t]) < 1e-9, (brute, evr[t])


def test_E2_information():
    """Full 2x2 Fisher matrix over (h, beta) equals E[score score^T], by finite differences (P16)."""
    beta = BANK.mu_b
    for h0 in [2.0, 6.0, 9.5]:
        I = design.info_matrix_bank(np.array([h0]), beta, BANK.x, BANK.u)[0]
        d = 1e-5
        tot = np.zeros((2, 2))
        for t in range(len(BANK.x)):
            p = lambda h, b: expit(b * (BANK.x[t] - h) + BANK.u[t]).mean()
            p0 = p(h0, beta)
            g = np.array([(p(h0 + d, beta) - p(h0 - d, beta)) / (2 * d), (p(h0, beta + d) - p(h0, beta - d)) / (2 * d)])
            tot += np.outer(g, g) / (p0 * (1 - p0))                     # E[score score^T] for a Bernoulli
        assert np.allclose(tot, I, rtol=1e-4, atol=1e-10), (h0, tot, I)


def test_E3_quantiles():
    uq = MFIT.tau * smodels.Z_GRID
    for t in [0, 50, 120, 200]:
        w = np.exp(MFIT.log_post[t] - MFIT.log_post[t].max()); w /= w.sum()
        mean, sd = w @ uq, np.sqrt(w @ uq**2 - (w @ uq) ** 2)
        assert abs(BANK.u[t].mean() - mean) < 1e-6, (t, BANK.u[t].mean(), mean)
        assert abs(BANK.u[t].std() / sd - 1) < 0.2, (t, BANK.u[t].std(), sd)   # K=10 points lose a little spread


def test_E4_calibration(n=60):
    rng = np.random.default_rng(1)
    cover, sds = 0, []
    for i in range(n):
        h_true = rng.uniform(0, 9)
        b_true = BANK.mu_b + BANK.sd_b * rng.standard_normal()
        u_true = BANK.u[np.arange(len(BANK.x)), rng.integers(0, hb.K_QUANTILES, len(BANK.x))]
        respond = lambda t: int(rng.random() < expit(b_true * (BANK.x[t] - h_true) + u_true[t]))
        pol = engine.InfoPolicy(BANK, rng, gamma=1.0)
        tr, stop, st = engine.run(BANK, GRID, PM, pol, respond, target_sd=0.4, max_tasks=60)
        lo, hi = tr[-1]["h_lo"], tr[-1]["h_hi"]
        cover += lo <= h_true <= hi
        sds.append(tr[-1]["h_sd"])
    rate = cover / n
    test_E4_calibration.report = {"coverage95": rate, "median_final_sd": float(np.median(sds))}
    assert 0.85 <= rate <= 1.0, rate


if __name__ == "__main__":
    only = sys.argv[1:]
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not only or any(o in name for o in only)):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s) {getattr(fn, 'report', '')}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(failed)
