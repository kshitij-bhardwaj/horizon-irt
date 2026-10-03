"""Correctness tests for the estimators. Run from study/:  python -m tests.test_models

Each test backs a specific claim made in the report:
  T1 analytic gradients are right            -> optimiser output is a true optimum
  T2 B1 == METR's sklearn estimator           -> 'baseline' really is the paper's method
  T3 quadrature error is negligible           -> marginal likelihoods / LRTs are numerically trustworthy
  T4 MML recovers tau; LRT is calibrated      -> tau-hat and the test of tau > 0 can be believed
  T5 p50 invariance, p80 attenuation          -> the analytical result of E3
  T6 M -> B1 as tau -> 0                      -> nesting used for the likelihood-ratio test
"""

import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, log_expit

from irt import data, models
from irt.data import REPO
from irt.simulate import simulate

sys.path.insert(0, str(REPO / "src"))
CELLS = data.load()


def _gradcheck(fun, th, n_coords=25, h=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    th = th + 0.05 * rng.standard_normal(len(th))
    _, g = fun(th)
    worst = 0.0
    for i in rng.choice(len(th), size=min(n_coords, len(th)), replace=False):
        e = np.zeros_like(th); e[i] = h
        num = (fun(th + e)[0] - fun(th - e)[0]) / (2 * h)
        worst = max(worst, abs(num - g[i]) / max(1.0, abs(num)))
    return worst


def test_T1_gradients():
    b1 = models.fit_map(CELLS, "agent")
    for kw in [dict(slope="agent", lam=0.1, weights="metr"), dict(slope="common", tau=1.5),
               dict(slope="agent", hier_kappa=5.0, tau=1.0, rank=2, rho=1.0, weights="count")]:
        fun, th0 = models.fit_map(CELLS, init=b1, _objective_only=True, **kw)
        err = _gradcheck(fun, th0)
        assert err < 1e-5, (kw, err)
    for kw in [dict(slope="agent"), dict(slope="none"), dict(slope="agent", hier_kappa=3.0, weights="count")]:
        fun, th0 = models.fit_mml(CELLS, init=b1, _objective_only=True, **kw)
        err = _gradcheck(fun, th0)
        assert err < 1e-5, (kw, err)


def test_T2_metr_equivalence():
    from horizon.utils.logistic import get_x_for_quantile, logistic_regression
    for lam in (1e-5, 0.1):
        fit = models.fit_map(CELLS, "agent", weights="metr", lam=lam, eps_alpha=0.0)
        mine = fit.horizons(CELLS.agents)["cond_p50"]
        J = lambda b0, b1, X, y, w: (-np.sum(w * (y * log_expit(b0 + b1 * X) + (1 - y) * log_expit(-(b0 + b1 * X))))
                                     + 0.5 * lam * b1**2)
        for i, ag in enumerate(CELLS.agents):
            g = CELLS.df[CELLS.df.a == i]
            X = np.repeat(g.x.values, g.n.values).reshape(-1, 1)
            y = np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for n, k in zip(g.n, g.k)])
            w = np.repeat(g.w_metr.values, g.n.values)
            m = logistic_regression(X, y, w, regularization=lam)
            ref = 2 ** get_x_for_quantile(m, 0.5)
            # sklearn stops at tol=1e-4, so compare optimality directly: ours must be at least as good,
            # and horizons must agree to 0.1%.
            j_ref = J(m.intercept_[0], m.coef_[0][0], X[:, 0], y, w)
            j_mine = J(fit.alpha[i] - fit.beta[i] * CELLS.xbar, fit.beta[i], X[:, 0], y, w)
            assert j_mine <= j_ref + 1e-10, (lam, ag, j_mine, j_ref)
            assert abs(mine[i] / ref - 1) < 1e-3, (lam, ag, mine[i], ref)


def test_T3_quadrature():
    m = models.fit_mml(CELLS, "agent", init=models.fit_map(CELLS, "agent"))
    df = CELLS.df
    for t in [0, 57, 120, 200]:
        g = df[df.t == t]
        e0 = m.eta0(g.a.values, g.x.values)
        def integrand(u):
            ll = np.sum(g.k * log_expit(e0 + u) + (g.n - g.k) * log_expit(-(e0 + u)))
            return np.exp(ll) * np.exp(-0.5 * (u / m.tau) ** 2) / (m.tau * np.sqrt(2 * np.pi))
        exact = np.log(quad(integrand, -12 * m.tau, 12 * m.tau, limit=400, epsabs=0, epsrel=1e-12)[0])
        grid = models.logsumexp(models.LOG_V + np.array(
            [np.sum(g.k * log_expit(e0 + m.tau * z) + (g.n - g.k) * log_expit(-(e0 + m.tau * z))) for z in models.Z_GRID]))
        assert abs(exact - grid) < 1e-4, (t, exact, grid)


def test_T4_recovery_and_lrt():
    b1 = models.fit_map(CELLS, "agent")
    m = models.fit_mml(CELLS, "agent", init=b1)
    taus = []
    for s in range(5):
        sim, _ = simulate(CELLS, m.alpha, m.beta, m.tau, seed=s)
        taus.append(models.fit_mml(sim, "agent", init=m).tau)
    rel = abs(np.mean(taus) / m.tau - 1)
    assert rel < 0.10, (m.tau, taus)
    # Null tau = 0: LRT statistic should follow 0.5 chi2_0 + 0.5 chi2_1 (5% critical value 2.71).
    stats = []
    for s in range(30):
        sim, _ = simulate(CELLS, b1.alpha, b1.beta, 0.0, seed=100 + s)
        f0 = models.fit_map(sim, "agent", init=b1)
        f1 = models.fit_mml(sim, "agent", init=f0, tau0=0.3)
        stats.append(max(0.0, 2 * (f1.loglik - f0.loglik)))
    rate = np.mean(np.array(stats) > 2.71)
    assert rate <= 0.2, (rate, stats)
    test_T4_recovery_and_lrt.report = {"tau_true": m.tau, "tau_hat": taus, "null_reject_rate": rate,
                                       "null_stats_mean": float(np.mean(stats))}


def test_T5_p50_invariance():
    m = models.fit_mml(CELLS, "agent", init=models.fit_map(CELLS, "agent"))
    for i in range(CELLS.A):
        c50, m50 = m.horizon(i, 0.5, "conditional"), m.horizon(i, 0.5, "marginal")
        assert abs(m50 / c50 - 1) < 1e-6, (i, c50, m50)
        assert m.horizon(i, 0.8, "marginal") < m.horizon(i, 0.8, "conditional")


def test_T6_nesting():
    b1 = models.fit_map(CELLS, "agent", eps_alpha=0.0)
    fun, th = models.fit_mml(CELLS, "agent", init=b1, _objective_only=True)
    th[-1] = np.log(1e-5)   # second-order term is ~ tau^2/2 * sum_t (sum_c residual_c)^2, negligible here
    nll_b1 = -np.sum(CELLS.df.k * log_expit(b1.eta0(CELLS.df.a.values, CELLS.df.x.values))
                     + (CELLS.df.n - CELLS.df.k) * log_expit(-b1.eta0(CELLS.df.a.values, CELLS.df.x.values)))
    assert abs(fun(th)[0] - nll_b1) < 1e-2, (fun(th)[0], nll_b1)


if __name__ == "__main__":
    only = sys.argv[1:]
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not only or any(o in name for o in only)):
            t0 = time.time()
            try:
                fn()
                extra = getattr(fn, "report", "")
                print(f"PASS {name} ({time.time() - t0:.1f}s) {extra}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(failed)
