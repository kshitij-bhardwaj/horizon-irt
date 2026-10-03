"""Estimators for the nested model family of decision D5.

    eta_at = alpha_a + beta_a (x_t - xbar) + u_t + p_a . q_t,   P(success) = sigmoid(eta_at)

Two estimation engines:
  * fit_map  : penalised maximum likelihood (B0, B1, B2, F_K; METR's own estimator is B1 with weights='metr').
  * fit_mml  : marginal maximum likelihood with u_t ~ N(0, tau^2) integrated out on a quadrature grid
               (R, L, M). Task posteriors over the grid are kept for prediction and horizons.
Both minimise  -sum_c omega_c [k_c log s(eta_c) + (n_c-k_c) log s(-eta_c)] + penalties  with analytic gradients.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, gammaln, log_expit, logsumexp

from .data import Cells

# Quadrature for u = tau * z, z ~ N(0,1): uniform grid on [-8, 8] (decision D7: fixed GH nodes are too coarse
# relative to the narrow task posteriors that 20 agents x 5 runs produce when tau is large).
Z_GRID = np.linspace(-8, 8, 321)
LOG_V = -0.5 * Z_GRID**2 - logsumexp(-0.5 * Z_GRID**2)   # normalised log N(0,1) masses on the grid

SLOPES = ("none", "common", "agent")


@dataclass
class Fit:
    model: str
    alpha: np.ndarray
    beta: np.ndarray                      # length A (zeros if slope='none', all equal if 'common')
    xbar: float
    tau: float = 0.0
    log_post: np.ndarray | None = None    # (T, Q) log posterior over grid for each task (MML fits)
    u: np.ndarray | None = None           # (T,) task effect: posterior mean (MML) or MAP value
    P: np.ndarray | None = None
    Q: np.ndarray | None = None
    objective: float = np.nan             # minimised objective (incl. penalties)
    loglik: float = np.nan                # log-likelihood incl. binomial constants (marginal for MML)
    n_params: int = 0
    info: dict = field(default_factory=dict)

    # ---------- prediction ----------
    def eta0(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.alpha[a] + self.beta[a] * (x - self.xbar)

    def predict(self, df, how: str = "posterior") -> np.ndarray:
        """P(success) for rows of a cells-like frame (needs columns a, t, x).

        how='posterior' : MML -> E[s(eta0+u_t) | training data of task t]; MAP -> plug-in incl. u_t, factors.
        how='marginal'  : MML -> E[s(eta0+u)] under the prior (the right predictive for an unseen task).
        how='plugin'    : MML -> s(eta0 + E[u_t | data]).
        how='fixed'     : ignore task-specific terms (u_t = 0, no factors) -> s(eta0).
        """
        a, t, x = df.a.values, df.t.values, df.x.values
        e0 = self.eta0(a, x)
        if how == "fixed" or (self.tau == 0 and self.u is None and self.P is None):
            return expit(e0)
        if self.log_post is not None:
            if how == "plugin":
                return expit(e0 + self.u[t])
            logw = LOG_V[None, :] if how == "marginal" else self.log_post[t]
            return np.exp(logsumexp(logw + log_expit(e0[:, None] + self.tau * Z_GRID[None, :]), axis=1))
        eta = e0 + (self.u[t] if self.u is not None else 0)
        if self.P is not None:
            eta = eta + np.einsum("ik,ik->i", self.P[a], self.Q[t])
        return expit(eta)

    # ---------- horizons ----------
    def horizon(self, a: int, p: float = 0.5, kind: str = "conditional") -> float:
        """Task length (minutes) at which agent a succeeds with probability p.

        conditional : for a task with u_t = 0 (a 'typical' task of that length).
        marginal    : averaged over task heterogeneity, E_u s(eta0 + u) = p.  Equal for p = 0.5 (symmetry).
        """
        al, be = self.alpha[a], self.beta[a]
        if kind == "conditional" or self.tau == 0:
            return float(2 ** (self.xbar + (np.log(p / (1 - p)) - al) / be))
        f = lambda x: np.exp(logsumexp(LOG_V + log_expit(al + be * (x - self.xbar) + self.tau * Z_GRID))) - p
        return float(2 ** brentq(f, -40, 60, xtol=1e-10))

    def horizons(self, agents: list[str], ps=(0.5, 0.8), kinds=("conditional",)) -> dict:
        return {f"{k[:4]}_p{int(p * 100)}": np.array([self.horizon(i, p, k) for i in range(len(agents))])
                for p in ps for k in kinds}


def _binom_const(df, w):
    return float(np.sum(w * (gammaln(df.n + 1) - gammaln(df.k + 1) - gammaln(df.n - df.k + 1))))


def _pack_slope(beta_free, slope, A):
    if slope == "none":
        return np.zeros(A)
    if slope == "common":
        return np.full(A, beta_free[0])
    return beta_free


def _n_beta(slope, A):
    return {"none": 0, "common": 1, "agent": A}[slope]


def _slope_grad(g_eta, xc, a, slope, A):
    if slope == "none":
        return np.zeros(0)
    if slope == "common":
        return np.array([np.sum(g_eta * xc)])
    return np.bincount(a, g_eta * xc, minlength=A)


# =====================================================================================================
# Penalised maximum likelihood (MAP)
# =====================================================================================================
def fit_map(cells: Cells, slope: str = "agent", weights: str = "none", lam: float = 0.0,
            hier_kappa: float | None = None, tau: float | None = None, rank: int = 0, rho: float = 1.0,
            eps_alpha: float = 1e-6, init: Fit | None = None, seed: int = 0, name: str | None = None,
            _objective_only: bool = False) -> Fit:
    """Penalties: lam/2 sum beta^2 (METR's ridge toward 0)  or  kappa/2 sum (beta_a - mu)^2 (hierarchical, D9);
    1/(2 tau^2) sum u^2 if tau is given (task effects as MAP); rho/2 (|P|^2 + |Q|^2) for rank-K factors;
    eps_alpha/2 sum alpha^2 guards against separation only.
    """
    df, A, T = cells.df, cells.A, cells.T
    a, t, n, k = df.a.values, df.t.values, df.n.values.astype(float), df.k.values.astype(float)
    xc = df.x.values - cells.xbar
    w = cells.weights(weights)
    nb = _n_beta(slope, A)
    use_mu = hier_kappa is not None and slope == "agent"
    use_u = tau is not None
    sizes = [A, nb, int(use_mu), T if use_u else 0, A * rank, T * rank]
    cuts = np.cumsum(sizes)[:-1]

    def unpack(th):
        al, bf, mu, u, P, Q = np.split(th, cuts)
        return al, _pack_slope(bf, slope, A), bf, mu, u, P.reshape(A, rank), Q.reshape(T, rank)

    def fun(th):
        al, be, bf, mu, u, P, Q = unpack(th)
        eta = al[a] + be[a] * xc
        if use_u:
            eta = eta + u[t]
        if rank:
            eta = eta + np.einsum("ik,ik->i", P[a], Q[t])
        nll = -np.sum(w * (k * log_expit(eta) + (n - k) * log_expit(-eta)))
        g = w * (n * expit(eta) - k)                     # d nll / d eta
        g_al = np.bincount(a, g, minlength=A) + eps_alpha * al
        g_bf = _slope_grad(g, xc, a, slope, A)
        pen = 0.5 * eps_alpha * al @ al
        g_mu = np.zeros(int(use_mu))
        if use_mu:
            d = bf - mu[0]
            pen += 0.5 * hier_kappa * d @ d
            g_bf = g_bf + hier_kappa * d
            g_mu = np.array([-hier_kappa * d.sum()])
        elif lam:
            pen += 0.5 * lam * bf @ bf
            g_bf = g_bf + lam * bf
        g_u = np.zeros(0)
        if use_u:
            pen += 0.5 * u @ u / tau**2
            g_u = np.bincount(t, g, minlength=T) + u / tau**2
        g_P = g_Q = np.zeros(0)
        if rank:
            pen += 0.5 * rho * (np.sum(P**2) + np.sum(Q**2))
            gP = np.zeros((A, rank)); np.add.at(gP, a, g[:, None] * Q[t])
            gQ = np.zeros((T, rank)); np.add.at(gQ, t, g[:, None] * P[a])
            g_P, g_Q = (gP + rho * P).ravel(), (gQ + rho * Q).ravel()
        return nll + pen, np.concatenate([g_al, g_bf, g_mu, g_u, g_P, g_Q])

    rng = np.random.default_rng(seed)
    th0 = np.zeros(cuts[-1] + sizes[-1])
    if init is not None:
        th0[:A] = init.alpha
        if nb == A:
            th0[A:A + nb] = init.beta
        elif nb == 1:
            th0[A] = init.beta.mean()
        if use_mu:
            th0[cuts[1]] = init.beta.mean()
        if use_u and init.u is not None:
            th0[cuts[2]:cuts[3]] = init.u
    if rank:
        th0[cuts[3]:] = 0.1 * rng.standard_normal(sizes[4] + sizes[5])
    if _objective_only:
        return fun, th0
    res = minimize(fun, th0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "gtol": 1e-9, "ftol": 1e-14})
    al, be, bf, mu, u, P, Q = unpack(res.x)
    eta = al[a] + be[a] * xc + (u[t] if use_u else 0) + (np.einsum("ik,ik->i", P[a], Q[t]) if rank else 0)
    ll = float(np.sum(w * (k * log_expit(eta) + (n - k) * log_expit(-eta)))) + _binom_const(df, w)
    return Fit(model=name or f"MAP[{slope}]", alpha=al, beta=be, xbar=cells.xbar, tau=0.0,
               u=u if use_u else None, P=P if rank else None, Q=Q if rank else None,
               objective=float(res.fun), loglik=ll, n_params=len(res.x),
               info={"converged": bool(res.success), "message": str(res.message), "mu": mu, "fixed_tau": tau})


# =====================================================================================================
# Marginal maximum likelihood with a Gaussian task random effect
# =====================================================================================================
class _TaskSum:
    """Sum rows of a (C, Q) array into (T, Q) by task, via a cached sort + reduceat."""

    def __init__(self, t, T):
        self.order = np.argsort(t, kind="stable")
        ts = t[self.order]
        self.present, self.starts = np.unique(ts, return_index=True)
        self.T = T

    def __call__(self, M):
        return np.add.reduceat(M[self.order], self.starts, axis=0)   # rows correspond to self.present


def _mml_terms(eta0, tau, n, k, w, tsum):
    E = eta0[:, None] + tau * Z_GRID[None, :]
    L = w[:, None] * (k[:, None] * log_expit(E) + (n - k)[:, None] * log_expit(-E))   # (C, Q)
    Lt = tsum(L)                                                                        # (T_present, Q)
    logpost_un = LOG_V[None, :] + Lt
    logml = logsumexp(logpost_un, axis=1)
    return E, logml, logpost_un - logml[:, None]


def fit_mml(cells: Cells, slope: str = "agent", weights: str = "none", lam: float = 0.0,
            hier_kappa: float | None = None, init: Fit | None = None, tau0: float = 1.0,
            name: str | None = None, tau_fixed: float | None = None, _objective_only: bool = False) -> Fit:
    """Maximise sum_t log int prod_{a} Bin(k_at | n_at, s(eta0_at + u)) N(u; 0, tau^2) du over (alpha, beta, tau).

    Gradient (Fisher identity): d log L_t / d theta = E_{u | data_t}[ d log p(data_t | u) / d theta ].
    """
    df, A, T = cells.df, cells.A, cells.T
    a, t = df.a.values, df.t.values
    n, k = df.n.values.astype(float), df.k.values.astype(float)
    xc = df.x.values - cells.xbar
    w = cells.weights(weights)
    nb = _n_beta(slope, A)
    use_mu = hier_kappa is not None and slope == "agent"
    tsum = _TaskSum(t, T)
    pos_of_cell = np.searchsorted(tsum.present, t)       # row of each cell's task in the (T_present, Q) arrays

    def unpack(th):
        return th[:A], _pack_slope(th[A:A + nb], slope, A), th[A:A + nb], th[A + nb:A + nb + int(use_mu)], th[-1]

    def fun(th):
        al, be, bf, mu, logtau = unpack(th)
        tau = np.exp(logtau)
        eta0 = al[a] + be[a] * xc
        E, logml, logpost = _mml_terms(eta0, tau, n, k, w, tsum)
        post = np.exp(logpost)                                   # (T_present, Q)
        R = w[:, None] * (k[:, None] - n[:, None] * expit(E))     # d loglik_cq / d eta
        g_eta = -np.einsum("cq,cq->c", post[pos_of_cell], R)      # d(-logL)/d eta0_c
        S = tsum(R)                                               # (T_present, Q)
        g_logtau = -tau * np.sum(post * S * Z_GRID[None, :])
        obj = -logml.sum()
        g_al = np.bincount(a, g_eta, minlength=A)
        g_bf = _slope_grad(g_eta, xc, a, slope, A)
        g_mu = np.zeros(int(use_mu))
        if use_mu:
            d = bf - mu[0]
            obj += 0.5 * hier_kappa * d @ d
            g_bf = g_bf + hier_kappa * d
            g_mu = np.array([-hier_kappa * d.sum()])
        elif lam:
            obj += 0.5 * lam * bf @ bf
            g_bf = g_bf + lam * bf
        return obj, np.concatenate([g_al, g_bf, g_mu, [g_logtau]])

    th0 = np.zeros(A + nb + int(use_mu) + 1)
    th0[-1] = np.log(tau0)
    if init is not None:
        th0[:A] = init.alpha
        if nb == A:
            th0[A:A + nb] = init.beta
        elif nb == 1:
            th0[A] = init.beta.mean()
        if use_mu:
            th0[A + nb] = init.beta.mean()
        if init.tau > 0:
            th0[-1] = np.log(init.tau)
    tau_bounds = (np.log(1e-3), np.log(50))
    if tau_fixed is not None:                      # profile likelihood in tau (E2)
        th0[-1] = np.log(tau_fixed)
        tau_bounds = (th0[-1], th0[-1])
    if _objective_only:
        return fun, th0
    res = minimize(fun, th0, jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] * (len(th0) - 1) + [tau_bounds],
                   options={"maxiter": 20000, "gtol": 1e-8, "ftol": 1e-14})
    al, be, bf, mu, logtau = unpack(res.x)
    tau = float(np.exp(logtau))
    _, logml, logpost_present = _mml_terms(al[a] + be[a] * xc, tau, n, k, w, tsum)
    log_post = np.tile(LOG_V, (T, 1))                 # tasks with no data keep the prior
    log_post[tsum.present] = logpost_present
    u_mean = np.exp(log_post) @ (tau * Z_GRID)
    ll = float(logml.sum()) + _binom_const(df, w)
    return Fit(model=name or f"MML[{slope}]", alpha=al, beta=be, xbar=cells.xbar, tau=tau,
               log_post=log_post, u=u_mean, objective=float(res.fun), loglik=ll, n_params=len(res.x),
               info={"converged": bool(res.success), "message": str(res.message), "mu": mu})


def posterior_given(fit: Fit, cells: Cells, weights: str = "none") -> Fit:
    """Recompute task posteriors under fixed (alpha, beta, tau) using the data in `cells` (used in E1-C)."""
    df = cells.df
    tsum = _TaskSum(df.t.values, cells.T)
    eta0 = fit.eta0(df.a.values, df.x.values)
    _, _, lp = _mml_terms(eta0, fit.tau, df.n.values.astype(float), df.k.values.astype(float),
                          cells.weights(weights), tsum)
    log_post = np.tile(LOG_V, (cells.T, 1))
    log_post[tsum.present] = lp
    return Fit(**{**fit.__dict__, "log_post": log_post, "u": np.exp(log_post) @ (fit.tau * Z_GRID)})
