"""HorizonCAT engine: sequential Bayesian estimation of an agent's 50% horizon (P5–P8).

Agent model (study model M, reparameterised by the horizon h = log2 H50):
    P(success on task t | h, beta, u_t) = sigmoid(beta * (x_t - h) + u_t),  beta < 0
Single-run predictive for a bank task integrates its difficulty posterior:
    p_t(h, beta) = (1/K) sum_k sigmoid(beta * (x_t - h) + u_tk)
Posterior over a (h, beta) grid; prior: h ~ Uniform[H_MIN, H_MAX] (P6), beta ~ N(mu_b, sd_b^2) from the bank.
Selection (P7): argmax_t  [Var(h) - E_y Var(h | y_t)] / cost_t^gamma   over not-yet-administered tasks.
Stopping (P8): SD(h) <= target, or budget exhausted, or saturation (no task can reduce Var(h) materially).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit
from scipy.stats import norm

H_MIN, H_MAX, H_STEP = -6.0, 18.0, 0.1       # log2 minutes: ~1 s .. ~180 days
N_BETA = 15


@dataclass
class Grid:
    h: np.ndarray        # (G,)
    beta: np.ndarray     # (G,)
    log_prior: np.ndarray


def make_grid(mu_b, sd_b, inflate=1.5):
    hs = np.arange(H_MIN, H_MAX + 1e-9, H_STEP)
    sd = sd_b * inflate
    qs = (np.arange(N_BETA) + 0.5) / N_BETA                  # equal-mass quantile nodes of the slope prior
    bs = mu_b + sd * norm.ppf(qs)
    bs = np.minimum(bs, -0.05)                               # longer tasks must be harder (beta < 0)
    H, B = np.meshgrid(hs, bs, indexing="ij")
    G = H.size
    return Grid(H.ravel(), B.ravel(), np.full(G, -np.log(G)))


GH_J = 7
_GH_Z, _GH_W = np.polynomial.hermite.hermgauss(GH_J)
GH_V = np.sqrt(2) * _GH_Z            # v = sigma_v * GH_V[j] for v ~ N(0, sigma_v^2)
GH_W = _GH_W / np.sqrt(np.pi)


def predictive_matrix(x, u, grid: Grid, sigma_v: float = 0.0):
    """(T, G) single-run success probabilities. Integrates the task's difficulty posterior (K equal-weight points)
    and, if sigma_v > 0, an agent-by-task interaction v ~ N(0, sigma_v^2) by 7-point Gauss-Hermite (P18, P19)."""
    base = grid.beta[None, :, None] * (x[:, None, None] - grid.h[None, :, None]) + u[:, None, :]
    if sigma_v <= 0:
        return expit(base).mean(axis=2)
    out = np.zeros(base.shape[:2])
    for vj, wj in zip(sigma_v * GH_V, GH_W):
        out += wj * expit(base + vj).mean(axis=2)
    return out


@dataclass
class State:
    logp: np.ndarray
    asked: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    spent: float = 0.0

    def post(self):
        w = np.exp(self.logp - self.logp.max())
        return w / w.sum()


def h_moments(w, grid):
    m = w @ grid.h
    return m, max(w @ grid.h**2 - m**2, 0.0)


def h_quantiles(w, grid, qs=(0.025, 0.5, 0.975)):
    hs = np.arange(H_MIN, H_MAX + 1e-9, H_STEP)
    marg = w.reshape(len(hs), -1).sum(1)
    c = np.cumsum(marg)
    return [float(np.interp(q, c, hs)) for q in qs]


def expected_var_reduction(w, Pm, grid):
    """For each task: Var(h) - E_y[Var(h | y)], vectorised over tasks. Pm: (T, G)."""
    m, v = h_moments(w, grid)
    a1 = Pm * w[None, :]
    p1 = a1.sum(1)
    a0 = w[None, :] - a1
    p0 = 1 - p1
    h, h2 = grid.h, grid.h**2
    def var_given(a, p):
        p = np.maximum(p, 1e-300)
        mu = (a @ h) / p
        return (a @ h2) / p - mu**2
    ev = p1 * var_given(a1, p1) + p0 * var_given(a0, p0)
    return v - ev


class Policy:
    name = "base"

    def __init__(self, bank, rng):
        self.bank, self.rng = bank, rng

    def choose(self, state, Pm, grid, available):
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "Random"

    def choose(self, state, Pm, grid, available):
        return int(self.rng.choice(available))


class StratifiedPolicy(Policy):
    """Cycle through 2-doubling log-length bins (METR-like coverage of all lengths), random within bin."""
    name = "Length-stratified"

    def __init__(self, bank, rng):
        super().__init__(bank, rng)
        self.bins = np.floor((bank.x - bank.x.min()) / 2).astype(int)
        self.order = rng.permutation(np.unique(self.bins))
        self.i = 0

    def choose(self, state, Pm, grid, available):
        for _ in range(len(self.order)):
            b = self.order[self.i % len(self.order)]
            self.i += 1
            cand = [t for t in available if self.bins[t] == b]
            if cand:
                return int(self.rng.choice(cand))
        return int(self.rng.choice(available))


class MidPassFilterPolicy(Policy):
    """Static baseline after Ndzomga (arXiv:2603.23749): only tasks with historical pass rate in [0.3, 0.7]."""
    name = "Static 30–70% filter"

    def choose(self, state, Pm, grid, available):
        pr = self.bank.pass_rate[available]
        cand = [t for t, p in zip(available, pr) if 0.3 <= p <= 0.7]
        return int(self.rng.choice(cand if cand else available))


class InfoPolicy(Policy):
    """Bayesian adaptive: maximise expected Var(h) reduction per cost^gamma (gamma = 0: item-count CAT)."""

    def __init__(self, bank, rng, gamma=1.0):
        super().__init__(bank, rng)
        self.gamma = gamma
        self.name = "Adaptive (info only)" if gamma == 0 else f"Adaptive, cost-aware (γ={gamma:g})"

    def choose(self, state, Pm, grid, available):
        evr = expected_var_reduction(state.post(), Pm[available], grid)
        score = evr / self.bank.cost[available] ** self.gamma
        return int(available[int(np.argmax(score))])


def update(state, Pm, t, y):
    p = np.clip(Pm[t], 1e-12, 1 - 1e-12)
    state.logp = state.logp + (np.log(p) if y else np.log1p(-p))
    state.asked.append(int(t))
    state.outcomes.append(int(y))


def saturated(state, Pm, grid, available, bank, rel=0.01):
    """Saturation (P8): even the most informative remaining task would cut Var(h) by < rel, and the posterior
    places material mass beyond the longest task in the bank."""
    w = state.post()
    _, v = h_moments(w, grid)
    best = expected_var_reduction(w, Pm[available], grid).max() if len(available) else 0.0
    beyond = float(w[grid.h > bank.x.max()].sum())
    return best < rel * v and beyond > 0.2, beyond


def run(bank, grid, Pm, policy, respond, target_sd=0.5, max_tasks=80, budget=np.inf, allowed=None):
    """Administer tasks one run at a time until a stopping rule fires. `respond(t)` returns 0/1."""
    state = State(grid.log_prior.copy())
    available = np.array(sorted(allowed if allowed is not None else range(len(bank.x))))
    trace = []
    stop = "max_tasks"
    while len(state.asked) < max_tasks and len(available):
        t = policy.choose(state, Pm, grid, available)
        y = respond(t)
        update(state, Pm, t, y)
        state.spent += bank.cost[t]
        available = available[available != t]
        w = state.post()
        m, v = h_moments(w, grid)
        lo, med, hi = h_quantiles(w, grid)
        sat, beyond = saturated(state, Pm, grid, available, bank)
        trace.append({"step": len(state.asked), "task": int(t), "y": int(y), "tokens": state.spent,
                      "h_mean": m, "h_sd": np.sqrt(v), "h_lo": lo, "h_med": med, "h_hi": hi,
                      "p_beyond_suite": beyond})
        if np.sqrt(v) <= target_sd:
            stop = "precision"
            break
        if state.spent >= budget:
            stop = "budget"
            break
        if sat:
            stop = "saturated"
            break
    return trace, stop, state
