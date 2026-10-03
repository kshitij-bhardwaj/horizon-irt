"""Measurement range of a task suite and how to extend it (P10, revised by P16).

For one run of task t by an agent (h, beta), the single-run predictive is p_t = E_u sigmoid(beta (x_t - h) + u), with
    dp/dh    = -beta * E_u[s(1-s)]          dp/dbeta = (x_t - h) * E_u[s(1-s)]
and the Bernoulli Fisher information matrix over theta = (h, beta) is  I_t = g g^T / (p(1-p)),  g = (dp/dh, dp/dbeta).
The slope is NOT known for a new agent, so the standard error of h must come from the inverse of the full matrix,
with the slope prior's precision added (P16):
    SE(h) = sqrt( [ (sum_t I_t + diag(0, 1/sd_beta^2))^{-1} ]_{hh} )
Ignoring the slope (SE = 1/sqrt(I_hh)) is badly optimistic for frontier agents: when nearly all tasks lie below h, the
data cannot separate "longer horizon" from "flatter curve".
"""

import numpy as np
from scipy.special import expit

Z = np.linspace(-6, 6, 241)
WZ = np.exp(-0.5 * Z**2); WZ /= WZ.sum()


def _terms_bank(h, beta, x, u):
    """p, E[s(1-s)] for each (h, task); u: (T, K) equal-weight points."""
    eta = beta * (x[None, :, None] - h[:, None, None]) + u[None, :, :]
    s = expit(eta)
    return s.mean(2), (s * (1 - s)).mean(2)


def info_matrix_bank(h, beta, x, u):
    """(H, 2, 2) Fisher information over (h, beta) from one run of each bank task."""
    p, v = _terms_bank(h, beta, x, u)
    gh, gb = -beta * v, (x[None, :] - h[:, None]) * v
    d = np.clip(p * (1 - p), 1e-12, None)
    return np.stack([np.stack([(gh * gh / d).sum(1), (gh * gb / d).sum(1)], -1),
                     np.stack([(gb * gh / d).sum(1), (gb * gb / d).sum(1)], -1)], -2)


def info_matrix_new(h, beta, L, tau):
    """(H, 2, 2) Fisher information from one run of one new task of log2-length L, u ~ N(0, tau^2)."""
    eta = beta * (L - h[:, None]) + tau * Z[None, :]
    s = expit(eta)
    p, v = s @ WZ, (s * (1 - s)) @ WZ
    gh, gb = -beta * v, (L - h) * v
    d = np.clip(p * (1 - p), 1e-12, None)
    return np.stack([np.stack([gh * gh / d, gh * gb / d], -1), np.stack([gb * gh / d, gb * gb / d], -1)], -2)


def se_h(I, sd_beta):
    """SE of h with the slope estimated (prior precision 1/sd_beta^2 on beta)."""
    J = I.copy()
    J[..., 1, 1] += 1.0 / sd_beta**2
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] ** 2
    return np.sqrt(J[..., 1, 1] / det)


def info_bank(h, beta, x, u):
    """Known-slope information I_hh (kept for comparison; see P16)."""
    return info_matrix_bank(h, beta, x, u)[..., 0, 0]


def tasks_needed(h_target, beta, sd_beta, x, u, L, tau, target_se=0.5, cap=10_000):
    """Smallest n new tasks of log2-length L (one run each) with SE(h_target) <= target_se, slope estimated."""
    h = np.array([h_target])
    I0 = info_matrix_bank(h, beta, x, u)[0]
    In = info_matrix_new(h, beta, L, tau)[0]
    se = lambda n: se_h((I0 + n * In)[None], sd_beta)[0]
    if se(0) <= target_se:
        return 0
    if se(cap) > target_se:
        return cap
    lo, hi = 0, 1
    while se(hi) > target_se:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if se(mid) > target_se else (lo, mid)
    return hi


def token_cost_model(x_bank, cost_bank):
    """log(tokens) ~ a + b * x, fitted on tasks of >= 16 min, used to price hypothetical new tasks (extrapolation)."""
    b, a = np.polyfit(x_bank, np.log(cost_bank), 1)
    return lambda L: float(np.exp(a + b * L)), {"a": float(a), "b": float(b)}
