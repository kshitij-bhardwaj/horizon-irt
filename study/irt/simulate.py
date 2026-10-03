"""Simulate binomial cells from model M on a fixed design (same agents, tasks, run counts, task lengths)."""

import numpy as np
from scipy.special import expit

from .data import Cells


def simulate(cells: Cells, alpha, beta, tau, seed=0) -> tuple[Cells, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = cells.df.copy()
    u = tau * rng.standard_normal(cells.T)
    eta = alpha[df.a.values] + beta[df.a.values] * (df.x.values - cells.xbar) + u[df.t.values]
    df["k"] = rng.binomial(df.n.values, expit(eta))
    return Cells(df, cells.agents, cells.tasks, cells.xbar, cells.dates), u
