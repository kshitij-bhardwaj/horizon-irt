"""Item bank for HorizonCAT, built from the study's model M (P2, P3).

Each task t carries:
  x_t         log2 human minutes (the anchor, METR's difficulty scale; BRIDGE's linearity result)
  u_tk, k=1..K  K equal-mass quantile points of the posterior of its residual difficulty u_t given the bank agents
  cost_t      median tokens per run across all METR runs of the task (compute-cost proxy, P4)
Population prior for slopes: beta ~ N(mu_b, sd_b^2) from the bank agents' M estimates.
"""

import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

STUDY = pathlib.Path(__file__).resolve().parents[2] / "study"
sys.path.insert(0, str(STUDY))
from irt import data as sdata, models as smodels  # noqa: E402

K_QUANTILES = 10


@dataclass
class Bank:
    task_ids: list[str]
    x: np.ndarray          # (T,)
    u: np.ndarray          # (T, K) equal-weight support points of p(u_t | bank data)
    cost: np.ndarray       # (T,) tokens
    pass_rate: np.ndarray  # (T,) historical pass rate among bank agents (for the static-filter baseline)
    source: list[str]
    family: list[str]
    mu_b: float
    sd_b: float
    tau: float
    agents: list[str]      # agents used to build the bank

    def to_json(self) -> dict:
        r = lambda a, d=3: np.round(a, d).tolist()
        return {"task_ids": self.task_ids, "x": r(self.x), "u": r(self.u), "cost": r(self.cost, 0),
                "pass_rate": r(self.pass_rate), "source": self.source, "family": self.family,
                "mu_b": round(self.mu_b, 4), "sd_b": round(self.sd_b, 4), "tau": round(self.tau, 4),
                "agents": self.agents}


def task_costs(runs: pd.DataFrame) -> pd.Series:
    t = runs[(runs.alias != "human") & runs.tokens_count.notna() & (runs.tokens_count > 0)]
    return t.groupby("task_id").tokens_count.median()


def quantile_points(log_post_row: np.ndarray, tau: float, K: int = K_QUANTILES) -> np.ndarray:
    """K equal-mass points: the conditional means of u within the K posterior quantile bins (preserves the mean)."""
    w = np.exp(log_post_row - log_post_row.max())
    w /= w.sum()
    u = tau * smodels.Z_GRID
    cdf = np.cumsum(w)
    edges = np.linspace(0, 1, K + 1)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # mass of each grid point falling in [lo, hi) of the CDF (handles partial cells)
        prev = np.concatenate([[0.0], cdf[:-1]])
        m = np.clip(np.minimum(cdf, hi) - np.maximum(prev, lo), 0, None)
        pts.append(np.sum(m * u) / m.sum())
    return np.array(pts)


def build(cells: "sdata.Cells", runs: pd.DataFrame, exclude_agent: str | None = None) -> tuple[Bank, smodels.Fit]:
    keep = np.ones(len(cells.df), bool) if exclude_agent is None else (cells.df.alias != exclude_agent).values
    C = cells.subset(keep)
    b1 = smodels.fit_map(C, "agent")
    m = smodels.fit_mml(C, "agent", init=b1)
    agents_in = [a for a in cells.agents if a != exclude_agent]
    idx = [cells.agents.index(a) for a in agents_in]
    U = np.array([quantile_points(m.log_post[t], m.tau) for t in range(cells.T)])
    costs = task_costs(runs).reindex(cells.tasks.task_id).values
    costs = np.where(np.isfinite(costs), costs, np.nanmedian(costs))
    pr = C.df.groupby("t").apply(lambda g: g.k.sum() / g.n.sum(), include_groups=False).reindex(range(cells.T)).values
    bank = Bank(task_ids=cells.tasks.task_id.tolist(), x=cells.tasks.x.values, u=U, cost=costs,
                pass_rate=np.nan_to_num(pr, nan=0.5), source=cells.tasks.source.tolist(),
                family=cells.tasks.family.tolist(), mu_b=float(np.mean(m.beta[idx])),
                sd_b=float(np.std(m.beta[idx], ddof=1)), tau=m.tau, agents=agents_in)
    return bank, m
