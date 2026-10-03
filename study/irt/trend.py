"""Doubling-time trend over SOTA agents, matching METR's definitions (validated in replication/VALIDATION.md).

SOTA: an agent whose horizon is >= the best horizon among agents released on or before its release date.
Doubling time: 1 / slope of OLS of log2(horizon) on release date (days).
"""

import numpy as np
import pandas as pd


def sota_agents(horizon: dict[str, float], dates: dict[str, str], after: str, before: str = "2030-01-01"):
    d = pd.DataFrame({"h": pd.Series(horizon), "date": [pd.Timestamp(dates[a]) for a in horizon]})
    d = d[(d.date >= pd.Timestamp(after)) & (d.date < pd.Timestamp(before))].sort_values("date")
    best, keep = -np.inf, []
    for _, g in d.groupby("date", sort=True):
        best = max(best, g.h.max())
        keep += list(g.index[g.h >= best])
    return keep


def doubling_days(horizon: dict[str, float], dates: dict[str, str], agents: list[str]) -> float:
    t = np.array([(pd.Timestamp(dates[a]) - pd.Timestamp("2019-01-01")).days for a in agents], float)
    y = np.log2([horizon[a] for a in agents])
    return float(1 / np.polyfit(t, y, 1)[0])
