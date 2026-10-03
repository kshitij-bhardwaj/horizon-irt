"""Load METR runs and aggregate to binomial (agent, task) cells (decision D2)."""

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2] / "eval-analysis-public"


@dataclass
class Cells:
    df: pd.DataFrame          # one row per (agent, task): a, t, n, k, x, w_metr, w_count, family, source
    agents: list[str]
    tasks: pd.DataFrame       # one row per task: task_id, x, family, source, minutes
    xbar: float               # centring constant for x (D5)
    dates: dict[str, str]

    @property
    def A(self) -> int:
        return len(self.agents)

    @property
    def T(self) -> int:
        return len(self.tasks)

    def weights(self, mode: str) -> np.ndarray:
        """Per-run weight of each cell under weighting mode D4."""
        return {"none": np.ones(len(self.df)), "metr": self.df.w_metr.values,
                "equal": self.df.w_equal.values, "count": self.df.w_count.values}[mode]

    def subset(self, mask: np.ndarray) -> "Cells":
        """Same agent/task indexing, fewer cells (for cross-validation)."""
        return Cells(self.df[mask].reset_index(drop=True), self.agents, self.tasks, self.xbar, self.dates)


def aggregate(runs: pd.DataFrame, dates: dict[str, str]) -> Cells:
    runs = runs[runs.alias != "human"]
    g = runs.groupby(["alias", "task_id"], sort=True)
    df = g.agg(n=("score_binarized", "size"), k=("score_binarized", "sum"),
               w_metr=("invsqrt_task_weight", "first"), w_equal=("equal_task_weight", "first"),
               family=("task_family", "first"),
               minutes=("human_minutes", "first"), source=("task_source", "first")).reset_index()
    agents = sorted(df.alias.unique())
    tasks = (df.groupby("task_id", sort=True)
               .agg(family=("family", "first"), minutes=("minutes", "first"), source=("source", "first"))
               .reset_index())
    tasks["x"] = np.log2(tasks.minutes)
    df["a"] = df.alias.map({s: i for i, s in enumerate(agents)})
    df["t"] = df.task_id.map({s: i for i, s in enumerate(tasks.task_id)})
    df["x"] = np.log2(df.minutes)
    # Count-scale weights (D4): keep METR's relative weights, rescale each agent to sum to its run count.
    n_runs = df.groupby("a").n.transform("sum")
    w_sum = (df.w_metr * df.n).groupby(df.a).transform("sum")
    df["w_count"] = df.w_metr * n_runs / w_sum
    return Cells(df, agents, tasks, float(tasks.x.mean()), dates)


def load(report: str = "time-horizon-1-1", root: pathlib.Path = REPO) -> Cells:
    runs = pd.read_json(root / f"reports/{report}/data/raw/runs.jsonl", lines=True)
    dates = yaml.safe_load(open(root / "data/external/release_dates.yaml"))["date"]
    return aggregate(runs, dates)


def load_runs(report: str = "time-horizon-1-1", root: pathlib.Path = REPO) -> tuple[pd.DataFrame, dict]:
    runs = pd.read_json(root / f"reports/{report}/data/raw/runs.jsonl", lines=True)
    dates = yaml.safe_load(open(root / "data/external/release_dates.yaml"))["date"]
    return runs[runs.alias != "human"].reset_index(drop=True), dates
