"""Three-level hierarchical bootstrap (family -> task -> run), as in METR, with one change (decision D12):
a family or task drawn more than once becomes a *distinct* pseudo-item (suffix '#j'), so random-effect models
give each copy its own u_t instead of double-counting one task's evidence."""

import numpy as np
import pandas as pd


class HierBootstrap:
    def __init__(self, runs: pd.DataFrame):
        self.runs = runs.reset_index(drop=True)
        self.fam_tasks = {f: g.task_id.unique() for f, g in self.runs.groupby("task_family")}
        self.task_rows = {}
        for t, g in self.runs.groupby("task_id"):
            g = g.sort_values("alias")
            _, starts, counts = np.unique(g.alias.values, return_index=True, return_counts=True)
            self.task_rows[t] = (g.index.values, np.repeat(starts, counts), np.repeat(counts, counts))
        self.fams = np.array(list(self.fam_tasks))

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        idx, tid, fid = [], [], []
        for j, f in enumerate(rng.choice(self.fams, len(self.fams))):
            tasks = self.fam_tasks[f]
            for i, t in enumerate(rng.choice(tasks, len(tasks))):
                rows, start, count = self.task_rows[t]
                pick = rows[start + (rng.random(len(rows)) * count).astype(int)]
                idx.append(pick)
                tid += [f"{t}#{j}.{i}"] * len(pick)
                fid += [f"{f}#{j}"] * len(pick)
        out = self.runs.iloc[np.concatenate(idx)].copy()
        out["task_id"], out["task_family"] = tid, fid
        return out
