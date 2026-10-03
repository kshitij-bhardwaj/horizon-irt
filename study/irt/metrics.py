"""Run-level predictive metrics on binomial cells, and a paired family-cluster bootstrap (decision D8)."""

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def cell_nll(p, n, k):
    """Sum over the cell's runs of -log P(outcome): Bernoulli per run, no binomial constant."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(k * np.log(p) + (n - k) * np.log1p(-p))


def summarise(df, p, weight_col: str | None = None) -> dict:
    """Mean per-run NLL, Brier score, AUROC and ECE (10 equal-width bins).

    weight_col=None   : every run counts equally.
    weight_col='w_metr': METR's diversity weights (each agent's runs sum to 1) -> per-agent-equal, task-weighted.
    """
    n, k = df.n.values.astype(float), df.k.values.astype(float)
    w = np.ones(len(df)) if weight_col is None else df[weight_col].values
    runs = w * n
    nll = np.sum(w * cell_nll(p, n, k)) / runs.sum()
    brier = np.sum(w * (k * (1 - p) ** 2 + (n - k) * p ** 2)) / runs.sum()
    y = np.r_[np.ones(len(p)), np.zeros(len(p))]
    sw = np.r_[w * k, w * (n - k)]
    keep = sw > 0
    auc = roc_auc_score(y[keep], np.r_[p, p][keep], sample_weight=sw[keep])
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += abs(np.sum(w[m] * k[m]) - np.sum(w[m] * n[m] * p[m])) / runs.sum()
    return {"nll": nll, "brier": brier, "auroc": auc, "ece": ece}


def paired_family_bootstrap(df, p_a, p_b, n_boot=2000, seed=0, weight_col=None):
    """95% CI for mean per-run NLL(model a) - NLL(model b), resampling task families (the top-level
    sampling unit of the task suite) with replacement. Negative = model a better."""
    n, k = df.n.values.astype(float), df.k.values.astype(float)
    w = np.ones(len(df)) if weight_col is None else df[weight_col].values
    d = w * (cell_nll(p_a, n, k) - cell_nll(p_b, n, k))
    fam_codes, fam = np.unique(df.family.values, return_inverse=True)
    D = np.bincount(fam, d, minlength=len(fam_codes))
    R = np.bincount(fam, w * n, minlength=len(fam_codes))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fam_codes), size=(n_boot, len(fam_codes)))
    boots = D[idx].sum(1) / R[idx].sum(1)
    return {"diff": D.sum() / R.sum(), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "p_one_sided": float(np.mean(boots >= 0))}
