"""E3c: out-of-suite test of E3b's claim that METR horizons carry a task-suite-specific component.

Prediction: if IRT (model M) removes the realised-task component, the *same model's* horizon should agree better
between the TH1.0 and TH1.1 task suites under M than under METR's estimator.
Metric: SD across agents of d_a = log2 H_a(TH1.1) - log2 H_a(TH1.0) after removing the mean shift (the mean absorbs
suite-wide differences such as the Vivaria -> Inspect migration, which affects both estimators alike).
Uncertainty: bootstrap over agents (paired across estimators).
"""

import json
import pathlib

import numpy as np
import pandas as pd

from irt import data, models, plotstyle
from irt.plotstyle import INK2, MUTED, SLOTS, plt

OUT = pathlib.Path(__file__).parent


def horizons(report):
    C = data.load(report)
    metr = models.fit_map(C, "agent", weights="metr", lam=1e-5)
    hier = models.fit_map(C, "agent", weights="count", hier_kappa=3000.0)
    b1c = models.fit_map(C, "agent", weights="count")
    m = models.fit_mml(C, "agent", weights="count", init=b1c)
    names = [a.replace(" (Inspect)", "") for a in C.agents]
    return pd.DataFrame({"METR": np.log2(metr.horizons(C.agents, ps=(0.5,))["cond_p50"]),
                         "METR + hier. slope": np.log2(hier.horizons(C.agents, ps=(0.5,))["cond_p50"]),
                         "IRT (M)": np.log2(m.horizons(C.agents, ps=(0.5,))["cond_p50"])}, index=names)


def main():
    h10, h11 = horizons("time-horizon-1-0"), horizons("time-horizon-1-1")
    common = sorted(set(h10.index) & set(h11.index))
    D = h11.loc[common] - h10.loc[common]
    D.to_csv(OUT / "results/e3c_cross_suite_diffs.csv")
    sd = D.std(ddof=1)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(5000):
        i = rng.integers(0, len(common), len(common))
        boots.append(D.iloc[i].std(ddof=1))
    B = pd.DataFrame(boots)
    ratio = B["IRT (M)"] / B["METR"]
    res = {"agents": common, "n": len(common), "mean_shift": D.mean().to_dict(), "sd_resid": sd.to_dict(),
           "sd_ci": {c: [float(B[c].quantile(.025)), float(B[c].quantile(.975))] for c in B},
           "ratio_IRT_over_METR": {"point": float(sd["IRT (M)"] / sd["METR"]),
                                   "ci": [float(ratio.quantile(.025)), float(ratio.quantile(.975))],
                                   "p_ratio_ge_1": float((ratio >= 1).mean())},
           "corr_across_suites": {c: float(np.corrcoef(h10.loc[common, c], h11.loc[common, c])[0, 1]) for c in D}}
    json.dump(res, open(OUT / "results/e3c_cross_suite.json", "w"), indent=2)
    print(D.round(3).to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "agents"}, indent=1))

    plotstyle.setup()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    order = D["METR"].sort_values().index
    y = np.arange(len(order))
    for col, c, mk, dy in [("METR", SLOTS[1], "s", -0.18), ("IRT (M)", SLOTS[0], "o", 0.18)]:
        v = D.loc[order, col] - D[col].mean()
        ax.plot(v, y + dy, mk, color=c, ms=4.5, mec="white", mew=0.6,
                label=f"{col}: SD = {sd[col]:.2f}")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, order, fontsize=6.5)
    ax.set_xlabel("log$_2$ H50(TH1.1) − log$_2$ H50(TH1.0), mean-centred")
    ax.set_title("Same model, two task suites", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures/e3c_cross_suite.pdf"); fig.savefig(OUT / "figures/e3c_cross_suite.png")


if __name__ == "__main__":
    main()
