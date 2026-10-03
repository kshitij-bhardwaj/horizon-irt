# Design decision log: IRT study of METR time horizons

Each entry records what was decided, why, the alternatives considered, and how it affects the report.
Entries are append-only. If a later finding overturns a decision, a new entry says so and links back.

Notation used throughout:
- Agents (AI models): a = 1..A.
- Tasks: t = 1..T, grouped into families f(t).
- Cells: (a, t) pairs, each with n_at runs and k_at successes.
- x_t = log2(human minutes for task t).
- σ(z) = 1/(1+e^{-z}), logit(p) = log(p/(1-p)).
- ω_at: per-run weight of cell (a, t).

---

## D1. Dataset: TH1.1 as primary, TH1.0 as replication (2026-10-01)

- **Decision.** The primary dataset is `reports/time-horizon-1-1/data/raw/runs.jsonl` at commit `52cb829`: 20 agents, 228 tasks, 79 families, about 23k runs. Headline results are replicated on TH1.0 (33 agents, 170 tasks).
- **Why.** TH1.1 is the current task suite. It has more long tasks (31 of 8 h or more) and is what METR now reports. TH1.0 is an independent replication with different tasks and more agents.
- **Excluded.** The `human` alias. Humans are baseliners, not agents being evaluated, and their success is defined differently.

## D2. Unit of analysis: binomial cells (2026-10-01)

- **Decision.** Runs are aggregated to (agent, task) cells with k_at successes out of n_at runs. The likelihood is binomial: runs are treated as conditionally independent given the cell's success probability.
- **Why.** The weight `invsqrt_task_weight` is constant within a cell (verified: within-cell std = 0), so aggregating loses nothing for any model here. It also cuts the data from about 23k rows to 4523 cells.
- **Caveat checked later (E2).** Only 15.9% of cells have mixed outcomes. If runs were binomial with a moderate p, more cells would be mixed. This points to agent×task interaction (over-dispersion) beyond what a task effect explains. Measured in E2 and modelled in E1 with the factor model.

## D3. Outcome: `score_binarized` (2026-10-01)

- **Decision.** Use `score_binarized`, as METR's headline does.
- **Note.** 13.5% of runs have a fractional `score_cont`. METR's code splits fractional y into weighted 0/1 pseudo-observations. That isn't needed for binarized scores.

## D4. Weighting modes and the scale of the penalty (2026-10-01)

Three per-run weight modes:

| Mode | ω for a run of agent a on task t | Use |
|---|---|---|
| `none` | 1 | Proper likelihood. Used for τ estimation, likelihood-ratio tests, and predictive comparisons. |
| `metr` | invsqrt_task_weight (sums to 1 per agent) | Exact reproduction of METR's estimator, including its penalty λ. |
| `count` | invsqrt_task_weight · N_a (sums to N_a, the agent's run count) | Diversity-weighted pseudo-likelihood on count scale. Used for weighted horizon estimates from joint models. |

- **Why `count` exists.** In a joint model with a prior on task effects, the absolute scale of the weights matters relative to the prior. With weights that sum to 1, the prior would swamp the data. `count` keeps METR's relative weighting while keeping the likelihood on the scale of the actual number of runs.
- **Consequence (used in E4).** METR's objective is Σ ω ℓ + (λ/2)β² with Σω = 1. Multiplying through by N_a gives an equivalent count-scale objective with penalty (λ N_a/2)β². So METR's λ is a Gaussian prior β ~ N(0, 1/(λ N_a)) on a count-scale likelihood. With N_a ≈ 1000 and λ = 0.1, the prior s.d. is about 0.1 logit per doubling of task length. That is informative compared with typical slopes.

## D5. Model family: one linear predictor, nested sub-models (2026-10-01)

Every model is a special case of

    η_at = α_a + β_a (x_t − x̄) + u_t + p_aᵀ q_t,     P(success) = σ(η_at)

| Code | Name | α_a | slope | u_t | p_aᵀq_t | Role |
|---|---|---|---|---|---|---|
| B0 | Agent-only | ✓ | 0 | – | – | Floor |
| B1 | METR per-agent logistic | ✓ | β_a | – | – | **The paper's method (baseline)** |
| B2 | Common-slope logistic | ✓ | β | – | – | Do per-agent slopes matter? |
| R | Rasch (1PL IRT), no length | ✓ | 0 | ~N(0, τ²) | – | Pure psychometric; ignores human time |
| L | Explanatory IRT (LLTM-R), common slope | ✓ | β | ~N(0, τ²) | – | Difficulty = length + residual |
| M | **Agent-slope LLTM-R (proposed)** | ✓ | β_a | ~N(0, τ²) | – | Nests B1 exactly at τ = 0 |
| F_K | M + rank-K interaction | ✓ | β_a | ~N(0, τ²) | rank K | Agent×task specialisation |

- **Why this family.** METR's model is a 2PL IRT model with *known* item difficulty x_t and an agent-specific discrimination β_a. The question "does difficulty depend on more than length?" becomes "is τ > 0?", tested inside a nested family. τ = 0 gives back METR exactly; as τ grows, the model approaches Rasch with a length covariate. References: Fischer (1973) LLTM; De Boeck (2008) random-item IRT; Janssen et al. (2000).
- **Why x is centred at x̄ (the mean of x_t over tasks).** For numerical conditioning only. Slopes are unchanged, and horizons are reported in original units.
- **Rejected: free fixed-effect difficulties b_t.** 76 of 228 tasks are never or always solved, so their maximum-likelihood b_t would be ±∞. The Gaussian random effect gives finite, shrunken estimates and a single interpretable variance τ².

## D6. Estimation: MML for random-effect models, MAP elsewhere (2026-10-01)

- **Decision.** Models R, L and M are fitted by **marginal maximum likelihood** (MML; Bock & Aitkin 1981). Each task's u_t is integrated out of the likelihood:
  log L = Σ_t log ∫ Π_a Bin(k_at | n_at, σ(η⁰_at + u)) N(u; 0, τ²) du.
  The likelihood is maximised over (α, β, τ). Each task's posterior over u_t is kept for prediction. Gradients come from Fisher's identity: ∂log L_t/∂θ = E_{u|data}[∂ log p(data_t | u)/∂θ]. They are verified by finite differences (test T1).
  B0, B1, B2 and F_K are fitted by penalised maximum likelihood (L-BFGS-B, gtol 1e-9).
- **Why MML and not the joint MAP over (α, β, u).** The joint MAP has no valid objective for τ: it collapses to τ → 0. MML gives a genuine likelihood for τ, so likelihood-ratio tests, AIC/BIC and profile CIs are all well defined.
- **Validation.** T4 simulates 5 datasets from the fitted M on the real design. The mean τ̂ is 2.87 against a true 2.92 (−1.7%). Under τ = 0, the boundary LRT rejected 2 of 30 times (6.7%) at nominal 5%, and the mean statistic was 0.62 against 0.5 theoretical. **The test is calibrated.**

## D7. Quadrature: uniform 321-point grid on z ∈ [−8, 8] (2026-10-01)

- **Decision.** Integrate u = τz on a fixed uniform grid with normalised N(0,1) masses, not Gauss–Hermite nodes.
- **Why.** τ̂ ≈ 2.9, and a task seen by 20 agents × 5 runs has a posterior s.d. of about 0.2–0.4 logits. Fixed GH nodes with Q ≈ 60 are spaced about 1 logit apart near the mode, too coarse for that. The uniform grid spacing is 0.05τ ≈ 0.15 logits. Adaptive GH (Pinheiro & Bates 1995) would also work but complicates the gradients.
- **Validation.** T3: per-task log marginal likelihoods agree with adaptive `scipy.integrate.quad` (rtol 1e-12) to within 1e-4.

## D8. Predictive evaluation and uncertainty (2026-10-01)

- **Metrics.** Mean per-run negative log-likelihood (NLL; the primary metric, a proper scoring rule), Brier score, AUROC and ECE (10 equal-width bins). Each is computed run-weighted (primary) and with METR's per-agent diversity weights (secondary, matching the estimand METR's horizons use).
- **Uncertainty.** Differences between models are paired by cell and summarised by a cluster bootstrap over **task families** (2000 resamples). Families are the top-level sampling unit of the suite, and tasks within a family are correlated, so a cell-level bootstrap would understate the CI width.

## D9. Penalty variants for the slope (2026-10-01)

- **Ridge toward 0, (λ/2)Σβ_a²** (METR's choice). Under D4's count-scale view this is a prior β ~ N(0, 1/(λN_a)) centred on "task length is irrelevant", which is not plausible.
- **Hierarchical, (κ/2)Σ(β_a − μ)²**, with μ estimated. This shrinks agents toward the population slope instead of toward 0. It is studied in E4 as a principled replacement.

## D10. In-sample evidence (E2) drives the next steps (2026-10-01)

- τ̂_M = 2.92 (profile 95% CI 2.59–3.31). Length explains 80% of latent difficulty variance, but the residual SD equals τ/|β_L| ≈ 2.4 doublings of task length.
- **Over-dispersion.** 15.9% of multi-run cells are mixed, against 22.3% expected under M (z = −14.7). Runs of the same agent on the same task agree more than a task-only effect allows, which points to agent×task interaction. **Decision:** include the factor models F_K in E1, and report this as a limit of any length-plus-task-effect model.
- **Shrinkage artefact.** For tasks that are always solved, the posterior mean û_t increases mechanically with η⁰ (the SWAA band in Fig. E2a). This is a property of the posterior, not evidence about the task. Figures and tables should show û_t with its posterior s.d. and solve rate, not on its own.

## D11. E1 protocols and fairness rules (2026-10-01)

- **A (held-out cells).** 5-fold CV over (agent, task) cells, seed 0. All runs of a cell go to the same fold. Random-effect models predict with the full posterior predictive E[σ(η⁰ + u_t) | training data of t]. MAP models (F_K) predict with plug-ins.
- **B (held-out task families).** 5-fold CV over families: everything about a held-out family is unseen. Random-effect models must predict with the **prior** predictive E_u σ(η⁰ + u), which is the correct Bayesian predictive for a new item. The plug-in u = 0 is reported too, to show why that matters.
- **C (new agent).** For each agent: fit on the other 19 agents, reveal m ∈ {8, 16, 32, 64} of its tasks uniformly at random (10 repeats), estimate (α, β), predict its hidden tasks, and recover METR's full-data p50 for that agent (its paper estimator on all of the agent's data).
  - **Fairness.** Both methods get a Gaussian prior on (α, β) fitted to the other agents' estimates *from the same model*, because conditional and marginal slopes live on different scales (D10/E3). METR is also shown with a flat prior (METR as published).
  - **Weights.** Count-scale (D4), because the target is the diversity-weighted horizon.
- **Factor models.** ρ ∈ {1, 3, 10, 30} and K ∈ {1, 2, 3} are chosen on an inner 10% split of each training fold, never on the test fold. τ for F_K is fixed at the fold's MML τ̂ (empirical-Bayes plug-in). F0 (the same MAP fitter, K = 0) separates "MAP plug-in vs integration" from "adding factors".
- **Observed.** Inner validation mostly picks ρ = 10, sometimes 30. At large ρ the ridge on (P, Q) acts as a nuclear-norm penalty on PQᵀ and zeroes the factors.

## D12. Bootstrap pseudo-items (2026-10-01)

- **Decision.** In the 3-level bootstrap, a family drawn twice, or a task drawn twice within a family draw, becomes a distinct pseudo-item (`task#j.i`).
- **Why.** METR's per-agent logistic is indifferent to this. A random-effect model, though, would otherwise give both copies one shared u_t and double-count that single task's evidence about u_t. Pseudo-items match the nonparametric-bootstrap logic of resampling items i.i.d.
- **Weights.** METR's original invsqrt weights are kept in the bootstrap, as METR does.

## D13. Common SOTA set for paired trend comparisons (2026-10-01)

- **Decision.** In E3 and E4, every estimator's doubling time is fitted on the **same** agents: the SOTA set defined by METR's full-data p50.
- **Why.** If each estimator chose its own frontier, differences in doubling time would mix estimator effects with agent-selection effects. Paired bootstrap differences then isolate the estimator.

## D14. Protocol C reports two horizon targets (2026-10-01)

- **Observation.** At m = 64, the IRT estimate is *further* from METR's full-data p50 than METR's own small-sample estimate is (median |error| 0.44 vs 0.35 log2), although IRT is better at m ≤ 32.
- **Cause.** The target is METR's estimator on full data. A small-sample METR fit converges to that target by construction. M converges to M's full-data p50, which differs systematically (E3).
- **Decision.** Report the error against both (i) METR's full-data p50 (a common yardstick) and (ii) each method's own full-data p50 (sampling error only). Interpret (ii) as efficiency and (i) − (ii) as between-estimator offset.

## D15. Explaining the METR-vs-M p50 offset: a simulation, not a story (2026-10-01)

- **Observation (E3).** M's p50 differs from METR's: −21% for Opus 4.5, −28% for GPT-5, +30% for GPT-4-era models. This holds even though p50 is exactly invariant *within* M (T5).
- **First hypothesis: realised task effects near each agent's horizon.** It explains only part of the gap: the correlation between the observed offset and the information-weighted mean û is 0.70. Frontier agents have a negative mean û but a positive offset.
- **Decision.** Test for a *structural* bias with a simulation from the fitted M (E3b). Data are drawn with fresh u_t, so the true p50 is known, and METR's estimator is applied. A second arm fixes u_t at the real posterior means to measure the "realised tasks" component separately.

## D16. Stability sweeps (E4) (2026-10-01)

- **Ridge grid.** λ ∈ {1e-5, …, 0.3}, on METR's own weight scale (Σω = 1 per agent), so values compare directly with METR's 1e-5 and 0.1.
- **First-order theory.** The pivot formula log2 H(λ) − x̄_v ≈ (log2 H(0) − x̄_v)(1 + λ/I_a) is checked against exact refits. Its maximum error is 2e-6 log2 at λ = 1e-5, 8e-4 at 1e-2, and 0.05 at 0.1. So the explanation is quantitatively right for the λ METR used.
- **Prior comparison.** On count scale (D4), because κ and λ are only comparable with a likelihood on the scale of the actual run counts. Selection uses unseen-family NLL (protocol B), METR-weighted, because that is the estimand.
- **κ grid extended.** The first run selected κ = 100 at the grid edge. The grid now runs to 1e4. As κ → ∞ the model becomes the common-slope B2.

## D17. Slope prior: hierarchical shrinkage is selected and kept (2026-10-01)

- **Result (E4c, extended grid).** Unseen-family NLL (METR-weighted) improves steadily as κ grows, then plateaus from κ ≈ 1000: ΔNLL = −0.0019 [−0.0036, −0.0002]. κ = 3000 is the point-estimate optimum: −0.0020 [−0.0041, +0.0001]. Ridge toward 0 never helps, and at λ_count = 1000 it hurts (+0.035).
- **Interpretation.** The between-agent spread of *marginal* slopes is small compared with how precisely each slope is estimated. This matches protocol A, where per-agent slopes add nothing out of sample without task effects. Shrinking to the common slope is the right default, and shrinking toward 0 (METR's ridge) is the wrong target.
- **Effect.** The frontier failure case stabilises. Opus 4.6's p50 goes from 719 min [325, 4608] to 525 [289, 1217]: CI width 3.8 → 2.1 log2. Doubling times barely move (2023+: 128.7 → 130.5 d; CIs overlap almost exactly).
- **Caveat recorded for the report.** Neighbouring κ values are statistically indistinguishable (the CV curve is flat), so κ should be reported as a range (10³ to 10⁴), not a sharp choice.

## D18. E3b/E3c outcomes and how they are reported (2026-10-01)

- **E3b, structural arm (fresh u).** METR's p50 is biased downward at the frontier when difficulty varies beyond length: −0.17 log2 for Opus 4.5, −0.39 for Opus 4.6, about +0.03 for GPT-4-era models. corr(bias, true log2 p50) = −0.89. The effect on the 2023+ doubling time is +4.8% (147 vs 141 d), which is 0.54 SD of its sampling spread: small next to noise.
  M's own estimator has a roughly flat offset of −0.08 log2. That offset does not affect trends, and it is recorded as a limitation (count-weighted pseudo-likelihood plus a slight underestimate of τ in MML; see T4).
- **E3b, realised arm.** With u_t fixed at the real posterior means, simulated METR-vs-truth shifts reproduce the *observed* METR-vs-M offsets with **R² = 0.995** (RMSE 0.028 log2). Conclusion: METR's per-agent horizons contain a component of up to ±0.5 log2 that depends on which tasks are in the suite.
- **E3c (out-of-suite test of that conclusion).** Across 16 agents present in both TH1.0 and TH1.1, the SD of the mean-centred cross-suite difference in log2 p50 is 0.46 for M vs 0.57 for METR (ratio 0.80, bootstrap 95% CI [0.50, 1.23], P(ratio ≥ 1) = 0.17).
  **Reported as directionally consistent but not significant.** Power is limited by n = 16 and by agent-specific infrastructure shifts (Vivaria → Inspect).
- **Not done (and why).** Pairing the suites on shared tasks would isolate the suite effect, but the shared tasks were re-versioned between suites (TH1.1 "updated 53"), so their identity is not guaranteed. Left as future work.

## D19. E3 trend result and figure correction (2026-10-01)

- **Paired bootstrap (200 replicates).** M p50 − METR p50 doubling time: +15.3 d [−6.2, +36.0] for 2023+ and +9.5 d [−9.8, +27.5] for 2024+. **Not significant.** It is reported as "the trend is robust; individual horizons are not".
- **Figure correction.** The first version of `e3_p80` labelled the p50 tick "both" and tested attenuation with METR's estimator. That was wrong after E3b: METR's p50 ≠ M's p50, and the attenuation theory concerns M's own marginal vs conditional curves. The figure now plots M-marginal (the theory test, which lies on the line s = 1.94) and METR (empirical, scattered by the suite-specific offset) separately.
- **Engineering.** `exp3_horizons.py --from-saved` summarises the saved bootstrap samples without refitting. A pandas MultiIndex join failed after the 35-minute bootstrap had finished, so this path was added to avoid rerunning it.

## D20. Related-work positioning: what is and is not new here (2026-10-01)

Prior work found after the study was run, which must be cited and positioned against:

- **BRIDGE** (Liu, Gala, Nilaksh, Bahdanau, Reddy, Larochelle; arXiv:2602.07267; Mila/McGill; code McGill-NLP/BRIDGE). Fits a 2PL IRT model jointly on METR's suites plus SWE-bench Verified, MLE-bench, GDPval and Cybench. Shows that latent difficulty is linear in log human time (R² = 0.81) and reproduces METR's trend (about 6-month doubling). Uses Bayesian MCMC and analyses 80% thresholds.
  It has no task random-effect variance or variance decomposition, no analysis of METR's estimator (bias, regularisation, suite-specific component), no adaptive or new-agent evaluation, and no test of unseen-task prediction against METR. Its future work lists "uncertainty-aware difficulty estimation".
- **Agent psychometrics** (Ge, Kryvosheieva, Fried, Girit, Hariharan; arXiv:2604.00594). IRT plus task features for agentic coding benchmarks, with a decomposition into LLM ability and scaffold ability.
- **CurveShift** (Xing et al.; arXiv:2608.00355). A Rasch model on METR data arguing that apparent shifts toward harder tasks are mostly ceiling effects.

**Consequence.** "IRT difficulty tracks log human time" (E2's R² ≈ 0.80) is a *replication*, not a contribution. The contributions to claim are:
1. A nested random-effects model with exact METR nesting, an MML τ, and calibrated tests.
2. The out-of-sample split showing IRT gains only for known tasks, plus the population-averaged equivalence for new tasks.
3. p50 invariance and the p80 attenuation reading.
4. METR estimator analysis: the closed-form ridge pivot, a structural frontier bias, and a suite-specific component (R² = 0.995).
5. The hierarchical-slope fix for the frontier failure case.
6. Better new-agent sample efficiency.

Also noted: METR staff have publicly said that many methodology critiques "are not new". External communication should lead with constructive, quantified items (4 and 5, and the adaptive-testing prototype), not with critique.
