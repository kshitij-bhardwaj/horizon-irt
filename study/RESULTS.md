# Beyond task length: an item-response analysis of METR's AI time horizons

Study of directions 1 (IRT difficulty) and 2 (estimator stability) on METR Time Horizon 1.1 data (20 agents, 228 tasks, 79 families, 23,235 runs). TH1.0 (33 agents, 170 tasks) is used for replication.

Supporting material:
- Design decisions: [DECISIONS.md](DECISIONS.md), entries D1–D18.
- Code: `irt/` (models, data, metrics, bootstrap, trend), `exp*.py` (experiments), `tests/test_models.py`.
- Every number below is in `results/`, every figure in `figures/` (PDF for LaTeX, PNG for preview), and every run log in `logs/`.

---

## 0. Headline findings

1. **Task difficulty is far from a function of length alone.** In the latent-variable model, length explains about 80% of task-difficulty variance. The residual SD is τ̂ = 2.92 logits (95% profile CI 2.59–3.31), equal to **2.4 doublings of task length**: a task one SD harder than its length predicts behaves like one **5× longer**. The likelihood-ratio test for τ > 0 gives χ² = 5536 (E2).
2. **Knowing which task it is beats knowing how long it takes.** For held-out agent×task cells, an IRT model cuts per-run NLL by **38%** against METR's per-agent logistic (0.310 → 0.192; AUROC 0.929 → 0.974). Almost all of that comes from task identity. A Rasch model that *ignores length entirely* captures 95% of the gain (E1-A).
3. **For brand-new tasks, METR's curve is already right.** Once the task effect is integrated out, the IRT predictive is statistically tied with METR's logistic (ΔNLL +0.003 [−0.005, +0.011]). Plugging in u = 0 instead is significantly worse (+0.064). METR's model *is* the population-averaged (marginal) predictor (E1-B).
4. **IRT makes new-agent evaluation about 2× more sample-efficient.** Measured against each method's own full-data estimate, IRT with 16 revealed tasks (median error 0.42 log2) beats METR with 32 (0.53). Against METR's yardstick, IRT is better for m ≤ 32 and loses at m = 64, because of the between-estimator offset explained in finding 6. IRT also predicts the agent's hidden tasks far better at every budget (NLL 0.20 vs 0.33) (E1-C).
5. **The 50% horizon is invariant to task heterogeneity, but the 80% horizon is not.** We prove and verify that p50 is unchanged by integrating a symmetric task effect. The p80 gap is stretched by s = √(1 + c²τ²) ≈ 1.94. For a typical task (u = 0), frontier models reach 80% reliability at about **⅓ of their p50** (e.g. GPT-5.2: 104 vs 304 min), not the ⅕–⅒ that METR's marginal p80 implies (E3).
6. **METR's per-agent horizons contain a task-suite component of up to ±0.5 log2 (±40%).** A simulation with the suite's actual task effects reproduces the observed METR-vs-IRT differences with **R² = 0.995**. Separately, METR's estimator is structurally biased downward at the frontier under heterogeneity (−0.39 log2 for Opus 4.6). The doubling-time trend is only mildly affected (E3, E3b). An out-of-suite check (TH1.0 vs TH1.1) points the same way but is not significant (E3c).
7. **METR's ridge penalty rotates each fitted curve about a pivot, and we derive the pivot in closed form.** A first-order formula predicts the refitted horizons to within 0.05 log2 at METR's own λ = 0.1. The rotation stretches horizons outward (strong agents up, weak down), shortening the doubling time: 128.7 → 122.6 days at λ = 0.1 (E4a).
8. **Shrinking slopes toward their *common mean* (not toward 0) is the principled fix.** It matches or beats METR on unseen families, narrows the frontier failure case (Opus 4.6's CI width drops from 3.8 to 2.1 log2), and leaves the trend unchanged: 2023+ doubling time 130.5 vs 128.7 days (E4c/d).

---

## 1. Model family and mathematics

### 1.1 Notation

- Agents a = 1..A; tasks t = 1..T; cells (a, t) with n_at runs and k_at successes.
- x_t = log2(human minutes of task t); x̄ = the mean of x_t over tasks.
- σ(z) = 1/(1+e^{−z}).

All models share one linear predictor:

    η_at = α_a + β_a (x_t − x̄) + u_t + p_aᵀ q_t ,     k_at | η_at ~ Binomial(n_at, σ(η_at))

| Model | α_a | slope | task effect u_t | factors | Free parameters (TH1.1) |
|---|---|---|---|---|---|
| B0 | ✓ | 0 | – | – | 20 |
| **B1 (METR)** | ✓ | β_a | – | – | 40 |
| B2 | ✓ | β | – | – | 21 |
| R (Rasch) | ✓ | 0 | N(0, τ²) | – | 21 |
| L (LLTM-R) | ✓ | β | N(0, τ²) | – | 22 |
| **M (proposed)** | ✓ | β_a | N(0, τ²) | – | 41 |
| F_K | ✓ | β_a | N(0, τ̂²) | rank K | 41 + K(A+T) |

METR's model is a 2-parameter logistic IRT model whose item difficulty is *fixed* to log task length, with an agent-specific discrimination β_a. Its p-horizon is

    log2 H_p(a) = x̄ + (logit p − α_a) / β_a .

M nests B1 exactly at τ = 0 (test T6). As τ → ∞ with β_a ≡ 0 it becomes a Rasch model (Fischer 1973; De Boeck 2008).

### 1.2 Estimation

- **B-models and F_K: penalised maximum likelihood.**

      min_θ  −Σ_c ω_c [k_c log σ(η_c) + (n_c − k_c) log σ(−η_c)] + penalties

  This uses L-BFGS-B with analytic gradients, where ∂/∂η_c = ω_c (n_c σ(η_c) − k_c).
- **R, L, M: marginal maximum likelihood.** The task effect is integrated out task by task:

      log L(α, β, τ) = Σ_t log ∫ Π_a Bin(k_at | n_at, σ(η⁰_at + u)) N(u; 0, τ²) du

  Gradients follow Fisher's identity: ∂ log L_t/∂θ = E_{u|data_t}[∂ log p(data_t | u)/∂θ]. For τ, with u = τz: ∂ log L_t/∂ log τ = τ E[z · Σ_{c∈t} ω_c (k_c − n_c σ(η⁰_c + τz))].

  The integral uses a 321-point uniform grid in z ∈ [−8, 8] (D7). The task posteriors p(u_t | data) are kept and drive prediction.
- **Weights (D4).** Proper likelihood (ω = 1) for tests and prediction; METR's diversity weights for horizons.

  METR normalises weights to Σω = 1 per agent, so its penalty (λ/2)β² is equivalent to a prior β ~ N(0, 1/(λ N_a)) on a likelihood over the agent's N_a runs. With N_a ≈ 1000 and λ = 0.1, that is a prior SD of about 0.1 logit per doubling.

### 1.3 Three analytic results used in the report

**(i) p50 invariance.** σ(−z) = 1 − σ(z), so for any symmetric task-effect distribution, E_u σ(u) = ½. The population-averaged curve m(x) = E_u σ(α + β(x − x̄) + u) therefore crosses ½ exactly where α + β(x − x̄) = 0. **The p50 of the typical task equals the marginal p50.** Verified numerically to 1e-6 for all agents (T5).

**(ii) Attenuation of the marginal curve (Zeger, Liang & Albert 1988).** Use σ(z) ≈ Φ(cz) with c = 16√3/(15π) ≈ 0.588, and the identity E_u Φ(c(η + u)) = Φ(cη/√(1 + c²τ²)). Then

    m(x) ≈ σ( (α + β(x − x̄)) / s ),     s = √(1 + c²τ²).

So the marginal slope is β/s, and the distance from p50 to any other quantile, in log2 units, is s times larger for the marginal curve. At τ̂ = 2.92, s ≈ 2.0. Fig. E2b confirms that the fitted conditional/marginal slope ratio sits on this line.

**(iii) Ridge as a rotation about a pivot.** Near the unpenalised optimum, METR's per-agent objective is quadratic with Hessian H = Σ ω p(1−p)[1, x; x, x²]. Profiling out the intercept gives:

- Fisher information for β: I_a = Σ ω p(1−p)(x − x̄_v)²
- Information-weighted pivot: x̄_v = Σ ω p(1−p) x / Σ ω p(1−p)
- Ridge solution: β_λ = β_0 · I_a/(I_a + λ), with the curve pivoting about x̄_v.

Hence

    log2 H50(λ) − x̄_v ≈ (log2 H50(0) − x̄_v)(1 + λ/I_a).

Horizons above the pivot (strong agents) move up, and those below it (weak agents) move down. The time trend steepens and the doubling time shortens.

---

## 2. Correctness checks (`python -m tests.test_models`)

| Test | Claim backed | Result |
|---|---|---|
| T1 | Analytic gradients (all estimators) | Max relative error < 1e-5 against central differences |
| T2 | B1 = METR's sklearn estimator | Our objective ≤ sklearn's for every agent; horizons agree to 0.1% at λ = 1e-5 and 0.1 (sklearn stops at tol 1e-4) |
| T3 | Quadrature accuracy | |grid − adaptive quad| < 1e-4 in per-task log marginal likelihood |
| T4 | MML recovers τ; the LRT is calibrated | τ̂ = 2.87 on average for a true 2.92 (−1.7%). Null rejection 2/30 = 6.7% at nominal 5%; mean statistic 0.62 against 0.5 for ½χ²₁ |
| T5 | p50 invariance; p80 attenuation | Marginal and conditional p50 agree to 1e-6; marginal p80 < conditional p80 for all 20 agents |
| T6 | M → B1 as τ → 0 | Objectives agree to 1e-2 at τ = 1e-5 |

---

## 3. E2: how much difficulty does length explain? (in-sample, unweighted likelihood)

| Model | k | log-lik | AIC | BIC | τ̂ |
|---|---|---|---|---|---|
| B0 agent-only | 20 | −12292.8 | 24625.6 | 24786.6 | – |
| B2 common slope | 21 | −5734.0 | 11509.9 | 11679.1 | – |
| B1 METR | 40 | −5687.4 | 11454.9 | 11777.0 | – |
| R Rasch (no length) | 21 | −3222.3 | 6486.6 | 6655.7 | 6.41 |
| L LLTM-R | 22 | −3056.1 | 6156.3 | 6333.4 | 2.74 |
| **M agent-slope LLTM-R** | 41 | **−2919.6** | **5921.1** | **6251.3** | 2.92 |

Likelihood-ratio tests:

| Comparison | Statistic | p | Note |
|---|---|---|---|
| B1 vs M (τ > 0) | 5536 | ≈ 0 | Boundary null ½χ²₀ + ½χ²₁ (Self & Liang 1987) |
| R vs L (length, given task effects) | 332 | 3e-74 | |
| L vs M (agent slopes) | 273, df 19 | 6e-47 | |
| B2 vs B1 (agent slopes) | 93, df 19 | 1e-11 | |

**Caveat.** These in-sample tests treat tasks as independent. Tasks within a family are correlated, so the tests are anti-conservative. E1 shows that per-agent slopes do *not* improve out-of-sample prediction without task effects. The τ test is so far beyond any plausible correction that its conclusion stands.

**Variance decomposition (model L, latent logit scale).**
- Length variance: β_L² Var(x_t) = 1.133² × 23.1 = 29.7.
- Residual variance: τ_L² = 7.48.
- R²_length = 0.80, close to METR's own R² ≈ 0.83 for success rate vs log length. The pseudo-R² against Rasch is 1 − τ_L²/τ_R² = 0.82.
- In length units, the residual SD is τ_L/|β_L| = 2.4 doublings.

**Over-dispersion.** Among cells with ≥ 2 runs, 15.9% have mixed outcomes. M predicts 22.3% (z = −14.7) and B1 predicts 40.2%. Runs of the same agent on the same task agree more than any task-level model predicts, which is evidence of agent×task interaction.

**Qualitative (Fig. `e2_task_effects_th11`).**
- *Harder than their length:* `blackbox/apron` and `blackbox/beach` (10 min, never solved by any agent); `request_routing_9` (0.3 min, 68% solved — very low for a 20-second task); `count_words/ai` (1 min, 54%).
- *Easier than their length:* `hackthebox/babyencryption` (45 min, 98%); `audio_classification/macaques` (5–6 h, 65%); `white_box_attack/untargeted` (10 h, 55%).
- *Shrinkage artefact:* for tasks that are always solved, û_t increases mechanically with η⁰ (the SWAA band). Always read û_t with its posterior SD.
- *Task source:* mean û is +0.24 (SE 0.18) for SWAA, −0.12 (0.24) for HCAST, +0.70 (0.57) for RE-Bench. No source is clearly harder for its length.

Figures: `e2_task_effects_th11` (task effects vs length; profile likelihood of τ) and `e2_slopes_th11` (conditional vs marginal slopes against the theoretical √(1+c²τ²) line).

---

## 4. E1: out-of-sample prediction (paired family-cluster bootstrap CIs, D8)

### Protocol A: known task, held-out cells (5-fold over 4523 cells)

| Model | NLL/run | NLL (METR wts) | Brier | AUROC | ECE | ΔNLL vs METR [95% CI] |
|---|---|---|---|---|---|---|
| F2 rank-2 factors | **0.1907** | **0.2447** | **0.0580** | **0.9745** | 0.0093 | −0.120 [−0.173, −0.082] |
| F1 rank-1 factors | 0.1909 | 0.2450 | 0.0581 | 0.9745 | 0.0099 | −0.120 |
| M agent-slope LLTM-R | 0.1924 | 0.2472 | 0.0586 | 0.9740 | 0.0085 | −0.118 [−0.171, −0.080] |
| F0 (M as MAP plug-in) | 0.1925 | 0.2473 | 0.0586 | 0.9740 | 0.0086 | −0.118 |
| L LLTM-R | 0.1969 | 0.2491 | 0.0594 | 0.9727 | **0.0076** | −0.114 |
| R Rasch (no length) | 0.1982 | 0.2502 | 0.0596 | 0.9722 | 0.0079 | −0.112 [−0.166, −0.075] |
| B2 common slope | 0.3090 | 0.3817 | 0.0981 | 0.9298 | 0.0177 | −0.0015 [−0.0040, 0.0008] |
| B1 METR (MLE) | 0.3101 | 0.3838 | 0.0988 | 0.9292 | 0.0160 | −0.0003 |
| **B1 METR (paper weights)** | 0.3105 | 0.3840 | 0.0986 | 0.9290 | 0.0172 | 0 |
| B0 agent-only | 0.5928 | 0.6790 | 0.2025 | 0.6493 | 0.0248 | +0.282 |

Ablation chain (paired ΔNLL, 95% CI). Each row adds one ingredient:

| Ingredient | ΔNLL [95% CI] |
|---|---|
| Task identity (Rasch vs METR) | −0.1123 [−0.1664, −0.0752] |
| + length covariate (L vs R) | −0.0013 [−0.0021, −0.0006] |
| + agent slopes (M vs L) | −0.0045 [−0.0097, −0.0003] |
| integration vs MAP plug-in (M vs F0) | −0.0001 [−0.0004, 0.0001] |
| + rank-1 / rank-2 factors (vs M) | −0.0016 [−0.0045, 0.0010] / −0.0017 [−0.0039, 0.0002] |
| agent slopes *without* task effects (B1 vs B2) | +0.0012 [−0.0005, 0.0033] |

Calibration (Fig. `e1_calibration_th11`): both METR and M are well calibrated (ECE < 0.02). The IRT gain is in **sharpness** (resolution), not calibration.

Factor models: inner validation chooses ρ = 10–30. At large ρ the ridge on (P, Q) acts as a nuclear-norm penalty and zeroes the factors. Low-rank agent×task structure is at most a small improvement. The over-dispersion found in E2 is therefore mostly *unstructured* cell-level noise (D11).

### Protocol B: unseen task families (5-fold over 79 families)

| Model | NLL/run | Brier | AUROC | ECE | ΔNLL vs METR [95% CI] |
|---|---|---|---|---|---|
| B2 common slope | 0.3128 | 0.0992 | 0.9282 | 0.0152 | −0.0002 [−0.0038, 0.0037] |
| **B1 METR (paper weights)** | 0.3130 | 0.0991 | 0.9281 | 0.0144 | 0 |
| B1 METR (MLE) | 0.3143 | 0.0997 | 0.9276 | 0.0111 | +0.0013 [−0.0016, 0.0051] |
| L LLTM-R [marginal] | 0.3148 | 0.1000 | 0.9281 | 0.0174 | +0.0018 [−0.0041, 0.0086] |
| M [marginal] | 0.3157 | 0.1014 | 0.9270 | 0.0198 | +0.0026 [−0.0045, 0.0113] |
| M [plug-in u = 0] | 0.3768 | 0.1092 | 0.9269 | **0.0677** | **+0.0638 [+0.0320, +0.1036]** |
| R Rasch [marginal] | 0.5990 | 0.2055 | 0.6439 | 0.0395 | +0.286 |
| B0 agent-only | 0.5958 | 0.2041 | 0.6441 | 0.0271 | +0.283 |

Interpretation:
- With nothing known about a task but its length, the correct IRT predictive E_u σ(η⁰ + u) is the population-averaged curve. METR's logistic estimates that curve directly, so the two tie.
- The plug-in u = 0 uses the conditional (steeper) curve and is overconfident. Its ECE is 4.7× higher. This is the attenuation result (ii) seen in out-of-sample data.

### Protocol C: a new agent with m revealed tasks (leave-one-agent-out, 10 random draws × 20 agents)

Median |log2 error| of p50, and mean NLL per run on the agent's hidden tasks:

| m | METR, flat prior (as published) | METR + population prior | **IRT (M) + population prior** | IRT vs its own full-data p50 | NLL: METR+prior / IRT |
|---|---|---|---|---|---|
| 8 | 1.21 | 1.03 | **0.94** | 0.69 | 0.351 / **0.222** |
| 16 | 0.85 | 0.73 | **0.58** | 0.42 | 0.327 / **0.201** |
| 32 | 0.61 | 0.53 | **0.50** | 0.30 | 0.319 / **0.193** |
| 64 | 0.38 | **0.35** | 0.44 | 0.24 | 0.312 / **0.190** |

Errors are measured against METR's full-data p50 except in the "own" column. The flat-prior METR NLL means are 1.10, 0.50, 0.34 and 0.32; they are inflated by separation at small m.

IRT is roughly twice as sample-efficient for its own estimand. The m = 64 reversal against METR's yardstick is the between-estimator offset explained in E3 (D14).

---

## 5. E3: horizons and the trend under task heterogeneity

Model M is fitted with METR's diversity weights (count scale), τ̂_w = 2.82, s = 1.94. Full table: `results/e3_horizons_th11.csv`.

| Agent | METR p50 | M p50 | METR p80 | M p80 (typical task) | M p80 (marginal) | p50/p80: METR / M typical |
|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 719 | 881 | 70 | 186 | 38 | 10.3 / 4.7 |
| GPT-5.2 | 352 | 304 | 66 | 104 | 35 | 5.3 / 2.9 |
| Claude Opus 4.5 | 293 | 231 | 49 | 77 | 25 | 5.9 / 3.0 |
| GPT-5 | 203 | 146 | 38 | 54 | 19 | 5.3 / 2.7 |
| o3 | 120 | 83 | 30 | 37 | 16 | 4.0 / 2.3 |
| Claude 3.7 Sonnet | 60 | 43 | 12 | 18 | 7 | 5.0 / 2.4 |
| GPT-4o | 7.0 | 8.3 | 1.3 | 3.7 | 1.7 | 5.5 / 2.2 |

- **p80 reading.** "Can the model do a typical task of length L with 80% reliability?" is answered by M's conditional p80, which is 2–3× shorter than p50. METR's p80 answers "what fraction of *all* tasks of length L", and that fraction mixes reliability with task heterogeneity. The theory predicts M's marginal p50/p80 ratio from the conditional ratio raised to the power s (e.g. GPT-5.2: 2.91^1.94 = 7.9, against 8.7 observed).
- **p50 offset between METR and M.** Despite exact within-model p50 invariance, the two estimators' p50s differ: M/METR ranges from 0.68 to 1.40, i.e. −32% to +40%. E3b explains why.

**Doubling times** (same SOTA agent set for all estimators, D13; 200 paired bootstraps):

| Estimator | 2023+ (95% CI) | 2024+ (95% CI) |
|---|---|---|
| METR p50 | 128.7 d (100–165) | 102.2 d (77–130) |
| M p50 | 140.7 d (107–185) | 108.2 d (79–140) |
| METR p80 | 133.4 d (110–189) | 102.5 d (84–130) |
| M p80 (typical task) | 155.9 d (122–203) | 118.6 d (93–152) |
| M p80 (marginal) | 175.3 d (139–244) | 131.3 d (101–184) |

**Paired difference, M p50 − METR p50 (same bootstrap samples):** +15.3 d [−6.2, +36.0] for 2023+ and +9.5 d [−9.8, +27.5] for 2024+.

The IRT correction moves individual horizons by up to 40%, but **does not significantly change the doubling time**. METR's headline trend is robust to task heterogeneity. The point estimates lean toward a slightly slower trend, consistent with the structural bias found in E3b.

The METR p50 CIs here (200 bootstraps, pseudo-item resampling D12) are a little wider than METR's own (105–157), because copies of a resampled task are treated as distinct items.

### E3b: why do METR's and M's p50s differ? (simulation from the fitted M, 60 replicates)

- **Structural arm (fresh u_t each replicate).** METR's estimator is biased downward as capability rises: about +0.03 log2 for GPT-4-era models, −0.17 for Opus 4.5, −0.22 for GPT-5.3-Codex, −0.39 for Opus 4.6. corr(bias, true log2 p50) = −0.89.
  - Effect on the 2023+ doubling time: METR estimates 147.4 ± 12.4 d against a true 140.7 d (+4.8%, about 0.5 SD).
  - M's own estimator has a flat −0.08 log2 offset, which leaves trends unaffected.
- **Realised arm (u_t fixed at this suite's posterior means).** METR's simulated shifts reproduce the *observed* METR-vs-M offsets almost perfectly: **R² = 0.995**, RMSE 0.028 log2. So METR's per-agent horizon contains a component of up to ±0.5 log2 that depends on which particular tasks are near that agent's curve.

### E3c: out-of-suite check (TH1.0 vs TH1.1, 16 shared agents)

- SD of mean-centred cross-suite differences in log2 p50: **0.46 for M vs 0.57 for METR**. Ratio 0.80, 95% CI [0.50, 1.23], P(ratio ≥ 1) = 0.17.
- This is consistent with E3b but **not significant**. Power is limited by n = 16 and by agent-specific infrastructure shifts (Vivaria → Inspect).

Figures: `e3_p80_th11`, `e3b_bias_th11`, `e3c_cross_suite`.

---

## 6. E4: stability of METR's estimator

**(a) Ridge λ** (METR scale, Σω = 1). Doubling time 2023+ / 2024+:

| λ | 1e-5 | 1e-3 | 1e-2 | 3e-2 | **0.1** | 0.3 |
|---|---|---|---|---|---|---|
| Doubling 2023+ / 2024+ (days) | 128.7 / 102.2 | 128.7 / 102.1 | 128.1 / 101.6 | 126.7 / 100.6 | **122.6 / 97.4** | 113.4 / 89.8 |
| Max error of the first-order pivot formula (log2) | 2e-6 | 8e-6 | 8e-4 | 6e-3 | **0.05** | 0.29 |

The formula of §1.3(iii) is accurate up to METR's own published setting, λ = 0.1 (Fig. `e4a_ridge_th11`). Earlier validation found that the change λ = 0.1 → 1e-5 between METR's January blog and the current repository moves Opus 4.5 from 320 to 293 min. That shift is this rotation.

**(b) Task weighting** (λ = 1e-5). Doubling time 2023+: 128.7 days (invsqrt), 127.2 (equal-task), 128.6 (per-run). Weighting barely matters.

**(c) Slope prior** (count scale, selected by unseen-family NLL with METR weights):

| Prior | Best strength | ΔNLL vs METR [95% CI] |
|---|---|---|
| Ridge toward 0 | λ = 10 | −0.0003 [−0.0012, 0.0004]; λ = 1000 *hurts*: +0.035 |
| **Hierarchical toward μ** | κ = 1000–3000 (flat beyond) | **−0.0019 [−0.0036, −0.0002]** at κ = 1000 |

Under the hierarchical prior (κ = 3000), compared with METR (300 paired bootstraps):

| Quantity | METR | Hierarchical |
|---|---|---|
| Opus 4.6 p50 | 719 [325, 4608] min | 525 [289, 1217] min |
| Opus 4.6 CI width | 3.8 log2 | 2.1 log2 |
| Opus 4.5 p50 | 293 [172, 733] | 285 [169, 628] |
| Doubling 2023+ | 128.7 d [99, 155] | 130.5 d [104, 157] |
| Doubling 2024+ | 102.2 d [77, 125] | 105.6 d [84, 129] |

**(d) Failure case: Claude Opus 4.6** (Fig. `e4cd_uncertainty_th11`).
- Only **9 of its tasks (6.3% of its task weight) are longer than its estimated horizon**. For every other agent the figure is ≥ 25%.
- Its 50% crossing is set by a handful of 10–30 h tasks. Each of the top families moves log2 p50 by about 0.2 when dropped: `continue_pattern` (18 h, never solved) +0.24; `robot_control` (30 h, always solved) −0.22.
- Leave-one-family-out jackknife SE: 0.80 log2 (×1.74). Bootstrap CI width: 3.8 log2 (×14).
- The estimator is extrapolating, not measuring. A hierarchical slope prior halves the CI width. Only more long tasks can fix it properly, as METR itself notes.

---

## 7. Replication on TH1.0 (33 agents, 170 tasks, 50 families; different suite and infrastructure)

| Quantity | TH1.1 | TH1.0 |
|---|---|---|
| τ̂_M (profile 95% CI) | 2.92 (2.59–3.31) | 2.57 (2.28–2.91) |
| R²_length (latent) / vs Rasch | 0.80 / 0.82 | 0.80 / 0.83 |
| Mixed cells: observed vs M-expected | 15.9% vs 22.3% (z = −14.7) | 18.9% vs 25.8% (z = −18.6) |
| E1-A NLL: METR → M (reduction) | 0.310 → 0.192 (−38%) | 0.299 → 0.192 (−36%) |
| E1-A AUROC: METR → M | 0.929 → 0.974 | 0.939 → 0.975 |
| E1-A share of gain from task identity (Rasch) | 95% | 93% |
| E1-B M [marginal] vs METR, ΔNLL [95% CI] | +0.003 [−0.005, +0.011] | −0.005 [−0.060, +0.033] |
| E1-B M [plug-in u = 0] vs METR | +0.064 [+0.032, +0.104] | +0.034 [−0.006, +0.078] |
| E1-A best factor model vs M | −0.0017 [−0.0039, 0.0002] (F2) | −0.0022 (F3, point estimate) |

Every qualitative conclusion of E1 and E2 replicates on the independent suite. Only the plug-in penalty in protocol B loses significance on TH1.0, where there are fewer families and so wider CIs.

---

## 8. Limitations

- **Binomial runs.** The over-dispersion check shows runs within an agent×task cell are more alike than any of these models assume. Run-level likelihoods and the LRT degrees of freedom should be read with that in mind. Bootstrap CIs (family-clustered) do not depend on the assumption.
- **Gaussian, length-independent task effects.** u_t ~ N(0, τ²) independent of x_t. A heteroscedastic τ(x) (e.g. more variability among long tasks) is a natural extension.
- **Pseudo-likelihood.** Diversity weights make the weighted horizon estimates pseudo-likelihood estimates. τ̂ differs slightly by weighting (2.92 unweighted vs 2.82 weighted).
- **Simulation scope.** E3b assumes model M is true. Its value is in showing what METR's estimator does *if* difficulty varies as M says, and the real-data match (R² = 0.995) supports that premise.
- **Underpowered out-of-suite test.** E3c has 16 agents and is confounded by the infrastructure migration.

---

## References

- Kwa et al. (2025), *Measuring AI Ability to Complete Long Tasks*, arXiv:2503.14499; METR (2026), *Time Horizon 1.1*.
- Fischer (1973), The linear logistic test model, *Acta Psychologica*.
- De Boeck (2008), Random item IRT models, *Psychometrika*.
- Bock & Aitkin (1981), Marginal maximum likelihood estimation of item parameters, *Psychometrika*.
- Zeger, Liang & Albert (1988), Models for longitudinal data: a GEE approach, *Biometrics*.
- Self & Liang (1987), Asymptotic properties of MLE and LRT under nonstandard conditions, *JASA*.
- Pinheiro & Bates (1995), Approximations to the log-likelihood function in the nonlinear mixed-effects model, *JCGS*.
