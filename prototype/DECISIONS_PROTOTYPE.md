# Design decision log: HorizonCAT prototype

**HorizonCAT** is a cost-aware adaptive evaluation and suite-audit tool for AI-agent time horizons. It builds on the IRT study in `../study/` (decisions D1–D20 there). Entries here are numbered P1, P2, …, are append-only, and are dated.

---

## P0. What is built, what it builds on, and what is new (2026-10-01)

### Builds on
- **METR** (Kwa et al. 2025, arXiv:2503.14499; Time Horizon 1.1, 2026). The task suite, human-time anchors, the per-agent logistic horizon definition, and the public run data (`eval-analysis-public`, TH1.1 `runs.jsonl`, 20 agents × 228 tasks).
- **BRIDGE** (Liu, Gala, Nilaksh, Bahdanau, Reddy, Larochelle 2026, arXiv:2602.07267; code McGill-NLP/BRIDGE). Their central result is that IRT latent difficulty is *linear in log human completion time*, so human time can serve as a calibrated difficulty scale. We use that result as the anchoring principle of our item bank. Where they differ from us:
  - BRIDGE fits free 2PL difficulties and regresses log-time on them afterwards. We keep log human time as the fixed anchor and model only the *residual* as a random effect, u_t ~ N(0, τ²) (study model M, D5/D6). Horizons are then in minutes by construction, and the residual uncertainty is propagated rather than regressed away.
  - BRIDGE's stated future work includes "uncertainty-aware difficulty estimation". Our item bank carries a full posterior for each task's difficulty, and every selection and stopping decision integrates over it.
- **Generic IRT-based efficient evaluation.**
  - Adaptive testing with Fisher-information item selection: ATLAS (Li et al., arXiv:2511.04689); Fluid Benchmarking.
  - Static anchor subsets: tinyBenchmarks (Polo et al.).
  - Static mid-difficulty filtering for agents: Ndzomga, *Efficient Benchmarking of AI Agents* (arXiv:2603.23749), "evaluate new agents only on tasks with intermediate historical pass rates (30–70%)".
  - Bayesian stopping for LLM evaluation (arXiv:2608.14425).
  - IRT benchmark auditing (arXiv:2605.30504).
  - **None of these is new here.** Adaptive item selection and IRT auditing are established ideas.

### New in this prototype, to the best of our search (2026-10-01)
1. **The estimand is a human-time-anchored horizon, not an ability score.** The target is log2 H50 = h, a functional of an agent's (level, slope). The agent is parameterised directly as η = β(x_t − h) + u_t, so the posterior over h (in minutes) is the output, with slope uncertainty integrated out.
2. **Cost-weighted selection with real compute costs.** Selection maximises expected posterior-variance reduction of h per unit cost^γ. The cost is each task's median token count from METR's runs, which spans about 2500× from 30-second to 30-hour tasks and correlates with log human time at r = 0.95. ATLAS-style CAT counts items; for agent tasks, items differ in cost by three orders of magnitude.
3. **Uncertainty-aware item bank.** Each task's difficulty is a posterior from leave-one-agent-out MML (study model M), not a point estimate. Predictive success probabilities integrate over it.
4. **Back-testing by replay on real agent runs**, not simulation alone. A held-out agent's actual run outcomes are replayed in whatever order the policy requests. We report cost to reach a target precision, error against the full-data estimate, and **credible-interval coverage**: is the uncertainty honest?
5. **Saturation detection and suite-extension design.** The test-information function of the suite over h shows where the suite can and cannot measure. The tool flags when an agent's horizon lies beyond it (the Opus 4.6 failure case in the study) and computes how many new tasks of what length would make a given horizon measurable to ±0.5 log2. This addresses METR's stated future work ("adding more long tasks") with a quantitative design.

### Not claimed
- That adaptive testing reduces evaluation cost in general (well known).
- That IRT difficulty tracks log human time (BRIDGE; METR).
- Results for models whose per-task runs are not public (see P9).

---

## P1. Data available for the newest models (2026-10-01)

- **Per-task runs** (needed for anything here) are public for 20 TH1.1 agents, up to Claude Opus 4.6 and GPT-5.3-Codex (`eval-analysis-public`, last data commit 2026-03-05).
- **METR's public `benchmark_results_1_1.yaml`** (time-horizons page, updated 2026-05-08) adds **headline numbers only** for GPT-5.4 (342 min), Gemini 3.1 Pro (384 min) and Claude Mythos Preview (early; 1045 min [509, 3304]). It notes that the doubling time "excludes points with central estimate p50 > 16 hrs", so METR itself treats the suite as saturated above 16 h.
- **GPT-5.6 Sol, GPT-6.1 Sol and Claude Opus 5.5** are not on METR's page as of this search.
- **Decision.** Back-tests use only agents with public runs. Newer or future models are handled in two ways:
  - a *hypothetical-agent* simulation mode with user-set (h, β);
  - the suite-design analysis (P10), which says what data would make such models measurable.
  No model's capability is invented: hypothetical agents are labelled by horizon only.

## P2. Item bank source: study model M, leave-one-agent-out (2026-10-01)

- **Decision.** Task parameters come from model M (agent-slope LLTM with random task effects, unweighted MML; study D5/D6), fitted on all agents *except* the one being evaluated.
- **Why leave-one-out.** If the evaluated agent's runs were in the bank, the bank's task posteriors would already contain its outcomes, and the back-test would leak.
- **Why M and not a free 2PL (as in BRIDGE).** M keeps human time as the anchor (BRIDGE's linearity result), so the horizon is in minutes without a post-hoc mapping. It also has a single, calibrated residual variance τ² (study T4).

## P3. Compressing task posteriors to K = 10 equal-mass points (2026-10-01)

- **Decision.** Each task's posterior p(u_t | bank) (321-point grid in the study) is reduced to the conditional means of its 10 posterior deciles.
- **Why.** The predictive p_t(h, β) = (1/K) Σ_k σ(β(x_t − h) + u_tk) becomes cheap and identical in Python and in the browser. The bank for 228 tasks is then 2280 numbers.
- **Check (test E3).** The posterior mean is preserved exactly. The SD is preserved to within 20%, since quantile compression slightly under-disperses. This makes predictions marginally overconfident about u, which is acceptable because calibration of the final h interval is checked end to end (E4, P11).

## P4. Cost of a task = median tokens per run (2026-10-01)

- **Observation.** `generation_cost` is zero for every public run (redacted). `tokens_count` is present for 95.5% of runs, and the per-task median exists for all 228 tasks. log(median tokens) correlates with log human time at r = 0.954, against r = 0.828 for wall-clock duration.
- **Decision.** cost_t = median tokens across all runs of t. It is agent-independent, because the agent being evaluated has not run yet.
- **Caveat.** The within-task SD of log tokens across agents is 1.38: costs vary a lot by agent. Token price per model is ignored, so for a single model the dollar cost is proportional to tokens.

## P5. Agent parameterisation by its horizon (2026-10-01)

- η = β(x_t − h) + u_t, the same model as M (α = β(x̄ − h)), with h = log2 H50 as an explicit coordinate.
- **Why.** The posterior over h is read directly. Stopping and selection target Var(h), the estimand, not a generic "ability".
- **Grid.** h ∈ [−6, 18] log2-minutes in steps of 0.1 (about 1 s to 180 days; 241 values) × 15 equal-mass quantile nodes of the slope prior, giving 3615 points. The prior SD of the slope is inflated ×1.5 for robustness to new kinds of agents, and the slope is constrained to β ≤ −0.05.

## P6. Prior on h: uniform over the grid (2026-10-01)

- **Decision.** h is uniform on [−6, 18], with no capability-informed prior.
- **Why.** A prior fitted to past agents (as in study protocol C) pulls new frontier agents toward historical horizons. That is exactly the wrong bias for the models this tool exists for. A flat prior lets the data decide, and the saturation rule (P8) catches the cases where the data cannot.
- **Validation (test E4).** On simulated agents drawn from the bank model, 95% credible intervals covered the true h in 58 of 60 runs (96.7%).

## P7. Selection rule: expected variance reduction of h per cost^γ (2026-10-01)

- score_t = [Var(h) − E_{y∼p(y|data)} Var(h | data, y_t = y)] / cost_t^γ, over tasks not yet administered.
  - γ = 0 is classical information-maximising CAT, counting items.
  - γ = 1 maximises information per token.
  - γ = 0.5 is in between.
- **Why variance reduction rather than Fisher information at a point estimate.** Early on, the posterior over h is wide and bimodal-prone, and a point-estimate Fisher criterion picks tasks for the wrong h. The expected posterior variance is exact for one-step lookahead under the full posterior, including slope uncertainty (test E1 checks it against brute force).
- **One run per task, no repeats.** Repeated runs of a task share u_t, so they are not conditionally independent given (h, β). Allowing repeats would need per-task posterior updating of u. Kept simple for the prototype and noted as future work.
- **Fair comparison.** Every policy, adaptive or not, uses the *same* Bayesian estimator, so differences come only from task selection.
- **Baselines.** Random; length-stratified (cycles through 2-doubling bins of task length, METR-like coverage); and the static 30–70% historical-pass-rate filter of Ndzomga (arXiv:2603.23749).

## P8. Stopping and saturation (2026-10-01)

- **Precision stop.** SD(h) ≤ 0.5 log2, i.e. the horizon is known within about ×1.4 at one SD. Back-tests record full trajectories (no early stop) and read off when the target is first reached.
- **Saturation stop.** The best remaining task would reduce Var(h) by less than 1%, **and** P(h > longest task in bank) > 0.2. This is the formal version of the study's failure case (Opus 4.6): the suite cannot pin the horizon because the agent succeeds on nearly everything it contains.

## P9. Newer models (GPT-5.6 Sol, GPT-6.1 Sol, Claude Opus 5.5, …): what testing them requires (2026-10-01)

Recorded here so the answer is traceable.

1. **Per-run outcomes on METR's task suite** (task id, run id, binary score). Not public for these models. Routes: a request to METR, or a future public data release.
2. **Or running the evaluations ourselves.** This needs:
   - API access to each model;
   - the task environments (most HCAST and RE-Bench tasks are held private to limit contamination);
   - METR's Inspect harness with sandboxed Docker execution;
   - compute far beyond this laptop (each frontier run on a long task takes hours of agent time and around 10⁶ tokens).
3. **Even with runs, the suite is the binding constraint.** Mythos Preview's public p50 (1045 min) already exceeds METR's own 16 h trend cut-off. P10 quantifies how many new long tasks, at what lengths, a 1–7-day horizon needs.
4. **What HorizonCAT does now.**
   - It simulates hypothetical agents at user-chosen horizons to show the measurement failure.
   - It ranks which existing tasks to run first, which minimises cost if runs can be bought.
   - It reports the suite extension needed.

## P10. Suite measurement range from test information (2026-10-01)

- **Decision.** Use the IRT test-information function of the suite (one run per task, population slope) to map where horizons can be measured to ±0.5 log2. Also compute how many new tasks of length L are needed to reach that precision at a target horizon h*. New tasks have unknown difficulty, so they enter with u ~ N(0, τ²): the predictive is integrated, which lowers their information.
- **Token cost of new tasks.** Extrapolated as log tokens ~ a + b·x, fitted on tasks of 16 minutes or longer (b = 0.645 per doubling). The page labels this as extrapolation. Observed tokens flatten between 4 h and 30 h tasks (median ≈ 0.9 M), so costs beyond 30 h are uncertain in both directions.

## P11. Back-test protocol (2026-10-01)

- 20 agents × 6 policies × 20 replays. Each policy gets up to 80 tasks with no precision stop; the step where SD(h) first reaches 0.5 is read from the trajectory. A full-suite baseline runs every available task once, 20 replays.
- **Metrics.** Reach rate; tasks and tokens to reach the target; tokens as a share of the full suite; |h − h_ref| at that point; 95% credible-interval coverage of h_ref.
- **Reference.** h_ref is model M fitted on all data, including the agent. It is a reference, not ground truth, so coverage is "agrees with the full-data estimate". METR's own p50 is also stored (`h_metr`) and shown on the page.

## P12. Hypothetical frontier agents (2026-10-01)

- Agents with horizons of 12 h, 24 h, 48 h and 1 week. Their slope is drawn from the population; each task's true difficulty is one of its K bank points, drawn at random.
- Runs use both γ = 1 and γ = 0 with no task cap (first version: γ = 1 only, capped at 120 tasks; see P17). Each is done on the current suite and on the suite plus 20 new tasks at 1.5× the horizon.
- These are **not** predictions about any named model. They show what the instrument can and cannot resolve.

## P13. Interactive page architecture (2026-10-01)

- **Static page plus one `data.json`.** It holds the per-agent leave-one-out banks, replay rates, back-test curves, frontier results and the audit table. No server, no runtime capabilities: everything runs in the viewer's browser.
- **The engine is ported to JavaScript** (`web/index.html`), mirroring `horizoncat/engine.py`: same grid, same quantile slope nodes (Acklam's inverse normal, relative error < 1.2e-9), same variance-reduction rule and same saturation rule. Parity is checked in P15.
- **Hypothetical-agent mode** simulates responses in the browser from user-set (h, β) with a seeded RNG (mulberry32). Replay uses each agent's real k/n per task.

## P14. Task-audit flags (2026-10-01)

Rules on the all-agent bank posterior (û_t, sd) and solve rates:

| Flag | Rule | Meaning |
|---|---|---|
| `unsolved-short` | solve rate 0 and human time ≤ 60 min | Candidate for a broken task or an under-estimated human time, e.g. `blackbox/apron` |
| `harder-than-length` | û_t < −τ and the task is solved at least once | Harder than its length predicts |
| `easier-than-length` | û_t > τ | Easier than its length predicts |
| `no-variation` | solve rate exactly 0 or 1 | Says nothing about any agent; low value per token |

The page also reports each task's Fisher information about a 12-hour agent's horizon, to rank tasks by usefulness at the frontier. Flags are prompts for human review, not verdicts.

## P15. JavaScript/Python parity (2026-10-01)

- The export writes a deterministic reference trace: the all-agent bank, γ = 1, and outcome rule "success iff x_t < 7.0". The page's engine must reproduce the same task sequence and posterior mean and SD of h.
- Checked once in the browser; result recorded in P18.

## P16. Correction: the standard error of h must account for the unknown slope (2026-10-01)

- **Bug found.** The first design calculation used SE = 1/√I_hh, i.e. it treated the agent's slope as known. It reported **zero** new tasks needed for 12–48 h horizons, while CAT simulations at those horizons stalled at SD 0.6–0.95.
- **Cause.** For frontier agents nearly every task lies below h, so the data cannot separate a longer horizon from a flatter curve: h and β are confounded.
- **Fix.** Use the full 2×2 Fisher matrix over (h, β), add the slope prior's precision 1/(1.5·sd_b)², and take SE(h) = √[(I + diag(0, 1/σ_β²))⁻¹]_hh. Test E2 now checks the full matrix against finite-difference E[score scoreᵀ].
- **Consequence.** The suite measures to ±0.5 log2 only up to about **17 h** (SE 0.50 at 17.1 h). That matches, independently, METR's own choice to exclude p50 > 16 h from its trend fit. At a 48 h horizon the SE is 0.73, not the 0.42 the known-slope formula gives.
- **Extension needed for ±0.5 (one run per task).** 24 h: 7 one-week, 8 64-hour or 11 32-hour tasks. 48 h: 19 one-week, 26 64-hour or 41 32-hour tasks (8-hour tasks: about 3300). 1 week: 48 one-week tasks.

## P17. Cost-aware selection under-buys long tasks at the frontier (2026-10-01)

- **Observation.** With γ = 1 and a 120-task cap, the first frontier simulations rarely reached ±0.5 even on the extended suite, below what the Fisher calculation allows. The expensive new long tasks were seldom chosen.
- **Decision.** Rerun the frontier simulations with γ = 1 and γ = 0 and no task cap, and report both.
- **Interpretation for the report.** γ trades precision for cost. For agents inside the suite's range this costs little (see back-test). At the frontier, only expensive long tasks are informative, so a cost-weighted policy needs a budget-aware variant: the next step is to maximise information subject to a token budget.

## P18. Back-test v1: efficient but miscalibrated off the information-only policy (2026-10-01)

Replay of 20 held-out agents × 20 replays, σ_v = 0. "Reached" means SD(h) ≤ 0.5 within 80 tasks; the other columns are measured at that point.

| Policy | Reached | Median tasks | Median tokens | Share of full suite | Median \|err\| | 95% coverage |
|---|---|---|---|---|---|---|
| Adaptive, info only (γ = 0) | 96% | 20 | 8.8 M | 6.5% | 0.26 | **94%** |
| Adaptive, γ = 0.5 | 90% | 22 | 1.5 M | 1.2% | 0.31 | 84% |
| Adaptive, γ = 1 | 65% | 27 | 0.46 M | 0.4% | 0.29 | **62%** |
| Static 30–70% filter | 72% | 32 | 16.3 M | 12.7% | 0.33 | 68% |
| Random | 70% | 54 | 29.6 M | 23.1% | 0.34 | 67% |
| Length-stratified | 68% | 51 | 28.3 M | 22.8% | 0.30 | 65% |
| Full suite (every task once) | – | 228 | 134.6 M | 100% | 0.12 | 99% (SD 0.24) |

- **Diagnosis.** Under the model (test E4), coverage was 97%. On real runs it falls below nominal for every policy except γ = 0.
- **Cause: misspecification.** The study found agent×task interaction (cells are more internally consistent than a task-only effect allows; study D10). Policies that rely on many tasks far from the agent's horizon, which are cheap or randomly chosen, see outcomes the model treats as near-certain. A surprise on such a task moves the posterior too far.
- **Decision.** Add an agent×task random effect v ~ N(0, σ_v²) to the single-run predictive, p_t = E_{u,v} σ(β(x_t − h) + u_t + v), integrated with 7-point Gauss–Hermite. Estimate σ_v by held-out predictive calibration (P19), then rerun the back-test (P20). Keep v1 for comparison.
- **Bayesian optional stopping.** Reading the posterior at the first step with SD ≤ 0.5 is valid under a correct model (likelihood principle). The undercoverage is therefore attributed to misspecification, not to the stopping rule.

## P19. σ_v selected by held-out predictive likelihood: 1.0, weakly identified (2026-10-01)

- **Held-out per-run log-likelihood** (20 agents × 3 half-splits; posterior from one replayed run per task in half A, all real runs in half B scored):

  | σ_v | 0 | 0.5 | **1.0** | 1.5 | 2.0 | 2.5 | 3.0 |
  |---|---|---|---|---|---|---|---|
  | Per-run log-lik | −0.19255 | −0.19223 | **−0.19213** | −0.19353 | −0.19677 | −0.20149 | −0.20721 |

- **Gain of σ_v = 1.0 over 0:** +14.3 total log-lik units, bootstrap 95% CI over agents [−34, +71]. **Not significant.** Predictive likelihood is driven by the centre of the predictive, while σ_v mainly affects the tails, which drive interval coverage.
- **Decision.** Use the likelihood-selected σ_v = 1.0 and *test* coverage independently in the v2 back-test (P20). σ_v is not tuned to hit 95% coverage, because coverage is the outcome being evaluated: tuning on it would be circular unless done by cross-validation across agents.

## P20. Correction: the "miscalibration" in P18 was a metric bug (2026-10-01)

- **Bug.** `summarise()` averaged interval coverage over *all* replays. Replays that never reached SD ≤ 0.5 have no interval at the reach point, and were silently counted as misses. The reported number was reach rate × coverage (γ = 1, v2: 0.558 × 0.955 = 0.533, exactly the value reported).
- **Fix.** Coverage is now computed among replays that reached the target. A second metric, `final_coverage95_ref`, measures every replay at its last step. Both are reported.
- **Corrected coverage of h_ref by 95% intervals:**

  | Policy | v1, σ_v = 0 (at reach) | v2, σ_v = 1 (at reach) | v2, final step, all replays |
  |---|---|---|---|
  | Adaptive, info only | 98.4% | 99.7% | 98.2% |
  | Adaptive, γ = 0.5 | 93.1% | 96.6% | 98.5% |
  | Adaptive, γ = 1 | 96.1% | 95.5% | 94.0% |
  | Static 30–70% filter | 93.8% | 97.0% | 96.8% |
  | Random | 95.3% | 96.8% | 96.5% |
  | Length-stratified | 95.9% | 96.8% | 96.5% |
  | Full suite | 99.3% | 98.8% | – |

- **What was wrong in P18.** Its diagnosis (agent×task misspecification) was unnecessary: the model was already calibrated. The lapse hypothesis was also tested and rejected. Model M's predictions are calibrated at the extremes (predicted 99.67% vs observed 99.64% success in the 99–99.9% band; 99.97% vs 99.97% above it).
- **Decision on σ_v.** Keep the default σ_v = 1.0. It was selected by a rule fixed in advance (held-out predictive likelihood, P19) that does not depend on coverage. It costs a little: reach at γ = 1 falls 65% → 56%, and info-only tokens rise 8.8 M → 10.4 M. It is slightly conservative. σ_v = 0 results are kept in `results/v1/` and shown on the page.
- **Lesson.** Verify a surprising metric from the raw traces before building a fix for it. The extra model (v) was built before the metric was checked.

## P21. Parity result and how it was run (2026-10-01)

- The local preview server could not be reached from the browser pane (sandboxed; connection refused). The page's engine section was therefore extracted **verbatim** from `web/index.html` (math through engine, no DOM code) and run in Node against `web/data.json`.
- **Result.** Same 25-task sequence as Python (both stop on precision at step 25). Max |Δ posterior mean of h| = 2.3e-4; max |Δ SD| = 6.5e-5. The residual comes from exporting u to 3 decimals, not from the algorithm.

## P22. Page defects found in the one rendered check, and fixed (2026-10-03)

- **Initialisation crash.** The precision slider's handler called `render()` before the first session existed (`S.trace` undefined), so the page never reached its opening state. Version 1 of the published page had this bug. Fixed in version 2.
- **Narrow layout.** Grid children lacked `min-width:0`, so the controls overflowed (673 px of content in a 274 px viewport).
- **Hidden fields.** `.field{display:flex}` overrode the `hidden` attribute; a page-level `[hidden]{display:none!important}` was added.
- **Label casing.** Uppercase labels turned "β" into "Β" and "days" into "DAYS". Values and Greek letters now opt out of the transform.
- **Verified** on a local copy with the data inlined: no console errors, opens with 12 tasks run (Claude Opus 4.5 replay), page width equals viewport width.

## P23. Final frontier results, σ_v = 1.0, no task cap (2026-10-03)

| Horizon | Suite | Selection | Reached ±0.5 | Saturated | Median SD | Coverage |
|---|---|---|---|---|---|---|
| 12 h | current | γ = 1 / γ = 0 | 67% / 70% | 10% / 10% | 0.50 / 0.50 | 90% / 93% |
| 24 h | current | γ = 1 / γ = 0 | 13% / 7% | 73% / 87% | 0.72 / 0.73 | 100% / 93% |
| 24 h | +20 tasks at 36 h | γ = 1 / γ = 0 | 37% / 37% | 53% / 60% | 0.53 / 0.63 | 93% / 97% |
| 48 h | current | both | 0% | 97–100% | 0.91–1.05 | 93–97% |
| 48 h | +20 tasks at 72 h | γ = 1 / γ = 0 | 27% / 17% | 57% / 67% | 0.65 / 0.66 | 90% / 97% |
| 1 week | current | both | 0% | 100% | 1.35–1.41 | 93–97% |
| 1 week | +20 tasks at 1.5 weeks | γ = 1 / γ = 0 | 0% / 3% | 67% / 73% | 0.79 / 0.84 | 97% / 83% |

- **Reading.** On today's suite, horizons of 24 h or more are flagged as *saturated* in 73–100% of runs: the tool correctly reports that it cannot measure them, rather than returning a falsely precise number. The intervals still cover the truth 93–100% of the time.
- **Adding 20 long tasks helps but is not enough.** The Fisher design (P16) says 7–11 one-week or 64-hour tasks for 24 h, and 19 one-week tasks for 48 h, in each case with every *other* task also run once. Adaptive runs stop earlier, which is why reach stays below 40%.
- **Claude Opus 4.6 replay** (real runs, cost-aware, up to 120 tasks): 29 of 30 runs hit the task cap without reaching ±0.5 (1 reached it). Median SD 0.71 log2. The slope-aware design SE at its full-data horizon (h = 10.04, about 17.5 h) is 0.50, so this agent sits exactly at the edge of the suite's measurable range.
