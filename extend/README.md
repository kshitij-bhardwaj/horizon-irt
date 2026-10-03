# Extending HorizonCAT to newer models

METR's public per-run data stops at Claude Opus 4.6 and GPT-5.3-Codex (TH1.1, March 2026). Newer models, such as GPT-5.4, Gemini 3.1 Pro, Claude Mythos Preview, GPT-5.6 Sol, GPT-6.1 Sol and Claude Opus 5.5, have only headline numbers or no public numbers at all. This folder is for researchers who can produce runs for such models.

## 1. Produce runs

You need:

- **Task environments.** Most HCAST and RE-Bench tasks are held privately by METR to limit contamination; ask METR about research access.
- **A harness.** METR's time-horizon runs use [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) with sandboxed containers.
- **Model API access.** Keep API keys in environment variables (for example `export OPENAI_API_KEY=...`) or a git-ignored `.env`. **Never commit keys.**

Write one JSON object per run:

```json
{"task_id": "hackthebox/babyencryption", "score_binarized": 1, "tokens_count": 412000}
```

`task_id` must match METR's TH1.1 task ids (listed in `docs/data.json` under `tasks.id`).

## 2. Run adaptively instead of exhaustively

```bash
.venv/bin/python extend/horizoncat_cli.py plan --next 5            # where to start (no runs yet)
.venv/bin/python extend/horizoncat_cli.py plan --runs runs.jsonl   # after each batch: estimate + next tasks
.venv/bin/python extend/horizoncat_cli.py estimate --runs runs.jsonl
```

In the replay back-test, information-driven selection (`--gamma 0`, the default) reached ±0.5 doublings with a median of 24 tasks and about 8% of the tokens needed to run every task once. If the tool reports **SATURATED**, the model's horizon is beyond what this suite can measure. `prototype/frontier.py` computes how many longer tasks would be needed.

## 3. Share results (if you can)

Open an issue using the **Share run data for a newer model** template. Aggregate results (per-task success counts) are enough to add a model to the back-test; raw transcripts are not needed.
