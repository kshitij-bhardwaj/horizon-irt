# Contributing

Thanks for helping. The most useful contributions are listed under [Call for support](README.md#call-for-support).

- **Bugs and questions:** open an issue with the command you ran and its output.
- **Code:** small, focused pull requests. Run the tests first:
  `cd study && ../.venv/bin/python -m tests.test_models` and `cd prototype && ../.venv/bin/python -m tests.test_engine`.
- **Design decisions:** add a dated entry to `study/DECISIONS.md` or `prototype/DECISIONS_PROTOTYPE.md`. Say what you decided, why, and what you rejected. If a result overturns an earlier entry, add a new entry rather than editing the old one.
- **Data:** do not commit METR's raw data or any third-party data without a licence that allows redistribution. Never commit API keys or credentials; use environment variables.
