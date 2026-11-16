# CLAUDE.md — predictagent

> **Scope:** PROJECT-LEVEL, extends `/workspace/CLAUDE.md`. Global workflow, git, TDD,
> Python style and security rules apply unchanged; only what differs lives here.
> Personal overrides go in `CLAUDE.local.md` (gitignored).

Forecasts numerical network values — cell load, PRB utilisation, demand — from VIAVI
RAN counter exports. An LSTM + GradientBoosting ensemble over engineered lag/rolling features, blended with
a fitted alpha and served behind FastAPI.
In the Observe → Reason → Act loop, this is the perception/reasoning component; it
predicts numbers, it does not decide actions.

---

## Layout

```
src/predictagent/
├── cli.py            # entry points: ingest / train / serve, each takes --config
├── config.py         # loads config/default.yaml into pydantic models
├── schemas.py        # the data contracts crossing module boundaries
├── exceptions.py     # domain errors
├── pipeline/
│   ├── ingestor.py   # CSV → rolled-up frame (epoch-second timestamps)
│   ├── features.py   # lag, rolling mean/std, EMA columns
│   ├── sequencer.py  # frame → (lookback, horizon) supervised windows
│   └── trainer.py    # Keras LSTM + sklearn GBR fit, blend weight, evaluation
├── registry/         # versioned model + scaler artefacts on disk
└── api/routers/      # FastAPI forecast endpoint
config/default.yaml   # single source of truth for paths, features, hyperparameters
viavi-dataset/        # raw/ CSV exports, processed/ derived frames — large, not source
```

## Commands

```bash
uv sync --extra dev                  # pytest lives in the dev extra; without it uv run
                                     # silently falls back to system pytest
uv run predictagent-ingest --config config/default.yaml
uv run predictagent-train  --config config/default.yaml
uv run predictagent-serve  --config config/default.yaml

uv run pytest                        # regression excluded by addopts
uv run pytest -m unit
uv run pytest -m regression          # full training run, slow, opt-in
uv run ruff check . && uv run ruff format .
```

## Rules specific to this project

- **Nothing is hardcoded that `config/default.yaml` already holds.** Paths, the site
  filter, the target column, the feature list, `lookback_steps`, hyperparameters — read
  them from the config object, never inline a literal. Adding a feature means adding it
  to `features.feature_columns`, not to a list in code.
- **Timestamps are epoch seconds throughout.** `ingestor.rollup_to_interval` normalises
  with `dt.as_unit("s")`; a pandas op that silently promotes to nanoseconds breaks the
  sequencer downstream. Keep the unit assertion in any test that touches timestamps.
- **`viavi-dataset/` is data, not source.** Never
  read a full raw CSV into context — sample it with `head`/pandas in a shell instead.
- **Tests never train for real.** Only the `regression` marker may fit a model; `unit`
  and `integration` build their frames from the `sample_csv`/`minimal_settings` fixtures
  in `tests/conftest.py` and write only to `tmp_path`. TensorFlow
  imports are slow — keep them out of unit-test import paths where practical.
- Model artefacts are written through `registry/model_registry.py`, which versions them.
  Do not write a `.keras`/`.joblib` file straight to `models/` from pipeline code.

## Known gap

Line coverage sits at **77%**, under the org's 80% floor. Raise it with tests for the
uncovered `pipeline/` branches; do not lower the floor.
