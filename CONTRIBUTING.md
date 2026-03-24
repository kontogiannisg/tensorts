# Contributing to tensorts

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Setup

```bash
git clone https://github.com/gkontogiannhs/tensorts.git
cd tensorts
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

Tests use a custom harness (not pytest). Run each suite directly:

```bash
python tests/test_tensorts.py                  # core + all transforms
python tests/test_feature_extractor.py         # FeatureExtractor modes
python tests/test_function_transform.py        # FunctionTransform wrapping
python tests/test_sklearn_compat_problems.py   # end-to-end sklearn integration
```

All four must pass before submitting a PR. CI runs them on Python 3.10–3.13.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check tensorts/ tests/          # lint
ruff format tensorts/ tests/         # format
```

CI will fail if either reports errors. Run both before pushing.

## Adding a new transform

1. Choose the right file in `tensorts/transforms/` based on category (preprocessing, temporal, spectral, shape, functional).
2. Subclass `Transform` and implement `fit()` and `transform()`. Add `inverse_transform()` if the operation is reversible.
3. Set `supports_ragged = True` if your transform handles variable-length input.
4. Export it in `tensorts/transforms/__init__.py` and `tensorts/__init__.py`.
5. Add tests covering dense input, ragged input (if supported), and inverse round-trip (if applicable).
6. Document it in `tensorts/transforms/README.md`.

## Pull requests

- Keep PRs focused — one feature or fix per PR.
- Include tests for any new or changed behavior.
- Make sure all 4 test suites pass and ruff is clean.
- Write a clear PR description explaining what changed and why.

## Design principles

- **NumPy only** in core. scipy, pandas, and scikit-learn are optional — use lazy imports and raise helpful errors if missing.
- **Non-destructive**: transforms return new TimeTensors via `.with_data()` or `.with_ragged()`. Never mutate input.
- **Ragged support**: most transforms should handle variable-length input. Follow existing patterns in `preprocessing.py`.
- **Channel-aware**: per-channel operations pool across batch and time axes, computing one statistic per channel.
