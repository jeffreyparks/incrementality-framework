# Incrementality Framework — Claude Code Guidance

Working notes for Claude Code taking over implementation. Design was settled in a
prior planning session; this file is the handoff. **Read it fully before writing code.**

## What this is

A reusable incrementality testing / evaluation harness for pulse and holdout
experiments, built on time-series counterfactual forecasting (Prophet by default).
The reusable pattern: *binary-treatment holdout → counterfactual forecast trained on
untreated periods → per-window lift → aggregation → bootstrap CI → business-metric
translation.* It generalizes to paid-media holdouts, geo holdouts, budget pulses, and
promo on/off tests.

Deliverable shape is **notebook-first**: a shared Python package plus narrated
notebooks. Intended for client deployment and as a public demonstration of method.

## Working agreement (important)

- **Do not write or modify files until User explicitly says "execute."** Preview the
  plan first — structure and design worked through before code.
- "execute" authorizes *the specific plan as previewed*, not the general area of work.
- User wants **critical review of his plans with real adjustments proposed**, not
  validation. Push back when something is over-built or wrong.
- Scope discipline is the point of v0.1. When in doubt, cut. See Non-goals.

## v0.1 scope — single unit, backtest only

One treated unit, multiple treatment windows, clean input data. This is deliberately
narrow. Data quality work, donors, and multi-unit are later slices.

### Components — objective and output

**`contracts.py`** — define and enforce the data contract.
Output: two pandera `DataFrameModel`s — series (`unit_id`, `timestamp` UTC, `value`)
and windows (`window_id`, `start`, `end`) — plus `BaselineState` enum, `RunConfig`,
`EstimateResult`. Validation coerces dtypes and checks columns, nulls, regular
interval, and that each window spans at least one bucket. **No statistical checks, no
anomaly detection, no imputation.** Use class-based `DataFrameModel` (subclassable
later), not `DataFrameSchema`.

**`label_roles()`** (in `estimate.py`) — resolve windows plus washout into per-bucket
roles.
Output: the series frame with a `role` column: `baseline | window | washout`.
This is the key structural decision: washout is **not** a cross-cutting concern
threaded through fit/score/aggregate. One upstream labeling step; everything
downstream filters on `role`. Later roles (`donor`, backtest folds) are new values,
not new plumbing.

**`models/base.py`** — the extension point, in neutral vocabulary.
Output: a protocol (`fit(history)`, `predict(timestamps)`), a forecast frame contract
(`timestamp`, `expected`, `lower`, `upper`), and a plain `dict` registry.
**Prophet's vocabulary must not leak into the interface.** No `yhat`,
`yhat_lower`, `make_future_dataframe` outside the Prophet adapter. This is the single
biggest extensibility risk in v0.1. No entry points or plugin machinery.

**`models/prophet.py`** — default backend.
Output: protocol-conforming adapter that renames Prophet's columns internally. ~40 lines.

**`models/naive.py`** — proves the protocol isn't Prophet-shaped.
Output: groupby-mean over baseline buckets by day-of-week × time-of-day. ~15 lines.
It exists to keep the interface honest and give the notebook a sanity comparator.

**`estimate.py`** — the pipeline and the only public entry point.
Output: `run(df, windows, config) -> EstimateResult`, containing a per-window frame
(actual, counterfactual, lift), pooled ratio, mean-of-ratios, bootstrap CI bounds,
model name, and the `RunConfig` object itself so a run is self-describing.

**`diagnostics.py`** — check the model, separately from the estimate.
Output: `backtest(df, config)` returning prediction-interval coverage and MAE on **one**
held-out contiguous slice of baseline buckets. One fold. Rolling-origin is a later
upgrade behind the same signature.

**`data/synthetic.py`** — the flagship dataset and the reason the harness is testable.
Output: `generate(...) -> (df, windows, truth)` where `truth` carries the planted
multiplicative effect and the window list. Clean data only: trend, weekly and daily
seasonality, noise, known effect inside declared windows. **Load-bearing — build this
first**, since every downstream test runs against it.

**`notebooks/01-single-unit-walkthrough.ipynb`** — end-to-end proof, unpolished.
Output: generate → run → backtest → effective-CPC translation. Business-metric
translation lives *here*, not in the library — it is presentation, not framework.

**`tests/test_recovery.py`** — definition of done.
Output: planted effect recovered within CI, on both backends.

### Layout

```
src/arjentic/incrementality/
  contracts.py
  models/
    base.py
    prophet.py
    naive.py
  estimate.py
  diagnostics.py
  data/synthetic.py
notebooks/01-single-unit-walkthrough.ipynb
tests/test_recovery.py
```

Do not split further. `lift`/`aggregate`/`uncertainty` are consecutive steps in one
narrative that always change together — they live in `estimate.py`. Split later only
when a seam proves itself.

Public export surface: `run`, `RunConfig`, `BaselineState`, `generate`. Nothing else.

## Method decisions that are settled

- **Fit on all baseline buckets, not pre-only.** Filter `role == baseline` to fit,
  predict `role == window`. Training on a pre-period and extrapolating weeks ahead to
  estimate short scattered windows throws away most of the information. Pre-only is a
  later option, not a config field today.
- **Sign resolution** — one branch on `BaselineState`, and the first unit test written:
  - `treated` (baseline is treatment-on, e.g. a paid-media pause): `(counterfactual − actual) / counterfactual`
  - `untreated` (baseline is treatment-off, e.g. a pulse-on test): `(actual − counterfactual) / counterfactual`

  Positive always means treatment increased the metric. Getting this backwards inverts
  every client answer and looks plausible while doing it.
- **Uncertainty** — percentile bootstrap resampling the window-level lift ratios.
  Guard `n_windows == 1` loudly (raise or fall back to forecast-uncertainty-only);
  never return a one-element bootstrap silently. Cluster bootstrap arrives with
  multi-unit — same function plus a group key. No interval decomposition in v0.1.
- **Aggregation** — report pooled ratio and mean-of-ratios side by side. **No gap
  threshold or flagging logic**; let the analyst read the difference.
- **Washout/carryover** — `washout_before` / `washout_after` in buckets, explicitly
  declared, **no zero default**.
- **`RunConfig` rule: no field that nothing reads.** v0.1 fields: metric column, unit
  id, baseline state, washout before/after, primary model + params, bootstrap seed and
  resample count, backtest holdout length. `primary_model` is required —
  pre-registration, not a default.
- **Synthetic flagship granularity: hourly buckets with multi-hour windows.** The
  contract stays interval-agnostic, but the generator's defaults are hourly.
  Sub-bucket windows are incoherent, and minute-level data makes Prophet slow enough to
  spoil the notebook. *(This was the last decision made and the one most worth
  revisiting with Jeff if it causes friction.)*
- **Nothing calibrated to one dataset.** No threshold in absolute time units or fixed
  counts — express in buckets, seasonality cycles, or fractions of window duration.

## Extensibility hooks

These five are the deliberate seams. Everything outside them can be rewritten freely.

1. Model protocol + registry → new backends (CausalImpact, BSTS).
2. `unit_id` already in the contract → multi-unit and donors need no schema migration.
3. Bootstrap takes a per-window frame → a group key makes it a cluster bootstrap.
4. `generate()` kwargs → messiness injectors added later.
5. pandera `DataFrameModel` subclassing → stricter contracts later.

## Non-goals for v0.1

Do not build, and push back if asked mid-stream: donor units, multi-unit, CausalImpact,
placebo tests, imputation or anomaly detection, CLI, dashboard, interval decomposition,
rolling-origin backtest, adstock/carryover modeling, threshold-based flagging.

## Packaging and tooling (settled — do not re-litigate)

- Distribution `arjentic-incrementality`, imported as `arjentic.incrementality`.
- **PEP 420 implicit namespace package.** Layout `src/arjentic/incrementality/`, with
  **no `__init__.py` in `src/arjentic/`**. Add a test asserting its absence — that file
  is the canonical way to break a namespace and it fails late, not at install.
- Build backend `uv_build` with
  `[tool.uv.build-backend] module-name = "arjentic.incrementality"`. Avoid the
  `namespace = true` escape hatch — it disables safety checks.
- **`uv` exclusively.** No pip, venv activation, pyenv, tox, or separate build
  frontend. Commit `uv.lock` and `.python-version`. CI: `astral-sh/setup-uv` +
  `uv sync --locked` + `uv run <tool>`. Build/publish via `uv build` / `uv publish`.
  Colab is the one exception (it ships its own pip) — Colab notebooks need a bootstrap
  cell installing the *published* package.
- **License GPL-3.0-or-later**, verbatim text from gnu.org, SPDX headers in source.

## Language and framing

- "**paid media**," not "paid search."
- **No "interview project" framing anywhere.** The flagship dataset is synthetic with a
  known planted effect. Messy field conditions are described as commonly seen in
  practice.

## Build order

1. `data/synthetic.py` — everything else is tested against it.
2. `contracts.py`
3. `models/base.py` + `models/naive.py` (naive first — it keeps the protocol honest
   before Prophet's shape can bias it)
4. `models/prophet.py`
5. `estimate.py` (`label_roles`, sign resolution, aggregation, bootstrap, `run`)
6. `diagnostics.py`
7. `tests/test_recovery.py`
8. `notebooks/01-single-unit-walkthrough.ipynb`

**Done when** `tests/test_recovery.py` recovers the planted effect within CI on both
backends. That single test is worth more than broad unit coverage at this stage.

## Prior context

The predecessor analysis lives in a separate repo (`horizon-ts-project`) and is
**read-only reference** — do not import from it or reproduce its data-cleaning work.
Its dataset contained deliberately planted anomalies; chasing those is what derailed the
first implementation attempt and prompted this reset.
