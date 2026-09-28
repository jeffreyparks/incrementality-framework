# arjentic-incrementality

A reusable incrementality testing harness for pulse and holdout experiments,
built on time-series counterfactual forecasting behind a pluggable model
interface (Prophet by default).

v0.1 covers a single treated unit with multiple treatment windows: a synthetic
dataset with a known planted effect, the estimator, and a counterfactual-model
backtest. Multi-unit and donor designs, geo holdouts, input data-quality checks
and a reporting dashboard are on the roadmap, not in this release.

## Method

The pattern is the same across a range of experiment designs:

> binary-treatment holdout → counterfactual forecast trained on untreated
> periods → per-window lift → aggregation → bootstrap confidence interval →
> business-metric translation

That generalizes to paid-media holdouts, geo holdouts, budget pulses, and promo
on/off tests. A counterfactual model is fit on the untreated ("baseline")
buckets of a single unit's time series and used to predict what the metric
would have been inside each declared treatment window. The gap between actual
and counterfactual is the lift; window-level lift ratios are aggregated and a
percentile bootstrap over those windows gives the interval.

Two design decisions:

- **Sign convention is explicit.** Whether the baseline period is
  treatment-*on* (e.g. a paid-media pause) or treatment-*off* (e.g. a pulse-on
  test) determines the sign of the estimate. It is declared, not inferred, so a
  positive number always means the treatment increased the metric.
- **Washout is declared, not defaulted.** Buckets adjacent to a window are
  labeled as washout up front and excluded from both fitting and scoring, with
  no silent zero default.

## v0.1

A single treated unit, multiple treatment windows, backtest validation only. 

- `contracts.py` — the data contract (`Observations`, `TreatmentWindows`,
  `validate()`), plus `RunConfig`, `BaselineState`, and `EstimateResult`.
- `models/` — a neutral `Forecaster` protocol and forecast-frame contract, with
  a naive (seasonal-mean) backend and a Prophet backend behind it.
- `estimate.py` — `run()`: role labeling (baseline / window / washout), the
  sign-resolution branch, pooled and mean-of-ratios aggregation, and a
  percentile bootstrap interval over windows — falling back to a
  forecast-uncertainty interval for a single-window design, since resampling
  one window is degenerate.
- `diagnostics.py` — `backtest()`, a single interior holdout reporting
  interval coverage, MAE, and bias (MAE alone can't distinguish a biased
  counterfactual from harmless noise, so both are reported).
- `data/synthetic.py` — the flagship dataset: an hourly series with trend,
  weekly and daily seasonality, noise, and a known multiplicative effect
  planted inside declared windows, so the harness can be checked against a
  known answer rather than a plausible-looking one.
- `notebooks/01-single-unit-demo.ipynb` — the end-to-end walkthrough: generate,
  estimate, backtest, and a business-metric translation.
- `tests/test_recovery.py` — the definition of done. Across 20 seeds, which
  move window placement as well as noise, both backends recover the planted
  effect without bias. The percentile bootstrap interval contains the truth on
  15–17 of 20 seeds against a nominal 95%, so it under-covers; read it as
  indicative. The measured figures are recorded in the test module.

## Install

```sh
uv sync
```

## Quickstart

```python
from arjentic.incrementality import BaselineState, RunConfig, backtest, generate, run

df, windows, truth = generate(effect_size=0.20)

config = RunConfig(
    unit_id="unit_0",
    baseline_state=BaselineState.UNTREATED,
    washout_before=3,
    washout_after=6,
    primary_model="prophet",
    backtest_holdout_buckets=168,
)

result = run(df, windows, config)
diagnostic = backtest(df, windows, config)
```

`df` holds `unit_id`, `timestamp` (UTC) and `value`; `windows` holds
`window_id`, `start` and `end` as half-open intervals; `truth` carries the
planted effect so a run can be scored against it. `result` is self-describing:
it carries `pooled_lift`, `mean_of_ratios_lift`, a confidence interval with the
method that produced it (`ci_method`), the full `series` with the
counterfactual on every bucket, and the `config` that produced the run.
`diagnostic` reports how well the model predicts baseline buckets it has not
seen: interval coverage, MAE and bias.

## Development

```sh
uv sync
uv run pytest
```

To regenerate the demo notebook after editing its paired `.py` source:

```sh
uv run jupytext --to ipynb --execute notebooks/01-single-unit-demo.py
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
