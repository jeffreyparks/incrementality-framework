# arjentic-incrementality

A reusable incrementality testing harness for variety of experiment designs,
including single and multi-unit holdout, donors and geo-holdout with a range 
of time-series counterfactual forecasting backends.  

Includes synthetic data generators, counterfactual model backtesting and
evaluation, validation checks against common data issues, and a standalone 
dashboard app for easy report generation.

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

## Status

Early — v0.1 is under construction and covers a single treated unit with
multiple treatment windows. What exists today:

- `arjentic.incrementality.data.synthetic` — the flagship synthetic dataset: an
  hourly series with trend, weekly and daily seasonality, noise, and a known
  multiplicative effect planted inside declared windows. Because the effect is
  planted, the harness can be validated against a known answer rather than
  against a plausible-looking one.

Still to land: the data contract, the model protocol and its Prophet and naive
backends, the estimation pipeline, backtest diagnostics, and the walkthrough
notebook.

## Install

```sh
uv sync
```

## Quickstart

```python
from arjentic.incrementality.data.synthetic import generate

df, windows, truth = generate(effect_size=0.20)
```

`df` holds `unit_id`, `timestamp` (UTC) and `value`; `windows` holds
`window_id`, `start` and `end` as half-open intervals; `truth` carries the
planted effect so a run can be scored against it.

## Development

```sh
uv sync
uv run pytest
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
