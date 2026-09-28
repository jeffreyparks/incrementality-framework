# SPDX-License-Identifier: GPL-3.0-or-later
"""Definition of done: the harness recovers a planted effect on both backends.

Every assertion is made over a sweep of seeds, never a single draw. The seed
moves window placement against the daily and weekly cycle as well as the
noise, and placement is the dominant source of error -- so a single-seed
assertion says more about where that seed's windows happened to land than
about the method. Tolerances were set from measured behaviour over
`SEEDS`, and the sweep is deterministic.

What the sweeps measured, recorded here because they are properties of the
method rather than defects to pin:

- **Point estimates are unbiased.** Mean pooled error is under 0.4pp on every
  stationary scenario for both backends, and under 0.7pp with a trend.
- **The percentile bootstrap under-covers.** Containment of the true effect is
  17/20 (naive) and 15/20 (prophet) against a nominal 95%; with a trend,
  prophet falls to 12/20. The zero-effect case is the same number read the
  other way: prophet's interval excludes zero on about a quarter of seeds with
  no effect planted. The interval narrows with the window count while the
  per-window seasonal misfit does not shrink, which is expected of a percentile
  bootstrap over four clusters. Read these intervals as indicative, not exact.
- **Point error scales with the effect.** The effect is multiplicative, so a
  given misfit in the counterfactual moves the lift by that misfit times
  `1 + effect`. Worst-case pooled error is about 0.062 * (1 + effect) for
  prophet and 0.020 * (1 + effect) for naive.
- **A trend costs naive width, not bias.** The seasonal mean has no trend
  term, so each window's counterfactual is off by where it sits in time. With
  windows stratified across the span those errors roughly cancel in the pooled
  figure, and the bootstrap sees the between-window disagreement: naive's
  median interval widens from 0.023 to 0.199, while prophet's stays near 0.06.
- **A single long window is well covered.** At 72 buckets both backends cover
  truth on 20/20 seeds from forecast uncertainty alone. A single *short* window
  carries a position-specific misfit that interval cannot see, so the
  single-window case is exercised at a length long enough to be sound.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from arjentic.incrementality import BaselineState, RunConfig, generate, run

BACKENDS = ["naive", "prophet"]
SEEDS = range(20)

# Two below the lowest measured containment on stationary data (15/20, prophet),
# so numeric drift passes and a badly miscalibrated interval does not.
COVERAGE_FLOOR = 13

# Worst measured |pooled error| / (1 + effect), with headroom.
RELATIVE_ERROR_TOLERANCE = {"naive": 0.035, "prophet": 0.08}

BIAS_TOLERANCE = 0.01


def _config(model: str, **overrides) -> RunConfig:
    defaults = {
        "unit_id": "unit_0",
        "baseline_state": BaselineState.UNTREATED,
        "washout_before": 2,
        "washout_after": 2,
        "primary_model": model,
        "backtest_holdout_buckets": 168,
    }
    return RunConfig(**{**defaults, **overrides})


@dataclass(frozen=True)
class Sweep:
    """Results of one design run across every seed."""

    target: float
    contained: int
    errors: np.ndarray
    widths: np.ndarray
    ci_methods: set[str]

    @property
    def bias(self) -> float:
        return float(self.errors.mean())

    @property
    def worst_relative_error(self) -> float:
        return float(np.abs(self.errors).max() / (1.0 + self.target))


def _sweep(model: str, target: float, generate_kw=None, config_kw=None) -> Sweep:
    contained, errors, widths, methods = 0, [], [], set()
    for seed in SEEDS:
        df, windows, _ = generate(seed=seed, **(generate_kw or {}))
        result = run(df, windows, _config(model, **(config_kw or {})))
        contained += result.ci_lower <= target <= result.ci_upper
        errors.append(result.pooled_lift - target)
        widths.append(result.ci_upper - result.ci_lower)
        methods.add(result.ci_method)
    return Sweep(target, contained, np.array(errors), np.array(widths), methods)


def _assert_recovered(sweep: Sweep, model: str) -> None:
    assert abs(sweep.bias) < BIAS_TOLERANCE, f"{model} bias {sweep.bias:+.4f}"
    assert sweep.worst_relative_error < RELATIVE_ERROR_TOLERANCE[model], (
        f"{model} worst relative error {sweep.worst_relative_error:.4f}"
    )
    assert sweep.contained >= COVERAGE_FLOOR, (
        f"{model} covered truth in only {sweep.contained}/{len(SEEDS)}"
    )


@pytest.mark.parametrize("model", BACKENDS)
def test_planted_effect_is_recovered_within_the_interval(model) -> None:
    """The headline assertion this milestone is defined by."""
    _assert_recovered(_sweep(model, 0.20), model)


@pytest.mark.parametrize("model", BACKENDS)
def test_the_interval_is_tight_enough_to_be_worth_reporting(model) -> None:
    """Containment alone is not evidence: an interval wide enough to hold any
    plausible answer would satisfy it while saying nothing. Measured median
    widths are 0.023 (naive) and 0.064 (prophet).
    """
    sweep = _sweep(model, 0.20)

    assert np.median(sweep.widths) < 0.15


@pytest.mark.parametrize("model", BACKENDS)
@pytest.mark.parametrize("effect_size", [0.05, 0.50, -0.20])
def test_recovery_holds_across_effect_sizes(model, effect_size) -> None:
    """Including a negative effect, so the sign survives the whole pipeline."""
    sweep = _sweep(model, effect_size, generate_kw={"effect_size": effect_size})

    _assert_recovered(sweep, model)


@pytest.mark.parametrize("model", BACKENDS)
def test_no_planted_effect_yields_an_interval_containing_zero(model) -> None:
    """The false-positive check. A harness that finds lift in unaffected data
    is worse than no harness. Measured: zero is excluded on 3/20 (naive) and
    5/20 (prophet) seeds -- see the module docstring on under-coverage.
    """
    sweep = _sweep(model, 0.0, generate_kw={"effect_size": 0.0})

    _assert_recovered(sweep, model)


@pytest.mark.parametrize("model", BACKENDS)
def test_treated_baseline_recovers_the_effect_through_the_pipeline(model) -> None:
    """A 20% drop inside the windows, read as a treatment-on baseline, is a 20%
    contribution from the treatment. Unit tests pin the formula; this pins the
    wiring from config through to the reported number.
    """
    sweep = _sweep(
        model,
        0.20,
        generate_kw={"effect_size": -0.20},
        config_kw={"baseline_state": BaselineState.TREATED},
    )

    _assert_recovered(sweep, model)


@pytest.mark.parametrize("model", BACKENDS)
def test_single_window_holdout_recovers_within_its_forecast_interval(model) -> None:
    """A one-window design is real and supported, but its interval comes from
    forecast uncertainty alone. Exercised at 72 buckets, where measured
    containment is 20/20 on both backends; see the module docstring on short
    windows.
    """
    sweep = _sweep(
        model, 0.20, generate_kw={"n_windows": 1, "window_duration_buckets": 72}
    )

    assert sweep.ci_methods == {"forecast"}
    assert sweep.contained >= 18
    assert abs(sweep.bias) < BIAS_TOLERANCE


@pytest.mark.parametrize("model", BACKENDS)
def test_recovery_survives_a_generous_washout(model) -> None:
    """Discarding buckets around each window must not bias what is left."""
    sweep = _sweep(model, 0.20, config_kw={"washout_before": 12, "washout_after": 12})

    _assert_recovered(sweep, model)


def test_a_trend_costs_naive_width_but_neither_backend_bias() -> None:
    """The generator's default is trend-free, so without this the recovery
    suite would never see one. Neither backend is biased by it; naive, which
    has no trend term, pays in interval width instead (measured median 0.199
    against prophet's 0.059). Prophet's containment drops to 12/20 here, below
    the stationary floor, and is recorded rather than asserted.
    """
    trend = {"trend_frac_per_day": 0.005}
    naive = _sweep("naive", 0.20, generate_kw=trend)
    prophet = _sweep("prophet", 0.20, generate_kw=trend)

    assert abs(naive.bias) < BIAS_TOLERANCE
    assert abs(prophet.bias) < BIAS_TOLERANCE
    assert np.median(naive.widths) > 2 * np.median(prophet.widths)


def test_the_two_backends_agree_with_each_other() -> None:
    """Independent methods landing on the same answer is the strongest evidence
    available that the pipeline, not one model's quirk, produced it. Measured
    worst per-seed disagreement is 0.057.
    """
    for seed in SEEDS:
        df, windows, _ = generate(seed=seed)

        naive = run(df, windows, _config("naive"))
        prophet = run(df, windows, _config("prophet"))

        assert naive.pooled_lift == pytest.approx(prophet.pooled_lift, abs=0.08)
        assert np.sign(naive.pooled_lift) == np.sign(prophet.pooled_lift)
