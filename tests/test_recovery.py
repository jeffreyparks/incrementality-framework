# SPDX-License-Identifier: GPL-3.0-or-later
"""Definition of done: the harness recovers a planted effect on both backends.

Every tolerance here was set from measured behaviour rather than guessed, and
the seeds are fixed so the results are deterministic.

Two measured limitations are recorded here rather than asserted, because they
are properties of the method and not defects to be pinned.

A *single* short window carries a position-specific seasonality misfit that the
forecast interval cannot see. Holding the design fixed and moving only where
one 6-bucket window lands moves Prophet's error from +8.1pp to -6.0pp, with
about 1.3pp of scatter within each position; coverage there falls to roughly
half. Multiple scattered windows average that misfit out, which is why the
multi-window results below cover truth and a lone short window need not. The
single-window case is therefore exercised at a window long enough to be sound.

The percentile bootstrap interval also under-covers. Measured across window
counts from 2 to 12, containment of the true effect runs 73-87% against a
nominal 95%, and it does not improve with more windows -- the interval narrows
roughly with the square root of the window count while the systematic misfit
above does not shrink. With only two windows the interval can collapse almost
to nothing (observed minimum width 0.0011). This is expected of a percentile
bootstrap over few clusters, so these intervals should be read as indicative
rather than exact.
"""

import numpy as np
import pytest

from arjentic.incrementality import BaselineState, RunConfig, generate, run

BACKENDS = ["naive", "prophet"]


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


@pytest.mark.parametrize("model", BACKENDS)
def test_planted_effect_is_recovered_within_the_interval(model) -> None:
    """The headline assertion this milestone is defined by."""
    df, windows, truth = generate()

    result = run(df, windows, _config(model))

    assert result.ci_lower <= truth["effect_size"] <= result.ci_upper
    assert result.pooled_lift == pytest.approx(truth["effect_size"], abs=0.03)
    assert result.mean_of_ratios_lift == pytest.approx(truth["effect_size"], abs=0.03)


@pytest.mark.parametrize("model", BACKENDS)
def test_the_interval_is_tight_enough_to_be_worth_reporting(model) -> None:
    """Containment alone is not evidence: an interval wide enough to hold any
    plausible answer would satisfy it while saying nothing. Measured widths are
    0.024 (naive) and 0.072 (prophet).
    """
    df, windows, _ = generate()

    result = run(df, windows, _config(model))

    assert (result.ci_upper - result.ci_lower) < 0.15


@pytest.mark.parametrize("model", BACKENDS)
@pytest.mark.parametrize("effect_size", [0.05, 0.20, 0.50, -0.20])
def test_recovery_holds_across_effect_sizes(model, effect_size) -> None:
    """Including a negative effect, so the sign survives the whole pipeline."""
    df, windows, truth = generate(effect_size=effect_size)

    result = run(df, windows, _config(model))

    assert result.ci_lower <= truth["effect_size"] <= result.ci_upper
    assert result.pooled_lift == pytest.approx(effect_size, abs=0.03)


@pytest.mark.parametrize("model", BACKENDS)
def test_no_planted_effect_yields_an_interval_containing_zero(model) -> None:
    """The false-positive check. A harness that finds lift in unaffected data
    is worse than no harness.
    """
    df, windows, _ = generate(effect_size=0.0)

    result = run(df, windows, _config(model))

    assert result.ci_lower <= 0.0 <= result.ci_upper
    assert result.pooled_lift == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize("model", BACKENDS)
def test_treated_baseline_recovers_the_effect_through_the_pipeline(model) -> None:
    """A 20% drop inside the windows, read as a treatment-on baseline, is a 20%
    contribution from the treatment. Unit tests pin the formula; this pins the
    wiring from config through to the reported number.
    """
    df, windows, _ = generate(effect_size=-0.20)

    result = run(df, windows, _config(model, baseline_state=BaselineState.TREATED))

    assert result.ci_lower <= 0.20 <= result.ci_upper
    assert result.pooled_lift == pytest.approx(0.20, abs=0.03)


@pytest.mark.parametrize("model", BACKENDS)
def test_interval_covers_truth_across_many_seeds(model) -> None:
    """Calibration, not luck. One passing seed says little about an interval.

    Measured containment is 18/20 (naive) and 20/20 (prophet). The floor sits
    at 16 to leave room for numeric drift while still catching a badly
    miscalibrated interval. Naive's shortfall is expected: a percentile
    bootstrap over four windows is known to under-cover.
    """
    contained = 0
    seeds = range(20)

    for seed in seeds:
        df, windows, truth = generate(seed=seed)
        result = run(df, windows, _config(model))
        contained += result.ci_lower <= truth["effect_size"] <= result.ci_upper

    assert contained >= 16, f"{model} covered truth in only {contained}/{len(seeds)}"


@pytest.mark.parametrize("model", BACKENDS)
def test_single_window_holdout_recovers_within_its_forecast_interval(model) -> None:
    """A one-window design is real and supported, but its interval comes from
    forecast uncertainty alone. Exercised at 72 buckets, where measured
    containment is 19/20; see the module docstring on short windows.
    """
    df, windows, truth = generate(n_windows=1, window_duration_buckets=72)

    result = run(df, windows, _config(model))

    assert result.ci_method == "forecast"
    assert result.ci_lower <= truth["effect_size"] <= result.ci_upper
    assert result.pooled_lift == pytest.approx(truth["effect_size"], abs=0.03)


@pytest.mark.parametrize("model", BACKENDS)
def test_recovery_survives_a_generous_washout(model) -> None:
    """Discarding buckets around each window must not bias what is left."""
    df, windows, truth = generate()

    result = run(df, windows, _config(model, washout_before=12, washout_after=12))

    assert result.ci_lower <= truth["effect_size"] <= result.ci_upper


def test_the_two_backends_agree_with_each_other() -> None:
    """Independent methods landing on the same answer is the strongest evidence
    available that the pipeline, not one model's quirk, produced it.
    """
    df, windows, truth = generate()

    naive = run(df, windows, _config("naive"))
    prophet = run(df, windows, _config("prophet"))

    assert naive.pooled_lift == pytest.approx(prophet.pooled_lift, abs=0.02)
    assert np.sign(naive.pooled_lift) == np.sign(prophet.pooled_lift)
