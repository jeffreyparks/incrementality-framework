# SPDX-License-Identifier: GPL-3.0-or-later
"""Backtest mechanics, and that its metrics detect what they claim to."""

import pandas as pd
import pytest

from arjentic.incrementality.contracts import BaselineState, Role, RunConfig, validate
from arjentic.incrementality.data.synthetic import generate
from arjentic.incrementality.diagnostics import backtest
from arjentic.incrementality.estimate import label_roles


def _config(**overrides) -> RunConfig:
    defaults = {
        "unit_id": "unit_0",
        "baseline_state": BaselineState.UNTREATED,
        "washout_before": 2,
        "washout_after": 2,
        "primary_model": "naive",
        "backtest_holdout_buckets": 168,
    }
    return RunConfig(**{**defaults, **overrides})


def _holdout_timestamps(df, windows, config) -> pd.Series:
    """Mirror the fold the implementation picks, to target it from a test."""
    observations, treatment_windows = validate(df, windows)
    labeled = label_roles(
        observations, treatment_windows, config.washout_before, config.washout_after
    )
    baseline = labeled[labeled["role"] == Role.BASELINE].reset_index(drop=True)
    start = (len(baseline) - config.backtest_holdout_buckets) // 2
    return baseline.iloc[start : start + config.backtest_holdout_buckets]["timestamp"]


@pytest.mark.parametrize("model", ["naive", "prophet"])
def test_backtest_reports_usable_metrics(model) -> None:
    df, windows, _ = generate()

    result = backtest(df, windows, _config(primary_model=model))

    assert result.model_name == model
    assert result.holdout_buckets == 168
    assert result.nominal_coverage == 0.95
    assert 0.0 <= result.coverage <= 1.0
    assert result.mae > 0.0


@pytest.mark.parametrize("model", ["naive", "prophet"])
def test_coverage_lands_near_nominal_on_clean_data(model) -> None:
    """Loose band: clean synthetic data should not badly miscalibrate either
    backend, but one fold of 168 buckets carries real sampling noise."""
    df, windows, _ = generate()

    result = backtest(df, windows, _config(primary_model=model))

    assert 0.85 <= result.coverage <= 1.0


@pytest.mark.parametrize("model", ["naive", "prophet"])
def test_bias_is_negligible_on_clean_data(model) -> None:
    df, windows, _ = generate()

    result = backtest(df, windows, _config(primary_model=model))

    assert abs(result.relative_bias) < 0.02


def test_bias_detects_a_counterfactual_that_runs_low() -> None:
    """The reason bias is reported at all: inflate the held-out actuals and the
    counterfactual is now systematically low, which is what would inflate a
    lift estimate. MAE alone could not distinguish this from added noise.
    """
    df, windows, _ = generate()
    config = _config()
    inflated = df.copy()
    target = inflated["timestamp"].isin(_holdout_timestamps(df, windows, config))
    inflated.loc[target, "value"] *= 1.10

    result = backtest(inflated, windows, config)

    # expected sits ~1/1.1 of actual, so the relative error is about -0.09.
    assert result.relative_bias == pytest.approx(-0.0909, abs=0.02)


def test_noise_raises_mae_without_creating_bias() -> None:
    """The complement: symmetric noise inflates MAE but leaves bias near zero,
    so the two metrics genuinely carry different information.
    """
    df, windows, _ = generate()
    config = _config()
    noisy = df.copy()
    target = noisy["timestamp"].isin(_holdout_timestamps(df, windows, config))
    alternating = pd.Series(
        [1.15 if i % 2 else 0.85 for i in range(int(target.sum()))],
        index=noisy.index[target],
    )
    noisy.loc[target, "value"] *= alternating

    clean_result = backtest(df, windows, config)
    noisy_result = backtest(noisy, windows, config)

    assert noisy_result.mae > clean_result.mae * 2
    assert abs(noisy_result.relative_bias) < 0.02


def test_holdout_is_interior_with_baseline_on_both_sides() -> None:
    """A tail fold would measure extrapolation, not the interpolation the
    estimator actually performs."""
    df, windows, _ = generate()
    config = _config()
    observations, treatment_windows = validate(df, windows)
    labeled = label_roles(observations, treatment_windows, 2, 2)
    baseline = labeled[labeled["role"] == Role.BASELINE].reset_index(drop=True)

    holdout = _holdout_timestamps(df, windows, config)

    assert baseline["timestamp"].min() < holdout.min()
    assert holdout.max() < baseline["timestamp"].max()


def test_holdout_excludes_window_and_washout_buckets() -> None:
    df, windows, _ = generate()
    config = _config()
    observations, treatment_windows = validate(df, windows)
    labeled = label_roles(observations, treatment_windows, 2, 2)
    non_baseline = labeled.loc[labeled["role"] != Role.BASELINE, "timestamp"]

    holdout = _holdout_timestamps(df, windows, config)

    assert not holdout.isin(non_baseline).any()


def test_holdout_larger_than_the_baseline_is_refused() -> None:
    df, windows, _ = generate(n_days=10)

    with pytest.raises(ValueError, match="must be smaller than"):
        backtest(df, windows, _config(backtest_holdout_buckets=10_000))


def test_backtest_rejects_an_unknown_unit() -> None:
    df, windows, _ = generate(n_days=30)

    with pytest.raises(ValueError, match="not present in the observations"):
        backtest(df, windows, _config(unit_id="nope"))


def test_backtest_is_deterministic() -> None:
    df, windows, _ = generate()
    config = _config()

    assert backtest(df, windows, config) == backtest(df, windows, config)


def test_washout_shrinks_the_available_baseline() -> None:
    """Washout buckets are excluded from the fold, exactly as in the estimate."""
    df, windows, _ = generate()
    observations, treatment_windows = validate(df, windows)

    generous = label_roles(observations, treatment_windows, 50, 50)
    none = label_roles(observations, treatment_windows, 0, 0)

    assert (generous["role"] == Role.BASELINE).sum() < (
        none["role"] == Role.BASELINE
    ).sum()
