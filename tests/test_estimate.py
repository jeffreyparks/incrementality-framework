# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline mechanics: sign resolution, role labeling, aggregation, intervals.

Recovery of the planted effect on both backends belongs to test_recovery.py;
these tests pin the machinery rather than the answer.
"""

import numpy as np
import pandas as pd
import pytest

from arjentic.incrementality.contracts import BaselineState, Role, RunConfig, validate
from arjentic.incrementality.data.synthetic import generate
from arjentic.incrementality.estimate import _signed_lift, label_roles, run


def _config(**overrides) -> RunConfig:
    defaults = {
        "unit_id": "unit_0",
        "baseline_state": BaselineState.UNTREATED,
        "washout_before": 2,
        "washout_after": 2,
        "primary_model": "naive",
        "backtest_holdout_buckets": 24,
    }
    return RunConfig(**{**defaults, **overrides})


# --- sign resolution -------------------------------------------------------
# The first test written: inverting this flips every client answer while the
# output still looks entirely reasonable.


def test_untreated_baseline_reports_a_gain_when_the_window_is_higher() -> None:
    """Treatment applied during the window and the metric rose."""
    assert _signed_lift(120.0, 100.0, BaselineState.UNTREATED) == pytest.approx(0.20)


def test_untreated_baseline_reports_a_loss_when_the_window_is_lower() -> None:
    assert _signed_lift(80.0, 100.0, BaselineState.UNTREATED) == pytest.approx(-0.20)


def test_treated_baseline_reports_a_gain_when_the_window_is_lower() -> None:
    """Treatment removed during the window and the metric fell, so treatment
    was contributing."""
    assert _signed_lift(80.0, 100.0, BaselineState.TREATED) == pytest.approx(0.20)


def test_treated_baseline_reports_a_loss_when_the_window_is_higher() -> None:
    assert _signed_lift(120.0, 100.0, BaselineState.TREATED) == pytest.approx(-0.20)


def test_the_two_baseline_states_are_mirror_images() -> None:
    for actual in (60.0, 100.0, 140.0):
        treated = _signed_lift(actual, 100.0, BaselineState.TREATED)
        untreated = _signed_lift(actual, 100.0, BaselineState.UNTREATED)
        assert treated == pytest.approx(-untreated)


# --- role labeling ---------------------------------------------------------


def test_role_constants_compare_equal_to_column_values(observations) -> None:
    """Regression guard. As a `str`-subclass Enum, `Role.WINDOW` compared equal
    to nothing in a pandas 3.0 `str` column, so every role filter silently
    returned an empty frame. Keep these plain strings.
    """
    obs, windows = observations
    labeled = label_roles(obs, windows, washout_before=2, washout_after=2)

    for role in (Role.BASELINE, Role.WINDOW, Role.WASHOUT):
        assert type(role) is str
        assert (labeled["role"] == role).sum() > 0


@pytest.fixture
def observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, windows, _ = generate(n_days=30, n_windows=2, window_duration_buckets=6)
    return validate(df, windows)


def test_every_bucket_gets_exactly_one_role(observations) -> None:
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=3, washout_after=3)

    assert len(labeled) == len(obs)
    assert set(labeled["role"]) == {
        Role.BASELINE,
        Role.WINDOW,
        Role.WASHOUT,
    }


def test_window_buckets_match_the_declared_windows(observations) -> None:
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=0, washout_after=0)
    window_buckets = labeled[labeled["role"] == Role.WINDOW]

    assert len(window_buckets) == 2 * 6
    assert set(window_buckets["window_id"]) == set(windows["window_id"])


def test_washout_count_follows_the_declared_buckets(observations) -> None:
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=3, washout_after=4)

    washout = labeled[labeled["role"] == Role.WASHOUT]
    assert len(washout) == 2 * (3 + 4)


def test_zero_washout_leaves_no_washout_buckets(observations) -> None:
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=0, washout_after=0)

    assert (labeled["role"] == Role.WASHOUT).sum() == 0


def test_washout_sits_immediately_around_each_window(observations) -> None:
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=2, washout_after=2)

    first = labeled.index[labeled["role"] == Role.WINDOW][0]
    last = labeled.index[labeled["window_id"] == windows.iloc[0]["window_id"]][-1]
    assert labeled.loc[first - 2 : first - 1, "role"].eq(Role.WASHOUT).all()
    assert labeled.loc[last + 1 : last + 2, "role"].eq(Role.WASHOUT).all()


def test_window_role_wins_over_a_neighbours_washout(observations) -> None:
    """Washout wide enough to reach the next window must not overwrite it."""
    obs, windows = observations
    gap_buckets = int(
        (windows.iloc[1]["start"] - windows.iloc[0]["end"]) / pd.Timedelta(hours=1)
    )

    labeled = label_roles(
        obs, windows, washout_before=gap_buckets + 10, washout_after=gap_buckets + 10
    )
    window_buckets = labeled[labeled["role"] == Role.WINDOW]

    assert len(window_buckets) == 2 * 6
    assert set(window_buckets["window_id"]) == set(windows["window_id"])


def test_washout_clips_at_the_series_edges(observations) -> None:
    """A washout wider than the series must not raise or wrap around."""
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=10_000, washout_after=10_000)

    assert (labeled["role"] == Role.BASELINE).sum() == 0
    assert len(labeled) == len(obs)


def test_label_roles_refuses_multiple_units(observations) -> None:
    obs, windows = observations
    two_units = pd.concat([obs, obs.assign(unit_id="unit_1")], ignore_index=True)

    with pytest.raises(ValueError, match="single unit"):
        label_roles(two_units, windows, washout_before=1, washout_after=1)


def test_non_window_buckets_carry_no_window_id(observations) -> None:
    obs, windows = observations

    labeled = label_roles(obs, windows, washout_before=2, washout_after=2)

    assert labeled.loc[labeled["role"] != Role.WINDOW, "window_id"].isna().all()


# --- run() -----------------------------------------------------------------


def test_run_returns_a_self_describing_result() -> None:
    df, windows, truth = generate(n_days=45)

    result = run(df, windows, _config())

    assert result.model_name == "naive"
    assert result.config.baseline_state == BaselineState.UNTREATED
    assert result.ci_method == "bootstrap"
    assert list(result.per_window.columns) == [
        "window_id",
        "actual",
        "counterfactual",
        "buckets",
        "lift",
    ]
    assert len(result.per_window) == len(windows)


def test_run_reports_both_aggregations(observations) -> None:
    df, windows, _ = generate(n_days=45)

    result = run(df, windows, _config())

    assert result.pooled_lift == pytest.approx(result.mean_of_ratios_lift, abs=0.05)
    assert result.ci_lower < result.pooled_lift < result.ci_upper


def test_run_recovers_the_planted_effect_within_the_interval() -> None:
    df, windows, truth = generate(n_days=45)

    result = run(df, windows, _config())

    assert result.ci_lower <= truth["effect_size"] <= result.ci_upper


def test_run_rejects_an_unknown_unit() -> None:
    df, windows, _ = generate(n_days=30)

    with pytest.raises(ValueError, match="not present in the observations"):
        run(df, windows, _config(unit_id="nope"))


def test_run_fits_on_baseline_and_excludes_washout() -> None:
    """Washout buckets must reach neither the fit nor the score."""
    df, windows, _ = generate(n_days=30, n_windows=2)
    observations, treatment_windows = validate(df, windows)

    labeled = label_roles(observations, treatment_windows, 5, 5)
    result = run(df, windows, _config(washout_before=5, washout_after=5))

    assert result.per_window["buckets"].sum() == (labeled["role"] == Role.WINDOW).sum()


# --- uncertainty -----------------------------------------------------------


def test_bootstrap_is_reproducible_under_a_fixed_seed() -> None:
    df, windows, _ = generate(n_days=45)

    first = run(df, windows, _config(bootstrap_seed=7))
    second = run(df, windows, _config(bootstrap_seed=7))
    other = run(df, windows, _config(bootstrap_seed=8))

    assert (first.ci_lower, first.ci_upper) == (second.ci_lower, second.ci_upper)
    assert (first.ci_lower, first.ci_upper) != (other.ci_lower, other.ci_upper)


def test_single_window_falls_back_to_forecast_uncertainty() -> None:
    """A one-window holdout is a real design, so it must produce an interval
    rather than a degenerate bootstrap."""
    df, windows, truth = generate(n_days=45, n_windows=1)

    result = run(df, windows, _config())

    assert result.ci_method == "forecast"
    assert result.ci_lower < result.pooled_lift < result.ci_upper
    assert result.ci_lower <= truth["effect_size"] <= result.ci_upper


def test_single_window_pooled_and_mean_of_ratios_coincide() -> None:
    df, windows, _ = generate(n_days=45, n_windows=1)

    result = run(df, windows, _config())

    assert result.pooled_lift == pytest.approx(result.mean_of_ratios_lift)


def test_forecast_interval_is_not_inflated_by_summing_bounds() -> None:
    """Summing per-bucket bounds would assume perfectly correlated errors and
    widen the interval by roughly sqrt(n_buckets)."""
    df, windows, _ = generate(n_days=45, n_windows=1, window_duration_buckets=16)
    observations, treatment_windows = validate(df, windows)
    labeled = label_roles(observations, treatment_windows, 0, 0)
    n_buckets = int((labeled["role"] == Role.WINDOW).sum())

    result = run(df, windows, _config(washout_before=0, washout_after=0))
    half_width = (result.ci_upper - result.ci_lower) / 2

    naive_sum_half_width = half_width * np.sqrt(n_buckets)
    assert half_width < naive_sum_half_width


def test_bootstrap_interval_widens_with_heterogeneous_windows() -> None:
    """Between-window variation is what the bootstrap is supposed to see."""
    uniform = run(*generate(n_days=60, n_windows=4)[:2], _config())
    df, windows, _ = generate(n_days=60, n_windows=4)
    scattered = df.copy()
    inside = (scattered["timestamp"] >= windows.iloc[0]["start"]) & (
        scattered["timestamp"] < windows.iloc[0]["end"]
    )
    scattered.loc[inside, "value"] *= 3.0

    heterogeneous = run(scattered, windows, _config())

    assert (heterogeneous.ci_upper - heterogeneous.ci_lower) > (
        uniform.ci_upper - uniform.ci_lower
    )


def test_public_surface_is_exactly_what_was_declared() -> None:
    import arjentic.incrementality as package

    assert set(package.__all__) == {"BaselineState", "RunConfig", "generate", "run"}
