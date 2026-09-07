# SPDX-License-Identifier: GPL-3.0-or-later
"""A contract whose checks are unverified is worse than no contract, since it
grants false confidence. Every check gets a case that must fail it.
"""

import dataclasses

import pandas as pd
import pandera.errors
import pandera.pandas as pa
import pandera.typing.pandas as pat
import pytest

from arjentic.incrementality.contracts import (
    BaselineState,
    EstimateResult,
    Observations,
    RunConfig,
    TreatmentWindows,
    validate,
)
from arjentic.incrementality.data.synthetic import generate


@pytest.fixture
def clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, windows, _ = generate(n_days=30, n_windows=2)
    return df, windows


def test_generator_output_satisfies_the_contract(clean) -> None:
    df, windows = clean

    observations, treatment_windows = validate(df, windows)

    assert len(observations) == len(df)
    assert len(treatment_windows) == len(windows)


def test_validate_coerces_timestamp_resolution(clean) -> None:
    """The generator emits microsecond resolution; the contract normalizes it."""
    df, windows = clean
    assert df["timestamp"].dtype == "datetime64[us, UTC]"

    observations, _ = validate(df, windows)

    assert observations["timestamp"].dtype == "datetime64[ns, UTC]"


def test_validate_returns_canonically_sorted_frames(clean) -> None:
    df, windows = clean
    shuffled = df.sample(frac=1.0, random_state=0)
    reversed_windows = windows.iloc[::-1]

    observations, treatment_windows = validate(shuffled, reversed_windows)

    assert observations["timestamp"].is_monotonic_increasing
    assert treatment_windows["start"].is_monotonic_increasing


def test_naive_timestamps_are_rejected(clean) -> None:
    df, windows = clean
    naive = df.assign(timestamp=df["timestamp"].dt.tz_localize(None))

    with pytest.raises(ValueError, match="timezone-aware"):
        validate(naive, windows)


def test_naive_window_bounds_are_rejected(clean) -> None:
    df, windows = clean
    naive = windows.assign(start=windows["start"].dt.tz_localize(None))

    with pytest.raises(ValueError, match="timezone-aware"):
        validate(df, naive)


def test_non_utc_timestamps_are_converted_not_relabeled(clean) -> None:
    df, windows = clean
    eastern = df.assign(timestamp=df["timestamp"].dt.tz_convert("US/Eastern"))

    observations, _ = validate(eastern, windows)

    assert observations["timestamp"].iloc[0] == df["timestamp"].iloc[0]


def test_duplicate_buckets_are_rejected(clean) -> None:
    df, windows = clean
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(pandera.errors.SchemaError):
        validate(duplicated, windows)


def test_all_identical_timestamps_are_rejected(clean) -> None:
    """The case that slips past a diff-based regular-interval check on its own."""
    df, windows = clean
    collapsed = df.assign(timestamp=df["timestamp"].iloc[0])

    with pytest.raises(pandera.errors.SchemaError):
        validate(collapsed, windows)


def test_irregular_interval_is_rejected(clean) -> None:
    df, windows = clean
    gapped = df.drop(index=5).reset_index(drop=True)

    with pytest.raises(pandera.errors.SchemaError):
        validate(gapped, windows)


def test_null_values_are_rejected(clean) -> None:
    df, windows = clean
    with_null = df.copy()
    with_null.loc[0, "value"] = None

    with pytest.raises(pandera.errors.SchemaError):
        validate(with_null, windows)


def test_missing_column_is_rejected(clean) -> None:
    df, windows = clean

    with pytest.raises(pandera.errors.SchemaError):
        validate(df.drop(columns=["value"]), windows)


def test_window_ending_before_it_starts_is_rejected(clean) -> None:
    df, windows = clean
    inverted = windows.copy()
    inverted.loc[0, ["start", "end"]] = [
        windows.loc[0, "end"],
        windows.loc[0, "start"],
    ]

    with pytest.raises(pandera.errors.SchemaError):
        validate(df, inverted)


def test_duplicate_window_id_is_rejected(clean) -> None:
    df, windows = clean
    duplicated = windows.copy()
    duplicated.loc[1, "window_id"] = duplicated.loc[0, "window_id"]

    with pytest.raises(pandera.errors.SchemaError):
        validate(df, duplicated)


def test_window_outside_the_observed_range_is_rejected(clean) -> None:
    """Long enough to span a bucket, but covering none of them."""
    df, windows = clean
    offset = pd.Timedelta(days=365)
    displaced = windows.copy()
    displaced.loc[0, ["start", "end"]] = [
        windows.loc[0, "start"] + offset,
        windows.loc[0, "end"] + offset,
    ]

    with pytest.raises(ValueError, match="cover no observation buckets"):
        validate(df, displaced)


def test_window_falling_between_buckets_is_rejected(clean) -> None:
    """Sub-bucket windows are judged by overlap, not duration: one landing
    strictly between two buckets covers nothing and is rejected."""
    df, windows = clean
    between = windows.copy()
    between.loc[0, ["start", "end"]] = [
        windows.loc[0, "start"] + pd.Timedelta(minutes=1),
        windows.loc[0, "start"] + pd.Timedelta(minutes=59),
    ]

    with pytest.raises(ValueError, match="cover no observation buckets"):
        validate(df, between)


def test_sub_bucket_window_aligned_to_a_boundary_is_accepted(clean) -> None:
    """The counterpart: it still covers its bucket, so the contract allows it.
    A duration-based check would have wrongly rejected this."""
    df, windows = clean
    narrow = windows.copy()
    narrow.loc[0, "end"] = narrow.loc[0, "start"] + pd.Timedelta(minutes=1)

    _, treatment_windows = validate(df, narrow)

    assert len(treatment_windows) == len(windows)


def test_models_are_subclassable_for_stricter_contracts(clean) -> None:
    """The documented extensibility hook: tighten the contract by subclassing."""
    df, windows = clean

    class NonNegativeObservations(Observations):
        value: pat.Series[float] = pa.Field(ge=0)

    assert NonNegativeObservations.validate(df) is not None

    negative = df.copy()
    negative.loc[0, "value"] = -1.0
    with pytest.raises(pandera.errors.SchemaError):
        NonNegativeObservations.validate(negative)


def test_run_config_rejects_negative_washout() -> None:
    with pytest.raises(ValueError, match="washout"):
        RunConfig(
            unit_id="unit_0",
            baseline_state=BaselineState.UNTREATED,
            washout_before=-1,
            washout_after=0,
            primary_model="naive",
            backtest_holdout_buckets=24,
        )


def test_run_config_rejects_non_positive_holdout_and_resamples() -> None:
    common = {
        "unit_id": "unit_0",
        "baseline_state": BaselineState.UNTREATED,
        "washout_before": 0,
        "washout_after": 0,
        "primary_model": "naive",
    }

    with pytest.raises(ValueError, match="backtest_holdout_buckets"):
        RunConfig(**common, backtest_holdout_buckets=0)

    with pytest.raises(ValueError, match="bootstrap_resamples"):
        RunConfig(**common, backtest_holdout_buckets=24, bootstrap_resamples=0)


def test_run_config_is_frozen_and_defaults_are_isolated() -> None:
    config = RunConfig(
        unit_id="unit_0",
        baseline_state=BaselineState.UNTREATED,
        washout_before=2,
        washout_after=2,
        primary_model="naive",
        backtest_holdout_buckets=24,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.unit_id = "other"

    other = RunConfig(
        unit_id="unit_1",
        baseline_state=BaselineState.TREATED,
        washout_before=0,
        washout_after=0,
        primary_model="naive",
        backtest_holdout_buckets=24,
    )
    assert config.model_params is not other.model_params


def test_baseline_state_values_match_the_declared_vocabulary() -> None:
    assert BaselineState.TREATED == "treated"
    assert BaselineState.UNTREATED == "untreated"


def test_estimate_result_comparison_does_not_raise_on_its_frame() -> None:
    """`eq=False` keeps the embedded frame from poisoning equality."""
    config = RunConfig(
        unit_id="unit_0",
        baseline_state=BaselineState.UNTREATED,
        washout_before=0,
        washout_after=0,
        primary_model="naive",
        backtest_holdout_buckets=24,
    )
    result = EstimateResult(
        per_window=pd.DataFrame({"window_id": ["w0"], "lift": [0.2]}),
        pooled_lift=0.2,
        mean_of_ratios_lift=0.2,
        ci_lower=0.1,
        ci_upper=0.3,
        model_name="naive",
        config=config,
    )

    assert result == result
    assert result != config
