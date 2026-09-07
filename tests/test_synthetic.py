# SPDX-License-Identifier: GPL-3.0-or-later
"""The synthetic generator is load-bearing.

Every downstream test asserts that the harness recovers this generator's
planted effect, so an error here surfaces as a confusing failure in the
estimator rather than in the data. These tests use a deliberately naive
day-of-week x hour-of-day baseline, never the library's own estimator, so
they can only fail for data reasons.
"""

import numpy as np
import pandas as pd
import pytest

from arjentic.incrementality.data.synthetic import generate


def _window_mask(timestamps: pd.Series, windows: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)
    for _, w in windows.iterrows():
        mask |= (timestamps >= w["start"]) & (timestamps < w["end"])
    return mask


def _naive_lift(df: pd.DataFrame, windows: pd.DataFrame) -> float:
    """Mean per-bucket lift against a day-of-week x hour-of-day baseline."""
    in_window = _window_mask(df["timestamp"], windows)
    frame = df.assign(
        dow=df["timestamp"].dt.dayofweek,
        hod=df["timestamp"].dt.hour,
        in_window=in_window,
    )
    baseline = frame[~frame["in_window"]].groupby(["dow", "hod"])["value"].mean()
    expected = frame.set_index(["dow", "hod"]).index.map(baseline)
    return float(((frame["value"] - expected) / expected)[in_window].mean())


def test_returns_contract_shaped_frames() -> None:
    df, windows, truth = generate(n_days=30)

    assert list(df.columns) == ["unit_id", "timestamp", "value"]
    assert list(windows.columns) == ["window_id", "start", "end"]
    assert len(df) == 30 * 24
    assert not df.isnull().any().any()
    assert truth["window_ids"] == windows["window_id"].tolist()


def test_series_has_regular_hourly_interval() -> None:
    df, _, _ = generate(n_days=30)

    assert df["timestamp"].diff().dropna().unique().tolist() == [pd.Timedelta(hours=1)]


def test_windows_are_ordered_non_overlapping_and_clear_of_edges() -> None:
    df, windows, _ = generate()

    assert (windows["end"] > windows["start"]).all()
    assert (windows["start"].shift(-1).dropna() >= windows["end"].iloc[:-1]).all()
    # every window needs baseline both before and after it
    assert windows["start"].min() > df["timestamp"].min()
    assert windows["end"].max() < df["timestamp"].max()


def test_planted_effect_is_recoverable() -> None:
    df, windows, truth = generate(effect_size=0.25, noise_std=0.01)

    assert _naive_lift(df, windows) == pytest.approx(truth["effect_size"], abs=0.02)


def test_no_lift_when_effect_size_is_zero() -> None:
    """Negative control: the naive baseline must not manufacture lift at the
    window positions on its own."""
    df, windows, _ = generate(effect_size=0.0, noise_std=0.01)

    assert _naive_lift(df, windows) == pytest.approx(0.0, abs=0.02)


def test_generation_is_deterministic_under_a_fixed_seed() -> None:
    first, first_windows, _ = generate(seed=7)
    second, second_windows, _ = generate(seed=7)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_windows, second_windows)


def test_seed_actually_drives_the_noise() -> None:
    first, _, _ = generate(seed=1)
    second, _, _ = generate(seed=2)

    assert not first["value"].equals(second["value"])
