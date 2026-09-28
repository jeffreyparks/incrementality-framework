# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic single-unit dataset with a planted, known treatment effect.

The flagship dataset for the harness: hourly buckets over a trend plus
weekly/daily seasonality plus noise, with a multiplicative effect applied
inside a set of scattered treatment windows. Everything downstream
(contracts, models, estimation) is tested against recovering that effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate(
    n_days: int = 90,
    unit_id: str = "unit_0",
    baseline_level: float = 100.0,
    trend_frac_per_day: float = 0.0,
    weekly_amplitude: float = 0.15,
    daily_amplitude: float = 0.30,
    noise_std: float = 0.03,
    n_windows: int = 4,
    window_duration_buckets: int = 6,
    effect_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate a synthetic hourly series with a planted multiplicative effect.

    Returns:
        df: columns `unit_id`, `timestamp` (UTC), `value`.
        windows: columns `window_id`, `start`, `end` (half-open: buckets with
            `start <= timestamp < end` fall inside the window).
        truth: `{"effect_size": ..., "window_ids": [...]}`.
    """
    rng = np.random.default_rng(seed)

    n_buckets = n_days * 24
    timestamps = pd.date_range("2024-01-01", periods=n_buckets, freq="h", tz="UTC")

    t_days = np.arange(n_buckets) / 24.0
    week_frac = timestamps.dayofweek.to_numpy() / 7.0
    day_frac = timestamps.hour.to_numpy() / 24.0

    trend = 1.0 + trend_frac_per_day * t_days
    weekly = 1.0 + weekly_amplitude * np.sin(2 * np.pi * week_frac)
    daily = 1.0 + daily_amplitude * np.sin(2 * np.pi * day_frac - np.pi / 2)
    noise = 1.0 + rng.normal(0.0, noise_std, size=n_buckets)

    value = baseline_level * trend * weekly * daily * noise

    windows = _place_windows(timestamps, n_windows, window_duration_buckets, rng)

    window_mask = np.zeros(n_buckets, dtype=bool)
    for start, end in zip(windows["start"], windows["end"]):
        window_mask |= (timestamps >= start) & (timestamps < end)
    value = np.where(window_mask, value * (1.0 + effect_size), value)

    df = pd.DataFrame({"unit_id": unit_id, "timestamp": timestamps, "value": value})

    truth = {"effect_size": effect_size, "window_ids": windows["window_id"].tolist()}

    return df, windows, truth


def _place_windows(
    timestamps: pd.DatetimeIndex,
    n_windows: int,
    window_duration_buckets: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Place non-overlapping windows at seed-driven positions, clear of the edges.

    Stratified: the usable span is cut into one slot per window, and each window
    lands at a random offset inside its slot. Where a window falls against the
    daily and weekly cycle is the dominant source of estimation error, so the
    seed has to move it -- fixed positions would let a many-seed test vary only
    the noise. Even spacing would also cancel a linear trend's bias across
    windows by symmetry and hide it.

    The central half of each slot's free space is eligible, so neighbouring
    windows keep a gap of at least half a slot's slack between them.
    """
    n_buckets = len(timestamps)
    edge_buffer = n_buckets // 10
    usable_span = n_buckets - 2 * edge_buffer

    slot = usable_span // n_windows
    slack = slot - window_duration_buckets
    if slack <= 0:
        raise ValueError("n_windows * window_duration_buckets too large for n_days")

    margin = slack // 4
    offsets = rng.integers(margin, slack - margin + 1, size=n_windows)

    rows = []
    for i, offset in enumerate(offsets):
        start_idx = edge_buffer + i * slot + int(offset)
        end_idx = start_idx + window_duration_buckets
        rows.append(
            {
                "window_id": f"w{i}",
                "start": timestamps[start_idx],
                "end": timestamps[end_idx],
            }
        )

    return pd.DataFrame(rows)
