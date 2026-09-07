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

    windows = _place_windows(timestamps, n_windows, window_duration_buckets)

    window_mask = np.zeros(n_buckets, dtype=bool)
    for start, end in zip(windows["start"], windows["end"]):
        window_mask |= (timestamps >= start) & (timestamps < end)
    value = np.where(window_mask, value * (1.0 + effect_size), value)

    df = pd.DataFrame({"unit_id": unit_id, "timestamp": timestamps, "value": value})

    truth = {"effect_size": effect_size, "window_ids": windows["window_id"].tolist()}

    return df, windows, truth


def _place_windows(
    timestamps: pd.DatetimeIndex, n_windows: int, window_duration_buckets: int
) -> pd.DataFrame:
    """Evenly space non-overlapping windows, clear of the series edges."""
    n_buckets = len(timestamps)
    edge_buffer = n_buckets // 10
    usable_span = n_buckets - 2 * edge_buffer

    total_window_buckets = n_windows * window_duration_buckets
    if total_window_buckets >= usable_span:
        raise ValueError("n_windows * window_duration_buckets too large for n_days")

    gap = (usable_span - total_window_buckets) // (n_windows + 1)

    rows = []
    cursor = edge_buffer + gap
    for i in range(n_windows):
        start_idx = cursor
        end_idx = start_idx + window_duration_buckets
        rows.append(
            {
                "window_id": f"w{i}",
                "start": timestamps[start_idx],
                "end": timestamps[end_idx],
            }
        )
        cursor = end_idx + gap

    return pd.DataFrame(rows)
