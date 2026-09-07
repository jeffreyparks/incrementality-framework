# SPDX-License-Identifier: GPL-3.0-or-later
"""A deliberately simple backend: the seasonal mean of untreated buckets.

Two jobs. It keeps `Forecaster` honest -- a protocol shaped around one library
would not accommodate something this plain -- and it gives the notebook a
sanity comparator for the default backend.
"""

from __future__ import annotations

import pandas as pd

from arjentic.incrementality.models.base import Z, forecast_frame

_MIN_OBSERVATIONS_PER_KEY = 2


class SeasonalMean:
    """Day-of-week x hour-of-day mean over the fitted history."""

    def fit(self, history: pd.DataFrame) -> None:
        grouped = history.groupby(_seasonal_key(history["timestamp"]))["value"]
        self._mean = grouped.mean()
        self._std = grouped.std()
        self._count = grouped.count()

    def predict(self, timestamps: pd.Series) -> pd.DataFrame:
        keys = _seasonal_key(timestamps)
        counts = keys.map(self._count).fillna(0)

        # Substituting a global mean for an unobserved slot would return a
        # plausible number with no basis, so refuse instead.
        thin = counts < _MIN_OBSERVATIONS_PER_KEY
        if thin.any():
            slots = sorted({(key // 24, key % 24) for key in keys[thin]})
            raise ValueError(
                f"baseline has fewer than {_MIN_OBSERVATIONS_PER_KEY} observations "
                f"for (day_of_week, hour) slots {slots}; the baseline is too short "
                "to forecast these windows with the seasonal mean"
            )

        expected = keys.map(self._mean).to_numpy()
        spread = Z * keys.map(self._std).to_numpy()
        return forecast_frame(timestamps, expected, expected - spread, expected + spread)


def _seasonal_key(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.dayofweek * 24 + timestamps.dt.hour
