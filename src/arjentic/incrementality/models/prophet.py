# SPDX-License-Identifier: GPL-3.0-or-later
"""The Prophet backend, and the only module permitted to speak Prophet.

Everything library-specific is confined here: its column names, its refusal of
timezone-aware timestamps, and its 0.80 default interval. Nothing outside this
file should need to know Prophet is what produced a forecast.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from prophet import Prophet

from arjentic.incrementality.models.base import (
    INTERVAL_WIDTH,
    forecast_frame,
    require_sorted,
)


class ProphetForecaster:
    """Counterfactual forecasts from Prophet.

    Seasonality is left to Prophet's own detection unless the caller pins it
    through `RunConfig.model_params`, so the harness carries no assumption
    about bucket size.
    """

    def __init__(self, **params: Any) -> None:
        self._params: dict[str, Any] = {"interval_width": INTERVAL_WIDTH, **params}
        self._model: Prophet | None = None

    def fit(self, history: pd.DataFrame) -> None:
        # A fitted Prophet cannot be refit, so each call gets a fresh instance.
        self._model = Prophet(**self._params)
        self._model.fit(
            pd.DataFrame(
                {
                    "ds": _to_naive(history["timestamp"]),
                    "y": history["value"].to_numpy(),
                }
            )
        )

    def predict(self, timestamps: pd.Series) -> pd.DataFrame:
        requested = require_sorted(timestamps)
        predicted = self._model.predict(pd.DataFrame({"ds": _to_naive(requested)}))
        return forecast_frame(
            requested,
            predicted["yhat"],
            predicted["yhat_lower"],
            predicted["yhat_upper"],
        )


def _to_naive(timestamps: pd.Series) -> pd.Series:
    """Prophet rejects timezone-aware input, so hand it UTC wall-clock values.

    Converting to UTC first makes the result independent of whatever offset the
    caller's timestamps carried.
    """
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None).reset_index(drop=True)
