# SPDX-License-Identifier: GPL-3.0-or-later
"""The model extension point, expressed in vocabulary no backend owns.

A backend satisfies `Forecaster` and returns frames matching `Forecast`. The
boundary only holds while no module outside the Prophet adapter names Prophet's
columns or calls its API, so a different backend can drop in without touching
anything downstream. `tests/test_models.py` enforces that as a standing check.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat

INTERVAL_WIDTH = 0.95
"""Nominal coverage that every backend's `lower`/`upper` must target.

Shared so that backtest coverage means the same thing across backends. Prophet
defaults to 0.80, so its adapter has to pass this value explicitly rather than
inherit its own default.
"""

Z = NormalDist().inv_cdf(0.5 + INTERVAL_WIDTH / 2)
"""Two-sided normal quantile for `INTERVAL_WIDTH`, derived rather than pasted."""


class Forecast(pa.DataFrameModel):
    """One counterfactual prediction per requested timestamp."""

    timestamp: pat.Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC"})
    expected: pat.Series[float]
    lower: pat.Series[float]
    upper: pat.Series[float]

    class Config:
        coerce = True

    @pa.dataframe_check
    def bounds_bracket_expected(cls, df: pd.DataFrame) -> pd.Series:
        return (df["lower"] <= df["expected"]) & (df["expected"] <= df["upper"])


@runtime_checkable
class Forecaster(Protocol):
    """A counterfactual backend: learn from untreated history, score timestamps."""

    def fit(self, history: pd.DataFrame) -> None:
        """Learn from observations whose role is baseline."""
        ...

    def predict(self, timestamps: pd.Series) -> pd.DataFrame:
        """Return a `Forecast`-shaped frame, one row per requested timestamp.

        `timestamps` must be sorted ascending. Backends may rely on that, so
        call `require_sorted()` rather than assuming it silently.
        """
        ...


def require_sorted(timestamps: pd.Series) -> pd.Series:
    """Enforce the ascending-order precondition and normalize the index.

    Sorted order is guaranteed upstream by `contracts.validate()`, and some
    backends align their output positionally because of it. Enforcing it here
    keeps that a stated invariant rather than an accident, and makes every
    backend fail the same way when it is violated. Duplicates are allowed.
    """
    requested = pd.Series(timestamps).reset_index(drop=True)
    if not requested.is_monotonic_increasing:
        raise ValueError(
            "timestamps must be sorted ascending; contracts.validate() returns "
            "observations in that order, so slice the request from a validated frame"
        )
    return requested


def forecast_frame(
    timestamps: pd.Series,
    expected: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
) -> pd.DataFrame:
    """Assemble and validate a forecast frame.

    The only place the forecast column names are written, so a backend never
    needs to know them. Values are taken positionally, since aligning a
    backend's output index against the requested timestamps would silently
    introduce nulls.
    """
    return Forecast.validate(
        pd.DataFrame(
            {
                "timestamp": pd.Series(timestamps).reset_index(drop=True),
                "expected": np.asarray(expected, dtype=float),
                "lower": np.asarray(lower, dtype=float),
                "upper": np.asarray(upper, dtype=float),
            }
        )
    )
