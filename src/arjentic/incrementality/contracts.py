# SPDX-License-Identifier: GPL-3.0-or-later
"""The data contract, plus the run's configuration and result types.

Enforces structure only -- columns, dtypes, nulls, keys, regular spacing, and
that every treatment window actually covers observations. No statistical
checks, no anomaly detection, no imputation: bad values are the caller's
problem, malformed shapes are this module's.

`validate()` also *normalizes*: it returns frames sorted into canonical order,
which is what lets everything downstream do positional arithmetic without
re-sorting defensively.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pat


class Observations(pa.DataFrameModel):
    """One metric value per unit per time bucket."""

    unit_id: pat.Series[str]
    timestamp: pat.Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC"})
    value: pat.Series[float]

    class Config:
        coerce = True
        # Without this, a frame whose timestamps are *all* identical satisfies
        # regular_interval below -- every diff is zero, so there is exactly one
        # unique diff. Duplicate buckets are the usual symptom of an upstream
        # join or groupby gone wrong.
        unique = ["unit_id", "timestamp"]

    @pa.dataframe_check
    def regular_interval(cls, df: pd.DataFrame) -> bool:
        for _, group in df.groupby("unit_id"):
            diffs = group.sort_values("timestamp")["timestamp"].diff().dropna().unique()
            if len(diffs) > 1:
                return False
        return True


class TreatmentWindows(pa.DataFrameModel):
    """Half-open treatment intervals: a bucket is inside when start <= ts < end."""

    window_id: pat.Series[str] = pa.Field(unique=True)
    start: pat.Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC"})
    end: pat.Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": "UTC"})

    class Config:
        coerce = True

    @pa.dataframe_check
    def end_after_start(cls, df: pd.DataFrame) -> pd.Series:
        return df["end"] > df["start"]


class Role:
    """What a bucket is used for, assigned once by `estimate.label_roles`.

    Lives here rather than with the estimator because diagnostics filters on
    roles too. Later roles -- donors, backtest folds -- are new members, not
    new plumbing.

    Plain string constants rather than an Enum on purpose. Under pandas 3.0 a
    `str`-subclass Enum member does not compare equal to values in a `str`
    dtype column, so `frame["role"] == Role.WINDOW` would match nothing and
    silently yield an empty selection. These are the strings themselves, and a
    mistyped attribute raises immediately instead.
    """

    BASELINE = "baseline"
    WINDOW = "window"
    WASHOUT = "washout"


class BaselineState(str, Enum):
    """Whether the baseline (untreated-by-construction) period had treatment on.

    Determines the sign of every estimate, so it is declared rather than
    inferred.
    """

    TREATED = "treated"
    UNTREATED = "untreated"


@dataclass(frozen=True)
class RunConfig:
    """A pre-registered description of one estimation run."""

    unit_id: str
    baseline_state: BaselineState
    washout_before: int
    washout_after: int
    primary_model: str
    backtest_holdout_buckets: int
    model_params: dict[str, Any] = field(default_factory=dict)
    bootstrap_seed: int = 0
    bootstrap_resamples: int = 1000

    def __post_init__(self) -> None:
        if self.washout_before < 0 or self.washout_after < 0:
            raise ValueError("washout_before and washout_after must be non-negative")
        if self.backtest_holdout_buckets <= 0:
            raise ValueError("backtest_holdout_buckets must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")


@dataclass(frozen=True, eq=False)
class EstimateResult:
    """The output of one run, carrying the config that produced it.

    `eq=False` because the default dataclass `__eq__` would compare the
    embedded frame with `==` and raise on the ambiguous truth value.
    """

    per_window: pd.DataFrame
    pooled_lift: float
    mean_of_ratios_lift: float
    ci_lower: float
    ci_upper: float
    ci_method: str
    """Which procedure produced the interval, and so what it measures.

    `"bootstrap"` resamples whole windows and brackets `pooled_lift`, capturing
    between-window variation. `"forecast"` is the single-window fallback and
    captures forecast uncertainty instead. The two are not comparable.
    """

    model_name: str
    config: RunConfig


def validate(
    observations: pd.DataFrame, windows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize the observation and treatment-window frames.

    Returns both frames coerced to contract dtypes and sorted into canonical
    order: observations by `(unit_id, timestamp)`, windows by `start`.

    Raises:
        ValueError: a timestamp column is not timezone-aware, or a window
            covers no observations.
        pandera.errors.SchemaError: a frame violates the schema.
    """
    _require_tz_aware(observations, ["timestamp"])
    _require_tz_aware(windows, ["start", "end"])

    validated_observations = (
        Observations.validate(observations)
        .sort_values(["unit_id", "timestamp"])
        .reset_index(drop=True)
    )
    validated_windows = (
        TreatmentWindows.validate(windows).sort_values("start").reset_index(drop=True)
    )

    _require_windows_cover_observations(validated_observations, validated_windows)

    return validated_observations, validated_windows


def _require_tz_aware(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Reject naive timestamps before pandera can silently relabel them.

    Coercion to a tz-aware dtype localizes naive values to UTC without shifting
    them, so local-time data saved without an offset would validate cleanly and
    be wrong by that offset. Parsing is the caller's job precisely because it is
    where the mistake happens.
    """
    for column in columns:
        if column not in df.columns:
            continue
        if not isinstance(df[column].dtype, pd.DatetimeTZDtype):
            raise ValueError(
                f"column {column!r} must be a timezone-aware datetime, got "
                f"{df[column].dtype}; parse it with "
                "pd.to_datetime(..., utc=True) before validating"
            )


def _require_windows_cover_observations(
    observations: pd.DataFrame, windows: pd.DataFrame
) -> None:
    """Every window must contain at least one observation bucket.

    Checked by overlap rather than by duration, so a window that is long enough
    but falls outside the observed range is caught here instead of producing an
    empty estimate deep in the pipeline.
    """
    timestamps = observations["timestamp"]
    uncovered = [
        row["window_id"]
        for _, row in windows.iterrows()
        if not ((timestamps >= row["start"]) & (timestamps < row["end"])).any()
    ]
    if uncovered:
        raise ValueError(
            f"treatment windows cover no observation buckets: {uncovered}; "
            "check the windows fall inside the observed time range"
        )
