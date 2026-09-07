# SPDX-License-Identifier: GPL-3.0-or-later
"""Model diagnostics, kept separate from the estimate.

Answers one question: can the counterfactual model predict untreated buckets it
has not seen? That is the assumption the whole estimate rests on, and on client
data it is the only check available, since there is no planted truth to compare
against.

It cannot validate the counterfactual *under* treatment, which is unobservable.
A promotion that coincided with a treatment window passes this cleanly. Passing
validates the forecasting model, not the causal claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arjentic.incrementality.contracts import Role, RunConfig, validate
from arjentic.incrementality.estimate import label_roles
from arjentic.incrementality.models import get_model
from arjentic.incrementality.models.base import INTERVAL_WIDTH


@dataclass(frozen=True)
class BacktestResult:
    """Forecast quality on one held-out slice of baseline buckets."""

    coverage: float
    """Fraction of held-out actuals falling inside the prediction interval.

    Directly load-bearing for single-window designs, whose reported interval is
    derived from the model's own prediction bounds. Multi-window intervals come
    from resampling windows instead and do not depend on this.
    """

    nominal_coverage: float
    mae: float

    bias: float
    """Mean signed error, `expected - actual`.

    Reported alongside MAE because the two have different consequences and MAE
    cannot separate them. Zero-mean noise on a window total shrinks with the
    square root of the bucket count and largely washes out; bias does not.
    """

    relative_bias: float
    """`bias` as a fraction of the mean actual, which passes into lift roughly
    one-for-one. A value of -0.02 means the counterfactual runs 2% low, which
    inflates the lift estimate by about 2 percentage points.
    """

    holdout_buckets: int
    model_name: str


def backtest(
    df: pd.DataFrame, windows: pd.DataFrame, config: RunConfig
) -> BacktestResult:
    """Score the primary model on one interior slice of baseline buckets.

    The slice is taken from the middle rather than the tail: the estimator fits
    on every baseline bucket and predicts windows sitting between them, so it
    interpolates. A tail fold would measure extrapolation, a harder task than
    the one the estimate performs, and would understate the model.

    One fold, deliberately. Rolling-origin is a later upgrade behind this same
    signature.
    """
    observations, treatment_windows = validate(df, windows)

    available = observations["unit_id"].unique()
    if config.unit_id not in set(available):
        raise ValueError(
            f"unit_id {config.unit_id!r} is not present in the observations; "
            f"available units: {sorted(available)}"
        )
    observations = observations[observations["unit_id"] == config.unit_id]

    labeled = label_roles(
        observations,
        treatment_windows,
        washout_before=config.washout_before,
        washout_after=config.washout_after,
    )
    baseline = labeled[labeled["role"] == Role.BASELINE].reset_index(drop=True)

    n_holdout = config.backtest_holdout_buckets
    if n_holdout >= len(baseline):
        raise ValueError(
            f"backtest_holdout_buckets ({n_holdout}) must be smaller than the "
            f"{len(baseline)} baseline buckets available to hold out from"
        )

    start = (len(baseline) - n_holdout) // 2
    held_out = baseline.index[start : start + n_holdout]
    holdout = baseline.loc[held_out]
    train = baseline.drop(held_out)

    model = get_model(config.primary_model, **config.model_params)
    model.fit(train)
    forecast = model.predict(holdout["timestamp"])

    actual = holdout["value"].to_numpy()
    error = forecast["expected"].to_numpy() - actual
    covered = (actual >= forecast["lower"].to_numpy()) & (
        actual <= forecast["upper"].to_numpy()
    )

    return BacktestResult(
        coverage=float(covered.mean()),
        nominal_coverage=INTERVAL_WIDTH,
        mae=float(np.abs(error).mean()),
        bias=float(error.mean()),
        relative_bias=float(error.mean() / actual.mean()),
        holdout_buckets=n_holdout,
        model_name=config.primary_model,
    )
