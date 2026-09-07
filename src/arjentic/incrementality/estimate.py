# SPDX-License-Identifier: GPL-3.0-or-later
"""The estimation pipeline, and the package's only public entry point.

`label_roles()` resolves treatment windows and washout into per-bucket roles in
a single upstream pass, so nothing after it needs to know washout exists -- it
filters on `role` and gets the right buckets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arjentic.incrementality.contracts import (
    BaselineState,
    EstimateResult,
    Role,
    RunConfig,
    validate,
)
from arjentic.incrementality.models import get_model
from arjentic.incrementality.models.base import INTERVAL_WIDTH, Z


def run(df: pd.DataFrame, windows: pd.DataFrame, config: RunConfig) -> EstimateResult:
    """Estimate incremental lift for one treated unit over its windows.

    Fits the counterfactual model on every baseline bucket -- not a pre-period
    only -- and scores the buckets inside each declared window.
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

    model = get_model(config.primary_model, **config.model_params)
    model.fit(labeled[labeled["role"] == Role.BASELINE])

    scored = labeled[labeled["role"] == Role.WINDOW].reset_index(drop=True)
    forecast = model.predict(scored["timestamp"])
    scored = scored.assign(
        counterfactual=forecast["expected"].to_numpy(),
        lower=forecast["lower"].to_numpy(),
        upper=forecast["upper"].to_numpy(),
    )

    per_window = _per_window_lift(scored, config.baseline_state)

    if len(per_window) > 1:
        ci_lower, ci_upper = _bootstrap_pooled_interval(per_window, config)
        ci_method = "bootstrap"
    else:
        ci_lower, ci_upper = _forecast_interval(scored, config.baseline_state)
        ci_method = "forecast"

    return EstimateResult(
        per_window=per_window,
        pooled_lift=_signed_lift(
            per_window["actual"].sum(),
            per_window["counterfactual"].sum(),
            config.baseline_state,
        ),
        mean_of_ratios_lift=float(per_window["lift"].mean()),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_method=ci_method,
        model_name=config.primary_model,
        config=config,
    )


def label_roles(
    observations: pd.DataFrame,
    windows: pd.DataFrame,
    washout_before: int,
    washout_after: int,
) -> pd.DataFrame:
    """Add `role` and `window_id` columns resolving windows plus washout.

    Washout is declared in buckets, so it is applied positionally. That is only
    sound because `contracts.validate()` guarantees ascending order and regular
    spacing. Window role takes precedence where a window and another window's
    washout overlap.
    """
    units = observations["unit_id"].unique()
    if len(units) > 1:
        raise ValueError(
            f"label_roles handles a single unit, got {sorted(units)}; filter to "
            "the treated unit first. Positional washout would otherwise run "
            "across the boundary between units."
        )

    labeled = observations.reset_index(drop=True).copy()
    labeled["role"] = Role.BASELINE
    labeled["window_id"] = pd.NA

    timestamps = labeled["timestamp"]
    in_any_window = np.zeros(len(labeled), dtype=bool)
    for _, window in windows.iterrows():
        in_window = (
            (timestamps >= window["start"]) & (timestamps < window["end"])
        ).to_numpy()
        labeled.loc[in_window, "window_id"] = window["window_id"]
        in_any_window |= in_window

    dilated = in_any_window.copy()
    for offset in range(1, washout_before + 1):
        dilated |= _shift(in_any_window, -offset)
    for offset in range(1, washout_after + 1):
        dilated |= _shift(in_any_window, offset)

    labeled.loc[dilated & ~in_any_window, "role"] = Role.WASHOUT
    labeled.loc[in_any_window, "role"] = Role.WINDOW

    return labeled


def _shift(mask: np.ndarray, offset: int) -> np.ndarray:
    """Move a boolean mask `offset` positions, padding with False.

    A positive offset marks buckets after the mask, a negative one buckets
    before it, so washout past either end of the series clips away.
    """
    shifted = np.zeros_like(mask)
    if offset > 0:
        shifted[offset:] = mask[:-offset]
    elif offset < 0:
        shifted[:offset] = mask[-offset:]
    return shifted


def _signed_lift(
    actual: float | pd.Series,
    counterfactual: float | pd.Series,
    baseline_state: BaselineState,
) -> float | pd.Series:
    """Resolve lift so that positive always means treatment raised the metric.

    Getting this backwards inverts every answer while still looking plausible,
    which is why it is one explicit branch and nothing else.
    """
    if baseline_state == BaselineState.TREATED:
        # Baseline had treatment on, so the window is treatment removed.
        return (counterfactual - actual) / counterfactual
    # Baseline had treatment off, so the window is treatment applied.
    return (actual - counterfactual) / counterfactual


def _per_window_lift(scored: pd.DataFrame, baseline_state: BaselineState) -> pd.DataFrame:
    """Total actual and counterfactual per window, then one ratio each.

    Summing before dividing rather than averaging per-bucket ratios, which are
    unstable wherever the counterfactual is small.
    """
    per_window = (
        scored.groupby("window_id", sort=True)
        .agg(
            actual=("value", "sum"),
            counterfactual=("counterfactual", "sum"),
            buckets=("value", "size"),
        )
        .reset_index()
    )
    per_window["lift"] = _signed_lift(
        per_window["actual"], per_window["counterfactual"], baseline_state
    )
    return per_window


def _bootstrap_pooled_interval(
    per_window: pd.DataFrame, config: RunConfig
) -> tuple[float, float]:
    """Percentile bootstrap of the pooled ratio, resampling whole windows.

    Resampling windows rather than buckets is what makes this a statement about
    between-window variation. A group key here is what turns it into a cluster
    bootstrap once there are multiple units.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    actual = per_window["actual"].to_numpy()
    counterfactual = per_window["counterfactual"].to_numpy()

    draws = rng.integers(
        0, len(per_window), size=(config.bootstrap_resamples, len(per_window))
    )
    pooled = _signed_lift(
        actual[draws].sum(axis=1),
        counterfactual[draws].sum(axis=1),
        config.baseline_state,
    )

    tail = (1.0 - INTERVAL_WIDTH) / 2.0 * 100.0
    lower, upper = np.percentile(pooled, [tail, 100.0 - tail])
    return float(lower), float(upper)


def _forecast_interval(
    scored: pd.DataFrame, baseline_state: BaselineState
) -> tuple[float, float]:
    """Single-window fallback: propagate the forecast's own uncertainty.

    Resampling one window is degenerate, so uncertainty comes from the model's
    prediction interval instead. Per-bucket standard errors combine in
    quadrature rather than by summing the bounds, since summing bounds assumes
    the forecast errors are perfectly correlated and inflates the interval by
    roughly the square root of the bucket count. Treating them as independent
    is the opposite approximation and somewhat optimistic, as forecast errors
    are serially correlated in practice.
    """
    sigma = ((scored["upper"] - scored["lower"]) / 2.0 / Z).to_numpy()
    sigma_total = float(np.sqrt(np.square(sigma).sum()))

    actual = float(scored["value"].sum())
    counterfactual = float(scored["counterfactual"].sum())
    lift = _signed_lift(actual, counterfactual, baseline_state)

    # Delta method: d/dC of (A/C - 1) has magnitude A/C^2 under either sign.
    standard_error = actual / counterfactual**2 * sigma_total
    return float(lift - Z * standard_error), float(lift + Z * standard_error)
