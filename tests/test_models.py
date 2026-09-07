# SPDX-License-Identifier: GPL-3.0-or-later
"""The model seam: protocol conformance, the forecast frame contract, and a
standing guard that Prophet's vocabulary stays inside its adapter.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from arjentic.incrementality.contracts import validate
from arjentic.incrementality.data.synthetic import generate
from arjentic.incrementality.models import REGISTRY, get_model
from arjentic.incrementality.models.base import (
    INTERVAL_WIDTH,
    Z,
    Forecast,
    Forecaster,
    forecast_frame,
)
from arjentic.incrementality.models.naive import SeasonalMean

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "arjentic" / "incrementality"

# Prophet's API surface, which must not appear outside its own adapter. Bare
# "prophet" is allowed: prose naming the boundary is how the boundary is
# documented.
PROPHET_VOCABULARY = (
    "yhat",
    "make_future_dataframe",
    "add_seasonality",
    "changepoint_prior_scale",
    "seasonality_mode",
)


@pytest.fixture
def baseline() -> pd.DataFrame:
    """Observations outside any treatment window."""
    df, windows, _ = generate(n_days=60)
    observations, treatment_windows = validate(df, windows)
    in_window = np.zeros(len(observations), dtype=bool)
    for _, window in treatment_windows.iterrows():
        in_window |= (observations["timestamp"] >= window["start"]) & (
            observations["timestamp"] < window["end"]
        )
    return observations[~in_window].reset_index(drop=True)


def test_seasonal_mean_satisfies_the_protocol() -> None:
    assert isinstance(SeasonalMean(), Forecaster)


def test_registry_exposes_the_naive_backend() -> None:
    assert isinstance(get_model("naive"), SeasonalMean)
    assert "naive" in REGISTRY


def test_unknown_model_name_lists_what_is_registered() -> None:
    with pytest.raises(ValueError, match="registered backends"):
        get_model("does-not-exist")


def test_predict_returns_a_contract_shaped_forecast(baseline) -> None:
    model = SeasonalMean()
    model.fit(baseline)
    requested = baseline["timestamp"].iloc[:48]

    forecast = model.predict(requested)

    assert list(forecast.columns) == ["timestamp", "expected", "lower", "upper"]
    assert len(forecast) == len(requested)
    assert forecast["timestamp"].tolist() == requested.tolist()
    assert Forecast.validate(forecast) is not None


def test_predict_tracks_the_seasonal_level(baseline) -> None:
    """Fitted on its own history, the seasonal mean should sit close to actuals."""
    model = SeasonalMean()
    model.fit(baseline)

    forecast = model.predict(baseline["timestamp"])
    relative_error = (
        (forecast["expected"].to_numpy() - baseline["value"].to_numpy())
        / baseline["value"].to_numpy()
    )

    assert np.abs(relative_error).mean() < 0.10


def test_predicted_interval_brackets_the_expectation(baseline) -> None:
    model = SeasonalMean()
    model.fit(baseline)

    forecast = model.predict(baseline["timestamp"].iloc[:48])

    assert (forecast["lower"] <= forecast["expected"]).all()
    assert (forecast["expected"] <= forecast["upper"]).all()


def test_predict_refuses_slots_the_baseline_never_covered(baseline) -> None:
    """The guard: a global-mean substitute would look plausible and mean nothing."""
    monday_mornings = _seasonal_slot(baseline["timestamp"]) == 9
    model = SeasonalMean()
    model.fit(baseline[~monday_mornings])

    with pytest.raises(ValueError, match="baseline is too short"):
        model.predict(baseline["timestamp"][monday_mornings])


def test_interval_width_and_quantile_agree() -> None:
    assert INTERVAL_WIDTH == 0.95
    assert Z == pytest.approx(1.959963, abs=1e-5)


def test_forecast_frame_ignores_the_caller_index(baseline) -> None:
    """Aligning a backend's output on index would silently produce nulls."""
    timestamps = baseline["timestamp"].iloc[10:14]
    values = np.array([1.0, 2.0, 3.0, 4.0])

    forecast = forecast_frame(timestamps, values, values - 1, values + 1)

    assert forecast["expected"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert not forecast.isnull().any().any()


def test_forecast_frame_rejects_swapped_bounds(baseline) -> None:
    timestamps = baseline["timestamp"].iloc[:3]
    expected = np.array([10.0, 10.0, 10.0])

    with pytest.raises(pandera.errors.SchemaError):
        forecast_frame(timestamps, expected, expected + 1, expected - 1)


def test_prophet_vocabulary_stays_inside_its_adapter() -> None:
    """CLAUDE.md names this the biggest extensibility risk in v0.1, so it is a
    standing check rather than a code-review promise."""
    adapter = PACKAGE_ROOT / "models" / "prophet.py"
    offenders = []

    for path in PACKAGE_ROOT.rglob("*.py"):
        if path == adapter:
            continue
        source = path.read_text()
        for token in PROPHET_VOCABULARY:
            if token in source:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {token}")
        if re.search(r"^\s*(from|import)\s+prophet", source, re.MULTILINE):
            offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: imports prophet")

    assert not offenders, f"Prophet vocabulary leaked outside its adapter: {offenders}"


def _seasonal_slot(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.dayofweek * 24 + timestamps.dt.hour
