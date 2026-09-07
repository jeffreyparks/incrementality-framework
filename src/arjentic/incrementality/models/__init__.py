# SPDX-License-Identifier: GPL-3.0-or-later
"""Backend registry.

Lives here rather than in `base.py` because the adapters import `base` for the
forecast-frame helper, so naming them there would be a circular import. A plain
dict, deliberately: registration is a line of code, not a plugin system.
"""

from __future__ import annotations

from typing import Any

from arjentic.incrementality.models.base import Forecaster
from arjentic.incrementality.models.naive import SeasonalMean

REGISTRY: dict[str, type[Forecaster]] = {
    "naive": SeasonalMean,
}


def get_model(name: str, **params: Any) -> Forecaster:
    """Instantiate a registered backend by name."""
    if name not in REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; registered backends: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**params)
