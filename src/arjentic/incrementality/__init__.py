# SPDX-License-Identifier: GPL-3.0-or-later
"""Incrementality testing harness for pulse and holdout experiments."""

from arjentic.incrementality.contracts import BaselineState, RunConfig
from arjentic.incrementality.data.synthetic import generate
from arjentic.incrementality.estimate import run

__all__ = ["BaselineState", "RunConfig", "generate", "run"]
