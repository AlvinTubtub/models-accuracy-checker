"""Naive lag-1 persistence benchmark."""

from __future__ import annotations

from typing import Sequence
import numpy as np


def generate_naive_predictions(
    origin_closes: Sequence[float],
) -> list[float]:
    """Naive model predicts Close_{t+1} = Close_t (origin_close)."""
    return [round(float(c), 4) for c in origin_closes]
