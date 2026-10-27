"""Shared input checks; statistical conventions live in the public modules."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
RngLike: TypeAlias = np.random.Generator | int | None


def integer(value: int, name: str, minimum: int = 1) -> int:
    """Return an integer at least minimum, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def finite(value: float, name: str) -> float:
    """Return a finite real scalar."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def probability(value: float, name: str) -> float:
    """Return a probability strictly between zero and one."""
    result = finite(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def vector(values: ArrayLike, name: str, minimum: int = 1) -> FloatArray:
    """Return a finite, nonempty one-dimensional float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length at least {minimum}")
    return array


def matrix(values: ArrayLike, minimum: int = 3) -> FloatArray:
    """Return finite observations by strategies, accepting a single vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if (array.ndim != 2 or array.shape[0] < minimum or array.shape[1] < 1
            or not np.isfinite(array).all()):
        raise ValueError("returns must be finite observations by strategies")
    return array
