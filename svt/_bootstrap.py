"""Circular stationary bootstrap shared by the multiple-testing procedures."""

from __future__ import annotations

import numpy as np

from ._validation import FloatArray, IntArray, RngLike, finite, integer


def stationary_bootstrap_indices(
    n_observations: int,
    *,
    n_resamples: int = 1999,
    block_length: float = 10.0,
    rng: RngLike = None,
) -> IntArray:
    """Draw circular stationary-bootstrap indices of shape (resamples, time).

    Politis and Romano (1994), "The Stationary Bootstrap", JASA 89.
    Each path starts uniformly; subsequent indices restart uniformly with
    probability 1 / block_length, otherwise advancing circularly. The expected
    block length must be at least one; one gives ordinary IID resampling.
    Pass a Generator or integer seed through rng for reproducibility.
    """
    n = integer(n_observations, "n_observations")
    count = integer(n_resamples, "n_resamples")
    length = finite(block_length, "block_length")
    if length < 1.0:
        raise ValueError("block_length must be at least one")
    generator = np.random.default_rng(rng)
    indices = generator.integers(n, size=(count, n), dtype=np.int64)
    restart = generator.random((count, n)) < 1.0 / length
    for time in range(1, n):
        indices[:, time] = np.where(
            restart[:, time], indices[:, time], (indices[:, time - 1] + 1) % n
        )
    return indices


def centered_bootstrap_means(
    values: FloatArray,
    n_resamples: int,
    block_length: float,
    rng: RngLike,
) -> FloatArray:
    """Resample joint centered means (Politis and Romano, 1994)."""
    integer(n_resamples, "n_resamples", 2)
    indices = stationary_bootstrap_indices(
        len(values), n_resamples=n_resamples, block_length=block_length, rng=rng
    )
    centered = values - values.mean(axis=0)
    means = np.empty((n_resamples, values.shape[1]), dtype=np.float64)
    batch = max(1, min(64, 1_000_000 // values.size))
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        means[start:stop] = centered[indices[start:stop]].mean(axis=1)
    return means
