"""White's one-sided Reality Check against an explicit benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._bootstrap import centered_bootstrap_means, stationary_bootstrap_indices
from ._validation import FloatArray, RngLike, finite, matrix, vector

__all__ = ["RealityCheckResult", "stationary_bootstrap_indices", "white_reality_check"]


@dataclass(frozen=True)
class RealityCheckResult:
    """Global test statistic, Monte Carlo p-value and bootstrap maxima."""

    statistic: float
    pvalue: float
    strategy_statistics: FloatArray
    bootstrap_maxima: FloatArray


def performance_differentials(returns: ArrayLike, benchmark: ArrayLike) -> FloatArray:
    """Compute joint return differentials for White (2000) and StepM (2005)."""
    values = matrix(returns)
    reference = np.asarray(benchmark, dtype=np.float64)
    if reference.ndim == 0:
        differences = values - finite(float(reference), "benchmark")
    else:
        reference = vector(reference, "benchmark", 3)
        if reference.size != len(values):
            raise ValueError("benchmark must have one value per observation")
        differences = values - reference[:, None]
    if not np.isfinite(differences).all():
        raise ValueError("performance differentials must be finite")
    return differences


def white_reality_check(
    returns: ArrayLike,
    *,
    benchmark: ArrayLike = 0.0,
    n_resamples: int = 1999,
    block_length: float = 10.0,
    rng: RngLike = None,
) -> RealityCheckResult:
    """Test H0: no candidate has positive expected benchmark-relative returns.

    White (2000), "A Reality Check for Data Snooping", Econometrica 68;
    Politis and Romano (1994), "The Stationary Bootstrap", JASA 89.
    Input is (observations, strategies), or a vector for one candidate.
    Benchmark is a finite scalar or aligned vector. All candidates use the
    same circular bootstrap indices, preserving cross-strategy dependence.
    Weak stationarity and suitable dependence/moment conditions are assumed.

    This is the unstudentized White statistic sqrt(T) * max(mean differential).
    Including the zero benchmark floors both observed and bootstrap maxima at
    zero. Bootstrap differentials are centered at their sample means under
    the least-favorable null. The p-value uses (exceedances + 1) / (B + 1),
    including ties. rng accepts a Generator or integer seed. A small p-value
    rejects the global null; it does not identify individual discoveries.
    """
    differences = performance_differentials(returns, benchmark)
    statistics = np.sqrt(len(differences)) * differences.mean(axis=0)
    observed = max(0.0, float(statistics.max()))
    means = centered_bootstrap_means(differences, n_resamples, block_length, rng)
    maxima = np.maximum(0.0, np.sqrt(len(differences)) * means.max(axis=1))
    pvalue = (1 + np.count_nonzero(maxima >= observed)) / (len(maxima) + 1)
    return RealityCheckResult(observed, float(pvalue), statistics, maxima)
