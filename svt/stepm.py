"""Romano-Wolf resampling stepdown inference for strategy families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._bootstrap import centered_bootstrap_means
from ._validation import FloatArray, IntArray, RngLike, probability
from .reality_check import performance_differentials


@dataclass(frozen=True)
class StepMResult:
    """Strategy-wise results in original column order; order sorts statistics."""

    statistics: FloatArray
    standard_errors: FloatArray
    adjusted_pvalues: FloatArray
    single_step_pvalues: FloatArray
    rejected: NDArray[np.bool_]
    order: IntArray
    bootstrap_statistics: FloatArray


def romano_wolf_stepm(
    returns: ArrayLike,
    *,
    benchmark: ArrayLike = 0.0,
    alpha: float = 0.05,
    n_resamples: int = 1999,
    block_length: float = 10.0,
    rng: RngLike = None,
) -> StepMResult:
    """Test each H0: expected strategy differential <= 0, controlling FWER.

    Romano and Wolf (2005), "Stepwise Multiple Testing as Formalized Data
    Snooping", Econometrica 73; adjusted p-values follow Romano and Wolf
    (2016), "Efficient Computation of Adjusted p-Values for Resampling-Based
    Stepdown Multiple Testing", Statistics & Probability Letters 113.
    The stationary bootstrap is Politis and Romano (1994).

    Input is (observations, strategies), with a scalar or aligned benchmark.
    Joint centered bootstrap means estimate each mean's standard error. Both
    observed and bootstrap means use these fixed standard errors. Descending
    observed statistics determine the stepdown order; bootstrap maxima cover
    the remaining hypotheses, and cumulative maxima enforce p-value
    monotonicity. P-values include ties and the (1 + count) / (B + 1)
    correction. Rejections use adjusted_pvalues <= alpha.

    FWER control is asymptotic under stationary-bootstrap regularity
    conditions, not a finite-sample guarantee. Constant candidates or a
    degenerate bootstrap raise ValueError because studentization is undefined.
    rng accepts a Generator or integer seed.
    """
    level = probability(alpha, "alpha")
    differences = performance_differentials(returns, benchmark)
    if (np.ptp(differences, axis=0) == 0.0).any():
        raise ValueError("studentization requires nonconstant candidates")
    means = centered_bootstrap_means(differences, n_resamples, block_length, rng)
    errors = means.std(axis=0, ddof=1)
    rounding_floor = 32 * np.finfo(np.float64).eps * differences.std(axis=0, ddof=1)
    if (errors <= rounding_floor).any() or not np.isfinite(errors).all():
        raise ValueError("bootstrap standard errors must be finite and positive")
    observed = differences.mean(axis=0) / errors
    boot = means / errors
    order = np.argsort(-observed, kind="stable").astype(np.int64)
    remaining = np.maximum.accumulate(boot[:, order][:, ::-1], axis=1)[:, ::-1]
    raw = (1 + np.sum(remaining >= observed[order], axis=0)) / (len(boot) + 1)
    adjusted = np.empty_like(observed)
    adjusted[order] = np.maximum.accumulate(raw)
    single_step = (1 + np.sum(boot.max(axis=1)[:, None] >= observed, axis=0)) / (len(boot) + 1)
    return StepMResult(
        observed, errors, adjusted, single_step, adjusted <= level, order, boot
    )
