"""Approximate record lengths and power for one-sided Sharpe inference."""

from __future__ import annotations

from math import ceil

import numpy as np
from scipy.stats import norm

from ._validation import finite, integer, probability
from .dsr import _variance_factor


def minimum_track_record_length(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """Return the minimum integer T whose plug-in PSR reaches confidence.

    Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15. MinTRL = ceil(1 + V(SR) *
    (Phi^-1(confidence) / (SR - benchmark))**2), with at least two values.
    Confidence must exceed 0.5 and Sharpe must exceed the benchmark.
    Sharpe is per observation; kurtosis is Pearson. This deterministic
    confidence calculation differs from planning a test's statistical power.
    """
    level = probability(confidence, "confidence")
    if level <= 0.5:
        raise ValueError("confidence must exceed 0.5")
    gap = finite(sharpe, "sharpe") - finite(benchmark, "benchmark")
    factor = _variance_factor(sharpe, skewness, kurtosis)
    if gap <= 0.0:
        raise ValueError("sharpe must exceed benchmark")
    return max(2, ceil(1.0 + factor * (norm.ppf(level) / gap) ** 2))


def sharpe_power(
    sharpe: float,
    n_observations: int,
    *,
    benchmark: float = 0.0,
    alpha: float = 0.05,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Approximate power of a one-sided normal Sharpe test at level alpha.

    Uses the asymptotic Sharpe variance in Bailey and Lopez de Prado (2012),
    "The Sharpe Ratio Efficient Frontier", Journal of Risk 15.
    Power = Phi((SR - SR0) * sqrt((T - 1) / V(SR)) - Phi^-1(1-alpha)).
    This is a normal planning approximation with fixed alternative moments;
    it assumes independent observations and does not model moment-estimation
    uncertainty. Inputs use per-observation Sharpe and Pearson kurtosis.
    """
    count = integer(n_observations, "n_observations", 2)
    level = probability(alpha, "alpha")
    gap = finite(sharpe, "sharpe") - finite(benchmark, "benchmark")
    factor = _variance_factor(sharpe, skewness, kurtosis)
    return float(norm.cdf(gap * np.sqrt((count - 1) / factor) - norm.isf(level)))


def required_observations(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    power: float = 0.8,
    alpha: float = 0.05,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """Return the minimum integer T reaching the requested approximate power.

    Inverts sharpe_power using Bailey and Lopez de Prado's (2012) asymptotic
    Sharpe variance, "The Sharpe Ratio Efficient Frontier", Journal of Risk.
    T = ceil(1 + V(SR) * ((z_(1-alpha) + z_power)/(SR-SR0))**2).
    Requires SR > SR0 and alpha < power < 1. All Sharpe inputs are per
    observation and kurtosis is Pearson. No random generator is needed.
    """
    level = probability(alpha, "alpha")
    target = probability(power, "power")
    gap = finite(sharpe, "sharpe") - finite(benchmark, "benchmark")
    factor = _variance_factor(sharpe, skewness, kurtosis)
    if gap <= 0.0 or target <= level:
        raise ValueError("sharpe must exceed benchmark and power must exceed alpha")
    quantile = norm.isf(level) + norm.ppf(target)
    return max(2, ceil(1.0 + factor * (quantile / gap) ** 2))
