"""Probabilistic and Deflated Sharpe Ratios, using per-observation units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from ._validation import finite, integer, vector


@dataclass(frozen=True)
class SharpeStatistics:
    """Sample Sharpe and empirical central moments (Pearson kurtosis)."""

    sharpe: float
    n_observations: int
    skewness: float
    kurtosis: float


def sharpe_statistics(returns: ArrayLike) -> SharpeStatistics:
    """Estimate the inputs used by Bailey and Lopez de Prado (2012, 2014).

    References: "The Sharpe Ratio Efficient Frontier" (2012), Journal of
    Risk; "The Deflated Sharpe Ratio" (2014), Journal of Portfolio Management.
    Input is a finite vector of excess returns, with at least three values
    and positive variance. Sharpe uses sample standard deviation (ddof=1);
    skewness and Pearson kurtosis use empirical central moments, without
    bias corrections. No annualization or serial-correlation adjustment is
    applied. These moment estimates are deterministic.
    """
    values = vector(returns, "returns", 3)
    centered = values - values.mean()
    deviation = values.std(ddof=1)
    if deviation <= 0.0:
        raise ValueError("Sharpe ratio requires positive return variance")
    standardized = centered / np.sqrt(np.mean(centered * centered))
    return SharpeStatistics(
        float(values.mean() / deviation), len(values),
        float(np.mean(standardized ** 3)), float(np.mean(standardized ** 4)),
    )


def _variance_factor(sharpe: float, skewness: float, kurtosis: float) -> float:
    """Return the Sharpe variance numerator from Bailey and Lopez de Prado (2012)."""
    ratio = finite(sharpe, "sharpe")
    skew = finite(skewness, "skewness")
    kurt = finite(kurtosis, "kurtosis")
    if kurt < 1.0:
        raise ValueError("Pearson kurtosis must be at least one")
    factor = 1.0 - skew * ratio + (kurt - 1.0) * ratio * ratio / 4.0
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("Sharpe variance factor must be finite and positive")
    return float(factor)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_observations: int,
    *,
    benchmark: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Return approximate confidence that the true Sharpe exceeds benchmark.

    Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15. PSR = Phi((SR - SR0) * sqrt(T - 1) /
    sqrt(1 - skewness * SR + (kurtosis - 1) * SR**2 / 4)).
    Inputs are unannualized Sharpe ratios, T >= 2, return skewness and Pearson
    kurtosis (normal = 3, not excess kurtosis). The result lies in [0, 1].
    This asymptotic moment approximation assumes independent observations;
    serial dependence is not corrected. No randomness is used.
    """
    count = integer(n_observations, "n_observations", 2)
    threshold = finite(benchmark, "benchmark")
    factor = _variance_factor(sharpe, skewness, kurtosis)
    return float(norm.cdf((sharpe - threshold) * np.sqrt((count - 1) / factor)))


def expected_max_sharpe(
    n_trials: int, sharpe_std: float, *, mean_sharpe: float = 0.0
) -> float:
    """Approximate the expected maximum Sharpe across independent trials.

    Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal
    of Portfolio Management 40. The normal order-statistic approximation is
    mean + std * ((1-gamma) * Phi^-1(1-1/N) +
    gamma * Phi^-1(1-1/(N*e))), with Euler's constant gamma.
    N is an integer independent/effective trial count; sharpe_std is the
    nonnegative cross-trial standard deviation of per-observation Sharpe
    estimates, not the standard deviation of returns. N=1 returns the mean.
    Trial dependence/effective counts must be assessed by the caller.
    """
    count = integer(n_trials, "n_trials")
    spread = finite(sharpe_std, "sharpe_std")
    location = finite(mean_sharpe, "mean_sharpe")
    if spread < 0.0:
        raise ValueError("sharpe_std must be nonnegative")
    if count == 1 or spread == 0.0:
        return location
    gamma = float(np.euler_gamma)
    quantile = ((1.0 - gamma) * norm.isf(1.0 / count)
                + gamma * norm.isf(1.0 / (count * np.e)))
    return float(location + spread * quantile)


def deflated_sharpe_ratio(
    sharpe: float,
    n_observations: int,
    n_trials: int,
    sharpe_std: float,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute PSR against the zero-mean expected maximum over N trials.

    Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal
    of Portfolio Management 40. Inputs have the units and assumptions of
    probabilistic_sharpe_ratio and expected_max_sharpe. The Sharpe spread
    must describe the full search, including discarded trials. N=1 reduces
    to PSR against zero. This is a confidence measure, not a p-value or an
    estimate of the probability of future profitability.
    """
    threshold = expected_max_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(
        sharpe, n_observations, benchmark=threshold,
        skewness=skewness, kurtosis=kurtosis,
    )
