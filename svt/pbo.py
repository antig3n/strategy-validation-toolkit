"""Combinatorially symmetric cross-validation of strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, IntArray, integer, matrix


@dataclass(frozen=True)
class PBOResult:
    """CSCV diagnostics in combination order; Sharpe ratios are per observation."""

    pbo: float
    logits: FloatArray
    relative_ranks: FloatArray
    selected_strategies: IntArray
    in_sample_sharpe: FloatArray
    out_of_sample_sharpe: FloatArray
    performance_degradation: FloatArray
    mean_degradation: float
    degradation_slope: float
    degradation_intercept: float
    loss_probability: float
    n_splits: int


def pbo(
    returns: ArrayLike, *, n_blocks: int = 8, max_splits: int = 100_000
) -> PBOResult:
    """Estimate PBO using every C(S, S/2) chronological block split.

    Bailey, Borwein, Lopez de Prado and Zhu (2014), "The Probability of
    Backtest Overfitting", subsequently Journal of Computational Finance (2017).
    Input is a finite (observations, strategies) matrix with at least two
    strategies. S must be even and divide the observation count exactly;
    each half needs at least two observations and positive strategy variances.
    No observations or duplicate strategies are silently removed.

    Selection maximizes mean / sample standard deviation in sample. Selection
    ties choose the first column; out-of-sample ties receive average ranks.
    Relative ranks are rank / (strategies + 1), with rank one the worst.
    PBO counts logits <= 0, including exact median ties. With odd strategy
    counts this convention can put the noise baseline slightly above 0.5.
    Degradation is IS minus OOS Sharpe; the regression is OOS = intercept +
    slope * IS, returning NaN coefficients when IS has no variation.
    The deterministic enumeration needs no random generator; max_splits
    rejects excessive enumerations before allocating their arrays.
    """
    values = matrix(returns, minimum=4)
    n, strategies = values.shape
    blocks = integer(n_blocks, "n_blocks", 2)
    limit = integer(max_splits, "max_splits")
    if strategies < 2:
        raise ValueError("at least two strategies are required")
    if blocks % 2 or n % blocks or n // 2 < 2:
        raise ValueError("n_blocks must be even and divide the observation count")
    count = comb(blocks, blocks // 2)
    if count > limit:
        raise ValueError("CSCV enumeration exceeds max_splits")

    location = values.mean(axis=0)
    centered = (values - location).reshape(blocks, n // blocks, strategies)
    block_sums = centered.sum(axis=1)
    block_squares = np.square(centered).sum(axis=1)
    masks = np.zeros((count, blocks), dtype=np.float64)
    for row, chosen in enumerate(combinations(range(blocks), blocks // 2)):
        masks[row, list(chosen)] = 1.0
    half = n // 2

    def performance(mask: FloatArray) -> FloatArray:
        """Compute split Sharpe ratios from centered sufficient statistics."""
        sums = mask @ block_sums
        variances = (mask @ block_squares - sums * sums / half) / (half - 1)
        if (variances <= 0.0).any():
            raise ValueError("every strategy needs positive variance in every half")
        return (location + sums / half) / np.sqrt(variances)

    inside, outside = performance(masks), performance(1.0 - masks)
    selected = np.argmax(inside, axis=1).astype(np.int64)
    rows = np.arange(count)
    best_inside, best_outside = inside[rows, selected], outside[rows, selected]
    below = (outside < best_outside[:, None]).sum(axis=1)
    equal = (outside == best_outside[:, None]).sum(axis=1)
    relative = (below + (equal + 1.0) / 2.0) / (strategies + 1.0)
    logits = np.log(relative / (1.0 - relative))
    delta = best_inside - best_outside
    centered_inside = best_inside - best_inside.mean()
    denominator = float(centered_inside @ centered_inside)
    slope = (float(centered_inside @ (best_outside - best_outside.mean()))
             / denominator if denominator > 0.0 else float("nan"))
    intercept = float(best_outside.mean() - slope * best_inside.mean())
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)), logits=logits,
        relative_ranks=relative, selected_strategies=selected,
        in_sample_sharpe=best_inside, out_of_sample_sharpe=best_outside,
        performance_degradation=delta, mean_degradation=float(delta.mean()),
        degradation_slope=slope, degradation_intercept=intercept,
        loss_probability=float(np.mean(best_outside < 0.0)), n_splits=count,
    )
