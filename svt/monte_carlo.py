"""Empirical trade resampling for maximum-drawdown distributions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, RngLike, integer, vector


def _drawdowns(paths: FloatArray, compounded: bool) -> FloatArray:
    """Measure peak-to-trough losses, including the initial equity peak."""
    initial = np.ones((len(paths), 1)) if compounded else np.zeros((len(paths), 1))
    with np.errstate(over="ignore", invalid="ignore"):
        equity = np.cumprod(1.0 + paths, axis=1) if compounded else np.cumsum(paths, axis=1)
    if not np.isfinite(equity).all() or (compounded and (equity <= 0.0).any()):
        raise ValueError("equity path overflowed or underflowed")
    equity = np.concatenate((initial, equity), axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    losses = 1.0 - equity / peaks if compounded else peaks - equity
    return losses.max(axis=1)


def max_drawdown(trades: ArrayLike, *, compounded: bool = False) -> float:
    """Return the largest peak-to-trough loss, including initial equity.

    Drawdown definition: Magdon-Ismail et al. (2004), "On the Maximum
    Drawdown of a Brownian Motion", Journal of Applied Probability 41.
    Input is a nonempty finite trade vector. Additive mode returns losses
    in trade units from cumulative P&L starting at zero. Compounded mode
    accepts fractional returns strictly greater than -1 and returns a
    fractional drawdown from initial wealth one. No Brownian model or
    randomness is assumed by this path calculation.
    """
    values = vector(trades, "trades")
    if compounded and (values <= -1.0).any():
        raise ValueError("compounded returns must be greater than -1")
    return float(_drawdowns(values[None, :], compounded)[0])


def drawdown_distribution(
    trades: ArrayLike,
    *,
    n_resamples: int = 1999,
    n_trades: int | None = None,
    method: Literal["bootstrap", "permutation"] = "bootstrap",
    compounded: bool = False,
    rng: RngLike = None,
) -> FloatArray:
    """Return one nonnegative maximum drawdown per resampled trade path.

    Efron (1979), "Bootstrap Methods: Another Look at the Jackknife", Annals
    of Statistics 7; drawdown definition follows Magdon-Ismail et al. (2004).
    Bootstrap samples trades with replacement, assuming IID trades. Random
    permutation instead preserves the observed trade multiset and measures
    order risk only. It requires n_trades equal to the input length.
    n_trades defaults to that length. Units and compounded-return restrictions
    follow max_drawdown. rng accepts a Generator or integer seed; the output
    is an empirical distribution, without parametric tail extrapolation.
    """
    values = vector(trades, "trades")
    count = integer(n_resamples, "n_resamples")
    length = len(values) if n_trades is None else integer(n_trades, "n_trades")
    if method not in ("bootstrap", "permutation"):
        raise ValueError("method must be bootstrap or permutation")
    if method == "permutation" and length != len(values):
        raise ValueError("permutation requires the original trade count")
    if compounded and (values <= -1.0).any():
        raise ValueError("compounded returns must be greater than -1")
    generator = np.random.default_rng(rng)
    result = np.empty(count, dtype=np.float64)
    batch = max(1, min(256, 1_000_000 // length))
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        if method == "bootstrap":
            paths = values[generator.integers(len(values), size=(stop - start, length))]
        else:
            paths = generator.permuted(np.broadcast_to(values, (stop - start, length)), axis=1)
        result[start:stop] = _drawdowns(paths, compounded)
    return result
