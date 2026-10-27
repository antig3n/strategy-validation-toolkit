"""Drawdown definitions, resampling limits and reproducibility."""

from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike

from svt import drawdown_distribution, max_drawdown


@pytest.mark.parametrize("trades, expected", [
    ([1, -2, 1, -3, 5], 4), ([-2, -3], 5), ([1, 2, 3], 0), ([0], 0),
])
def test_known_additive_drawdowns(trades: list[float], expected: float) -> None:
    assert max_drawdown(trades) == expected


def test_compounded_drawdown_includes_initial_peak() -> None:
    assert max_drawdown([-0.1, 0.2, -0.25], compounded=True) == pytest.approx(0.25)
    assert max_drawdown([-0.1, -0.2], compounded=True) == pytest.approx(0.28)


@pytest.mark.parametrize("method", ["bootstrap", "permutation"])
def test_reproducible_empirical_distribution(method: Literal["bootstrap", "permutation"]) -> None:
    trades = [2, -1, -3, 4, -2]
    first = drawdown_distribution(trades, n_resamples=100, method=method, rng=9)
    second = drawdown_distribution(trades, n_resamples=100, method=method, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (100,)
    assert (first >= 0).all()
    if method == "permutation":
        assert first.max() <= 6
    else:
        assert first.max() <= 15


def test_resampling_extreme_paths_and_custom_horizon() -> None:
    np.testing.assert_array_equal(drawdown_distribution([1, 2], rng=1), 0)
    np.testing.assert_array_equal(drawdown_distribution([-2], n_trades=5, rng=1), 10)
    np.testing.assert_allclose(drawdown_distribution([-0.1], n_trades=2, compounded=True, rng=1), 0.19)
    permutations = drawdown_distribution([-2, -3], method="permutation", rng=2)
    np.testing.assert_array_equal(permutations, 5)


@pytest.mark.parametrize("trades, options", [
    ([], {}), ([np.nan], {}), ([np.inf], {}), ([[1, 2]], {}),
    ([-1], {"compounded": True}), ([1], {"n_resamples": 0}),
    ([1], {"n_trades": 0}), ([1], {"method": "other"}),
    ([1, 2], {"method": "permutation", "n_trades": 3}),
])
def test_invalid_drawdown_inputs(trades: ArrayLike, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        drawdown_distribution(trades, **options)


def test_equity_overflow_and_underflow_are_errors() -> None:
    with pytest.raises(ValueError, match="overflowed"):
        max_drawdown([1e200, 1e200], compounded=True)
    with pytest.raises(ValueError, match="underflowed"):
        max_drawdown([-0.999] * 200, compounded=True)
