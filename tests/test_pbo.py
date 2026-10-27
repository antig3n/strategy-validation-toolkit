"""CSCV identities, null behavior and input contracts."""

from itertools import combinations
from math import comb

import numpy as np
import pytest
from scipy.stats import rankdata

from svt import pbo


def test_known_reversing_winners() -> None:
    returns = np.array([[1, 0], [2, 1], [1, 0], [2, 1],
                        [0, 1], [1, 2], [0, 1], [1, 2]], dtype=float)
    result = pbo(returns, n_blocks=2)
    assert result.pbo == 1.0
    np.testing.assert_array_equal(result.selected_strategies, [0, 1])
    np.testing.assert_allclose(result.logits, -np.log(2))
    assert result.mean_degradation > 0.0
    assert result.loss_probability == 0.0
    assert np.isnan(result.degradation_slope)


def test_exhaustive_results_match_direct_slicing() -> None:
    values = np.random.default_rng(8).normal(size=(24, 5))
    result = pbo(values, n_blocks=6)
    inside, outside, ranks, selected = [], [], [], []
    for chosen in combinations(range(6), 3):
        mask = np.repeat(np.isin(np.arange(6), chosen), 4)
        train, test = values[mask], values[~mask]
        train_sr = train.mean(axis=0) / train.std(axis=0, ddof=1)
        test_sr = test.mean(axis=0) / test.std(axis=0, ddof=1)
        best = int(np.argmax(train_sr))
        selected.append(best)
        inside.append(train_sr[best])
        outside.append(test_sr[best])
        ranks.append(rankdata(test_sr)[best] / 6)
    assert result.n_splits == comb(6, 3)
    np.testing.assert_array_equal(result.selected_strategies, selected)
    np.testing.assert_allclose(result.in_sample_sharpe, inside)
    np.testing.assert_allclose(result.out_of_sample_sharpe, outside)
    np.testing.assert_allclose(result.relative_ranks, ranks)
    np.testing.assert_allclose(result.performance_degradation, np.subtract(inside, outside))
    slope, intercept = np.polyfit(inside, outside, 1)
    assert result.degradation_slope == pytest.approx(slope)
    assert result.degradation_intercept == pytest.approx(intercept)


def test_noise_baseline_across_independent_experiments() -> None:
    estimates = [pbo(np.random.default_rng(seed).normal(size=(320, 20))).pbo
                 for seed in range(32)]
    assert np.mean(estimates) == pytest.approx(0.5, abs=0.09)


def test_clear_persistent_edge() -> None:
    values = np.random.default_rng(19).normal(size=(640, 12))
    values[:, 0] += 1.5
    result = pbo(values)
    assert result.pbo == 0.0
    assert result.loss_probability == 0.0
    np.testing.assert_array_equal(result.selected_strategies, 0)


def test_sixteen_blocks_enumerate_all_combinations() -> None:
    values = np.random.default_rng(1).normal(size=(160, 4))
    assert pbo(values, n_blocks=16).n_splits == 12870


def test_constant_half_is_rejected() -> None:
    values = np.random.default_rng(2).normal(size=(8, 2))
    values[:4, 0] = 0
    with pytest.raises(ValueError, match="variance"):
        pbo(values, n_blocks=2)


def test_ties_use_average_rank_and_first_selection() -> None:
    values = np.tile(np.arange(16.0)[:, None], (1, 4))
    result = pbo(values, n_blocks=4)
    np.testing.assert_array_equal(result.selected_strategies, 0)
    np.testing.assert_array_equal(result.relative_ranks, 0.5)
    assert result.pbo == 1.0


@pytest.mark.parametrize("values, options", [
    (np.ones((16, 3)), {}),
    (np.arange(16.0), {}),
    (np.empty((0, 3)), {}),
    (np.full((16, 3), np.nan), {}),
    (np.full((16, 3), np.inf), {}),
    (np.arange(48.0).reshape(16, 3), {"n_blocks": 3}),
    (np.arange(48.0).reshape(16, 3), {"n_blocks": 6}),
    (np.arange(48.0).reshape(16, 3), {"n_blocks": True}),
    (np.arange(48.0).reshape(16, 3), {"max_splits": 1}),
])
def test_invalid_inputs(values: np.ndarray, options: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        pbo(values, **options)
