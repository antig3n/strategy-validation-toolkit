"""Stationary-bootstrap sampling properties."""

import numpy as np
import pytest

from svt import stationary_bootstrap_indices


def test_seed_and_generator_reproducibility() -> None:
    first = stationary_bootstrap_indices(100, n_resamples=50, rng=16)
    second = stationary_bootstrap_indices(100, n_resamples=50, rng=np.random.default_rng(16))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, stationary_bootstrap_indices(100, n_resamples=50, rng=17))
    assert first.shape == (50, 100)
    assert first.min() >= 0 and first.max() < 100


def test_geometric_continuation_and_circular_wraparound() -> None:
    indices = stationary_bootstrap_indices(500, n_resamples=200, block_length=8, rng=7)
    follows = indices[:, 1:] == (indices[:, :-1] + 1) % 500
    assert follows.mean() == pytest.approx(7 / 8 + 1 / (8 * 500), abs=0.01)
    assert np.any((indices[:, :-1] == 499) & (indices[:, 1:] == 0))


def test_iid_bootstrap_has_uniform_marginals_and_low_continuation() -> None:
    indices = stationary_bootstrap_indices(10, n_resamples=4000, block_length=1, rng=5)
    frequency = np.bincount(indices.ravel(), minlength=10) / indices.size
    np.testing.assert_allclose(frequency, 0.1, atol=0.006)
    continuation = (indices[:, 1:] == (indices[:, :-1] + 1) % 10).mean()
    assert continuation == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("options", [
    {"n_observations": 0}, {"n_observations": 1.5},
    {"n_observations": True}, {"n_resamples": 0},
    {"block_length": 0.9}, {"block_length": float("nan")},
])
def test_invalid_bootstrap_inputs(options: dict[str, float]) -> None:
    arguments = {"n_observations": 10, "n_resamples": 10, **options}
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(**arguments)
