"""Known Sharpe probabilities and inverse record-length relationships."""

import numpy as np
import pytest
from scipy.stats import norm
from numpy.typing import ArrayLike

from svt import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    required_observations,
    sharpe_power,
    sharpe_statistics,
)


def test_psr_known_normal_and_nonnormal_moments() -> None:
    assert probabilistic_sharpe_ratio(0, 100) == 0.5
    assert probabilistic_sharpe_ratio(0.1, 101) == pytest.approx(norm.cdf(1 / np.sqrt(1.005)))
    factor = 1 - (-0.4) * 0.2 + (5 - 1) * 0.2 ** 2 / 4
    expected = norm.cdf((0.2 - 0.05) * np.sqrt(199 / factor))
    assert probabilistic_sharpe_ratio(0.2, 200, benchmark=0.05, skewness=-0.4, kurtosis=5) == pytest.approx(expected)


def test_expected_max_and_dsr_penalize_more_trials() -> None:
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert expected_max_sharpe(100, 0.0, mean_sharpe=0.2) == 0.2
    gamma = np.euler_gamma
    expected = 0.05 * ((1 - gamma) * norm.ppf(0.99) + gamma * norm.ppf(1 - 1 / (100 * np.e)))
    assert expected_max_sharpe(100, 0.05) == pytest.approx(expected)
    trials = [1, 2, 10, 100, 1000]
    probabilities = [deflated_sharpe_ratio(0.12, 300, count, 0.05) for count in trials]
    assert probabilities[0] == probabilistic_sharpe_ratio(0.12, 300)
    assert np.all(np.diff(probabilities) < 0)
    assert probabilities[-1] < 0.5
    assert np.isfinite(expected_max_sharpe(10**18, 0.05))


def test_empirical_moments_use_sample_sharpe_and_pearson_kurtosis() -> None:
    values = np.array([-2, -1, 0, 1, 2], dtype=float) + 0.1
    result = sharpe_statistics(values)
    assert result.n_observations == 5
    assert result.sharpe == pytest.approx(0.1 / values.std(ddof=1))
    assert result.skewness == pytest.approx(0, abs=1e-14)
    assert result.kurtosis == pytest.approx(1.7)


def test_mintrl_inverts_psr_at_the_integer_boundary() -> None:
    arguments = {"benchmark": 0.03, "skewness": -0.2, "kurtosis": 4.0}
    length = minimum_track_record_length(0.15, confidence=0.95, **arguments)
    assert probabilistic_sharpe_ratio(0.15, length, **arguments) >= 0.95
    assert probabilistic_sharpe_ratio(0.15, length - 1, **arguments) < 0.95
    assert minimum_track_record_length(0.3) < minimum_track_record_length(0.15)


def test_power_inversion_and_null_size() -> None:
    assert sharpe_power(0, 100, alpha=0.05) == pytest.approx(0.05)
    arguments = {"benchmark": 0.02, "alpha": 0.05, "skewness": 0.3, "kurtosis": 4.0}
    length = required_observations(0.15, power=0.8, **arguments)
    assert sharpe_power(0.15, length, **arguments) >= 0.8
    assert sharpe_power(0.15, length - 1, **arguments) < 0.8
    assert required_observations(0.15, power=0.9) > required_observations(0.15, power=0.8)
    assert required_observations(0.15, alpha=0.01) > required_observations(0.15)


@pytest.mark.parametrize("options", [
    {"sharpe": np.nan}, {"n_observations": 1}, {"n_observations": 10.5},
    {"benchmark": np.inf}, {"skewness": np.nan}, {"kurtosis": 0},
    {"sharpe": 2, "skewness": 2, "kurtosis": 1},
])
def test_invalid_psr_inputs(options: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(**{"sharpe": 0.1, "n_observations": 100, **options})


@pytest.mark.parametrize("count, spread", [(0, 0.1), (1.5, 0.1), (2, -1), (2, np.nan)])
def test_invalid_trial_inputs(count: int, spread: float) -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(count, spread)


@pytest.mark.parametrize("values", [[], [1, 2], [1, 1, 1], [0, 1, np.nan], [[1, 2, 3]]])
def test_invalid_sharpe_series(values: ArrayLike) -> None:
    with pytest.raises(ValueError):
        sharpe_statistics(values)


def test_invalid_record_length_and_power_targets() -> None:
    for ratio in (0.0, -0.1):
        with pytest.raises(ValueError):
            minimum_track_record_length(ratio)
        with pytest.raises(ValueError):
            required_observations(ratio)
    for confidence in (0.0, 0.5, 1.0):
        with pytest.raises(ValueError):
            minimum_track_record_length(0.1, confidence=confidence)
    with pytest.raises(ValueError):
        required_observations(0.1, power=0.01, alpha=0.05)
    with pytest.raises(ValueError):
        sharpe_power(0.1, 100, alpha=0)
