"""Reference statistics, global-null behavior and stepdown inference."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike

from svt import romano_wolf_stepm, stationary_bootstrap_indices, white_reality_check


def test_white_statistic_and_centered_resampling_by_direct_calculation() -> None:
    values = np.array([[1, -1], [3, 0], [-1, 2], [2, -2], [0, 1]], dtype=float)
    indices = stationary_bootstrap_indices(5, n_resamples=39, block_length=2, rng=4)
    expected = np.maximum(0, np.sqrt(5) * (values - values.mean(axis=0))[indices].mean(axis=1).max(axis=1))
    result = white_reality_check(values, n_resamples=39, block_length=2, rng=4)
    np.testing.assert_allclose(result.bootstrap_maxima, expected)
    assert result.statistic == pytest.approx(np.sqrt(5))
    assert result.pvalue == (1 + np.count_nonzero(expected >= result.statistic)) / 40


def test_centered_noise_does_not_reject() -> None:
    values = np.random.default_rng(6).normal(size=(300, 10))
    values -= values.mean(axis=0) + 1e-10
    result = white_reality_check(values, n_resamples=199, rng=2)
    assert result.pvalue == 1.0
    assert result.statistic == 0.0
    assert not romano_wolf_stepm(values, n_resamples=199, rng=2).rejected.any()


def test_null_rejection_rates_across_fixed_independent_experiments() -> None:
    white_rejections, family_errors = 0, 0
    for seed in range(40):
        values = np.random.default_rng(seed).normal(size=(200, 6))
        white_rejections += white_reality_check(values, n_resamples=199, rng=seed + 100).pvalue <= 0.05
        family_errors += bool(romano_wolf_stepm(values, n_resamples=199, rng=seed + 100).rejected.any())
    # Sampling variation prevents requiring exactly 5% in a small experiment.
    assert white_rejections <= 5
    assert family_errors <= 5


def test_edge_is_detected_without_false_noise_discoveries() -> None:
    values = np.random.default_rng(42).normal(size=(800, 10))
    values -= values.mean(axis=0)
    values[:, 0] += 0.5
    reality = white_reality_check(values, n_resamples=399, rng=31)
    step = romano_wolf_stepm(values, n_resamples=399, rng=31)
    assert reality.pvalue < 0.05
    np.testing.assert_array_equal(step.rejected, [True] + [False] * 9)
    assert np.all(np.diff(step.adjusted_pvalues[step.order]) >= 0)
    assert np.all(step.adjusted_pvalues <= step.single_step_pvalues + 1e-15)
    assert np.all(step.adjusted_pvalues >= 1 / 400)


def test_stepdown_matches_remaining_family_maxima() -> None:
    values = np.random.default_rng(3).normal(size=(60, 4)) + [0.5, 0.2, 0.1, -0.2]
    result = romano_wolf_stepm(values, n_resamples=99, rng=9)
    raw = []
    for rank, strategy in enumerate(result.order):
        remaining = result.bootstrap_statistics[:, result.order[rank:]].max(axis=1)
        raw.append((1 + np.count_nonzero(remaining >= result.statistics[strategy])) / 100)
    np.testing.assert_allclose(result.adjusted_pvalues[result.order], np.maximum.accumulate(raw))
    np.testing.assert_allclose(result.statistics, values.mean(axis=0) / result.standard_errors)


def test_duplicate_candidates_share_bootstrap_draws() -> None:
    values = np.random.default_rng(9).normal(size=100)
    joint = romano_wolf_stepm(np.column_stack([values, values]), n_resamples=99, rng=8)
    single = romano_wolf_stepm(values, n_resamples=99, rng=8)
    np.testing.assert_array_equal(joint.bootstrap_statistics[:, 0], joint.bootstrap_statistics[:, 1])
    np.testing.assert_allclose(joint.adjusted_pvalues, single.adjusted_pvalues[0])


@pytest.mark.parametrize("procedure", [white_reality_check, romano_wolf_stepm])
def test_benchmark_subtraction_and_reproducibility(procedure: Callable[..., Any]) -> None:
    values = np.random.default_rng(4).normal(size=(100, 3))
    benchmark = np.linspace(-0.3, 0.3, 100)
    first = procedure(values, benchmark=benchmark, n_resamples=99, rng=11)
    second = procedure(values - benchmark[:, None], n_resamples=99, rng=np.random.default_rng(11))
    field = "bootstrap_maxima" if procedure is white_reality_check else "bootstrap_statistics"
    np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    shifted = procedure(values + 2, benchmark=2, n_resamples=99, rng=11)
    original = procedure(values, n_resamples=99, rng=11)
    np.testing.assert_allclose(getattr(shifted, field), getattr(original, field), atol=1e-12)


@pytest.mark.parametrize("procedure", [white_reality_check, romano_wolf_stepm])
@pytest.mark.parametrize("values, options", [
    ([], {}), ([1, 2], {}), ([1, np.nan, 2], {}),
    ([1, 2, 3], {"benchmark": [1, 2, 3, 4]}),
    ([1, 2, 3], {"benchmark": [[1, 2, 3]]}),
    ([1, 2, 3], {"benchmark": np.inf}),
    ([1, 2, 3], {"n_resamples": 1}),
    ([1, 2, 3], {"block_length": 0}),
])
def test_invalid_multiple_testing_inputs(
    procedure: Callable[..., Any], values: ArrayLike, options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        procedure(values, **options)


def test_stepm_rejects_constant_candidates_and_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="nonconstant"):
        romano_wolf_stepm(np.ones((10, 2)))
    with pytest.raises(ValueError, match="alpha"):
        romano_wolf_stepm(np.arange(10.0), alpha=1.0)


def test_stepm_rejects_numerically_degenerate_bootstrap() -> None:
    values = np.random.default_rng(5).normal(size=(100, 2))
    with pytest.raises(ValueError, match="standard errors"):
        romano_wolf_stepm(values, n_resamples=19, block_length=1e30, rng=2)


def test_all_negative_white_statistic_has_pvalue_one() -> None:
    result = white_reality_check([-3, -2, -1], n_resamples=19, rng=0)
    assert result.statistic == 0.0 and result.pvalue == 1.0
