"""Statistical checks for backtest selection bias.

Return matrices use rows for observations and columns for strategies. Sharpe
ratios are per observation; kurtosis is Pearson (normal = 3). Randomized
functions accept a NumPy Generator or integer seed through rng. Inputs are
finite and no strategy deduplication or missing-value filling is implicit.
"""

from .dsr import (
    SharpeStatistics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_statistics,
)
from .holdout import HoldoutConsumedError, HoldoutRecord, SealedHoldout, parameter_fingerprint
from .monte_carlo import drawdown_distribution, max_drawdown
from .pbo import PBOResult, pbo
from .power import minimum_track_record_length, required_observations, sharpe_power
from .reality_check import RealityCheckResult, stationary_bootstrap_indices, white_reality_check
from .stepm import StepMResult, romano_wolf_stepm

__all__ = [
    "HoldoutConsumedError", "HoldoutRecord", "PBOResult", "RealityCheckResult",
    "SealedHoldout", "SharpeStatistics", "StepMResult", "deflated_sharpe_ratio",
    "drawdown_distribution", "expected_max_sharpe", "max_drawdown",
    "minimum_track_record_length", "parameter_fingerprint", "pbo",
    "probabilistic_sharpe_ratio", "required_observations", "romano_wolf_stepm",
    "sharpe_power", "sharpe_statistics", "stationary_bootstrap_indices",
    "white_reality_check",
]
