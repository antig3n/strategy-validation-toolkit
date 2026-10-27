"""Deterministic synthetic illustration; all observations are fictional."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svt import (  # noqa: E402
    HoldoutConsumedError,
    SealedHoldout,
    deflated_sharpe_ratio,
    drawdown_distribution,
    minimum_track_record_length,
    pbo,
    probabilistic_sharpe_ratio,
    required_observations,
    romano_wolf_stepm,
    sharpe_statistics,
    white_reality_check,
)
from svt.holdout import JSONValue  # noqa: E402


def synthetic_holdout_mean(parameters: Mapping[str, JSONValue]) -> float:
    """Evaluate fresh fictional data for the single-use holdout illustration."""
    seed = parameters["seed"]
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return float(np.random.default_rng(seed).normal(0.5, 1.0, 100).mean())


def main() -> None:
    """Contrast noise and a stable edge using the cited svt procedures."""
    generator = np.random.default_rng(42)
    noise = generator.normal(size=(1200, 20))
    edge = noise.copy()
    edge[:, 0] += 0.5
    print("Fictional data: 1,200 observations, 20 candidate strategies.")
    for label, values in (("Noise", noise), ("Edge ", edge)):
        selection = pbo(values)
        reality = white_reality_check(values, n_resamples=499, rng=17)
        step = romano_wolf_stepm(values, n_resamples=499, rng=17)
        ratios = values.mean(axis=0) / values.std(axis=0, ddof=1)
        winner = sharpe_statistics(values[:, int(np.argmax(ratios))])
        arguments = {"skewness": winner.skewness, "kurtosis": winner.kurtosis}
        psr = probabilistic_sharpe_ratio(winner.sharpe, len(values), **arguments)
        dsr = deflated_sharpe_ratio(
            winner.sharpe, len(values), values.shape[1], float(ratios.std(ddof=1)), **arguments
        )
        print(f"{label}: PBO={selection.pbo:.3f}, RC p={reality.pvalue:.3f}, "
              f"StepM discoveries={np.count_nonzero(step.rejected)}, "
              f"winner PSR/DSR={psr:.3f}/{dsr:.3f}")

    moments = sharpe_statistics(edge[:, 0])
    arguments = {"skewness": moments.skewness, "kurtosis": moments.kurtosis}
    print(f"Edge planning: MinTRL={minimum_track_record_length(moments.sharpe, **arguments)}, "
          f"80% power needs {required_observations(moments.sharpe, **arguments)} observations.")
    drawdowns = drawdown_distribution(edge[:200, 0], n_resamples=499, rng=23)
    print(f"Edge trade bootstrap: 95% drawdown quantile={np.quantile(drawdowns, 0.95):.2f} synthetic units.")
    with TemporaryDirectory(prefix="demo-", dir=Path(__file__).parent) as directory:
        holdout = SealedHoldout(Path(directory) / "audit.sqlite3", "2040-01-01", "2040-12-31")
        holdout.evaluate({"candidate": 0, "seed": 31}, synthetic_holdout_mean)
        try:
            holdout.evaluate({"candidate": 1}, lambda _: None)
        except HoldoutConsumedError:
            print("Holdout: first evaluation recorded; second evaluation refused.")


if __name__ == "__main__":
    main()
