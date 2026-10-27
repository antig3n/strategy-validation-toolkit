# strategy-validation-toolkit

Statistical tests that answer one question before any trading strategy, or any model picked from many candidates, is trusted: **is this result real, or is it the best of many lucky draws?**

When you try 20, 200 or 2,000 variants and keep the best one, its backtest is biased upward by construction. This toolkit implements the published methods that correct for that selection bias, with tests and a demo that shows them telling noise from a genuine edge.

## What's inside

| Module | Method | Question it answers |
|---|---|---|
| `svt.pbo` | **CSCV / PBO**: Bailey, Borwein, López de Prado & Zhu (2014) | How often does the in-sample winner fall below the median out of sample? |
| `svt.reality_check` | **White's Reality Check** (2000) with the Politis–Romano stationary bootstrap | Does the best strategy beat the benchmark once data snooping is accounted for? |
| `svt.stepm` | **Romano–Wolf StepM** (2005) | *Which* strategies beat the benchmark, with family-wise error control? |
| `svt.dsr` | **Probabilistic & Deflated Sharpe Ratio**: Bailey & López de Prado (2012, 2014) | Is this Sharpe significant given skew, fat tails and the number of trials? |
| `svt.power` | **Minimum Track Record Length** and power | How many observations are needed before a Sharpe can be trusted? |
| `svt.monte_carlo` | Trade-order bootstrap | What drawdown should I expect, beyond the single historical path? |
| `svt.holdout` | **Sealed holdout** | Has this final test period already been used? Single-use, logged in SQLite, refused on reuse. |

The sealed holdout is the practical safeguard: the final out-of-sample period can be evaluated **once**. Each attempt is recorded with a fingerprint of the parameters, and any overlapping second attempt is refused. Peeking stops being a matter of discipline.

## Demo

```bash
pip install numpy scipy pytest
python examples/demo.py
```

```
Fictional data: 1,200 observations, 20 candidate strategies.
Noise: PBO=0.586, RC p=0.430, StepM discoveries=0, winner PSR/DSR=0.972/0.357
Edge : PBO=0.000, RC p=0.002, StepM discoveries=1, winner PSR/DSR=1.000/1.000
Edge planning: MinTRL=12, 80% power needs 25 observations.
Edge trade bootstrap: 95% drawdown quantile=5.90 synthetic units.
Holdout: first evaluation recorded; second evaluation refused.
```

Read the noise line carefully. The best of 20 random strategies shows a PSR of 0.97, which looks convincing on its own. Once the 20 trials are accounted for, the Deflated Sharpe drops to 0.36, PBO is above 0.5 and the Reality Check finds nothing. With a real edge planted in one column, every test agrees.

## Tests

```bash
python -m pytest -q     # 109 tests, about 2 s
```

The tests are deterministic (seeded generators, exhaustive CSCV enumeration) and check known closed-form values, edge cases and input validation, along with each method's behaviour on pure noise versus planted signal.

## Notes

- Sharpe ratios are per observation; kurtosis is Pearson (normal = 3).
- The methods are asymptotic and assume what their papers assume. Serial dependence is handled by the bootstrap tests, not by PSR/DSR.
- Every function cites its source in the docstring.

## About

Generic extract of the validation layer I use to judge systematic strategies. The same discipline applies to choosing between LLM prompts, models or agent configurations. Designed and reviewed by me; implementation produced through a supervised multi-agent pipeline (specification → generation → independent review → tests rerun before acceptance).

MIT licence.
