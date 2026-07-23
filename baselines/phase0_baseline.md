# Phase 0 baseline

Captured before any generalization work begins (per the implementation plan,
`~/.claude/plans/vivid-gliding-snowflake.md`, Section 8 Phase 0). This is the
non-regression reference point for later phases — in particular Phase 9's
"benchmark regresses <2% vs. Phase-0 baseline" check.

## Test suite

- `pytest` (fast tier): **23 passed** in 226.54s.
- `pytest -m slow` (full iteration budget, tight EV floors): **3 passed**, 20
  deselected, in 205.51s.

Both green, unmodified, before any code changes — this repo's starting state.

## EV / timing baseline

Captured via `baselines/phase0_capture.py` → `baselines/phase0_baseline.json`
(`offer_opt.metrics.benchmark`, `n_reps=1`, hard case at
`max_iters=400, repair_every=20` matching `test_end_to_end.py`'s slow-tier
config; low/med at solver defaults). Machine has no CUDA; `mps` below is
Apple's Metal GPU backend.

| Case | Device | Median solve time (s) | Total EV | Iterations | Converged | Verifier OK |
|---|---|---:|---:|---:|---|---|
| low | cpu | 13.12 | 3,760,723.22 | 1200 | False | True |
| low | mps | 5.09 | 3,760,723.22 | 1200 | False | True |
| med | cpu | 12.49 | 190,066.76 | 1200 | False | True |
| med | mps | 24.79 | 190,066.76 | 1200 | False | True |
| hard | cpu | 260.02 | 1,308,276,376.68 | 400 | False | True |
| hard | mps | 123.09 | 1,308,276,376.68 | 400 | False | True |

Notes:
- EV is identical CPU vs. MPS per case (as expected — same deterministic
  algorithm, `dtype_for()` just drops to float32 on MPS). Only timing differs.
- `converged=False` everywhere is expected and already tolerated by the
  existing test suite (EV-floor assertions, not a `converged==True`
  assertion) — the subgradient loop hits `max_iters` before the strict
  `stable_patience` threshold, which is normal for this problem size.
- MPS is faster than CPU on `low` and `hard` (5-6x, more parallelism to
  exploit at scale) but *slower* on `med` (24.8s vs 12.5s) — `med` is the
  smallest offer table (154,659 rows) of the three, where MPS dispatch
  overhead outweighs the parallelism benefit. Not a regression to fix now,
  just a real characteristic of this hardware/problem-size combination worth
  remembering when Phase 9 compares against this baseline (a small case
  running on GPU is not automatically the fast configuration).

## Reproduce

```
PYTHONPATH=. .venv/bin/python3 baselines/phase0_capture.py
```
