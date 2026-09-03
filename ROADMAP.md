# Roadmap

Dependency order: descriptors before surrogate (the GP needs feature
vectors) → surrogate *and* evaluation before the orchestrator (it wires both
together) → orchestrator before the API (the API just wraps it) → everything
before any efficiency claims (can't honestly report iteration savings until
the loop has actually run against a baseline).

| Phase | Focus | Depends on | Status |
|---|---|---|---|
| 0 | Repo & environment scaffolding | — | ✅ done |
| 1 | Materials core (descriptors, space) | 0 | ✅ done |
| 2 | Surrogate model & acquisition | 1 | ✅ done |
| 3 | Evaluation backends | 1 | ✅ done |
| 4 | Closed-loop orchestrator | 2, 3 | not started |
| 5 | API + CI/CD | 4 | not started |
| 6 | Benchmarking, docs, polish | 4, 5 | not started |
| 7 (stretch) | Multi-objective, real DFT, batch acquisition | 6 | not started |

## Phase 0 — definition of done

- [x] `pip install -e ".[dev]"` succeeds in a clean virtualenv
- [x] Empty-but-green test suite runs and passes
- [x] CI workflow configured (pytest + ruff on push/PR)
- [ ] Pushed to GitHub with your own commit identity (your step — see setup notes)
- [ ] Materials Project account + API key registered (needed by Phase 3, worth doing now)

## Phase 1 — definition of done

- [x] `descriptors.py` — validated composition dataclass, 14-dim feature vector
- [x] `space.py` — combinatorial composition-grid generator
- [x] Tolerance/octahedral factor checked against 3 known real materials (MAPbI3, CsPbBr3, FASnI3)
- [x] Invalid input (bad fractions, unknown ions) raises a clear error instead of silently producing garbage
- [x] Whole generated grid produces finite features with no NaNs (11/11 tests passing)
- [ ] `crystal_builder.py` intentionally skipped — only needed for real DFT (Phase 7 stretch), not for mock-mode v1

## Phase 2 — definition of done

- [x] `gp_model.py` — validated GP wrapper (rejects wrong feature shape, predict-before-fit)
- [x] `acquisition.py` — Target-EI, UCB, LCB, plus a `score_candidates()` dispatcher for Phase 4
- [x] `uncertainty.py` — empirical coverage / calibration check
- [x] Uncertainty confirmed to shrink at observed points and grow far from data
- [x] Acquisition scoring confirmed by hand-derivation, not just "it ran" (9 new tests, 20/20 passing)
- [x] `N_FEATURES` centralized in `perov_core.descriptors` instead of hardcoded in two places

## Phase 3 — definition of done

- [x] `base.py` — shared `EvaluationBackend` protocol both evaluators implement
- [x] `benchmark_engine.py` — offline evaluator: all 18 pure end-members (3 A x 2 B x 3 X) hardcoded, multilinear interpolation with a real optical bowing correction (Eg = linear − b·x·(1−x)), local disk caching. Bowing parameters for X-site (halide) mixing are literature-cited (MA: 0.33 eV, per Trends in Chemistry 2020 — see code comments); A/B-site bowing parameters are flagged placeholders, not citations.
- [x] `materials_project_backend.py` — live MP evaluator with graceful fallback: no key, no internet, an unsearchable composition (mixed, or organic A-site), or any live-query exception all fall through to the offline backend rather than crashing. **The live query path itself is unverified — no real API key was available to test it against.** The fallback mechanism is fully tested (5/5 passing).
- [x] `log_parser.py` — real bug caught and fixed in Phase 3's first pass (HOMO/LUMO regex matching inside QE's combined summary line); still holds
- [x] 36/36 tests passing, including a hand-verified bowing calculation (not just "it ran")
- [x] `mp-api` added as an optional dependency group (`pip install -e ".[live]"`) rather than a hard requirement, since the default install and test suite don't need it

## Notes / known modeling limitations (carried forward from planning)

- Ionic radii for mixed organic A-site cations (MA/FA) are an approximation,
  not a single physical constant the way they are for a monatomic ion —
  flag this in Phase 1.
- Materials Project (computed DFT properties) and the Perovskite Database
  Project (community-curated experimental device data) serve different
  roles in Phase 3's benchmark engine: exact/near lookups against the former
  where structures exist, interpolation/nearest-neighbor over the latter for
  the main mock-mode sweep.
- The ~65% iteration-savings figure quoted in early planning is a target,
  not a measured result — Phase 6 replaces it with a number from an actual
  BO-vs-random-search run on the mock dataset.
