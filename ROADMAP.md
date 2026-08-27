# Roadmap

Dependency order: descriptors before surrogate (the GP needs feature
vectors) → surrogate *and* evaluation before the orchestrator (it wires both
together) → orchestrator before the API (the API just wraps it) → everything
before any efficiency claims (can't honestly report iteration savings until
the loop has actually run against a baseline).

| Phase | Focus | Depends on | Status |
|---|---|---|---|
| 0 | Repo & environment scaffolding | — | ✅ done |
| 1 | Materials core (descriptors, space) | 0 | not started |
| 2 | Surrogate model & acquisition | 1 | not started |
| 3 | Evaluation backends | 1 | not started |
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
