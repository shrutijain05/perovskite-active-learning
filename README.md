# Perov-ActiveLearn

Closed-loop active learning and Bayesian optimization for perovskite bandgap
engineering. Guides a Gaussian Process surrogate toward compositions that hit
a target electronic bandgap, using far fewer simulated/looked-up evaluations
than grid or random screening.

**Status:** Phase 0 — repo scaffolding. Modules are stubs; see `ROADMAP.md`
for the full build plan and what lands in each phase. The real quickstart
example and benchmark numbers will replace this section in Phase 6.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
```

## Project layout

```
perov_core/    materials descriptors & composition space
surrogate/     Gaussian Process model & acquisition functions
dft_runner/    DFT log parsing & benchmark/mock evaluation backend
pipeline/      closed-loop orchestrator & visualization
api/           FastAPI service
tests/         pytest suite
```

## License

MIT — see `LICENSE`.
