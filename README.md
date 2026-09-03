# Perov-ActiveLearn

Closed-loop active learning and Bayesian optimization for perovskite bandgap
engineering. Guides a Gaussian Process surrogate toward compositions that hit
a target electronic bandgap, using far fewer simulated/looked-up evaluations
than grid or random screening.

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

