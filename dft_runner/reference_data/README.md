# Reference data

`cache.json` lands here automatically the first time `BenchmarkEvaluationBackend`
evaluates a composition — it's gitignored, since it's just a locally
regenerable cache, not source data. `REFERENCE_BANDGAPS`, the actual
curated dataset it's built from, lives directly in `benchmark_engine.py`.
