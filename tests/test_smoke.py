"""Phase 0 smoke test.

Confirms the package structure installs and every top-level package
imports cleanly. This is intentionally trivial right now — each package
is an empty stub. As Phase 1-5 land real modules, this test (or new ones
alongside it) should start asserting real behavior instead of just imports.
"""


def test_package_imports():
    import api  # noqa: F401
    import dft_runner  # noqa: F401
    import perov_core  # noqa: F401
    import pipeline  # noqa: F401
    import surrogate  # noqa: F401
