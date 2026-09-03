"""
diagnose_mp.py
Standalone check for the live Materials Project connection, isolated from
the search loop entirely. Run this directly when `--live` in main.py
doesn't seem to be doing anything — it prints exactly what's happening
at each step instead of silently falling back.

Usage:
    python diagnose_mp.py
"""

import os

from dft_runner.materials_project_backend import (
    MaterialsProjectEvaluationBackend,
    _ensure_dotenv_loaded,
)
from perov_core.descriptors import PerovskiteComposition


def main():
    print("1. Checking for MP_API_KEY...")
    _ensure_dotenv_loaded()
    key = os.environ.get("MP_API_KEY")
    if not key:
        print("   FAILED: MP_API_KEY not found in the environment or .env file.")
        print("   Fix: cp .env.example .env, then add MP_API_KEY=yourkey to it.")
        return
    print(f"   OK — key loaded (starts with {key[:4]}..., {len(key)} chars total)")

    print("\n2. Checking mp-api is installed...")
    try:
        import mp_api  # noqa: F401
    except ImportError:
        print("   FAILED: mp-api is not installed.")
        print("   Fix: pip install -e \".[live]\"")
        return
    print("   OK — mp-api is installed")

    print("\n3. Running a live query for CsPbI3 (a composition inside the live-searchable")
    print("   scope: pure Cs on A-site, one ion per site) with debug=True so any error")
    print("   is printed instead of silently swallowed...")
    backend = MaterialsProjectEvaluationBackend(debug=True)
    comp = PerovskiteComposition(a_site={"Cs": 1.0}, b_site={"Pb": 1.0}, x_site={"I": 1.0})
    bandgap = backend.evaluate(comp)

    print(f"\n4. Result: bandgap = {bandgap:.4f} eV")
    print(f"   Source: {backend.last_source}")
    if backend.last_source == "live":
        print("   SUCCESS — this came from a real Materials Project query.")
    else:
        print("   This came from the offline fallback, not a live query.")
        print("   Look at the [MP debug] line(s) printed above step 4 for the reason why.")


if __name__ == "__main__":
    main()
