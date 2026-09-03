"""
main.py
Runs the full Perov-ActiveLearn closed loop end-to-end, on the offline
benchmark evaluator by default, and prints/saves real results so you can
see the whole pipeline work together for the first time — not just pass
tests in isolation.

Usage:
    python main.py
    python main.py --target 1.9 --iterations 25
    python main.py --live          # try the Materials Project evaluator first
                                    # (falls back to offline automatically)
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from dft_runner.benchmark_engine import BenchmarkEvaluationBackend
from dft_runner.materials_project_backend import MaterialsProjectEvaluationBackend
from perov_core.space import generate_composition_space
from pipeline.loop import ActiveLearningLoop, run_random_search_baseline
from pipeline.visualizer import plot_bo_vs_random, plot_optimization_trajectory, plot_parity

# Sparse-data GP fits routinely push a feature's length-scale to the search
# boundary early on (the model saying "not enough data yet to tell if this
# dimension matters") — expected behavior, not a bug, and it self-corrects
# as more points come in. Suppressed here so it doesn't drown out the
# actual run output; still visible if you run with `python -W default main.py`.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def build_search_space(resolution: float):
    return generate_composition_space(
        a_site_ions=["Cs", "MA", "FA"],
        b_site_ions=["Pb", "Sn"],
        x_site_ions=["Cl", "Br", "I"],
        resolution=resolution,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the Perov-ActiveLearn closed loop.")
    parser.add_argument("--target", type=float, default=1.9, help="Target bandgap in eV.")
    parser.add_argument("--iterations", type=int, default=20, help="Loop iterations after seeding.")
    parser.add_argument(
        "--n-initial", type=int, default=5, help="Random seed points before the GP kicks in."
    )
    parser.add_argument(
        "--resolution", type=float, default=0.1, help="Composition grid resolution."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use MaterialsProjectEvaluationBackend (tries a real MP lookup, "
             "falls back to offline automatically) instead of the pure offline evaluator.",
    )
    parser.add_argument(
        "--compare-random", action="store_true",
        help="Also run a random-search baseline on the same budget and plot both.",
    )
    parser.add_argument("--outdir", type=str, default="outputs", help="Where to save plots.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Perov-ActiveLearn — target bandgap: {args.target} eV")
    print(f"Building composition search space (resolution={args.resolution})...")
    space = build_search_space(args.resolution)
    print(f"  {len(space)} candidate compositions generated.\n")

    evaluator = MaterialsProjectEvaluationBackend() if args.live else BenchmarkEvaluationBackend()
    evaluator_note = (
        " (live-first, falls back to offline automatically)"
        if args.live
        else " (offline reference book)"
    )
    print(f"Evaluator: {type(evaluator).__name__}{evaluator_note}")

    loop = ActiveLearningLoop(
        target_bandgap_eV=args.target,
        candidate_space=space,
        evaluator=evaluator,
        n_initial=args.n_initial,
    )

    print(f"\nSeeding {args.n_initial} random points, then running up to {args.iterations} "
          f"active-learning iterations...\n")
    print(f"{'Eval #':>7}  {'Bandgap (eV)':>13}  {'|Error| (eV)':>13}  {'Best so far':>12}  Note")
    print("-" * 68)

    start = time.time()
    result = loop.run(n_iterations=args.iterations)
    elapsed = time.time() - start

    for i, h in enumerate(loop.history):
        if h.iteration == 0:
            note = "seed"
        elif h.was_random_exploration:
            note = "random explore"
        else:
            note = ""
        print(
            f"{i:>7}  {h.bandgap_eV:>13.4f}  {h.error_eV:>13.4f}  "
            f"{h.best_error_so_far_eV:>12.4f}  {note}"
        )

    print("-" * 68)
    print(f"\nDone in {elapsed:.2f}s, {result['n_evaluations']} total evaluations.")
    print(f"Best composition found: bandgap = {result['best_bandgap_eV']:.4f} eV "
          f"(target {args.target} eV, error {result['best_error_eV']:.4f} eV)")
    comp = result["best_composition"]
    print(f"  A-site: {comp.a_site}")
    print(f"  B-site: {comp.b_site}")
    print(f"  X-site: {comp.x_site}")

    traj_path = outdir / "trajectory.png"
    plot_optimization_trajectory(loop.history, args.target, save_path=traj_path)
    print(f"\nSaved optimization trajectory plot -> {traj_path}")

    X_obs = np.array(loop.X_observed)
    y_obs = np.array(loop.y_observed)
    parity_path = outdir / "parity.png"
    plot_parity(loop.surrogate, X_obs, y_obs, save_path=parity_path)
    print(f"Saved parity plot -> {parity_path}")

    if args.compare_random:
        print(
            f"\nRunning random-search baseline on the same budget "
            f"({result['n_evaluations']} evals)..."
        )
        random_evaluator = BenchmarkEvaluationBackend()
        random_result = run_random_search_baseline(
            target_bandgap_eV=args.target,
            candidate_space=space,
            evaluator=random_evaluator,
            n_evaluations=result["n_evaluations"],
        )
        print(f"  Random search best error: {random_result['best_error_eV']:.4f} eV")
        print(f"  Active learning best error: {result['best_error_eV']:.4f} eV")
        compare_path = outdir / "bo_vs_random.png"
        plot_bo_vs_random(
            loop.history, random_result["history"], args.target, save_path=compare_path
        )
        print(f"  Saved comparison plot -> {compare_path}")


if __name__ == "__main__":
    main()
