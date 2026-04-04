"""
End-to-end dataset generation pipeline.

Steps:
  1. Generate RSA fiber microstructures (Python)
  2. Run ABAQUS FEM simulations (Abaqus CAE)
  3. Extract stress + coordinates from .odb files (Abaqus Python)
  4. Interpolate FEM data to 128x128 grids and build .npz dataset (Python)

Usage:
    python data/run_dataset.py --n_fibers 6  --n_samples 100
    python data/run_dataset.py --n_fibers 20 --n_samples 100
    python data/run_dataset.py --n_fibers 50 --n_samples 50

Flags:
    --skip_fiber_gen     Skip Step 1 (fiber generation already done)
    --skip_sim           Skip Steps 2+3 (simulations already run)
    --skip_build         Skip Step 4 (dataset already built)
    --start_id N         Resume from sample N (for interrupted runs)
"""

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: str, check: bool = True) -> int:
    """Print and execute a shell command, return exit code."""
    print(f"  >> {cmd}")
    ret = os.system(cmd)
    if check and ret != 0:
        print(f"  ERROR: command returned {ret}", file=sys.stderr)
    return ret


def abaqus_available() -> bool:
    """Return True if the 'abaqus' executable is on PATH."""
    import subprocess
    try:
        subprocess.run(
            "abaqus information=release",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False, shell=True,
        )
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate composite FEM dataset")
    parser.add_argument("--n_fibers",       type=int, default=6,
                        choices=[6, 10, 20, 50, 100])
    parser.add_argument("--n_samples",      type=int, default=100,
                        help="Number of unique simulations (default: 100)")
    parser.add_argument("--start_id",       type=int, default=0,
                        help="Resume from this sample ID (default: 0)")
    parser.add_argument("--skip_fiber_gen", action="store_true",
                        help="Skip Step 1: fiber generation")
    parser.add_argument("--skip_sim",       action="store_true",
                        help="Skip Steps 2+3: ABAQUS simulation + extraction")
    parser.add_argument("--skip_build",     action="store_true",
                        help="Skip Step 4: dataset building")
    args = parser.parse_args()

    nf = args.n_fibers
    ns = args.n_samples

    print("=" * 60)
    print(f"Dataset Generation Pipeline")
    print(f"  Fiber system : {nf} fibers")
    print(f"  Samples      : {ns}")
    print(f"  Start ID     : {args.start_id}")
    print("=" * 60)

    # ── Step 1: Generate fiber microstructures ──────────────────────────────
    if not args.skip_fiber_gen:
        print("\n[Step 1] Generating fiber configurations ...")
        run(f"python data/generate_fibers.py --n_fibers {nf} --n_samples {ns}")
    else:
        print("\n[Step 1] Skipped (--skip_fiber_gen)")

    # ── Steps 2+3: ABAQUS simulation then stress extraction ─────────────────
    if not args.skip_sim:
        if not abaqus_available():
            print("\nERROR: 'abaqus' not found on PATH — cannot run simulations.",
                  file=sys.stderr)
            sys.exit(1)

        print(f"\n[Step 2+3] Running {ns - args.start_id} simulations ...")
        for i in range(args.start_id, ns):
            print(f"\n--- Sample {i}/{ns - 1} ---")

            # 2a. Run FEM simulation
            sim_ret = run(
                f"abaqus cae noGUI=data/abaqus_simulation.py -- {i} {nf}"
            )
            if sim_ret != 0:
                print(f"  Simulation failed for sample {i} — skipping extraction.")
                continue

            # 2b. Extract stress + coordinates
            run(
                f"abaqus python data/extract_stress_coords.py {i} {nf}"
            )
    else:
        print("\n[Step 2+3] Skipped (--skip_sim)")

    # ── Step 4: Build 128x128 .npz dataset ──────────────────────────────────
    if not args.skip_build:
        print("\n[Step 4] Building interpolated dataset ...")
        run(f"python preprocessing/build_dataset.py --n_fibers {nf}")
    else:
        print("\n[Step 4] Skipped (--skip_build)")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Dataset saved to : outputs/dataset/nf{nf}.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
