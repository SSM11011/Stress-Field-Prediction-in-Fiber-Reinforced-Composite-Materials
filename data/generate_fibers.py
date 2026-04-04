"""
Random Sequential Addition (RSA) fiber placement for composite microstructures.

Generates non-overlapping circular fiber configurations at ~30% volume fraction
for N-fiber systems. Fiber radius is computed per fiber count to maintain VF.

Usage:
    python data/generate_fibers.py --n_fibers 6  --n_samples 100
    python data/generate_fibers.py --n_fibers 20 --n_samples 100
    python data/generate_fibers.py --n_fibers 50 --n_samples 50
"""

import argparse
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOMAIN = 1.0          # RVE side length (m)
VF_TARGET = 0.30      # Target fiber volume fraction
MIN_GAP = 0.005       # Minimum surface-to-surface gap between fibers (m)
MAX_TRIES = 100_000   # RSA rejection limit per fiber
SEED_BASE = 42        # Reproducibility base seed


def fiber_radius(n_fibers: int) -> float:
    """Compute fiber radius to achieve TARGET_VF for `n_fibers` in unit domain."""
    return math.sqrt(VF_TARGET / (n_fibers * math.pi))


def generate_one_config(n_fibers: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """
    RSA placement for `n_fibers` circles of `radius` in [0,1]^2.

    Returns
    -------
    fibers : ndarray, shape (n_fibers, 2)  — (x, y) centre coordinates
    Raises RuntimeError if placement fails after MAX_TRIES total insertions.
    """
    lo = radius
    hi = DOMAIN - radius
    if lo >= hi:
        raise ValueError(
            f"Fiber radius {radius:.4f} too large for domain {DOMAIN}. "
            f"Reduce n_fibers or VF_TARGET."
        )

    min_dist = 2.0 * radius + MIN_GAP
    centers = []
    total_tries = 0

    while len(centers) < n_fibers:
        if total_tries > MAX_TRIES:
            raise RuntimeError(
                f"RSA failed after {MAX_TRIES} attempts — "
                f"packed {len(centers)}/{n_fibers} fibers. "
                "Reduce VF_TARGET or MIN_GAP."
            )
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        total_tries += 1

        overlap = any(
            math.hypot(x - cx, y - cy) < min_dist for cx, cy in centers
        )
        if not overlap:
            centers.append((x, y))

    return np.array(centers, dtype=np.float64)


def save_config(centers: np.ndarray, out_dir: str, sample_id: int) -> None:
    path = os.path.join(out_dir, f"fibers_{sample_id}.npy")
    np.save(path, centers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RSA fiber configurations")
    parser.add_argument("--n_fibers",  type=int, default=6,
                        choices=[6, 10, 20, 50, 100],
                        help="Number of fibers per RVE (default: 6)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of unique configurations to generate (default: 100)")
    parser.add_argument("--vf",        type=float, default=VF_TARGET,
                        help=f"Fiber volume fraction (default: {VF_TARGET})")
    parser.add_argument("--out_root",  type=str, default="outputs/microstructures",
                        help="Root output directory (default: outputs/microstructures)")
    args = parser.parse_args()

    radius = math.sqrt(args.vf / (args.n_fibers * math.pi))
    actual_vf = args.n_fibers * math.pi * radius ** 2 / DOMAIN ** 2
    out_dir = os.path.join(args.out_root, f"nf{args.n_fibers}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Fiber system  : {args.n_fibers} fibers")
    print(f"Radius        : {radius:.5f} m")
    print(f"Actual VF     : {actual_vf * 100:.1f}%")
    print(f"Samples       : {args.n_samples}")
    print(f"Output dir    : {out_dir}")
    print()

    failures = 0
    for i in range(args.n_samples):
        rng = np.random.default_rng(SEED_BASE + i)
        try:
            centers = generate_one_config(args.n_fibers, radius, rng)
            save_config(centers, out_dir, i)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Generated sample {i + 1:4d}/{args.n_samples}")
        except RuntimeError as exc:
            print(f"  WARNING: sample {i} failed — {exc}", file=sys.stderr)
            failures += 1

    print(f"\nDone. {args.n_samples - failures}/{args.n_samples} samples saved to {out_dir}/")
    if failures:
        print(f"  {failures} sample(s) failed RSA placement.", file=sys.stderr)


if __name__ == "__main__":
    main()
