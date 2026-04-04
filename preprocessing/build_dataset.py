"""
Preprocess raw ABAQUS FEM output into 128x128 training images.

For each sample:
  1. Load scattered element centroid coordinates + von Mises stress.
  2. Interpolate onto a regular 128x128 grid (linear, NaN-filled by nearest).
  3. Render the binary fiber microstructure image at 128x128.
    4. Apply configurable geometric augmentation (none / 4-fold / 8-fold) and optional elastic warp.

Output .npz format (compatible with training/train_unet.py):
  X          : (N, 1, 128, 128) float32  — binary microstructure [0, 1]
  Y          : (N, 1, 128, 128) float32  — von Mises stress (raw, Pa)
  sample_ids : (N,)             int32    — original sample index (pre-aug)
  aug_ids    : (N,)             int32    — augmentation index 0-7
  n_fibers   : scalar int
  Y_global_min : scalar float  — for denormalization
  Y_global_max : scalar float  — for denormalization
  fiber_radius : scalar float

Usage:
    python preprocessing/build_dataset.py --n_fibers 6
    python preprocessing/build_dataset.py --n_fibers 20 --no_aug
    python preprocessing/build_dataset.py --n_fibers 6 --grid_size 256
"""

import argparse
import math
import os
import sys

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN     = 1.0
VF_TARGET  = 0.30
GRID_SIZE  = 128          # Default; override with --grid_size
N_AUG_4    = 4            # 4-fold flips
N_AUG_8    = 8            # 8-fold augmentation (4 flips × {0, 90deg})


def fiber_radius(n_fibers: int) -> float:
    return math.sqrt(VF_TARGET / (n_fibers * math.pi))


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def interpolate_to_grid(coords: np.ndarray, stress: np.ndarray,
                        grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Interpolate scattered FEM element-centroid stress values onto a regular grid.

    Parameters
    ----------
    coords : (N_elem, 2)  — (x, y) centroid positions in [0, 1]
    stress : (N_elem,)    — von Mises stress values (Pa)
    grid_size : int

    Returns
    -------
    grid : (grid_size, grid_size) float32
        Row 0 → y ≈ 0 (domain bottom); row -1 → y ≈ 1 (domain top).
    """
    xi = np.linspace(0, DOMAIN, grid_size)
    yi = np.linspace(0, DOMAIN, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)   # (H, W)

    # Primary: linear interpolation (accurate inside convex hull)
    stress_grid = griddata(coords, stress, (grid_x, grid_y), method="linear")

    # Fill exterior NaNs with nearest-neighbour (avoids artefacts at edges)
    nan_mask = np.isnan(stress_grid)
    if nan_mask.any():
        stress_nn = griddata(coords, stress, (grid_x, grid_y), method="nearest")
        stress_grid[nan_mask] = stress_nn[nan_mask]

    return stress_grid.astype(np.float32)


# ---------------------------------------------------------------------------
# Binary microstructure rendering
# ---------------------------------------------------------------------------
def render_microstructure(fibers: np.ndarray, radius: float,
                          grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Rasterise circular fibers onto a binary image.

    Returns
    -------
    micro : (grid_size, grid_size) float32  — 1.0 inside fiber, 0.0 matrix
    """
    px = (np.arange(grid_size) + 0.5) / grid_size  # pixel centres in [0,1]
    XX, YY = np.meshgrid(px, px)                   # (H, W)
    micro = np.zeros((grid_size, grid_size), dtype=np.float32)
    for fx, fy in fibers:
        dist2 = (XX - fx) ** 2 + (YY - fy) ** 2
        micro[dist2 < radius ** 2] = 1.0
    return micro


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def _flip_variants(arr: np.ndarray):
    """Return original + LR + UD + LR+UD variants."""
    return [
        arr.copy(),
        np.fliplr(arr).copy(),
        np.flipud(arr).copy(),
        np.fliplr(np.flipud(arr)).copy(),
    ]


def elastic_deform_pair(micro: np.ndarray,
                        stress: np.ndarray,
                        alpha: float,
                        sigma: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the same smooth random elastic warp to both microstructure and stress.

    Parameters
    ----------
    alpha : displacement scale in pixels
    sigma : Gaussian smoothing for displacement fields in pixels
    """
    h, w = micro.shape
    dx = gaussian_filter((rng.random((h, w)) * 2.0 - 1.0), sigma=sigma, mode="reflect")
    dy = gaussian_filter((rng.random((h, w)) * 2.0 - 1.0), sigma=sigma, mode="reflect")
    dx *= alpha
    dy *= alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    sample_x = np.clip(x + dx, 0, w - 1)
    sample_y = np.clip(y + dy, 0, h - 1)
    coords = np.vstack([sample_y.ravel(), sample_x.ravel()])

    # Binary map uses nearest interpolation to preserve sharp interfaces.
    micro_warp = map_coordinates(micro, coords, order=0, mode="nearest").reshape(h, w)
    # Stress map uses linear interpolation for smooth fields.
    stress_warp = map_coordinates(stress, coords, order=1, mode="nearest").reshape(h, w)
    return micro_warp.astype(np.float32), stress_warp.astype(np.float32)


def build_augmented_pairs(micro: np.ndarray,
                          stress: np.ndarray,
                          aug_mode: str,
                          use_elastic: bool,
                          elastic_alpha: float,
                          elastic_sigma: float,
                          rng: np.random.Generator):
    """Generate augmented (micro, stress) pairs according to the selected mode."""
    if aug_mode == "none":
        pairs = [(0, micro.copy(), stress.copy())]
    else:
        rots = [False] if aug_mode == "4fold" else [False, True]
        pairs = []
        aug_id = 0
        for rot in rots:
            m0 = np.rot90(micro, 1) if rot else micro
            s0 = np.rot90(stress, 1) if rot else stress
            for m_v, s_v in zip(_flip_variants(m0), _flip_variants(s0)):
                if use_elastic:
                    m_v, s_v = elastic_deform_pair(
                        m_v, s_v,
                        alpha=elastic_alpha,
                        sigma=elastic_sigma,
                        rng=rng,
                    )
                pairs.append((aug_id, m_v.astype(np.float32), s_v.astype(np.float32)))
                aug_id += 1
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build 128×128 npz dataset from FEM output")
    parser.add_argument("--n_fibers",   type=int,   default=6,
                        choices=[6, 10, 20, 50, 100])
    parser.add_argument("--grid_size",  type=int,   default=GRID_SIZE,
                        help=f"Output image size (default: {GRID_SIZE})")
    parser.add_argument("--no_aug",     action="store_true",
                        help="Disable augmentation (legacy alias of --aug_mode none)")
    parser.add_argument("--aug_mode", type=str, default="4fold",
                        choices=["none", "4fold", "8fold"],
                        help="Geometric augmentation mode (default: 4fold)")
    parser.add_argument("--elastic", action="store_true",
                        help="Apply elastic deformation after geometric transforms")
    parser.add_argument("--elastic_alpha", type=float, default=2.0,
                        help="Elastic displacement scale in pixels (default: 2.0)")
    parser.add_argument("--elastic_sigma", type=float, default=6.0,
                        help="Elastic smoothing sigma in pixels (default: 6.0)")
    parser.add_argument("--elastic_seed", type=int, default=1234,
                        help="Random seed for elastic deformation (default: 1234)")
    parser.add_argument("--radius",      type=float, default=None,
                        help="Fiber radius (m). Defaults to VF=30%% computed value. "
                             "Pass --radius 0.04 if using legacy data generated with old scripts.")
    parser.add_argument("--stress_root", type=str, default="outputs/stress_maps",
                        help="Root dir for raw stress/coord .npy files")
    parser.add_argument("--micro_root",  type=str, default="outputs/microstructures",
                        help="Root dir for fiber .npy files")
    parser.add_argument("--out_dir",     type=str, default="outputs/dataset",
                        help="Output directory for .npz files (default: outputs/dataset)")
    args = parser.parse_args()

    nf          = args.n_fibers
    gs          = args.grid_size
    aug_mode    = "none" if args.no_aug else args.aug_mode
    radius      = args.radius if args.radius is not None else fiber_radius(nf)
    stress_dir  = os.path.join(args.stress_root, f"nf{nf}")
    micro_dir   = os.path.join(args.micro_root,  f"nf{nf}")
    radius_src  = "user-specified" if args.radius else f"VF={VF_TARGET*100:.0f}% computed"
    actual_vf   = nf * math.pi * radius ** 2 / DOMAIN ** 2
    if aug_mode == "none":
        n_aug_folds = 1
    elif aug_mode == "4fold":
        n_aug_folds = N_AUG_4
    else:
        n_aug_folds = N_AUG_8
    rng = np.random.default_rng(args.elastic_seed)

    if not os.path.isdir(stress_dir):
        print(f"ERROR: stress directory not found: {stress_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Fiber system  : {nf} fibers")
    print(f"Fiber radius  : {radius:.5f} m  ({radius_src}, VF={actual_vf*100:.1f}%)")
    print(f"Grid size     : {gs}x{gs}")
    elastic_desc = " + elastic" if args.elastic and aug_mode != "none" else ""
    print(f"Augmentation  : {aug_mode}{elastic_desc}")

    # ── Discover available samples ──────────────────────────────────────────
    import glob as _glob
    stress_files = sorted(
        _glob.glob(os.path.join(stress_dir, "stress_*.npy")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
    )
    if not stress_files:
        print(f"ERROR: no stress files in {stress_dir}", file=sys.stderr)
        sys.exit(1)

    sample_ids_found = [
        int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
        for p in stress_files
    ]
    print(f"Found {len(sample_ids_found)} sample(s) in {stress_dir}")

    # ── Pre-scan to find global stress range (for normalisation metadata) ──
    print("Scanning stress range ...")
    global_min =  float("inf")
    global_max = -float("inf")
    for sid in sample_ids_found:
        s = np.load(os.path.join(stress_dir, f"stress_{sid}.npy"))
        global_min = min(global_min, float(s.min()))
        global_max = max(global_max, float(s.max()))
    print(f"  Global stress range: {global_min:.3e} – {global_max:.3e} Pa")

    # ── Build arrays ────────────────────────────────────────────────────────
    n_total = len(sample_ids_found) * n_aug_folds
    X_all   = np.zeros((n_total, 1, gs, gs), dtype=np.float32)
    Y_all   = np.zeros((n_total, 1, gs, gs), dtype=np.float32)
    sid_out = np.zeros(n_total, dtype=np.int32)
    aug_out = np.zeros(n_total, dtype=np.int32)

    write_idx = 0
    for progress, sid in enumerate(sample_ids_found):
        coords_path = os.path.join(stress_dir, f"coords_{sid}.npy")
        stress_path = os.path.join(stress_dir, f"stress_{sid}.npy")
        fibers_path = os.path.join(micro_dir,  f"fibers_{sid}.npy")

        if not os.path.exists(coords_path):
            print(f"  WARNING: coords missing for sample {sid} — skipping.")
            continue
        if not os.path.exists(fibers_path):
            print(f"  WARNING: fibers missing for sample {sid} — skipping.")
            continue

        coords  = np.load(coords_path)   # (N_elem, 2)
        stress  = np.load(stress_path)   # (N_elem,)
        fibers  = np.load(fibers_path)   # (n_fibers, 2)

        # Interpolate stress to regular grid
        stress_grid = interpolate_to_grid(coords, stress, gs)   # (H, W)

        # Binary microstructure image
        micro_img  = render_microstructure(fibers, radius, gs)  # (H, W)

        # Augmentation
        aug_pairs = build_augmented_pairs(
            micro_img,
            stress_grid,
            aug_mode=aug_mode,
            use_elastic=bool(args.elastic and aug_mode != "none"),
            elastic_alpha=args.elastic_alpha,
            elastic_sigma=args.elastic_sigma,
            rng=rng,
        )

        for aug_idx, m, s in aug_pairs:
            X_all[write_idx, 0] = m
            Y_all[write_idx, 0] = s
            sid_out[write_idx]  = sid
            aug_out[write_idx]  = aug_idx
            write_idx += 1

        if (progress + 1) % 10 == 0 or progress == 0:
            print(f"  Processed {progress + 1}/{len(sample_ids_found)} samples ...")

    # Trim in case some samples were skipped
    X_all   = X_all[:write_idx]
    Y_all   = Y_all[:write_idx]
    sid_out = sid_out[:write_idx]
    aug_out = aug_out[:write_idx]

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, f"nf{nf}.npz")
    np.savez_compressed(
        out_path,
        X=X_all,
        Y=Y_all,
        sample_ids=sid_out,
        aug_ids=aug_out,
        n_fibers=np.int32(nf),
        fiber_radius=np.float32(radius),
        Y_global_min=np.float32(global_min),
        Y_global_max=np.float32(global_max),
        grid_size=np.int32(gs),
    )
    print(f"\nDataset saved: {out_path}")
    print(f"  X shape : {X_all.shape}   (N, 1, H, W)")
    print(f"  Y shape : {Y_all.shape}")
    print(f"  Samples : {len(sample_ids_found)} unique × {n_aug_folds} aug = {write_idx} total")
    print(f"  Stress  : {global_min:.3e} – {global_max:.3e} Pa")


if __name__ == "__main__":
    main()
