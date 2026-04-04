"""
Consolidated visualisation for the composite stress-prediction pipeline.

Four modes — select with --mode:
  raw_fem        Triangulated heat maps from raw FEM extraction (coords + stress .npy)
  dataset        Overview of processed 128x128 .npz dataset (microstructure + stress pairs)
  predictions    Side-by-side GT vs U-Net predictions with error maps
  training_curves  Loss + metric curves from a training log JSON

Usage examples:
  python visualization/visualize.py --mode raw_fem   --n_fibers 6
  python visualization/visualize.py --mode raw_fem   --n_fibers 6 --sample_id 3
  python visualization/visualize.py --mode dataset   --n_fibers 6 --n_show 16
  python visualization/visualize.py --mode predictions --n_fibers 6 \\
          --checkpoint outputs/checkpoints/best_attention_unet_nf6_scratch.pth
  python visualization/visualize.py --mode training_curves \\
          --log outputs/checkpoints/log_attention_unet_nf6_scratch.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # non-interactive backend; remove if Jupyter is used
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# Allow imports from project root when running as: python visualization/visualize.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Configure output directory ────────────────────────────────────────────────
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN = 1.0
VF_TARGET = 0.30


def _fiber_radius(n_fibers: int) -> float:
    return math.sqrt(VF_TARGET / (n_fibers * math.pi))


def _save(fig: plt.Figure, name: str, dpi: int = 200) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# MODE 1: raw_fem
# ===========================================================================
def _load_raw_sample(sid: int, n_fibers: int, stress_root: str, micro_root: str):
    sdir = os.path.join(stress_root, f"nf{n_fibers}")
    mdir = os.path.join(micro_root,  f"nf{n_fibers}")
    s = np.load(os.path.join(sdir, f"stress_{sid}.npy"))
    c = np.load(os.path.join(sdir, f"coords_{sid}.npy"))
    f = np.load(os.path.join(mdir, f"fibers_{sid}.npy"))
    return c, s, f


def _plot_heatmap(ax, coords, stress, fibers, radius,
                  title="", vmin=None, vmax=None):
    """Triangulated von Mises heat map with fiber outlines."""
    x, y = coords[:, 0], coords[:, 1]
    s_mpa = stress / 1e6

    triang = tri.Triangulation(x, y)
    # Mask long-edge triangles (boundary artefacts)
    tri_pts = triang.triangles
    xi, yi = x[tri_pts], y[tri_pts]
    mask = ((xi.max(1) - xi.min(1)) > 0.04) | ((yi.max(1) - yi.min(1)) > 0.04)
    triang.set_mask(mask)

    tc = ax.tripcolor(triang, s_mpa, shading="gouraud",
                      cmap="jet", vmin=vmin, vmax=vmax)
    for fx, fy in fibers:
        ax.add_patch(patches.Circle((fx, fy), radius, fill=False,
                                    edgecolor="white", lw=0.8, ls="--"))
    ax.set_xlim(0, DOMAIN); ax.set_ylim(0, DOMAIN)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return tc


def mode_raw_fem(args):
    """Visualise raw FEM stress maps (triangulated) for one or all samples."""
    nf = args.n_fibers

    # Discover available samples
    sdir = os.path.join(args.stress_root, f"nf{nf}")
    if not os.path.isdir(sdir):
        print(f"ERROR: {sdir} not found. Run extract_stress_coords.py first.")
        sys.exit(1)

    import glob as _g
    coord_files = sorted(
        _g.glob(os.path.join(sdir, "coords_*.npy")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
    )
    all_ids = [int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
               for p in coord_files
               if os.path.exists(os.path.join(sdir,
                   os.path.basename(p).replace("coords", "stress")))]

    if not all_ids:
        print("No samples with both coords and stress found.")
        sys.exit(1)

    radius = _fiber_radius(nf)

    # ── Single sample: high-res individual plot ──────────────────────────────
    if args.sample_id is not None:
        sid = args.sample_id
        coords, stress, fibers = _load_raw_sample(
            sid, nf, args.stress_root, args.micro_root)
        fig, ax = plt.subplots(figsize=(6, 6))
        tc = _plot_heatmap(ax, coords, stress, fibers, radius,
                           title=f"Sample {sid} — Von Mises (MPa)")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04, label="MPa")
        fig.tight_layout()
        _save(fig, f"heatmap_nf{nf}_sample{sid}.png", dpi=250)
        return

    # ── All samples: load once for consistent colour scale ───────────────────
    data = {}
    all_s = []
    for sid in all_ids:
        c, s, f = _load_raw_sample(sid, nf, args.stress_root, args.micro_root)
        data[sid] = (c, s, f)
        all_s.append(s)

    all_s_mpa = np.concatenate(all_s) / 1e6
    vmin = np.percentile(all_s_mpa, 1)
    vmax = np.percentile(all_s_mpa, 99)

    # Individual files
    for sid, (c, s, f) in data.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        tc = _plot_heatmap(ax, c, s, f, radius,
                           title=f"Sample {sid}", vmin=vmin, vmax=vmax)
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04, label="MPa")
        fig.tight_layout()
        _save(fig, f"heatmap_nf{nf}_sample{sid}.png", dpi=200)

    # Grid overview
    n = len(all_ids)
    cols = min(6, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    tc_last = None
    for idx, sid in enumerate(all_ids):
        ax = axes[idx // cols, idx % cols]
        tc_last = _plot_heatmap(ax, *data[sid], radius,
                                title=f"Sample {sid}", vmin=vmin, vmax=vmax)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    if tc_last is not None:
        fig.colorbar(tc_last, ax=axes.ravel().tolist(),
                     fraction=0.02, pad=0.02, label="Von Mises (MPa)")
    fig.suptitle(f"FEM Stress Maps — {nf}-fiber system", fontsize=14)
    fig.tight_layout()
    _save(fig, f"heatmap_grid_nf{nf}.png", dpi=180)

    # Summary statistics (bar charts + histograms)
    means = [d[1].mean() / 1e6 for d in data.values()]
    maxes = [d[1].max()  / 1e6 for d in data.values()]
    stds  = [d[1].std()  / 1e6 for d in data.values()]
    ids   = list(data.keys())

    fig, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, label, color in zip(
            axes2,
            [means, maxes, stds],
            ["Mean Mises (MPa)", "Max Mises (MPa)", "Std Dev (MPa)"],
            ["steelblue", "coral", "mediumpurple"]):
        ax.bar(ids, vals, color=color, edgecolor="white")
        ax.axhline(np.mean(vals), color="red", ls="--", lw=1,
                   label=f"avg = {np.mean(vals):.1f}")
        ax.set_xlabel("Sample ID"); ax.set_ylabel(label)
        ax.legend(fontsize=9)
    fig.suptitle(f"Stress Statistics — {nf}-fiber FEM Dataset", fontsize=13)
    fig.tight_layout()
    _save(fig, f"stress_stats_nf{nf}.png")
    print(f"All figures saved to {FIG_DIR}/")


# ===========================================================================
# MODE 2: dataset
# ===========================================================================
def mode_dataset(args):
    """Overview of a processed 128x128 .npz dataset."""
    npz_path = os.path.join(args.dataset_dir, f"nf{args.n_fibers}.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run build_dataset.py first.")
        sys.exit(1)

    data    = np.load(npz_path)
    X       = data["X"]              # (N, 1, H, W)
    Y       = data["Y"]              # (N, 1, H, W) raw Pa
    sids    = data["sample_ids"]
    augs    = data["aug_ids"]
    Y_min   = float(data["Y_global_min"])
    Y_max   = float(data["Y_global_max"])
    nf      = int(data["n_fibers"])
    n       = len(X)
    n_show  = min(args.n_show, n)

    print(f"Dataset: {npz_path}")
    print(f"  Samples  : {n}  (X: {X.shape}, Y: {Y.shape})")
    print(f"  Stress   : {Y_min:.3e} – {Y_max:.3e} Pa")

    # ── Input/Output pairs ───────────────────────────────────────────────────
    cols = 8
    rows = math.ceil(n_show / cols) * 2   # input row + stress row per group
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes)
    indices = np.linspace(0, n - 1, n_show, dtype=int)

    for i, idx in enumerate(indices):
        col  = i % cols
        rgrp = (i // cols) * 2   # first row of pair
        # Microstructure
        axes[rgrp,     col].imshow(X[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[rgrp,     col].set_title(f"s{sids[idx]} a{augs[idx]}", fontsize=7)
        axes[rgrp,     col].axis("off")
        # Stress
        im = axes[rgrp + 1, col].imshow(Y[idx, 0] / 1e6, cmap="jet")
        axes[rgrp + 1, col].axis("off")

    fig.colorbar(im, ax=axes[-1, :].tolist(), fraction=0.02, pad=0.02,
                 label="Von Mises (MPa)", orientation="horizontal")
    fig.suptitle(f"Dataset Overview — {nf}-fiber | {n} samples "
                 f"(top: microstructure, bottom: stress)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, f"dataset_overview_nf{nf}.png")

    # ── Augmentation check (same base sample, variable augment count) ───────
    aug_count = int(augs.max()) + 1 if len(augs) else 0
    unique_sids = np.unique(sids)
    for uid in unique_sids:
        mask = sids == uid
        if mask.sum() >= aug_count and aug_count > 0:
            aug_check_sid = uid
            aug_indices   = np.where(mask)[0][:aug_count]
            break
    else:
        aug_indices = None

    if aug_indices is not None:
        fig2, axes2 = plt.subplots(2, aug_count, figsize=(2.4 * aug_count, 5))
        if aug_count == 1:
            axes2 = np.array(axes2).reshape(2, 1)
        aug_names = ["orig", "flipLR", "flipUD", "flip180",
                     "rot90", "rot90+LR", "rot90+UD", "rot90+180"][:aug_count]
        for k, idx in enumerate(aug_indices):
            axes2[0, k].imshow(X[idx, 0], cmap="gray", vmin=0, vmax=1)
            axes2[0, k].set_title(aug_names[k], fontsize=8)
            axes2[0, k].axis("off")
            axes2[1, k].imshow(Y[idx, 0] / 1e6, cmap="jet")
            axes2[1, k].axis("off")
        fig2.suptitle(f"{aug_count}-fold Augmentation — sample {aug_check_sid}", fontsize=12)
        fig2.tight_layout()
        _save(fig2, f"augmentation_check_nf{nf}.png")

    # ── Stress distribution histogram ────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(Y.flatten() / 1e6, bins=200, color="steelblue", edgecolor="none")
    ax3.set_xlabel("Von Mises Stress (MPa)")
    ax3.set_ylabel("Pixel count")
    ax3.set_title(f"Global Stress Distribution — {nf}-fiber dataset")
    ax3.axvline(Y_min / 1e6, color="red",  ls="--", lw=1, label=f"min {Y_min/1e6:.1f} MPa")
    ax3.axvline(Y_max / 1e6, color="blue", ls="--", lw=1, label=f"max {Y_max/1e6:.1f} MPa")
    ax3.legend()
    fig3.tight_layout()
    _save(fig3, f"stress_distribution_nf{nf}.png")
    print(f"All figures saved to {FIG_DIR}/")


# ===========================================================================
# MODE 3: predictions
# ===========================================================================
def mode_predictions(args):
    """Compare U-Net predictions against FEM ground truth."""

    if args.checkpoint is None:
        print("ERROR: --checkpoint required for predictions mode.")
        sys.exit(1)

    import torch
    from models.unet import build_model
    from torch.utils.data import TensorDataset, DataLoader

    nf      = args.n_fibers
    npz     = os.path.join(args.dataset_dir, f"nf{nf}.npz")
    data    = np.load(npz)
    X       = torch.tensor(data["X"], dtype=torch.float32)
    Y_raw   = data["Y"]
    Y_min   = float(data["Y_global_min"])
    Y_max   = float(data["Y_global_max"])

    Y_norm  = (Y_raw - Y_min) / (Y_max - Y_min + 1e-10)
    Y_t     = torch.tensor(Y_norm, dtype=torch.float32)

    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    # Infer architecture from checkpoint args if available
    ckpt_args     = ckpt.get("args", {})
    arch          = ckpt_args.get("arch",          "attention_unet")
    base_features = ckpt_args.get("base_features", 64)
    dropout       = ckpt_args.get("dropout",       0.1)
    model = build_model(arch=arch, in_channels=1, out_channels=1,
                        base_features=base_features, dropout=dropout)
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(TensorDataset(X, Y_t), batch_size=16, shuffle=False)
    preds, targets, inputs_list = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu())
            targets.append(yb)
            inputs_list.append(xb)

    pred   = torch.cat(preds).numpy()    # normalised
    target = torch.cat(targets).numpy()
    inp    = torch.cat(inputs_list).numpy()

    # Denormalise to Pa
    pred_pa   = pred   * (Y_max - Y_min) + Y_min
    target_pa = target * (Y_max - Y_min) + Y_min

    n_show = min(args.n_show, len(pred))
    indices = np.random.default_rng(0).choice(len(pred), n_show, replace=False)

    fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Microstructure", "FEM Ground Truth (MPa)",
                  "U-Net Prediction (MPa)", "Abs Error (MPa)"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=10, fontweight="bold")

    for row, idx in enumerate(indices):
        micro  = inp[idx, 0]
        gt     = target_pa[idx, 0] / 1e6
        pr     = pred_pa[idx, 0]   / 1e6
        err    = np.abs(gt - pr)
        vmax   = max(gt.max(), pr.max())

        axes[row, 0].imshow(micro, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")

        im1 = axes[row, 1].imshow(gt,  cmap="jet", vmin=0, vmax=vmax)
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(pr,  cmap="jet", vmin=0, vmax=vmax)
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        im3 = axes[row, 3].imshow(err, cmap="hot")
        axes[row, 3].axis("off")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
        axes[row, 3].set_title(f"max err: {err.max():.2f} MPa", fontsize=7)

    ckpt_name = Path(args.checkpoint).stem
    fig.suptitle(f"Predictions — {arch}, {nf}-fiber system", fontsize=13)
    fig.tight_layout()
    _save(fig, f"predictions_{ckpt_name}.png", dpi=180)

    # Scatter: predicted vs actual (thin sample for speed)
    step = max(1, pred_pa.size // 50_000)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(target_pa.flatten()[::step] / 1e6,
                pred_pa.flatten()[::step]   / 1e6,
                alpha=0.2, s=1, c="steelblue", rasterized=True)
    lim = max(target_pa.max(), pred_pa.max()) / 1e6
    ax2.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    ax2.set_xlabel("FEM Stress (MPa)"); ax2.set_ylabel("Predicted Stress (MPa)")
    ax2.set_title(f"Scatter: Predicted vs FEM — {nf}-fiber")
    ax2.legend(); ax2.set_aspect("equal")
    fig2.tight_layout()
    _save(fig2, f"scatter_{ckpt_name}.png")
    print(f"All figures saved to {FIG_DIR}/")


# ===========================================================================
# MODE 4: training_curves
# ===========================================================================
def mode_training_curves(args):
    """Plot loss and metric curves from a training log JSON."""
    if args.log is None:
        print("ERROR: --log required for training_curves mode.")
        sys.exit(1)
    with open(args.log) as f:
        log = json.load(f)

    history  = log["history"]
    epochs   = range(1, len(history["train"]) + 1)
    train_l  = [h["loss"]      for h in history["train"]]
    val_l    = [h["loss"]      for h in history["val"]]
    train_s  = [1 - h["ssim_loss"] for h in history["train"]]
    val_s    = [1 - h["ssim_loss"] for h in history["val"]]
    lr_hist  = history.get("lr", [None] * len(epochs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_l, label="Train", lw=2)
    axes[0].plot(epochs, val_l,   label="Val",   lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Combined Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_s, label="Train SSIM", lw=2, color="green")
    axes[1].plot(epochs, val_s,   label="Val SSIM",   lw=2, color="orange")
    axes[1].axhline(0.95, color="red", ls="--", lw=1, label="Target 0.95")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM During Training"); axes[1].legend(); axes[1].grid(alpha=0.3)

    if any(v is not None for v in lr_hist):
        axes[2].plot(epochs, lr_hist, lw=2, color="purple")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("LR Schedule (cosine)"); axes[2].grid(alpha=0.3)
        axes[2].set_yscale("log")
    else:
        axes[2].axis("off")

    run = log.get("run_name", Path(args.log).stem)
    met = log.get("test_metrics", {})
    subtitle = (f"Test NRMSE={met.get('nrmse', 0)*100:.2f}%  "
                f"SSIM={met.get('ssim_score', 0):.4f}  "
                f"Peak LocErr={met.get('peak_loc_err', 0):.1f}px")
    fig.suptitle(f"{run}\n{subtitle}", fontsize=11)
    fig.tight_layout()

    out_name = f"training_curves_{run}.png"
    _save(fig, out_name)
    print(f"All figures saved to {FIG_DIR}/")


# ===========================================================================
# CLI entry point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualise pipeline outputs (FEM data, dataset, predictions, curves)")

    parser.add_argument("--mode", required=True,
                        choices=["raw_fem", "dataset", "predictions", "training_curves"],
                        help="Visualisation mode")

    # Shared
    parser.add_argument("--n_fibers",    type=int, default=6,
                        choices=[6, 10, 20, 50, 100])
    parser.add_argument("--n_show",      type=int, default=16,
                        help="Number of samples to show in grid plots (default: 16)")

    # raw_fem
    parser.add_argument("--sample_id",  type=int, default=None,
                        help="[raw_fem] Single sample to plot (default: all)")
    parser.add_argument("--stress_root", type=str,
                        default="outputs/stress_maps")
    parser.add_argument("--micro_root",  type=str,
                        default="outputs/microstructures")

    # dataset
    parser.add_argument("--dataset_dir", type=str, default="outputs/dataset")

    # predictions
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[predictions] Path to .pth checkpoint")

    # training_curves
    parser.add_argument("--log",  type=str, default=None,
                        help="[training_curves] Path to JSON training log")

    args = parser.parse_args()

    dispatch = {
        "raw_fem":        mode_raw_fem,
        "dataset":        mode_dataset,
        "predictions":    mode_predictions,
        "training_curves": mode_training_curves,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
