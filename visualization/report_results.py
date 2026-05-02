"""
Aggregate experiment logs into consolidated result tables, confidence intervals,
and pretrained-vs-scratch significance tests; also generate per-fiber-system
qualitative error-analysis figures.

Usage:
    python visualization/report_results.py
    python visualization/report_results.py --alpha 0.05 --n_show 6
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model


ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
DATASET_DIR = ROOT / "outputs" / "dataset"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"

METRICS = ["nrmse", "wmse", "ssim_score", "median_max_err", "peak_loc_err"]
LOWER_IS_BETTER = {"nrmse", "wmse", "median_max_err", "peak_loc_err"}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_test_onesample_zero(values: list[float]) -> float:
    """Two-sided p-value for H0: mean(values)=0 (fallback normal approx)."""
    n = len(values)
    if n < 2:
        return 1.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std <= 1e-12:
        return 0.0 if abs(mean) > 0 else 1.0
    t = mean / (std / math.sqrt(n))

    try:
        from scipy.stats import t as t_dist
        return float(2.0 * (1.0 - t_dist.cdf(abs(t), df=n - 1)))
    except Exception:
        # Conservative fallback: normal approximation.
        return float(2.0 * (1.0 - _normal_cdf(abs(t))))


def _safe_float(v, default=float("nan")):
    try:
        return float(v)
    except Exception:
        return default


def parse_run_metadata(run_name: str, args: dict) -> dict:
    seed = args.get("seed")
    if seed is None:
        m = re.search(r"_seed(\d+)$", run_name)
        seed = int(m.group(1)) if m else None

    source = None
    m_src = re.search(r"_from_nf(\d+)_", run_name)
    if m_src:
        source = int(m_src.group(1))

    return {
        "run_name": run_name,
        "arch": args.get("arch"),
        "n_fibers": args.get("n_fibers"),
        "seed": seed,
        "train_samples": args.get("train_samples"),
        "transfer_strategy": args.get("transfer_strategy") if args.get("pretrained") else "scratch",
        "pretrained": bool(args.get("pretrained")),
        "source_n_fibers": source,
    }


def load_all_runs() -> list[dict]:
    rows = []
    for log_path in sorted(CKPT_DIR.glob("log_*.json")):
        with open(log_path, "r") as f:
            log = json.load(f)
        run_name = log.get("run_name", log_path.stem.replace("log_", ""))
        args = log.get("args", {})
        meta = parse_run_metadata(run_name, args)
        metrics = log.get("test_metrics", {})

        row = {
            "log_path": str(log_path),
            "checkpoint_path": str(CKPT_DIR / f"best_{run_name}.pth"),
            **meta,
            "best_val_loss": _safe_float(log.get("best_val_loss")),
        }
        for m in METRICS:
            row[m] = _safe_float(metrics.get(m))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        key = (
            r["n_fibers"],
            r["arch"],
            r["pretrained"],
            r["source_n_fibers"],
            r["transfer_strategy"],
            r["train_samples"],
        )
        groups.setdefault(key, []).append(r)

    summary = []
    z = 1.96  # 95% CI
    for key, group in groups.items():
        base = {
            "n_fibers": key[0],
            "arch": key[1],
            "pretrained": key[2],
            "source_n_fibers": key[3],
            "transfer_strategy": key[4],
            "train_samples": key[5],
            "n_runs": len(group),
        }
        for metric in METRICS:
            vals = np.array([_safe_float(g[metric]) for g in group], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                mean = std = ci = float("nan")
            elif len(vals) == 1:
                mean = float(vals[0])
                std = 0.0
                ci = 0.0
            else:
                mean = float(vals.mean())
                std = float(vals.std(ddof=1))
                ci = float(z * std / math.sqrt(len(vals)))
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_ci95"] = ci
        summary.append(base)
    return sorted(summary, key=lambda r: (r["n_fibers"], str(r["arch"]), str(r["transfer_strategy"]), str(r["source_n_fibers"]), str(r["train_samples"])))


def significance_tests(rows: list[dict], alpha: float) -> list[dict]:
    scratch_map = {}
    for r in rows:
        if r["pretrained"]:
            continue
        key = (r["n_fibers"], r["arch"], r["train_samples"], r["seed"])
        scratch_map[key] = r

    buckets = {}
    for r in rows:
        if not r["pretrained"]:
            continue
        key_base = (r["n_fibers"], r["arch"], r["train_samples"], r["seed"])
        scratch = scratch_map.get(key_base)
        if scratch is None:
            continue
        bucket_key = (
            r["n_fibers"],
            r["arch"],
            r["source_n_fibers"],
            r["transfer_strategy"],
            r["train_samples"],
        )
        buckets.setdefault(bucket_key, []).append((scratch, r))

    tests = []
    for bkey, pairs in buckets.items():
        for metric in METRICS:
            gains = []
            for scratch, tl in pairs:
                s = _safe_float(scratch.get(metric))
                t = _safe_float(tl.get(metric))
                if not np.isfinite(s) or not np.isfinite(t):
                    continue
                gain = (s - t) if metric in LOWER_IS_BETTER else (t - s)
                gains.append(gain)

            if not gains:
                continue
            p_val = _t_test_onesample_zero(gains)
            tests.append({
                "n_fibers": bkey[0],
                "arch": bkey[1],
                "source_n_fibers": bkey[2],
                "transfer_strategy": bkey[3],
                "train_samples": bkey[4],
                "metric": metric,
                "n_pairs": len(gains),
                "mean_gain": float(np.mean(gains)),
                "p_value": float(p_val),
                "significant": bool(p_val < alpha),
                "alpha": alpha,
            })
    return sorted(tests, key=lambda r: (r["n_fibers"], str(r["metric"]), str(r["source_n_fibers"]), str(r["transfer_strategy"]), str(r["train_samples"])))


def _choose_best_run_for_nf(rows: list[dict], n_fibers: int) -> dict | None:
    candidates = [r for r in rows if int(r.get("n_fibers", -1)) == n_fibers]
    candidates = [r for r in candidates if np.isfinite(_safe_float(r.get("nrmse")))]
    if not candidates:
        return None
    candidates.sort(key=lambda r: _safe_float(r.get("nrmse")))
    return candidates[0]


def _load_model_from_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ckpt_args = ckpt.get("args", {})

    arch = ckpt_args.get("arch", "attention_unet")
    base_features = ckpt_args.get("base_features", 64)
    dropout = ckpt_args.get("dropout", 0.1)
    resnet_pretrained = bool(ckpt_args.get("resnet_pretrained", False))

    model = build_model(
        arch=arch,
        in_channels=1,
        out_channels=1,
        base_features=base_features,
        dropout=dropout,
        resnet_pretrained=resnet_pretrained,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _load_test_sample_ids_from_log(log_path: Path) -> list[int] | None:
    """Return test sample_ids from a training log, if present."""
    try:
        with open(log_path, "r") as f:
            log = json.load(f)
        split = log.get("split", {})
        test_ids = split.get("test_sample_ids")
        if not test_ids:
            return None
        return [int(x) for x in test_ids]
    except Exception:
        return None


def generate_qualitative_figures(rows: list[dict], n_show: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for nf in [6, 10, 20, 50, 100]:
        best = _choose_best_run_for_nf(rows, nf)
        if best is None:
            continue

        ckpt_path = Path(best["checkpoint_path"])
        data_path = DATASET_DIR / f"nf{nf}.npz"
        if not ckpt_path.exists() or not data_path.exists():
            continue

        model = _load_model_from_checkpoint(ckpt_path).to(device)
        data = np.load(data_path)
        X = data["X"]
        Y = data["Y"]
        y_min = float(data["Y_global_min"])
        y_max = float(data["Y_global_max"])

        sample_ids = data["sample_ids"] if "sample_ids" in data.files else None
        aug_ids    = data["aug_ids"]    if "aug_ids"    in data.files else None

        # Prefer test-set qualitative analysis (avoids train/val contamination).
        sel_idx = np.arange(len(X))
        test_ids = _load_test_sample_ids_from_log(Path(best["log_path"]))
        if test_ids is not None and sample_ids is not None:
            mask = np.isin(sample_ids, np.asarray(test_ids))
            idx = np.nonzero(mask)[0]
            if len(idx):
                sel_idx = idx

        X_sel = X[sel_idx]
        Y_sel = Y[sel_idx]
        sids_sel = sample_ids[sel_idx] if sample_ids is not None else None
        augs_sel = aug_ids[sel_idx] if aug_ids is not None else None

        x_t = torch.tensor(X_sel, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_n = model(x_t).cpu().numpy()

        pred = pred_n * (y_max - y_min) + y_min
        err = np.abs(pred - Y_sel)
        err_max = err.reshape(err.shape[0], -1).max(axis=1)

        top_local = np.argsort(-err_max)[:n_show]

        fig, axes = plt.subplots(len(top_local), 3, figsize=(10, 3.2 * len(top_local)))
        if len(top_local) == 1:
            axes = axes[np.newaxis, :]

        for row_i, j in enumerate(top_local):
            micro = X_sel[j, 0]
            gt = Y_sel[j, 0] / 1e6
            pr = pred[j, 0] / 1e6
            ae = np.abs(gt - pr)
            vmax = max(float(gt.max()), float(pr.max()))

            axes[row_i, 0].imshow(micro, cmap="gray", vmin=0, vmax=1)
            if sids_sel is not None and augs_sel is not None:
                axes[row_i, 0].set_title(f"s{sids_sel[j]} a{augs_sel[j]}")
            else:
                axes[row_i, 0].set_title(f"Index {int(sel_idx[j])}")
            axes[row_i, 0].axis("off")

            im1 = axes[row_i, 1].imshow(gt, cmap="jet", vmin=0, vmax=vmax)
            axes[row_i, 1].set_title("Ground truth (MPa)")
            axes[row_i, 1].axis("off")
            plt.colorbar(im1, ax=axes[row_i, 1], fraction=0.046, pad=0.04)

            im2 = axes[row_i, 2].imshow(ae, cmap="hot")
            axes[row_i, 2].set_title(f"Abs error (MPa), max={ae.max():.2f}")
            axes[row_i, 2].axis("off")
            plt.colorbar(im2, ax=axes[row_i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Qualitative Error Analysis — nf={nf} | best run: {best['run_name']}",
            fontsize=12,
        )
        fig.tight_layout()
        out_path = FIG_DIR / f"qualitative_error_nf{nf}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate logs and build report tables")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--n_show", type=int, default=4,
                        help="Rows in qualitative error figure per fiber system (default: 4)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_all_runs()
    if not rows:
        print("No log files found in outputs/checkpoints; nothing to aggregate.")
        return

    all_cols = [
        "run_name", "log_path", "checkpoint_path", "arch", "n_fibers", "seed",
        "train_samples", "pretrained", "source_n_fibers", "transfer_strategy",
        "best_val_loss", *METRICS,
    ]
    write_csv(REPORT_DIR / "all_runs.csv", rows, all_cols)

    summary = summarize_rows(rows)
    summary_cols = [
        "n_fibers", "arch", "pretrained", "source_n_fibers", "transfer_strategy",
        "train_samples", "n_runs",
    ]
    for m in METRICS:
        summary_cols.extend([f"{m}_mean", f"{m}_std", f"{m}_ci95"])
    write_csv(REPORT_DIR / "summary_metrics.csv", summary, summary_cols)

    tests = significance_tests(rows, alpha=args.alpha)
    test_cols = [
        "n_fibers", "arch", "source_n_fibers", "transfer_strategy", "train_samples",
        "metric", "n_pairs", "mean_gain", "p_value", "significant", "alpha",
    ]
    write_csv(REPORT_DIR / "significance_tests.csv", tests, test_cols)

    print(f"Saved: {REPORT_DIR / 'all_runs.csv'}")
    print(f"Saved: {REPORT_DIR / 'summary_metrics.csv'}")
    print(f"Saved: {REPORT_DIR / 'significance_tests.csv'}")

    generate_qualitative_figures(rows, n_show=max(1, args.n_show))


if __name__ == "__main__":
    main()
