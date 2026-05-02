"""
Enhanced U-Net training pipeline for composite von Mises stress prediction.

Improvements over the base paper (Jiang et al.):
  • Attention gates on skip connections   (AttentionUNet)
    • Optional ResNet-34 encoder backbone   (ResNet34AttentionUNet)
  • Combined loss: α·WeightedMSE + β·(1 − SSIM)
  • AdamW optimizer with cosine-annealing LR schedule
    • Configurable geometric/elastic augmentation (already baked into .npz)
  • Transfer learning: --pretrained <path> freezes encoder for fine-tuning

Dataset format expected (from preprocessing/build_dataset.py):
  .npz with keys: X, Y, Y_global_min, Y_global_max, n_fibers

Usage:
    # Train on 6-fiber data from scratch
    python training/train_unet.py --n_fibers 6

    # Fine-tune on 20-fiber data using pretrained 6-fiber encoder
    python training/train_unet.py --n_fibers 20 --pretrained outputs/checkpoints/best_nf6.pth

    # Ablation: baseline U-Net without attention
    python training/train_unet.py --n_fibers 6 --arch unet --alpha 1.0 --beta 0.0
"""

import sys
import os
import json
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# Allow imports from project root when running as: python training/train_unet.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model


# ============================================================================
# Dataset
# ============================================================================
class StressDataset(Dataset):
    """
    Loads a pre-built .npz dataset and normalises stress to [0, 1].

    The binary microstructure X is already in [0, 1] (binary float32).
    Normalisation parameters (Y_global_min, Y_global_max) are stored in the
    .npz so predictions can be denormalised back to Pa.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 Y_min: float, Y_max: float):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-10)
        self.Y = torch.as_tensor(Y_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_dataset(npz_path: str):
    data    = np.load(npz_path)
    X       = data["X"]                          # (N, 1, H, W)
    Y       = data["Y"]                          # (N, 1, H, W) raw Pa
    Y_min   = float(data["Y_global_min"])
    Y_max   = float(data["Y_global_max"])
    return X, Y, Y_min, Y_max


# ============================================================================
# Loss functions
# ============================================================================
class WeightedMSELoss(nn.Module):
    """
    Pixel-wise MSE weighted by true stress magnitude.
    Higher weights on high-stress (failure-critical) regions.

    L_wmse = mean( w(y_true) * (y_pred - y_true)^2 )
    where w(y_true) = 1 + gamma * y_true
    """

    def __init__(self, gamma: float = 4.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.gamma * target          # emphasise high-stress pixels
        return (weight * (pred - target) ** 2).mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss — maximises perceptual quality.
    L_ssim = 1 − SSIM(pred, target)

    Uses a Gaussian window over local patches.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        self.register_buffer(
            "kernel",
            kernel.unsqueeze(0).unsqueeze(0)   # (1, 1, W, W)
        )
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return torch.outer(g, g)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Works on (B, 1, H, W) tensors
        pad = self.window_size // 2
        kernel = self.kernel.to(pred.device)

        mu1    = nn.functional.conv2d(pred,   kernel, padding=pad, groups=1)
        mu2    = nn.functional.conv2d(target, kernel, padding=pad, groups=1)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12   = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(pred   ** 2, kernel, padding=pad, groups=1) - mu1_sq
        sigma2_sq = nn.functional.conv2d(target ** 2, kernel, padding=pad, groups=1) - mu2_sq
        sigma12   = nn.functional.conv2d(pred * target, kernel, padding=pad, groups=1) - mu12

        ssim_map = ((2 * mu12 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """L = alpha * WeightedMSE + beta * (1 − SSIM)"""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 4.0):
        super().__init__()
        self.alpha   = alpha
        self.beta    = beta
        self.wmse    = WeightedMSELoss(gamma=gamma)
        self.ssim    = SSIMLoss()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l_wmse = self.wmse(pred, target)
        l_ssim = self.ssim(pred, target)
        total  = self.alpha * l_wmse + self.beta * l_ssim
        return total, {"wmse": l_wmse.item(), "ssim_loss": l_ssim.item()}


# ============================================================================
# Evaluation metrics
# ============================================================================
@torch.no_grad()
def compute_metrics(model: nn.Module, loader: DataLoader,
                    device: torch.device,
                    Y_min: float, Y_max: float,
                    gamma: float = 4.0) -> dict:
    """
    Compute all five project metrics on a DataLoader.
    Returns dict with: nrmse, wmse, ssim_score, median_max_err, peak_loc_err
    """
    model.eval()
    stress_range = Y_max - Y_min + 1e-10

    all_pred, all_tgt = [], []
    for X_b, Y_b in loader:
        X_b = X_b.to(device)
        all_pred.append(model(X_b).cpu())
        all_tgt.append(Y_b)

    pred = torch.cat(all_pred)   # (N, 1, H, W) normalised
    tgt  = torch.cat(all_tgt)

    # Denormalise back to Pa-scale for engineering metrics
    pred_pa = pred * (Y_max - Y_min) + Y_min
    tgt_pa  = tgt  * (Y_max - Y_min) + Y_min

    err = pred_pa - tgt_pa                          # (N, 1, H, W)

    # 1. Normalised RMSE (target < 3%)
    nrmse = torch.sqrt((err ** 2).mean()).item() / stress_range

    # 2. Weighted MSE (normalised space, matches training loss)
    weight  = 1.0 + gamma * tgt
    wmse    = (weight * (pred - tgt) ** 2).mean().item()

    # 3. Median max absolute error across samples (as fraction of stress range)
    max_errs = torch.abs(err).flatten(1).max(dim=1).values   # (N,)
    median_max_err = torch.median(max_errs).item() / stress_range

    # 4. SSIM score (target > 0.95)
    ssim_fn    = SSIMLoss()
    ssim_loss  = ssim_fn(pred.cpu(), tgt.cpu()).item()
    ssim_score = 1.0 - ssim_loss

    # 5. Peak location error (Euclidean pixel distance, target < 5 px)
    N, _, H, W = pred_pa.shape
    pred_flat = pred_pa.view(N, -1)
    tgt_flat  = tgt_pa.view(N, -1)
    pred_peak_idx = pred_flat.argmax(dim=1)
    tgt_peak_idx  = tgt_flat.argmax(dim=1)
    pred_row = pred_peak_idx // W;  pred_col = pred_peak_idx % W
    tgt_row  = tgt_peak_idx  // W;  tgt_col  = tgt_peak_idx  % W
    peak_loc_err = torch.sqrt(
        (pred_row.float() - tgt_row.float()) ** 2 +
        (pred_col.float() - tgt_col.float()) ** 2
    ).mean().item()

    return {
        "nrmse":          nrmse,
        "wmse":           wmse,
        "ssim_score":     ssim_score,
        "median_max_err": median_max_err,
        "peak_loc_err":   peak_loc_err,
    }


# ============================================================================
# Training loop
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    total_loss = wmse_sum = ssim_sum = 0.0
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b)
        loss, parts = criterion(pred, Y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        wmse_sum   += parts["wmse"]
        ssim_sum   += parts["ssim_loss"]

    n = len(loader)
    return {"loss": total_loss / n, "wmse": wmse_sum / n, "ssim_loss": ssim_sum / n}


@torch.no_grad()
def eval_loss(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = wmse_sum = ssim_sum = 0.0
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        loss, parts = criterion(model(X_b), Y_b)
        total_loss += loss.item()
        wmse_sum   += parts["wmse"]
        ssim_sum   += parts["ssim_loss"]

    n = len(loader)
    return {"loss": total_loss / n, "wmse": wmse_sum / n, "ssim_loss": ssim_sum / n}


# ============================================================================
# Transfer-learning utilities
# ============================================================================
def load_pretrained_encoder(model: nn.Module, ckpt_path: str) -> None:
    """Copy encoder weights from a pretrained checkpoint (different fiber system)."""
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    encoder_keys = [
        k for k in state
        if k.startswith(("enc", "pool", "bottleneck", "stem", "layer1", "layer2", "layer3", "layer4"))
    ]
    model_state  = model.state_dict()
    loaded, skipped = 0, 0
    for k in encoder_keys:
        if k in model_state and model_state[k].shape == state[k].shape:
            model_state[k] = state[k]
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    print(f"  Pretrained encoder: loaded {loaded} tensors, skipped {skipped}.")


def load_pretrained_all(model: nn.Module, ckpt_path: str) -> None:
    """Load all shape-compatible tensors from a pretrained checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()

    loaded, skipped = 0, 0
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    print(f"  Pretrained full model: loaded {loaded} tensors, skipped {skipped}.")


def freeze_encoder(model: nn.Module) -> None:
    """Freeze all encoder + bottleneck weights."""
    for name, param in model.named_parameters():
        if name.startswith((
            "enc", "pool", "bottleneck", "stem", "layer1", "layer2", "layer3", "layer4"
        )):
            param.requires_grad = False

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen: {frozen:,} params | Trainable: {trainable:,} params")


def freeze_all_except_last_decoder(model: nn.Module) -> None:
    """Freeze everything except the last decoder stage and prediction head."""
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_tokens = ("dec1", "conv1", "up1", "cv1", "up0", "head")
    for name, param in model.named_parameters():
        if any(tok in name for tok in trainable_tokens):
            param.requires_grad = True

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Last-decoder tuning: Frozen {frozen:,} | Trainable {trainable:,} params")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train enhanced U-Net for stress prediction")

    # Data
    parser.add_argument("--n_fibers",   type=int,   default=6,
                        choices=[6, 10, 20, 50, 100])
    parser.add_argument("--dataset_dir", type=str,  default="outputs/dataset")
    parser.add_argument("--train_frac",  type=float, default=0.80)
    parser.add_argument("--val_frac",    type=float, default=0.10)
    parser.add_argument("--train_samples", type=int, default=None,
                        help="Limit number of training samples (for low-data transfer sweeps)")

    # Model
    parser.add_argument("--arch",          type=str,  default="attention_unet",
                        choices=["unet", "attention_unet", "resnet34_attention_unet", "cnn"])
    parser.add_argument("--base_features", type=int,  default=64)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--resnet_pretrained", action="store_true",
                        help="Use ImageNet-pretrained ResNet34 encoder for resnet34_attention_unet")

    # Transfer learning
    parser.add_argument("--pretrained",    type=str,  default=None,
                        help="Path to pretrained checkpoint (encoder will be loaded + frozen)")
    parser.add_argument("--unfreeze_epoch", type=int, default=20,
                        help="Unfreeze encoder after this many epochs (default: 20)")
    parser.add_argument("--transfer_strategy", type=str, default="encoder_freeze",
                        choices=["full_finetune", "encoder_freeze", "last_decoder"],
                        help="Transfer strategy used when --pretrained is provided")

    # Loss
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for Weighted MSE loss (default: 0.7)")
    parser.add_argument("--beta",  type=float, default=0.3,
                        help="Weight for SSIM loss (default: 0.3)")
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="Stress-emphasis factor in Weighted MSE (default: 4.0)")

    # Optimiser
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--batch_size", type=int,  default=16)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Output
    parser.add_argument("--out_dir",   type=str,  default="outputs/checkpoints")
    parser.add_argument("--run_name",  type=str,  default=None,
                        help="Run name for checkpoint/log files (auto-generated if None)")

    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Seed          : {args.seed}")

    # ── Run name ────────────────────────────────────────────────────────────
    pretrain_tag = "tl" if args.pretrained else "scratch"
    run_name = args.run_name or f"{args.arch}_nf{args.n_fibers}_{pretrain_tag}"
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_best = os.path.join(args.out_dir, f"best_{run_name}.pth")
    log_path  = os.path.join(args.out_dir, f"log_{run_name}.json")

    print(f"Run name      : {run_name}")
    print(f"Checkpoint    : {ckpt_best}")

    # ── Load data ───────────────────────────────────────────────────────────
    npz_path = os.path.join(args.dataset_dir, f"nf{args.n_fibers}.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: dataset not found at {npz_path}", file=sys.stderr)
        print("Run: python preprocessing/build_dataset.py --n_fibers", args.n_fibers)
        sys.exit(1)

    print(f"\nLoading       : {npz_path}")
    X, Y, Y_min, Y_max = load_dataset(npz_path)
    print(f"  Total samples : {len(X)}")
    print(f"  X shape       : {X.shape}")
    print(f"  Y range       : {Y_min:.3e} – {Y_max:.3e} Pa")

    dataset = StressDataset(X, Y, Y_min, Y_max)

    if args.train_frac + args.val_frac >= 1.0:
        print("ERROR: train_frac + val_frac must be < 1.0 to leave room for test set.",
              file=sys.stderr)
        sys.exit(1)

    n_total = len(dataset)
    n_train = int(args.train_frac * n_total)
    n_val   = int(args.val_frac   * n_total)
    n_test  = n_total - n_train - n_val

    # Ensure at least 1 sample in each split
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1
    n_train = n_total - n_val - n_test
    if n_train < 1:
        print("ERROR: dataset too small for train/val/test split "
              f"(n_total={n_total}).", file=sys.stderr)
        sys.exit(1)

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    if args.train_samples is not None:
        n_use = max(1, min(args.train_samples, len(train_ds)))
        subset_idx = torch.randperm(
            len(train_ds),
            generator=torch.Generator().manual_seed(args.seed + 101)
        )[:n_use].tolist()
        train_ds = Subset(train_ds, subset_idx)
        print(f"  Train subset  : {len(train_ds)} samples (requested {args.train_samples})")

    print(f"  Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_gen = torch.Generator().manual_seed(args.seed + 202)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True,
                              generator=train_gen)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"\nArchitecture  : {args.arch}")
    model = build_model(
        arch=args.arch,
        in_channels=1,
        out_channels=1,
        base_features=args.base_features,
        dropout=args.dropout,
        resnet_pretrained=bool(args.resnet_pretrained),
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Transfer learning: load pretrained encoder and (optionally) freeze it
    encoder_frozen = False
    if args.pretrained:
        print(f"\nTransfer learning from: {args.pretrained}")
        load_pretrained_all(model, args.pretrained)
        if args.transfer_strategy == "encoder_freeze":
            freeze_encoder(model)
            encoder_frozen = True
        elif args.transfer_strategy == "last_decoder":
            freeze_all_except_last_decoder(model)
        else:
            print("  Strategy: full_finetune (all layers trainable).")

    # ── Loss + Optimiser ────────────────────────────────────────────────────
    criterion = CombinedLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )

    print(f"\nLoss          : α={args.alpha}·WeightedMSE + β={args.beta}·(1−SSIM)")
    print(f"Optimiser     : AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"LR schedule   : CosineAnnealing (T_max={args.epochs})")

    # ── Training ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Epoch':>5} {'LR':>8} {'Train':>10} {'Val':>10} {'SSIM':>7}")
    print("=" * 65)

    best_val_loss = float("inf")
    history = {"train": [], "val": [], "lr": []}

    for epoch in range(1, args.epochs + 1):

        # Unfreeze encoder mid-training for transfer learning
        if encoder_frozen and epoch == args.unfreeze_epoch + 1:
            for param in model.parameters():
                param.requires_grad = True
            # Re-init optimiser to include newly unfrozen params
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr * 0.1,  # lower LR for fine-tune
                weight_decay=args.weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch + 1), eta_min=args.lr * 1e-4
            )
            encoder_frozen = False
            print(f"\n  [epoch {epoch}] Encoder unfrozen — fine-tuning all layers.\n")

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats   = eval_loss(model, val_loader, criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train"].append(train_stats)
        history["val"].append(val_stats)
        history["lr"].append(current_lr)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         best_val_loss,
                "Y_min":            Y_min,
                "Y_max":            Y_max,
                "args":             vars(args),
            }, ckpt_best)

        if epoch % 5 == 0 or epoch == 1:
            val_ssim = 1.0 - val_stats["ssim_loss"]
            print(f"{epoch:>5} {current_lr:>8.2e} "
                  f"{train_stats['loss']:>10.6f} "
                  f"{val_stats['loss']:>10.6f} "
                  f"{val_ssim:>7.4f}")

    print("=" * 65)
    print(f"Best val loss : {best_val_loss:.6f}  → saved to {ckpt_best}")

    # ── Final evaluation on test set ────────────────────────────────────────
    print("\n[Test Set Evaluation]")
    ckpt = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = compute_metrics(model, test_loader, device, Y_min, Y_max,
                              gamma=args.gamma)
    print(f"  NRMSE             : {metrics['nrmse'] * 100:.2f}%  (target < 3%)")
    print(f"  Weighted MSE      : {metrics['wmse']:.6f}")
    print(f"  SSIM score        : {metrics['ssim_score']:.4f}  (target > 0.95)")
    print(f"  Median max error  : {metrics['median_max_err'] * 100:.2f}%  (target < 5%)")
    print(f"  Peak loc error    : {metrics['peak_loc_err']:.2f} px  (target < 5 px)")

    # ── Save training log ───────────────────────────────────────────────────
    log = {
        "run_name":   run_name,
        "args":       vars(args),
        "best_val_loss": best_val_loss,
        "test_metrics":  metrics,
        "history":       history,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved: {log_path}")


if __name__ == "__main__":
    main()
