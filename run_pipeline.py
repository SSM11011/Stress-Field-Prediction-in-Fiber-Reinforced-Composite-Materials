"""
run_pipeline.py — Master orchestration script for the Composite Stress Prediction Pipeline
============================================================================================

Run from the project root (ME 228/) directory:
    python run_pipeline.py                     # full pipeline (requires ABAQUS)
    python run_pipeline.py --skip_sim          # skip ABAQUS steps (use existing data)
    python run_pipeline.py --start_step 4      # start from dataset building
    python run_pipeline.py --dry_run           # print commands without executing

============================================================
SCIENTIFIC BACKGROUND & GOVERNING EQUATIONS
============================================================

─── 1. COMPOSITE MICROSTRUCTURE ────────────────────────────────────────────────

A 2D representative volume element (RVE) models a cross-section of a
unidirectional fiber-reinforced polymer. The domain is a unit square Ω = [0,1]²
containing N circular glass/epoxy fibers of radius R embedded in an epoxy matrix.

Fiber volume fraction (VF):
    VF = N · π · R² / A_domain        where A_domain = 1 m²

We target VF = 30%, so for each fiber count N:
    R(N) = sqrt(VF / (N · π))

    N =   6  →  R ≈ 0.1262 m
    N =  10  →  R ≈ 0.0977 m
    N =  20  →  R ≈ 0.0691 m
    N =  50  →  R ≈ 0.0437 m
    N = 100  →  R ≈ 0.0309 m

─── 2. RANDOM SEQUENTIAL ADDITION (RSA) PLACEMENT ──────────────────────────────

Fibers are placed one at a time. A candidate centre (x,y) is accepted if:
    ‖(x,y) − (xⱼ,yⱼ)‖₂ ≥ 2R + δ     for all j already placed
    δ = 0.005 m  (minimum surface-to-surface gap)

This guarantees a non-overlapping, non-touching configuration.

─── 3. FINITE ELEMENT METHOD — PLANE-STRAIN LINEAR ELASTICITY ──────────────────

Under uniaxial tensile loading in the x-direction with plane-strain assumption
(ε₃₃ = γ₁₃ = γ₂₃ = 0), each material phase obeys linear elasticity:

    Constitutive law:   σᵢⱼ = Cᵢⱼₖₗ · εₖₗ        (Generalised Hooke's Law)

    For an isotropic material (E, ν):
        σₓₓ = E/((1+ν)(1-2ν)) · [ (1-ν)εₓₓ + ν·εᵧᵧ ]
        σᵧᵧ = E/((1+ν)(1-2ν)) · [ ν·εₓₓ + (1-ν)εᵧᵧ ]
        σₓᵧ = E/(2(1+ν)) · γₓᵧ
        σ₃₃ = ν(σₓₓ + σᵧᵧ)           (plane-strain out-of-plane stress)

Material properties:
    Phase       E (GPa)    ν        Representative of
    ─────────────────────────────────────────────────
    Epoxy matrix    3.0    0.35     LY1564 epoxy resin
    Glass fiber    70.0    0.20     E-glass fibre

Boundary conditions (uniaxial tension, 1% strain):
    Left edge  (x=0): u₁ = 0   (fixed, no x-displacement)
    Bottom-left corner: u₂ = 0 (no y-displacement, prevents rigid body motion)
    Right edge (x=1): u₁ = 0.01 m (applied displacement = 1% strain)

Mesh: CPE4R elements (4-node quadrilateral, plane-strain, reduced integration)
      with CPE3 triangles for irregular regions.

─── 4. VON MISES STRESS (FAILURE CRITERION) ─────────────────────────────────────

The von Mises stress is computed from the principal stresses (σ₁, σ₂, σ₃):

    σ_vm = √[ ½ · ((σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²) ]

In 2D plane-strain, σ₃ = ν(σ₁+σ₂), so:

    σ_vm = √[ σₓₓ² - σₓₓσᵧᵧ + σᵧᵧ² + 3σₓᵧ² + σ₃₃² - σ₃₃(σₓₓ+σᵧᵧ) ]

High σ_vm values locate failure-critical regions (fiber-matrix interfaces).

─── 5. GRID INTERPOLATION (SCATTERED DATA → REGULAR IMAGE) ──────────────────────

ABAQUS outputs stress at element centroids — scattered, unstructured points.
We interpolate onto a regular 128×128 grid using triangulation:

    Step 1: Delaunay triangulation of centroid positions { (xᵢ, yᵢ) }
    Step 2: For each grid point (ξ,η), find the enclosing triangle
    Step 3: Barycentric interpolation:
                σ(ξ,η) = λ₁σᵢ + λ₂σⱼ + λ₃σₖ
            where (λ₁, λ₂, λ₃) are barycentric coordinates summing to 1
    Step 4: For points outside the convex hull (boundary), nearest-neighbour
            fallback fills the NaN values.

─── 6. U-NET ARCHITECTURE ────────────────────────────────────────────────────────

The U-Net maps: f: X ∈ {0,1}^(128×128) → Ŷ ∈ ℝ^(128×128)

    Input: binary microstructure image X (1 = fiber, 0 = matrix)
    Output: predicted von Mises stress field Ŷ (normalised [0,1])

Architecture (base features b=64):
    Encoder:
        Level 1: ConvBlock(1→ b)   then MaxPool2d → 64  channels,  64×64
        Level 2: ConvBlock(b→2b)   then MaxPool2d → 128 channels,  32×32
        Level 3: ConvBlock(2b→4b)  then MaxPool2d → 256 channels,  16×16
        Level 4: ConvBlock(4b→8b)  then MaxPool2d → 512 channels,   8×8
    Bottleneck:
        ConvBlock(8b→16b)          → 1024 channels,  4×4
    Decoder (with Attention Gates on skip connections):
        Level 4: UpConv(16b→8b), AG(g=8b, x=8b), concat, ConvBlock(16b→8b)
        Level 3: UpConv(8b→4b),  AG(g=4b, x=4b), concat, ConvBlock(8b→4b)
        Level 2: UpConv(4b→2b),  AG(g=2b, x=2b), concat, ConvBlock(4b→2b)
        Level 1: UpConv(2b→b),   AG(g=b,  x=b),  concat, ConvBlock(2b→b)
    Head:
        Conv2d(b→1, kernel=1) → 128×128 output

Each ConvBlock = [ Conv3×3→BN→ReLU → Conv3×3→BN→ReLU ] (with Dropout2d)

─── 7. ATTENTION GATE (Oktay et al., 2018) ─────────────────────────────────────

Attention gates selectively amplify skip-connection features at
fiber-matrix interface regions, where stress concentrations localise.

Given:
    g = gating signal from decoder (F_g channels)
    x = skip features from encoder  (F_l channels)

    W_g·g  and  W_x·x  are projected to F_int channels
    α = σ( ψᵀ · ReLU(W_g·g + W_x·x) )    α ∈ [0,1] per pixel
    output = α ⊙ x                          (element-wise scale)

─── 8. COMBINED LOSS FUNCTION ───────────────────────────────────────────────────

    L(ŷ, y) = α · L_wmse(ŷ, y) + β · L_ssim(ŷ, y)

    Default: α = 0.7, β = 0.3

Weighted MSE (emphasises failure-critical high-stress regions):
    L_wmse = mean[ w(y) · (ŷ - y)² ]
    w(y)   = 1 + γ · y      where γ = 4.0
    (pixels with y = 1.0 get 5× the loss weight of y = 0.0 pixels)

SSIM loss (preserves spatial structure of stress gradients):
    L_ssim = 1 − SSIM(ŷ, y)

    SSIM(ŷ, y) = [2μŷμy + C₁][2σŷy + C₂]
                 ─────────────────────────────────
                 [μŷ² + μy² + C₁][σŷ² + σy² + C₂]

    computed over local 11×11 Gaussian windows (σ=1.5)
    C₁ = (0.01·L)², C₂ = (0.03·L)², L = dynamic range

─── 9. OPTIMISER ─────────────────────────────────────────────────────────────────

AdamW update rule:
    m_t = β₁·m_{t-1} + (1-β₁)·∇L_t
    v_t = β₂·v_{t-1} + (1-β₂)·∇L_t²
    θ_t = θ_{t-1} − lr_t · m̂_t / (√v̂_t + ε) − lr_t · λ · θ_{t-1}

    β₁=0.9, β₂=0.999, ε=1e-8, weight decay λ=1e-4

Cosine annealing learning rate schedule:
    lr_t = lr_min + ½(lr_max − lr_min)(1 + cos(π · t / T_max))
    lr_max = 1e-3, lr_min = 1e-6, T_max = num_epochs

─── 10. TRANSFER LEARNING ────────────────────────────────────────────────────────

Phase 1 — Pretraining (source domain D_s):
    Train on N_s-fiber dataset (e.g. 6-fiber, 100 samples × 4 aug = 400 images)
    Encoder learns general stress-field features (gradients, interfaces)

Phase 2 — Fine-tuning (target domain D_t):
    Epochs 1 → K_freeze (frozen encoder):
        Only decoder weights updated; lr = 0.1 × lr_source
        Rapid adaptation of decoding pathway to new fiber density

    Epochs K_freeze+1 → T_max (full fine-tuning):
        All weights updated with lower lr
        Refined encoding of higher-density microstructures

    Freezing strategy evaluated: (a) full, (b) encoder frozen, (c) last decoder only

─── 11. EVALUATION METRICS ──────────────────────────────────────────────────────

    Metric              Formula                                Target
    ─────────────────────────────────────────────────────────────────
    NRMSE               √(MSE) / (σ_max − σ_min)              < 3%
    Weighted MSE        mean[w(y)·(ŷ−y)²],  w=1+γy            ↓ vs baseline
    Median Max Error    median_i[max_{px}|ŷ_i − y_i|] / range < 5%
    SSIM Score          1 − L_ssim                             > 0.95
    Peak Location Error Euclidean dist (px) to σ_vm peak       < 5 px
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ─── Project root = directory containing this script ──────────────────────────
ROOT = Path(__file__).resolve().parent

# ─── Dataset plan (per project specification) ─────────────────────────────────
#  6-fiber:  100 unique FEM sims → 400 training images (4-fold + elastic)
# 10-fiber:  100 unique FEM sims → 400 training images
# 20-fiber:  100 unique FEM sims → 400 training images
# 50-fiber:   50 unique FEM sims → 200 images (transfer learning eval)
# 100-fiber:  50 unique FEM sims → 200 images (transfer learning eval)

FIBER_SYSTEMS = [
    {"n_fibers":   6, "n_samples": 100},
    {"n_fibers":  10, "n_samples": 100},
    {"n_fibers":  20, "n_samples": 100},
    {"n_fibers":  50, "n_samples":  50},
    {"n_fibers": 100, "n_samples":  50},
]

# Source domains (pretrained) → target domains (fine-tuned)
TRANSFER_PAIRS = [
    {"source": 6,  "target": 20},
    {"source": 6,  "target": 50},
    {"source": 6,  "target": 100},
    {"source": 10, "target": 20},
    {"source": 10, "target": 50},
    {"source": 10, "target": 100},
]

# Progressive low-data transfer settings
TRANSFER_TRAIN_SIZES = [4, 20, 40, 60, 80, 100]
TRANSFER_STRATEGIES = ["full_finetune", "encoder_freeze", "last_decoder"]


# ===========================================================================
# Helpers
# ===========================================================================

def banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def parse_seed_list(seed_csv: str) -> list[int]:
    seeds = [int(s.strip()) for s in seed_csv.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def run(cmd: str, dry_run: bool = False, check: bool = True) -> bool:
    """
    Print and optionally execute a shell command.
    Returns True on success, False on failure.
    """
    print(f"\n  $ {cmd}")
    if dry_run:
        return True
    t0 = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0
    if ret != 0:
        msg = f"  [FAILED — exit {ret} in {elapsed:.1f}s]"
        print(msg, file=sys.stderr)
        if check:
            raise RuntimeError(f"Command failed: {cmd}")
        return False
    print(f"  [OK — {elapsed:.1f}s]")
    return True


def abaqus_available() -> bool:
    """Cross-platform check for the Abaqus executable on PATH."""
    try:
        subprocess.run(
            "abaqus information=release",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, shell=True
        )
        return True
    except FileNotFoundError:
        return False


def checkpoint_path(n_fibers: int, arch: str = "attention_unet",
                    pretrain_tag: str = "scratch", seed: int | None = None) -> str:
    suffix = f"_seed{seed}" if seed is not None else ""
    return str(Path("outputs") / "checkpoints" /
               f"best_{arch}_nf{n_fibers}_{pretrain_tag}{suffix}.pth")


def log_path(n_fibers: int, arch: str = "attention_unet",
             pretrain_tag: str = "scratch", seed: int | None = None) -> str:
    suffix = f"_seed{seed}" if seed is not None else ""
    return str(Path("outputs") / "checkpoints" /
               f"log_{arch}_nf{n_fibers}_{pretrain_tag}{suffix}.json")


# ===========================================================================
# Step 1  — Generate fiber microstructures
# ===========================================================================

def step1_generate_fibers(dry_run: bool) -> None:
    """
    Random Sequential Addition (RSA) fiber placement.

    For each fiber count N and volume fraction VF = 30%:
        R(N) = sqrt(VF / (N·π))

    Acceptance condition for candidate centre (x,y):
        ‖(x,y) − (xⱼ,yⱼ)‖₂ ≥ 2R + δ    (δ = 0.005 m gap)

    Output: outputs/microstructures/nf{N}/fibers_{i}.npy
            shape (N, 2) — (x, y) centre coordinates, dtype float64
    """
    banner("STEP 1 — RSA Fiber Placement")
    print("""
  Physics / Math:
    Volume fraction    VF = N·π·R² / A_domain  (A_domain = 1 m²)
    Target VF = 30%%:  R(N)  = sqrt(0.30 / (N·π))
    Non-overlap check: ‖cᵢ − cⱼ‖₂ ≥ 2R + 0.005 m  for all j < i
""")

    for fs in FIBER_SYSTEMS:
        nf, ns = fs["n_fibers"], fs["n_samples"]
        import math
        R = math.sqrt(0.30 / (nf * math.pi))
        print(f"  nf={nf:3d}  samples={ns:3d}  R={R:.4f} m  "
              f"VF_check={nf * math.pi * R**2 * 100:.1f}%")
        run(
            f"python data/generate_fibers.py --n_fibers {nf} --n_samples {ns}",
            dry_run=dry_run
        )


# ===========================================================================
# Step 2  — ABAQUS FEM simulation
# ===========================================================================

def step2_simulate(dry_run: bool) -> None:
    """
    Plane-strain linear-elastic FEM for each (fiber system, sample).

    Governing PDE (static equilibrium):
        ∇·σ = 0  in Ω

    Constitutive law (plane-strain, isotropic):
        σₓₓ = E/((1+ν)(1-2ν)) · [(1-ν)εₓₓ + ν·εᵧᵧ]
        σᵧᵧ = E/((1+ν)(1-2ν)) · [ν·εₓₓ + (1-ν)εᵧᵧ]
        σₓᵧ = G·γₓᵧ,    G = E/(2(1+ν))
        σ₃₃ = ν(σₓₓ + σᵧᵧ)   (non-zero due to plane-strain constraint)

    Material moduli:
        Epoxy matrix: E = 3.0 GPa, ν = 0.35
        Glass fiber:  E = 70.0 GPa, ν = 0.20

    Boundary conditions:
        u₁(x=0) = 0          (left edge: no x-displacement)
        u₂(x=0, y=0) = 0     (bottom-left corner: no y-displacement)
        u₁(x=1) = 0.01 m     (right edge: 1% applied tensile strain)

    Elements: CPE4R (plane-strain, 4-node quad, reduced integration)
    Mesh seed: h = min(R/2, 0.015) m  (finer near small fibers)

    Output: job_nf{N}_{i}.odb
    """
    banner("STEP 2 — ABAQUS Plane-Strain FEM Simulations")
    print("""
  Physics:
    Plane-strain assumption:  ε₃₃ = γ₁₃ = γ₂₃ = 0
    Applied nominal strain:   ε_nom = Δu / L = 0.01 / 1.0 = 1%%
    FEM weak form:            ∫_Ω σ:δε dΩ = ∫_∂Ω t·δu dΓ

  Element: CPE4R (Abaqus)
    C  = Continuum
    PE = Plane Strain
    4  = 4-node (bilinear shape functions)
    R  = Reduced integration (prevents volumetric locking)
""")

    if not abaqus_available():
        print("  WARNING: 'abaqus' not found on PATH — skipping simulations.")
        print("  Use pre-extracted stress data or re-run with ABAQUS installed.")
        return

    for fs in FIBER_SYSTEMS:
        nf, ns = fs["n_fibers"], fs["n_samples"]
        print(f"\n  --- {nf}-fiber system ({ns} simulations) ---")
        for i in range(ns):
            run(
                f"abaqus cae noGUI=data/abaqus_simulation.py -- {i} {nf}",
                dry_run=dry_run, check=False      # don't abort on single failure
            )


# ===========================================================================
# Step 3  — Extract von Mises stress from ODB files
# ===========================================================================

def step3_extract(dry_run: bool) -> None:
    """
    Extract von Mises stress and element centroid coordinates from each ODB.

    Von Mises stress (scalar invariant of the stress deviator):
        σ_vm = sqrt[ ½ · ((σ₁−σ₂)² + (σ₂−σ₃)² + (σ₃−σ₁)²) ]

    In plane-strain (with σ₃ = ν(σ₁+σ₂)):
        σ_vm = sqrt[ σₓₓ² − σₓₓσᵧᵧ + σᵧᵧ² + 3σₓᵧ²
                     + σ₃₃² − σ₃₃(σₓₓ+σᵧᵧ) ]

    Element centroid coordinates:
        (cₓ, cᵧ) = (1/n_nodes) · Σⱼ (xⱼ, yⱼ)

    Output:
        outputs/stress_maps/nf{N}/stress_{i}.npy  — von Mises (Pa), shape (N_elem,)
        outputs/stress_maps/nf{N}/coords_{i}.npy  — centroids (m),  shape (N_elem, 2)
    """
    banner("STEP 3 — Stress Extraction from ODB Files")
    print("""
  Physics:
    σ_vm = sqrt[ ½·((σ₁−σ₂)² + (σ₂−σ₃)² + (σ₃−σ₁)²) ]

    High σ_vm → yielding / fracture initiation (von Mises yield criterion)
    Critical for failure prediction in fiber-matrix debonding and matrix cracking.
""")

    if not abaqus_available():
        print("  WARNING: 'abaqus' not found — skipping extraction.")
        return

    for fs in FIBER_SYSTEMS:
        nf = fs["n_fibers"]
        print(f"\n  --- {nf}-fiber system ---")
        run(
            f"abaqus python data/extract_stress_coords.py all {nf}",
            dry_run=dry_run, check=False
        )


# ===========================================================================
# Step 4  — Build 128×128 training datasets
# ===========================================================================

def step4_build_datasets(dry_run: bool) -> None:
    """
    Interpolate scattered FEM centroids → regular 128×128 pixel grid,
    render binary microstructure images, apply 4-fold flips + elastic deformation,
    and save to .npz format.

    Grid interpolation (Delaunay / barycentric):
        σ(ξ,η) = λ₁·σ_A + λ₂·σ_B + λ₃·σ_C    for grid point (ξ,η)
                                                  inside triangle ABC
        where  λ₁+λ₂+λ₃ = 1,  λᵢ ≥ 0   (barycentric coordinates)

    Binary microstructure rendering:
        X[row,col] = 1  if  (xₚₓ − fₓ)² + (yₚₓ − fᵧ)² < R²
                        (pixel centre (xₚₓ, yₚₓ) is inside a fiber)
        X[row,col] = 0  otherwise  (matrix)

    4-fold flip augmentation:
        aug 0: original
        aug 1: flip left–right      (σ field is symmetric under LR flip)
        aug 2: flip up–down         (symmetric under UD flip)
        aug 3: flip LR + UD         (180° rotation)

    Elastic deformation:
        Smooth random displacement field d(x,y) is applied to both X and Y
        to improve robustness to local geometric perturbations while preserving
        sample-level labels.

    Stress normalisation metadata saved to .npz:
        Y_global_min, Y_global_max  → used during training to map Y to [0,1]

    Output: outputs/dataset/nf{N}.npz
        X          : (N_total, 1, 128, 128)  float32  microstructure
        Y          : (N_total, 1, 128, 128)  float32  stress (raw Pa)
        sample_ids : (N_total,)              int32
        aug_ids    : (N_total,)              int32
    """
    banner("STEP 4 — Build 128×128 Training Datasets")
    print("""
  Data pipeline:
    Scattered FEM points  → Delaunay triangulation → bilinear grid interpolation
    Fiber centres (.npy)  → rasterisation on 128×128 pixel grid
    Raw images × 8        → augmented dataset saved as .npz

    Dataset sizes (4-fold + elastic augmentation):
        6-fiber:  100 × 4 =  400 training images
        10-fiber: 100 × 4 =  400 training images
        20-fiber: 100 × 4 =  400 training images
        50-fiber:  50 × 4 =  200 training images
     100-fiber:  50 × 4 =  200 training images
""")

    for fs in FIBER_SYSTEMS:
        nf = fs["n_fibers"]
        stress_dir = ROOT / "outputs" / "stress_maps" / f"nf{nf}"
        if not dry_run and not stress_dir.exists():
            print(f"  SKIP nf{nf}: stress directory not found ({stress_dir})")
            print(f"         Run Steps 2+3 with ABAQUS first, then re-run Step 4.")
            continue
        run(
            f"python preprocessing/build_dataset.py --n_fibers {nf} --aug_mode 4fold --elastic",
            dry_run=dry_run
        )


# ===========================================================================
# Step 5  — Visualise raw FEM and dataset
# ===========================================================================

def step5_visualise_data(dry_run: bool) -> None:
    """
    Generate all exploratory figures:
      - Triangulated von Mises heat maps per sample
      - Dataset overview: microstructure ↔ stress pairs with augmentations
    """
    banner("STEP 5 — Visualise FEM Data & Dataset Overview")
    print("""
  Outputs:
    outputs/figures/heatmap_grid_nf{N}.png    — all FEM heat maps (common scale)
    outputs/figures/heatmap_nf{N}_sample*.png — individual samples
    outputs/figures/stress_stats_nf{N}.png    — mean/max/std summary
    outputs/figures/dataset_overview_nf{N}.png — 128×128 grid
    outputs/figures/augmentation_check_nf{N}.png — geometric augmentations side-by-side
""")

    for fs in FIBER_SYSTEMS:
        nf = fs["n_fibers"]
        # Raw FEM heat maps (only if coordinates extracted)
        coords_dir = ROOT / "outputs" / "stress_maps" / f"nf{nf}"
        if coords_dir.exists() and list(coords_dir.glob("coords_*.npy")):
            run(
                f"python visualization/visualize.py --mode raw_fem --n_fibers {nf}",
                dry_run=dry_run, check=False
            )
        # Dataset overview (only if dataset built)
        dataset_file = ROOT / "outputs" / "dataset" / f"nf{nf}.npz"
        if dataset_file.exists() or dry_run:
            run(
                f"python visualization/visualize.py --mode dataset --n_fibers {nf} --n_show 16",
                dry_run=dry_run, check=False
            )


# ===========================================================================
# Step 6  — Train base models (source domains: 6-fiber, 10-fiber)
# ===========================================================================

def step6_train_base_models(dry_run: bool, epochs: int, seeds: list[int]) -> None:
    """
    Train AttentionUNet from scratch on 6-fiber and 10-fiber datasets.

    Training loop:
        For each epoch t = 1 … T:
            Forward:   Ŷ = f_θ(X)
            Loss:      L = α·L_wmse(Ŷ, Y) + β·(1 − SSIM(Ŷ, Y))
            Backward:  ∇_θ L  via automatic differentiation (backprop)
            Update:    θ ← θ − lr_t · AdamW_update(∇_θ L)
            Schedule:  lr_t = lr_min + ½(lr_max−lr_min)(1+cos(πt/T))

    Best checkpoint selected by minimum validation loss.

    These are the "source domain" models for transfer learning.
    """
    banner("STEP 6 — Train Base Models from Scratch (6- and 10-fiber)")
    print(f"""
    Architecture: AttentionUNet (base_features=64, dropout=0.1)
  Loss:         L = 0.7·WeightedMSE + 0.3·(1−SSIM)
  Optimiser:    AdamW  (lr=1e-3, weight_decay=1e-4)
  Scheduler:    Cosine annealing  (T_max={epochs}, lr_min=1e-6)
  Epochs:       {epochs}
    Augmentation: 4-fold + elastic deformation (already baked into .npz)

  WeightedMSE:  L_wmse = mean[ (1 + 4·y)·(ŷ − y)² ]
  SSIM:         L_ssim = 1 − SSIM(ŷ, y)   (11×11 Gaussian window)
""")

    for nf in [6, 10]:
        for seed in seeds:
            run(
                f"python training/train_unet.py"
                f"  --n_fibers {nf}"
                f"  --arch attention_unet"
                f"  --epochs {epochs}"
                f"  --batch_size 16"
                f"  --lr 1e-3"
                f"  --alpha 0.7  --beta 0.3  --gamma 4.0"
                f"  --dropout 0.1"
                f"  --train_frac 0.80  --val_frac 0.10"
                f"  --seed {seed}"
                f"  --run_name attention_unet_nf{nf}_scratch_seed{seed}",
                dry_run=dry_run
            )


# ===========================================================================
# Step 7  — Ablation: baseline U-Net (replicates reference paper)
# ===========================================================================

def step7_ablation(dry_run: bool, epochs: int, seeds: list[int]) -> None:
    """
    Ablation studies to isolate contributions of each enhancement.

    Experiments:
        (a) Baseline U-Net, pure MSE — reference replication
        (b) AttentionUNet, pure Weighted MSE — isolate SSIM contribution
        (c) Standard U-Net + combined loss — isolate attention gates
        (d) ResNet34 Attention U-Net + combined loss — backbone contribution
        (e) CNN baseline (no skip connections) — architecture benchmark

    Results isolate:
        Δ_attn, Δ_ssim, Δ_backbone and gain over a no-skip CNN baseline.
    """
    banner("STEP 7 — Ablation Studies (6-fiber baseline)")
    print("""
  Experiment (a): Baseline U-Net + MSE
  Experiment (b): AttentionUNet + WMSE
  Experiment (c): Baseline U-Net + WMSE+SSIM
  Experiment (d): ResNet34 Attention U-Net + WMSE+SSIM
  Experiment (e): CNN baseline + WMSE+SSIM
""")

    ablations = [
        {"arch": "unet",           "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
         "tag": "unet_mse",        "desc": "Baseline U-Net, pure MSE"},
        {"arch": "attention_unet", "alpha": 1.0, "beta": 0.0, "gamma": 4.0,
         "tag": "attn_wmse",       "desc": "AttentionUNet, Weighted MSE only"},
        {"arch": "unet",           "alpha": 0.7, "beta": 0.3, "gamma": 4.0,
         "tag": "unet_combined",   "desc": "Baseline U-Net, combined loss"},
        {"arch": "resnet34_attention_unet", "alpha": 0.7, "beta": 0.3, "gamma": 4.0,
         "tag": "resnet34_attn_combined", "desc": "ResNet34 Attention U-Net, combined loss",
         "extra": "--resnet_pretrained"},
        {"arch": "cnn", "alpha": 0.7, "beta": 0.3, "gamma": 4.0,
         "tag": "cnn_combined", "desc": "CNN baseline, combined loss"},
    ]

    for ab in ablations:
        print(f"\n  Running: {ab['desc']}")
        extra = ab.get("extra", "")
        for seed in seeds:
            run(
                f"python training/train_unet.py"
                f"  --n_fibers 6"
                f"  --arch {ab['arch']}"
                f"  --epochs {epochs}"
                f"  --alpha {ab['alpha']}  --beta {ab['beta']}  --gamma {ab['gamma']}"
                f"  --seed {seed}"
                f"  {extra}"
                f"  --run_name {ab['tag']}_nf6_seed{seed}",
                dry_run=dry_run, check=False
            )


# ===========================================================================
# Step 8  — Transfer learning experiments
# ===========================================================================

def step8_transfer_learning(dry_run: bool, epochs: int,
                             unfreeze_epoch: int, seeds: list[int]) -> None:
    """
    Systematic transfer learning study.

    For each (source, target) pair and each train size in
    {4, 20, 40, 60, 80, 100}:
        1. Train scratch baseline
        2. Fine-tune pretrained model with each strategy:
            - full_finetune
            - encoder_freeze (then unfreeze at K)
            - last_decoder

    Rationale:
        The encoder has learned low-level stress field features
        (gradients at interfaces, field continuity) that transfer across
        different fiber densities. The decoder adapts to the target
        fiber spatial distribution.

    Transfer learning gain:
        TL_gain(n, strategy) = metric_pretrained(n, strategy) − metric_scratch(n)
        where n ∈ {4, 20, 40, 60, 80, 100}
    """
    banner("STEP 8 — Transfer Learning (source → target fiber systems)")
    print(f"""
    Strategies:
        full_finetune   : all layers trainable from epoch 1
        encoder_freeze  : encoder frozen until epoch {unfreeze_epoch}, then unfrozen
        last_decoder    : only last decoder block + head trainable

  Pairs:
    Source 6-fiber  → Target 20, 50, 100-fiber
    Source 10-fiber → Target 20, 50, 100-fiber

    Train sizes:
        {TRANSFER_TRAIN_SIZES}
""")

    for pair in TRANSFER_PAIRS:
        src, tgt = pair["source"], pair["target"]
        for seed in seeds:
            src_ckpt = checkpoint_path(src, "attention_unet", "scratch", seed=seed)

            if not os.path.exists(src_ckpt) and not dry_run:
                print(f"  SKIP: pretrained checkpoint for {src}-fiber not found: {src_ckpt}")
                continue

            for n_train in TRANSFER_TRAIN_SIZES:
                # Scratch baseline for this data size
                run(
                    f"python training/train_unet.py"
                    f"  --n_fibers {tgt}"
                    f"  --arch attention_unet"
                    f"  --epochs {epochs}"
                    f"  --train_samples {n_train}"
                    f"  --seed {seed}"
                    f"  --run_name attention_unet_nf{tgt}_scratch_n{n_train}_seed{seed}",
                    dry_run=dry_run, check=False
                )

                # Transfer runs for each freezing strategy
                for strategy in TRANSFER_STRATEGIES:
                    run(
                        f"python training/train_unet.py"
                        f"  --n_fibers {tgt}"
                        f"  --arch attention_unet"
                        f"  --epochs {epochs}"
                        f"  --train_samples {n_train}"
                        f"  --seed {seed}"
                        f"  --pretrained {src_ckpt}"
                        f"  --transfer_strategy {strategy}"
                        f"  --unfreeze_epoch {unfreeze_epoch}"
                        f"  --run_name attention_unet_nf{tgt}_from_nf{src}_{strategy}_n{n_train}_seed{seed}",
                        dry_run=dry_run, check=False
                    )


# ===========================================================================
# Step 9  — Visualise predictions and training curves
# ===========================================================================

def step9_visualise_results(dry_run: bool) -> None:
    """
    Generate prediction comparison and training curve figures for every
    trained checkpoint found in outputs/checkpoints/.
    """
    banner("STEP 9 — Visualise Model Predictions & Training Curves")
    print("""
  Outputs per checkpoint:
    predictions_{run_name}.png   — 8 samples: microstructure | GT | pred | error
    scatter_{run_name}.png       — σ_pred vs σ_GT scatter (all test pixels)
    training_curves_{run_name}.png — loss + SSIM vs epoch + LR schedule
""")

    ckpt_dir = Path("outputs") / "checkpoints"
    if not ckpt_dir.exists() and not dry_run:
        print("  No checkpoints found — skipping.")
        return

    # Training curve plots (from JSON logs)
    log_files = list(ckpt_dir.glob("log_*.json")) if not dry_run else \
        [ckpt_dir / "log_attention_unet_nf6_scratch.json"]
    for lf in sorted(log_files):
        run(
            f"python visualization/visualize.py"
            f"  --mode training_curves"
            f"  --log {lf}",
            dry_run=dry_run, check=False
        )

    # Prediction figures (for each checkpoint + matching dataset)
    ckpt_files = list(ckpt_dir.glob("best_*.pth")) if not dry_run else \
        [ckpt_dir / "best_attention_unet_nf6_scratch.pth"]
    for cf in sorted(ckpt_files):
        # Extract n_fibers from filename, e.g. best_attention_unet_nf6_...
        parts = cf.stem.split("_nf")
        if len(parts) < 2:
            continue
        nf_str = parts[1].split("_")[0]
        try:
            nf = int(nf_str)
        except ValueError:
            continue

        dataset_file = ROOT / "outputs" / "dataset" / f"nf{nf}.npz"
        if not dataset_file.exists() and not dry_run:
            print(f"  SKIP predictions for {cf.name} — dataset nf{nf}.npz not found")
            continue

        run(
            f"python visualization/visualize.py"
            f"  --mode predictions"
            f"  --n_fibers {nf}"
            f"  --checkpoint {cf}"
            f"  --n_show 8",
            dry_run=dry_run, check=False
        )


# ===========================================================================
# Step 10 — Aggregate results and statistical reporting
# ===========================================================================

def step10_aggregate_reports(dry_run: bool, alpha: float) -> None:
    """
    Build consolidated result tables, confidence intervals, significance tests,
    and per-fiber-system qualitative error figures.
    """
    banner("STEP 10 — Aggregate Results and Statistical Reporting")
    print(f"""
  Reporting outputs:
    outputs/reports/all_runs.csv
    outputs/reports/summary_metrics.csv
    outputs/reports/significance_tests.csv
    outputs/figures/qualitative_error_nf*.png

  Statistical settings:
    Confidence intervals: 95%%
    Significance threshold: p < {alpha}
""")
    run(
        f"python visualization/report_results.py --alpha {alpha}",
        dry_run=dry_run, check=False
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the full composite stress-prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Generate RSA fiber configurations  (Python only)
  2  ABAQUS FEM simulations             (requires ABAQUS)
  3  Extract von Mises stress from ODB  (requires ABAQUS)
  4  Build 128×128 .npz datasets        (Python only)
  5  Visualise FEM data                 (Python only)
  6  Train base models (nf=6,10)        (Python + PyTorch)
  7  Ablation studies                   (Python + PyTorch)
  8  Transfer learning experiments      (Python + PyTorch)
  9  Visualise results                  (Python + PyTorch)
 10  Aggregate reports + statistics     (Python + PyTorch)
        """
    )
    parser.add_argument("--start_step",     type=int, default=1, choices=range(1, 11),
                        help="Start from this step (default: 1)")
    parser.add_argument("--end_step",       type=int, default=10, choices=range(1, 11),
                        help="Stop after this step (default: 10)")
    parser.add_argument("--skip_sim",       action="store_true",
                        help="Skip Steps 2+3 (ABAQUS simulation + extraction)")
    parser.add_argument("--skip_ablation",  action="store_true",
                        help="Skip Step 7 (ablation studies)")
    parser.add_argument("--skip_reporting", action="store_true",
                        help="Skip Step 10 (aggregate reporting)")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Print all commands without executing them")
    parser.add_argument("--epochs",         type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--unfreeze_epoch", type=int, default=20,
                        help="Epoch to unfreeze encoder in transfer learning (default: 20)")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated seeds for repeated runs (default: 42,43,44)")
    parser.add_argument("--significance_alpha", type=float, default=0.05,
                        help="Significance threshold for report tests (default: 0.05)")
    args = parser.parse_args()
    seeds = parse_seed_list(args.seeds)

    # Force UTF-8 output so Unicode math symbols render on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(__doc__)   # Print the full physics/math reference at the top

    t_pipeline_start = time.time()

    steps = {
        1: ("RSA Fiber Generation",         lambda: step1_generate_fibers(args.dry_run)),
        2: ("ABAQUS Simulations",           lambda: step2_simulate(args.dry_run)),
        3: ("Stress Extraction from ODB",   lambda: step3_extract(args.dry_run)),
        4: ("Build 128×128 Datasets",       lambda: step4_build_datasets(args.dry_run)),
        5: ("Visualise FEM Data",           lambda: step5_visualise_data(args.dry_run)),
        6: ("Train Base Models",            lambda: step6_train_base_models(args.dry_run, args.epochs, seeds)),
        7: ("Ablation Studies",             lambda: step7_ablation(args.dry_run, args.epochs, seeds)),
        8: ("Transfer Learning",            lambda: step8_transfer_learning(args.dry_run, args.epochs, args.unfreeze_epoch, seeds)),
        9: ("Visualise Results",            lambda: step9_visualise_results(args.dry_run)),
        10: ("Aggregate Reports",           lambda: step10_aggregate_reports(args.dry_run, args.significance_alpha)),
    }

    if args.skip_sim:
        steps.pop(2, None)
        steps.pop(3, None)
        print("  [--skip_sim] Skipping ABAQUS simulation and extraction steps.")

    if args.skip_ablation:
        steps.pop(7, None)
        print("  [--skip_ablation] Skipping ablation studies.")
    if args.skip_reporting:
        steps.pop(10, None)
        print("  [--skip_reporting] Skipping aggregate reporting.")

    active_steps = {k: v for k, v in steps.items()
                    if args.start_step <= k <= args.end_step}

    print(f"\n  Pipeline steps to run: {sorted(active_steps.keys())}")
    if args.dry_run:
        print("  *** DRY RUN — no commands will be executed ***")

    # Change working directory to project root so all relative paths resolve
    os.chdir(ROOT)

    for step_num in sorted(active_steps.keys()):
        step_name, step_fn = active_steps[step_num]
        t0 = time.time()
        try:
            step_fn()
        except RuntimeError as e:
            print(f"\n  FATAL ERROR in Step {step_num}: {e}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.time() - t0
        print(f"\n  [Step {step_num} complete in {elapsed:.1f}s]")

    total = time.time() - t_pipeline_start
    banner(f"Pipeline complete  ({total:.1f}s total)")
    print("""
  Key outputs:
    outputs/dataset/nf*.npz                   — training-ready datasets
    outputs/checkpoints/best_*.pth            — trained model weights
    outputs/checkpoints/log_*.json            — training logs with metrics
    outputs/figures/                          — all visualisation plots

  Target metrics (test set):
    NRMSE            < 3%%
    SSIM score       > 0.95
    Median max error < 5%% of stress range
    Peak loc. error  < 5 pixels
""")


if __name__ == "__main__":
    main()
