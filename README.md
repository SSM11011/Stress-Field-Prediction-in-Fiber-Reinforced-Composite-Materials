# Final Project Report  
## Stress-Field Prediction in Fiber-Reinforced Composite Materials

## 1. Project Objective and Scope

This project delivers a full **FEM-to-ML surrogate pipeline** that predicts the 2D von Mises stress field of fiber-reinforced composite RVEs from binary microstructure images, avoiding repeated expensive ABAQUS solves during design exploration.

- **Input:** binary microstructure image (fiber=1, matrix=0), typically `128x128`
- **Output:** continuous von Mises stress field image, same resolution
- **Core task type:** dense image-to-image regression

The implementation now spans:
1. stochastic microstructure generation,
2. ABAQUS simulation and ODB extraction,
3. interpolation and dataset construction with augmentation,
4. multiple deep architectures and training strategies,
5. ablations and transfer-learning orchestration,
6. visualization and reporting automation.

---

## 2. Complete Repository Implementation Map

| Area | File | What is implemented |
|---|---|---|
| Data generation | `data/generate_fibers.py` | Random Sequential Addition (RSA) placement, non-overlap + minimum gap, VF-based radius, reproducible seeds, per-system sample generation |
| FEM solve | `data/abaqus_simulation.py` | 2D plane-strain ABAQUS model build, matrix/fiber materials, BCs, mesh, job submission, per-sample ODB generation |
| ODB extraction | `data/extract_stress_coords.py` | Final-frame von Mises extraction, element-centroid computation, stress/coords `.npy` export |
| Data pipeline wrapper | `data/run_dataset.py` | End-to-end dataset generation orchestration with skip/resume flags |
| Preprocessing | `preprocessing/build_dataset.py` | Scattered-to-grid interpolation, binary rasterization, augmentation (`none/4fold/8fold`), optional elastic deformation, compressed `.npz` dataset output |
| Models | `models/unet.py`, `models/__init__.py` | UNet, Attention UNet, ResNet34 Attention UNet, SimpleCNN baseline, model factory |
| Training | `training/train_unet.py` | Combined loss (WMSE+SSIM), metrics suite, grouped split policy, low-data `train_samples`, transfer loading/freezing strategies, checkpoint/logging |
| Visualization | `visualization/visualize.py` | Raw FEM heatmaps, dataset overview plots, prediction-vs-GT/error plots, scatter plots, training curve plots |
| Reporting | `visualization/report_results.py` | Log aggregation to CSV, confidence intervals, pretrained-vs-scratch significance tests, qualitative error figures |
| Master orchestration | `run_pipeline.py` | 10-step full project runner from microstructure to reporting, dry-run support, seed sweeps, transfer matrix setup |
| Abaqus replay artifact | `abaqus.rpy` | Confirmed CAE replay execution example of simulation script |

---

## 3. Scientific and Computational Formulation

### 3.1 Composite geometry
- Domain: unit square RVE, `Omega = [0,1]^2`
- Fiber systems: `N in {6, 10, 20, 50, 100}`
- Target fiber volume fraction: `VF = 0.30`
- Radius per system:
  - `R = sqrt(VF / (N*pi))`
  - approx values:  
    - `N=6 -> R=0.1262`  
    - `N=10 -> R=0.0977`  
    - `N=20 -> R=0.0691`  
    - `N=50 -> R=0.0437`  
    - `N=100 -> R=0.0309`

### 3.2 RSA microstructure generation
- Candidate center accepted if distance to all existing centers satisfies:
  - `||c_i - c_j||_2 >= 2R + delta`
  - with `delta = 0.005`
- Deterministic reproducibility via seeded RNG (`SEED_BASE + sample_id`).

### 3.3 ABAQUS plane-strain mechanics setup
- Linear elastic, isotropic:
  - Matrix: `E=3.0 GPa`, `nu=0.35`
  - Fiber: `E=70.0 GPa`, `nu=0.20`
- BCs:
  - left edge `u1=0`
  - bottom-left corner `u2=0`
  - right edge `u1=0.01` (1% tensile strain)
- Elements: `CPE4R` (+ `CPE3` fallback)
- Mesh seed size: `min(R/2, 0.015)`.

### 3.4 Stress extraction and field construction
- Extract final-frame von Mises scalar field from ODB.
- Compute element centroids by averaging node coordinates.
- Interpolate scattered stress values to regular grid:
  - primary: linear interpolation
  - fallback: nearest-neighbor for NaNs.

### 3.5 Dataset representation and augmentation
- Final dataset format (`.npz`):
  - `X`: `(N,1,H,W)` binary microstructure
  - `Y`: `(N,1,H,W)` stress (Pa)
  - `sample_ids`, `aug_ids`, `n_fibers`, `fiber_radius`, `Y_global_min`, `Y_global_max`, `grid_size`
- Augmentations:
  - `none`, `4fold`, `8fold`
  - optional elastic pair warp (`alpha`, `sigma`, shared displacement field for X/Y consistency).

---

## 4. ML Architectures, Losses, and Metrics Implemented

### 4.1 Architectures
- `UNet`
- `AttentionUNet` (attention gates on all skips)
- `ResNet34AttentionUNet` (optional ImageNet-pretrained encoder)
- `SimpleCNN` baseline (no skip connections)

### 4.2 Losses and optimization
- Weighted MSE: `mean((1 + gamma*y_true)*(y_pred - y_true)^2)`
- SSIM loss: `1 - SSIM`
- Combined objective: `L = alpha*WMSE + beta*SSIMLoss`
- Defaults:
  - `alpha=0.7`, `beta=0.3`, `gamma=4.0`
  - AdamW (`lr=1e-3`, `weight_decay=1e-4`)
  - cosine annealing LR schedule.

### 4.3 Evaluation metrics implemented
- NRMSE
- Weighted MSE
- SSIM score
- Median max absolute error (normalized)
- Peak-location error (pixels)

### 4.4 Transfer learning strategies implemented
- `full_finetune`
- `encoder_freeze` (+ scheduled unfreeze)
- `last_decoder`

---

## 5. End-to-End Pipeline Coverage (run_pipeline.py)

The project is fully wired with 10 orchestration steps:
1. RSA fiber generation
2. ABAQUS simulation
3. ODB stress extraction
4. Dataset building
5. Data visualization
6. Base model training
7. Ablation studies
8. Transfer learning experiments
9. Result visualization (predictions/scatter/curves)
10. Aggregated reporting (CSV + significance + qualitative errors)

Configured experiment grids include:
- fiber systems: `6, 10, 20, 50, 100`
- seeds (default): `42,43,44`
- transfer pair list and low-data train sizes (`4,20,40,60,80,100`).

---

## 6. Outputs Folder Audit (Current Snapshot)

### 6.1 Folder status
- `outputs/figures/` exists and contains generated plots.
- `outputs/reports/` exists but has no CSV files in this snapshot.

### 6.2 Total output file count
- **508 PNG files** in `outputs/figures`.

### 6.3 Plot category inventory

| Plot family | Count | Details |
|---|---:|---|
| `heatmap_nf*_sample*.png` | 316 | Per-sample raw FEM heatmaps |
| `heatmap_grid_nf*.png` | 5 | One FEM grid summary per fiber system |
| `dataset_overview_nf*.png` | 5 | One dataset visualization per fiber system |
| `augmentation_check_nf*.png` | 5 | One augmentation diagnostic per fiber system |
| `stress_distribution_nf*.png` | 5 | Stress histogram per fiber system |
| `stress_stats_nf*.png` | 5 | Mean/max/std summary per fiber system |
| `qualitative_error_nf*.png` | 3 | Qualitative error summaries (nf6, nf10, nf20) |
| `training_curves_*.png` | 54 | Training history plots |
| `predictions_*.png` | 55 | Prediction-vs-GT comparison plots |
| `scatter_*.png` | 55 | Predicted-vs-true scatter plots |

### 6.4 Raw FEM sample coverage by fiber system

| Fiber count | Sample heatmaps present | Sample ID range |
|---:|---:|---|
| 6 | 100 | 0-99 |
| 10 | 16 | 0-15 |
| 20 | 100 | 0-99 |
| 50 | 50 | 0-49 |
| 100 | 50 | 0-49 |

---

## 7. Experiment Execution Coverage Inferred from Output Plots

From prediction/scatter artifacts (55 runs), the completed run families include:

| Experiment family | Runs detected |
|---|---:|
| Attention UNet scratch (nf6) | 4 |
| Attention UNet scratch (nf10) | 4 |
| UNet MSE ablation (nf6) | 4 |
| UNet combined-loss ablation (nf6) | 4 |
| Attention UNet WMSE-only ablation (nf6) | 4 |
| ResNet34 Attention UNet ablation (nf6) | 3 |
| CNN combined-loss baseline (nf6) | 3 |
| nf20 scratch low-data sweeps | 8 |
| nf6 -> nf20 transfer sweeps (3 strategies, multiple sizes/seeds) | 21 |

Transfer sweep details detected in figures:
- strategies: `full_finetune`, `encoder_freeze`, `last_decoder`
- train sizes: `4, 20, 40, 60, 80, 100`
- seeds present in transfer runs: `42, 43`

---

## 8. Plot Gallery (Embedded from outputs/figures)

### 8.1 Raw FEM and stress-field visuals

**FEM grid summaries**

![FEM grid nf6](outputs/figures/heatmap_grid_nf6.png)
![FEM grid nf10](outputs/figures/heatmap_grid_nf10.png)
![FEM grid nf20](outputs/figures/heatmap_grid_nf20.png)
![FEM grid nf50](outputs/figures/heatmap_grid_nf50.png)
![FEM grid nf100](outputs/figures/heatmap_grid_nf100.png)

**Example individual FEM samples**

![FEM sample nf6](outputs/figures/heatmap_nf6_sample0.png)
![FEM sample nf10](outputs/figures/heatmap_nf10_sample0.png)
![FEM sample nf20](outputs/figures/heatmap_nf20_sample0.png)
![FEM sample nf50](outputs/figures/heatmap_nf50_sample0.png)
![FEM sample nf100](outputs/figures/heatmap_nf100_sample0.png)

### 8.2 Dataset and augmentation visuals

![Dataset overview nf6](outputs/figures/dataset_overview_nf6.png)
![Dataset overview nf10](outputs/figures/dataset_overview_nf10.png)
![Dataset overview nf20](outputs/figures/dataset_overview_nf20.png)
![Dataset overview nf50](outputs/figures/dataset_overview_nf50.png)
![Dataset overview nf100](outputs/figures/dataset_overview_nf100.png)

![Augmentation check nf6](outputs/figures/augmentation_check_nf6.png)
![Augmentation check nf10](outputs/figures/augmentation_check_nf10.png)
![Augmentation check nf20](outputs/figures/augmentation_check_nf20.png)
![Augmentation check nf50](outputs/figures/augmentation_check_nf50.png)
![Augmentation check nf100](outputs/figures/augmentation_check_nf100.png)

### 8.3 Stress distribution/statistics visuals

![Stress distribution nf6](outputs/figures/stress_distribution_nf6.png)
![Stress distribution nf10](outputs/figures/stress_distribution_nf10.png)
![Stress distribution nf20](outputs/figures/stress_distribution_nf20.png)
![Stress distribution nf50](outputs/figures/stress_distribution_nf50.png)
![Stress distribution nf100](outputs/figures/stress_distribution_nf100.png)

![Stress stats nf6](outputs/figures/stress_stats_nf6.png)
![Stress stats nf10](outputs/figures/stress_stats_nf10.png)
![Stress stats nf20](outputs/figures/stress_stats_nf20.png)
![Stress stats nf50](outputs/figures/stress_stats_nf50.png)
![Stress stats nf100](outputs/figures/stress_stats_nf100.png)

### 8.4 Model training and prediction visuals

**Base/ablation examples**

![Training curves - attention nf6](outputs/figures/training_curves_attention_unet_nf6_scratch_seed42.png)
![Training curves - unet mse nf6](outputs/figures/training_curves_unet_mse_nf6_seed42.png)
![Training curves - attn wmse nf6](outputs/figures/training_curves_attn_wmse_nf6_seed42.png)
![Training curves - resnet34 attn nf6](outputs/figures/training_curves_resnet34_attn_combined_nf6_seed42.png)

![Predictions - attention nf6](outputs/figures/predictions_best_attention_unet_nf6_scratch_seed42.png)
![Scatter - attention nf6](outputs/figures/scatter_best_attention_unet_nf6_scratch_seed42.png)
![Predictions - resnet34 attn nf6](outputs/figures/predictions_best_resnet34_attn_combined_nf6_seed42.png)
![Scatter - resnet34 attn nf6](outputs/figures/scatter_best_resnet34_attn_combined_nf6_seed42.png)

**Transfer-learning examples**

![Training curves - transfer nf20](outputs/figures/training_curves_attention_unet_nf20_from_nf6_encoder_freeze_n20_seed42.png)
![Predictions - transfer nf20](outputs/figures/predictions_best_attention_unet_nf20_from_nf6_encoder_freeze_n20_seed42.png)
![Scatter - transfer nf20](outputs/figures/scatter_best_attention_unet_nf20_from_nf6_encoder_freeze_n20_seed42.png)

### 8.5 Qualitative error analysis

![Qualitative error nf6](outputs/figures/qualitative_error_nf6.png)
![Qualitative error nf10](outputs/figures/qualitative_error_nf10.png)
![Qualitative error nf20](outputs/figures/qualitative_error_nf20.png)

---

## 9. Quantitative Results (Recorded from Existing Project Logs/Report Snapshots)

The following run-level test metrics were previously documented in the project and remain the explicit numeric baseline currently available in-repo:

| Run | NRMSE | WMSE | SSIM | Median Max Error | Peak Location Error |
|---|---:|---:|---:|---:|---:|
| `unet_mse_nf6` | 3.81% | 0.00231 | 0.8007 | 17.28% | 44.32 px |
| `unet_combined_nf6` | 4.10% | 0.00293 | 0.8037 | 20.60% | 52.35 px |
| `attn_wmse_nf6` | **3.64%** | **0.00209** | 0.8005 | **16.76%** | **41.42 px** |
| `attention_unet_nf6_scratch` | 4.17% | 0.00338 | **0.8065** | 21.45% | 47.73 px |
| `attention_unet_nf10_scratch` | 8.19% | 0.01271 | 0.6687 | 37.49% | 60.80 px |

Observed pattern:
- nf6 baseline/ablation runs show meaningful learning but still above target engineering thresholds.
- nf10 is visibly harder than nf6 under current data/training setup.
- attention + weighted loss improved some peak-critical metrics; SSIM-best and NRMSE-best runs are not identical.

---

## 10. What Is Fully Implemented in This Project

1. **A complete modular codebase** from physics-based data generation to ML training and reporting.
2. **Abaqus simulation automation** with consistent material/BC/mesh setup across fiber systems.
3. **Robust preprocessing and augmentation stack** with geometric and elastic transforms.
4. **Multiple neural architectures** (UNet, Attention UNet, ResNet34-attention, CNN baseline).
5. **Engineering-relevant loss and metrics** beyond plain MSE.
6. **Ablation infrastructure** to isolate architecture/loss contributions.
7. **Transfer-learning infrastructure** with freeze strategies + progressive low-data sweeps.
8. **Comprehensive visualization outputs** (raw FEM, dataset, predictions, scatter, training curves, qualitative errors).
9. **Automated orchestration** through `run_pipeline.py` including seeds and optional step skipping.

---

## 11. Current Gaps in This Snapshot

- `outputs/reports/` CSV artifacts (`all_runs.csv`, `summary_metrics.csv`, `significance_tests.csv`) are not present in the current committed snapshot.
- Qualitative error plots are present for nf6, nf10, nf20 only (not nf50/nf100 in current figure inventory).
- nf10 raw FEM heatmaps currently cover 16 samples in this snapshot, while other systems show larger planned coverage.

---

## 12. Reproduction Commands

### 12.1 Full pipeline
```bash
python run_pipeline.py
```

### 12.2 Skip ABAQUS (use existing extracted data)
```bash
python run_pipeline.py --skip_sim
```

### 12.3 Train a single model
```bash
python training/train_unet.py --n_fibers 6 --arch attention_unet --epochs 100
```

### 12.4 Generate prediction and curve plots for a run
```bash
python visualization/visualize.py --mode predictions --n_fibers 6 --checkpoint outputs/checkpoints/best_attention_unet_nf6_scratch_seed42.pth --n_show 8
python visualization/visualize.py --mode training_curves --log outputs/checkpoints/log_attention_unet_nf6_scratch_seed42.json
```

### 12.5 Aggregate reports
```bash
python visualization/report_results.py --alpha 0.05 --n_show 4
```

---

## Appendix A — Complete Model Run Names with Prediction/Scatter Figures (55)

```text
attention_unet_nf10_scratch
attention_unet_nf10_scratch_seed42
attention_unet_nf10_scratch_seed43
attention_unet_nf10_scratch_seed44
attention_unet_nf20_from_nf6_encoder_freeze_n100_seed42
attention_unet_nf20_from_nf6_encoder_freeze_n20_seed42
attention_unet_nf20_from_nf6_encoder_freeze_n40_seed42
attention_unet_nf20_from_nf6_encoder_freeze_n4_seed42
attention_unet_nf20_from_nf6_encoder_freeze_n4_seed43
attention_unet_nf20_from_nf6_encoder_freeze_n60_seed42
attention_unet_nf20_from_nf6_encoder_freeze_n80_seed42
attention_unet_nf20_from_nf6_full_finetune_n100_seed42
attention_unet_nf20_from_nf6_full_finetune_n20_seed42
attention_unet_nf20_from_nf6_full_finetune_n40_seed42
attention_unet_nf20_from_nf6_full_finetune_n4_seed42
attention_unet_nf20_from_nf6_full_finetune_n4_seed43
attention_unet_nf20_from_nf6_full_finetune_n60_seed42
attention_unet_nf20_from_nf6_full_finetune_n80_seed42
attention_unet_nf20_from_nf6_last_decoder_n100_seed42
attention_unet_nf20_from_nf6_last_decoder_n20_seed42
attention_unet_nf20_from_nf6_last_decoder_n40_seed42
attention_unet_nf20_from_nf6_last_decoder_n4_seed42
attention_unet_nf20_from_nf6_last_decoder_n4_seed43
attention_unet_nf20_from_nf6_last_decoder_n60_seed42
attention_unet_nf20_from_nf6_last_decoder_n80_seed42
attention_unet_nf20_scratch_n100_seed42
attention_unet_nf20_scratch_n20_seed42
attention_unet_nf20_scratch_n20_seed43
attention_unet_nf20_scratch_n40_seed42
attention_unet_nf20_scratch_n4_seed42
attention_unet_nf20_scratch_n4_seed43
attention_unet_nf20_scratch_n60_seed42
attention_unet_nf20_scratch_n80_seed42
attention_unet_nf6_scratch
attention_unet_nf6_scratch_seed42
attention_unet_nf6_scratch_seed43
attention_unet_nf6_scratch_seed44
attn_wmse_nf6
attn_wmse_nf6_seed42
attn_wmse_nf6_seed43
attn_wmse_nf6_seed44
cnn_combined_nf6_seed42
cnn_combined_nf6_seed43
cnn_combined_nf6_seed44
resnet34_attn_combined_nf6_seed42
resnet34_attn_combined_nf6_seed43
resnet34_attn_combined_nf6_seed44
unet_combined_nf6
unet_combined_nf6_seed42
unet_combined_nf6_seed43
unet_combined_nf6_seed44
unet_mse_nf6
unet_mse_nf6_seed42
unet_mse_nf6_seed43
unet_mse_nf6_seed44
```

---

## Appendix B — Naming Patterns for All 508 Output Figures

- `outputs/figures/heatmap_nf{nf}_sample{id}.png`
- `outputs/figures/heatmap_grid_nf{nf}.png`
- `outputs/figures/dataset_overview_nf{nf}.png`
- `outputs/figures/augmentation_check_nf{nf}.png`
- `outputs/figures/stress_distribution_nf{nf}.png`
- `outputs/figures/stress_stats_nf{nf}.png`
- `outputs/figures/training_curves_{run_name}.png`
- `outputs/figures/predictions_best_{run_name}.png`
- `outputs/figures/scatter_best_{run_name}.png`
- `outputs/figures/qualitative_error_nf{nf}.png`

