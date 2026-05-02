# Stress-Field Prediction in Fiber-Reinforced Composite Materials

End-to-end FEM-to-ML pipeline for predicting 2D von Mises stress fields from binary microstructure images.

---

## 1. What is done (current project status)

This repository now includes a complete workflow:

1. **Microstructure generation** (RSA non-overlapping fibers, VF=30%)
2. **ABAQUS simulation** (2D plane-strain linear elasticity)
3. **ODB extraction** (element centroids + von Mises stress)
4. **Dataset building** (`.npz`, 128x128, augmentation + optional elastic)
5. **Model training** (UNet, Attention UNet, ResNet34-Attention UNet, CNN baseline)
6. **Ablations + transfer learning** (multiple seeds, train sizes, freeze strategies)
7. **Visualization** (FEM maps, dataset checks, predictions, scatter, curves)
8. **Reporting** (`all_runs.csv`, `summary_metrics.csv`, `significance_tests.csv`)

---

## 2. Latest outputs snapshot (from `outputs/`)

### 2.1 Artifact totals

| Artifact group | Count |
|---|---:|
| Total files in `outputs/` | 1602 |
| `.npy` files | 1032 |
| `.png` files | 508 |
| `.json` files | 54 |
| `.npz` files | 5 |
| `.csv` files | 3 |

### 2.2 Output directories

- `outputs/checkpoints/`
- `outputs/dataset/`
- `outputs/figures/`
- `outputs/microstructures/`
- `outputs/stress_maps/`
- `outputs/reports/`

### 2.3 Datasets (`outputs/dataset/*.npz`)

| Fiber system | Images in dataset | Unique simulations | Augmentations | Shape |
|---:|---:|---:|---|---|
| nf6 | 400 | 100 | 4-fold (`aug_ids` 0-3) | `(400,1,128,128)` |
| nf10 | 64 | 16 | 4-fold (`aug_ids` 0-3) | `(64,1,128,128)` |
| nf20 | 400 | 100 | 4-fold (`aug_ids` 0-3) | `(400,1,128,128)` |
| nf50 | 200 | 50 | 4-fold (`aug_ids` 0-3) | `(200,1,128,128)` |
| nf100 | 200 | 50 | 4-fold (`aug_ids` 0-3) | `(200,1,128,128)` |

Total dataset images available now: **1264**.

### 2.4 Raw data coverage

| Fiber system | Microstructures (`fibers_*.npy`) | Stress maps (`stress_*.npy`) | Coord maps (`coords_*.npy`) |
|---:|---:|---:|---:|
| nf6 | 100 | 100 | 100 |
| nf10 | 100 | 16 | 16 |
| nf20 | 100 | 100 | 100 |
| nf50 | 50 | 50 | 50 |
| nf100 | 50 | 50 | 50 |

### 2.5 Figures coverage (`outputs/figures`)

| Figure family | Count |
|---|---:|
| `heatmap_nf*_sample*.png` | 316 |
| `heatmap_grid_nf*.png` | 5 |
| `dataset_overview_nf*.png` | 5 |
| `augmentation_check_nf*.png` | 5 |
| `stress_distribution_nf*.png` | 5 |
| `stress_stats_nf*.png` | 5 |
| `qualitative_error_nf*.png` | 3 |
| `predictions_*.png` | 55 |
| `scatter_*.png` | 55 |
| `training_curves_*.png` | 54 |
| **Total PNG figures** | **508** |

### 2.6 Checkpoint/report artifacts

| Path | Current files |
|---|---:|
| `outputs/checkpoints/log_*.json` | 54 |
| `outputs/checkpoints/best_*.pth` | 0 (not present in current snapshot) |
| `outputs/reports/*.csv` | 3 |

Report files present:
- `outputs/reports/all_runs.csv`
- `outputs/reports/summary_metrics.csv`
- `outputs/reports/significance_tests.csv`

---

## 3. Experiment coverage (from `all_runs.csv`)

### 3.1 Total executed runs in report

- **54 runs** recorded

### 3.2 Runs by fiber system

| n_fibers | Runs |
|---:|---:|
| 6 | 22 |
| 10 | 4 |
| 20 | 28 |

### 3.3 Runs by architecture

| Architecture | Runs |
|---|---:|
| attention_unet | 40 |
| unet | 8 |
| resnet34_attention_unet | 3 |
| cnn | 3 |

### 3.4 Transfer vs scratch runs

| Type | Runs |
|---|---:|
| Scratch | 33 |
| Pretrained/transfer | 21 |

---

## 4. Latest metrics (proper numbers from reports)

### 4.1 Best run per fiber system (by NRMSE)

| n_fibers | Run | NRMSE | WMSE | SSIM | Median Max Error | Peak Loc Error |
|---:|---|---:|---:|---:|---:|---:|
| 6 | `attn_wmse_nf6_seed42` | **2.30%** | 0.000772 | 0.8999 | 12.24% | 25.08 px |
| 10 | `attention_unet_nf10_scratch_seed42` | **6.19%** | 0.007643 | 0.7851 | 27.64% | 22.51 px |
| 20 | `attention_unet_nf20_from_nf6_encoder_freeze_n100_seed42` | **2.07%** | 0.000639 | 0.9194 | 15.94% | 26.30 px |

### 4.2 Global best values across all runs

| Metric | Best value | Run |
|---|---:|---|
| NRMSE (lower better) | **2.07%** | `attention_unet_nf20_from_nf6_encoder_freeze_n100_seed42` |
| WMSE (lower better) | **0.000551** | `unet_mse_nf6_seed42` |
| SSIM (higher better) | **0.9205** | `resnet34_attn_combined_nf6_seed43` |
| Median Max Error (lower better) | **10.53%** | `resnet34_attn_combined_nf6_seed42` |
| Peak Location Error (lower better) | **21.39 px** | `attention_unet_nf20_from_nf6_encoder_freeze_n60_seed42` |

### 4.3 Scratch-model summary (mean +/- 95% CI from `summary_metrics.csv`)

| n_fibers | Architecture | n_runs | NRMSE mean +/- CI95 | SSIM mean | Peak Loc Error mean |
|---:|---|---:|---:|---:|---:|
| 6 | attention_unet | 8 | 2.99% +/- 0.43% | 0.8786 | 35.10 px |
| 6 | unet | 8 | 3.02% +/- 0.43% | 0.8788 | 40.55 px |
| 6 | resnet34_attention_unet | 3 | 2.75% +/- 0.29% | 0.9150 | 38.70 px |
| 6 | cnn | 3 | 2.76% +/- 0.47% | 0.8943 | 30.78 px |
| 10 | attention_unet | 4 | 7.19% +/- 0.92% | 0.7527 | 47.04 px |

### 4.4 Significance test summary (`significance_tests.csv`)

- Total tests: **90**
- Significant at `alpha=0.05`: **3**
- All significant findings are in **peak location error** (n_pairs=2, low statistical power).

Significant entries:
1. `encoder_freeze`, train_samples=4: mean gain `+11.34 px`, `p=0.00665`
2. `full_finetune`, train_samples=4: mean gain `+13.70 px`, `p=0.02753`
3. `last_decoder`, train_samples=4: mean gain `-2.17 px`, `p=0.01434`

---

## 5. Key figures

### Raw FEM and dataset checks

![FEM grid nf6](outputs/figures/heatmap_grid_nf6.png)
![FEM grid nf20](outputs/figures/heatmap_grid_nf20.png)
![Dataset overview nf6](outputs/figures/dataset_overview_nf6.png)
![Augmentation check nf6](outputs/figures/augmentation_check_nf6.png)

### Model results

![Training curves attention nf6](outputs/figures/training_curves_attention_unet_nf6_scratch_seed42.png)
![Predictions attention nf6](outputs/figures/predictions_best_attention_unet_nf6_scratch_seed42.png)
![Scatter attention nf6](outputs/figures/scatter_best_attention_unet_nf6_scratch_seed42.png)
![Qualitative error nf20](outputs/figures/qualitative_error_nf20.png)

---

## 6. Pipeline scripts and responsibilities

| Stage | Script |
|---|---|
| RSA generation | `data/generate_fibers.py` |
| ABAQUS simulation | `data/abaqus_simulation.py` |
| ODB extraction | `data/extract_stress_coords.py` |
| End-to-end data generation wrapper | `data/run_dataset.py` |
| Interpolation + dataset build | `preprocessing/build_dataset.py` |
| Models | `models/unet.py`, `models/__init__.py` |
| Training + transfer | `training/train_unet.py` |
| Visualization | `visualization/visualize.py` |
| Aggregated reporting | `visualization/report_results.py` |
| Master orchestration | `run_pipeline.py` |

---

## 7. How to run

### 7.1 Full pipeline
```bash
python run_pipeline.py
```

### 7.2 Skip ABAQUS solves/extraction
```bash
python run_pipeline.py --skip_sim
```

### 7.3 Train one model
```bash
python training/train_unet.py --n_fibers 6 --arch attention_unet --epochs 100
```

### 7.4 Generate plots for a run
```bash
python visualization/visualize.py --mode predictions --n_fibers 6 --checkpoint outputs/checkpoints/best_attention_unet_nf6_scratch_seed42.pth --n_show 8
python visualization/visualize.py --mode training_curves --log outputs/checkpoints/log_attention_unet_nf6_scratch_seed42.json
```

### 7.5 Rebuild aggregated reports
```bash
python visualization/report_results.py --alpha 0.05 --n_show 4
```

---

## 8. Current limitations visible from metrics

Even with strong NRMSE improvements (best 2.07%), current peak-location errors remain much larger than the target engineering threshold (<5 px). Further work should focus on peak-localization-aware training and broader multi-seed transfer coverage.

