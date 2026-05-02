# Project Progress Report

## 1. Problem Understanding

### What problem are we solving?
We are solving a computational mechanics surrogate-modeling problem: predict the 2D von Mises stress field in a fiber-reinforced composite microstructure under uniaxial tensile loading without running a full ABAQUS FEM simulation every time.

- Input: binary microstructure image (fiber = 1, matrix = 0), size 128x128.
- Output: continuous stress image (von Mises, Pa), size 128x128.

This is an image-to-image regression task (dense prediction), not classification.

### Why this matters
A full FEM solve is expensive. If a neural surrogate can map microstructure to stress quickly, it can accelerate design-space exploration, optimization, and transfer studies across fiber densities.

### What has been explored since proposal submission
The project has moved from concept to a working end-to-end pipeline:

1. RSA microstructure generation implemented.
2. ABAQUS simulation scripting implemented (geometry, materials, BCs, meshing, job submission).
3. ODB stress/centroid extraction implemented.
4. Interpolation + dataset builder implemented.
5. Training pipeline implemented with custom losses and engineering metrics.
6. Visualization pipeline implemented for FEM maps, datasets, predictions, and curves.
7. Ablation runs completed for core baseline/attention/loss variants.
8. Pipeline expanded to include:
   - ResNet-34 attention U-Net architecture support.
   - CNN baseline support.
   - 4-fold/8-fold/none augmentation modes.
   - Optional elastic deformation augmentation.
   - Transfer strategies and progressive training-size support.


## 2. Approach

## 2.1 Modeling choices and rationale

### Base and enhanced models
We currently support the following architectures:

- U-Net baseline (`arch=unet`): reference architecture for dense regression.
- Attention U-Net (`arch=attention_unet`): adds attention gates on skip connections to focus on stress-critical interface regions.
- ResNet34 Attention U-Net (`arch=resnet34_attention_unet`): replaces hand-crafted encoder with ResNet-34 features (optionally ImageNet pretrained).
- CNN baseline (`arch=cnn`): no skip connections, used as a simpler benchmark.

### Loss and optimization
- Combined loss:
  - Weighted MSE: focuses on high-stress zones.
  - SSIM loss: preserves spatial field structure.
  - Final: `L = alpha * WMSE + beta * (1 - SSIM)`.
- Optimizer: AdamW, weight decay = 1e-4.
- Scheduler: Cosine annealing learning-rate schedule.

### Transfer-learning setup
Implemented support for:
- `full_finetune`
- `encoder_freeze`
- `last_decoder`

and progressive train-size experiments via `--train_samples` (4, 20, 40, 60, 80, 100).

## 2.2 Techniques beyond class syllabus and understanding level
In class, we covered basic NN/SVM/logistic/linear regression; this project uses several advanced techniques. Current understanding:

1. U-Net and skip connections: strong conceptual and implementation-level understanding.
2. Attention gates: understood mathematically and architecturally; integrated and used in training/ablation.
3. SSIM-based loss: understood as structure-preserving objective; implemented and used in combined loss.
4. AdamW + cosine schedule: understood and used as standard modern training setup.
5. Transfer learning with staged freezing: understood and implemented with multiple strategies.
6. ResNet encoder integration: implemented and understood at practical level (feature hierarchy, stem adaptation for 1-channel input).
7. Elastic deformation augmentation: implemented with shared spatial warp for input and target to preserve supervision consistency.


## 3. Data Status

## 3.1 Dataset source
Primary data is generated through ABAQUS FEM as proposed:

- Fiber systems: 6, 10, 20, 50, 100 fibers.
- Geometry: unit square with randomly placed non-overlapping circular fibers.
- Volume fraction target: ~30%.
- Mechanics setup: 2D plane strain, uniaxial tensile loading.
- Ground truth: von Mises stress from final load-step frame.

## 3.2 Planned and implemented dataset sizes
The proposal-aligned target after augmentation is:

- 6-fiber: 100 simulations -> 400 images
- 10-fiber: 100 simulations -> 400 images
- 20-fiber: 100 simulations -> 400 images
- 50-fiber: 50 simulations -> 200 images
- 100-fiber: 50 simulations -> 200 images

This corresponds to 4-fold flips. Code now also supports 8-fold mode for optional comparison.

## 3.3 Preprocessing and pipeline details

1. Microstructure generation (`data/generate_fibers.py`)
- Random Sequential Addition (RSA).
- Radius derived from target volume fraction.
- Enforces non-overlap and minimum gap.

2. FEM solve (`data/abaqus_simulation.py`)
- Matrix and fiber elastic properties assigned.
- Left edge fixed in x, bottom-left constrained in y, right edge prescribed displacement.
- Plane-strain element setup and meshing.

3. Stress extraction (`data/extract_stress_coords.py`)
- Reads ODB final frame.
- Extracts von Mises per element.
- Computes element centroids.

4. Interpolation and rasterization (`preprocessing/build_dataset.py`)
- Scattered stress -> regular grid via linear interpolation, nearest-neighbor fallback for NaNs.
- Binary microstructure rendering on same grid.

5. Normalization and augmentation
- Stress normalization to [0, 1] during training using dataset-level `Y_global_min/max`.
- Augmentation modes implemented:
  - none
  - 4fold (proposal default)
  - 8fold
- Elastic deformation implemented as optional post-geometric transform with shared warp for X and Y.

6. Dataset format
Saved in compressed `.npz` with keys:
- `X`, `Y`, `sample_ids`, `aug_ids`, `n_fibers`, `fiber_radius`, `Y_global_min`, `Y_global_max`, `grid_size`.


## 4. Initial Experiments

We already have completed training logs/checkpoints for:

- `attention_unet_nf6_scratch`
- `attention_unet_nf10_scratch`
- `unet_mse_nf6`
- `unet_combined_nf6`
- `attn_wmse_nf6`

## 4.1 Observed results (from logs)

### nf=6 experiments

1. U-Net + MSE (`unet_mse_nf6`)
- NRMSE: 3.81%
- WMSE: 0.00231
- SSIM: 0.8007
- Median Max Error: 17.28%
- Peak Location Error: 44.32 px

2. U-Net + combined loss (`unet_combined_nf6`)
- NRMSE: 4.10%
- WMSE: 0.00293
- SSIM: 0.8037
- Median Max Error: 20.60%
- Peak Location Error: 52.35 px

3. Attention U-Net + WMSE (`attn_wmse_nf6`)
- NRMSE: 3.64% (best among current logged nf6 runs)
- WMSE: 0.00209 (best among current logged nf6 runs)
- SSIM: 0.8005
- Median Max Error: 16.76% (best among current logged nf6 runs)
- Peak Location Error: 41.42 px (best among current logged nf6 runs)

4. Attention U-Net + combined (`attention_unet_nf6_scratch`)
- NRMSE: 4.17%
- WMSE: 0.00338
- SSIM: 0.8065 (best SSIM among current logged nf6 runs)
- Median Max Error: 21.45%
- Peak Location Error: 47.73 px

### nf=10 experiment

5. Attention U-Net + combined (`attention_unet_nf10_scratch`)
- NRMSE: 8.19%
- WMSE: 0.01271
- SSIM: 0.6687
- Median Max Error: 37.49%
- Peak Location Error: 60.80 px

## 4.2 What we observed

1. Model learns nontrivial stress structure (loss and validation reduce across epochs).
2. Current best nf6 run favors WMSE-driven objective with attention (good high-stress accuracy trend).
3. SSIM and pixel RMSE show a tradeoff under current hyperparameter settings.
4. Current performance is still below target thresholds (especially peak location and max-error metrics).
5. nf10 appears significantly harder than nf6 with current setup.


## 5. Progress and Next Steps

## 5.1 Completed objectives

- End-to-end simulation-to-training pipeline implemented.
- Proposal-aligned data generation and preprocessing largely in place.
- Multiple model families integrated (U-Net, attention U-Net, ResNet backbone variant, CNN baseline).
- Ablation infrastructure implemented.
- Transfer-learning infrastructure implemented (strategies + progressive sample sizes).

## 5.2 Pending objectives

1. Run the newly implemented experiments end-to-end (not just support them in code):
- ResNet34 attention U-Net ablations.
- CNN baseline ablation runs.
- Full transfer matrix across source-target pairs, train sizes, and freeze strategies.

2. Statistical significance for transfer gain:
- Repeat runs with multiple seeds.
- Perform significance testing (`p < 0.05`) for pretrained-vs-scratch gains.

3. Improve metric performance toward targets:
- Tune `alpha/beta/gamma` in combined loss.
- Tune augmentation intensity and probability of elastic deformation.
- Try longer training / adjusted LR schedules / regularization.
- Consider peak-focused auxiliary losses (localized supervision around high-stress pixels).

4. Strengthen evaluation and reporting:
- Generate consolidated comparison tables across all runs.
- Add confidence intervals over seeds.
- Include per-fiber-system qualitative error analysis.

## 5.3 Execution plan for upcoming phase

1. Regenerate proposal-aligned 4-fold + elastic datasets for all fiber systems.
2. Re-run base and ablation models with consistent seeds and fixed split policy.
3. Run transfer experiments for all source-target-size-strategy combinations.
4. Aggregate logs to a unified results table and perform significance tests.
5. Finalize model selection using engineering metrics (not only loss).


## 6. Short Summary

The project has moved from planning to a functioning, modular FEM-to-ML pipeline with real initial results and expanded experiment infrastructure. The main remaining work is large-scale execution of the newly implemented transfer/ablation matrix and statistically rigorous comparison to validate gains and approach target engineering performance.
