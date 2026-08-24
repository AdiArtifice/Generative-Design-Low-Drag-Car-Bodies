# Project Roadmap: From Multi-Config Dataset to Aerodynamic Optimization

This document outlines the project roadmap following the successful completion of the **Multi-Config Preprocessing Pipeline** and **Conditional Triplane VAE (C-VAE) Refactor**.

---

## 🟢 Completed: Phase 6A: Multi-Config Preprocessing & Dataset Scaling
**Goal:** Expand the dataset from 692 Fastbacks (`F_S_WWC_WM`) to a 7-configuration vehicle dataset covering Fastback, Estateback, and Notchback body types.
* **What we did:** 
  1. Processed all 7 target vehicle configurations (`E_S_WW_WM`, `E_S_WWC_WM`, `F_S_WWC_WM`, `F_S_WWS_WM`, `N_S_WW_WM`, `N_S_WWC_WM`, `N_S_WWS_WM`) using a stream-and-delete parallel pipeline.
  2. Preprocessed **4,165 unique vehicle meshes** into 50,000-point surface point clouds (`.ply`) and 2,048 query point occupancy grids (`.npz`).
  3. Built master metadata file `metadata/metadata.csv` with a deterministic 80/10/10 split (3,332 Train / 416 Val / 417 Test) and class index mappings (`0: Fastback`, `1: Estateback`, `2: Notchback`).
* **Outcome:** Clean, standardized dataset of 4,165 samples ready for multi-class C-VAE training.

---

## 🟢 Completed: Phase 6B: C-VAE Architecture Refactor
**Goal:** Prevent geometric mode collapse when generating diverse vehicle body shapes by conditioning the model on body type class embeddings.
* **What we did:**
  1. Integrated `nn.Embedding(num_classes=3, embed_dim=16)` into `src/models/triplane.py` and `src/models/vae.py`.
  2. Conditioned PointNet Encoder features and Triplane Decoder latents on vehicle category embeddings.
  3. Updated `src/dataset.py` to extract `class_idx` and `scripts/train_triplane.py` to support class-conditioned training.
  4. Verified CPU training via `--smoke_test`.

---

## 🟢 Completed: Phase 6C: C-VAE Training & Drag Regressor Retraining on GPU
**Goal:** Train the C-VAE and Latent Drag Regressor on the complete multi-config dataset.
* **What we did:**
  1. Trained the C-VAE model on GPU (`triplane_vae_best_80.pth`) across the multi-config dataset to learn a smooth, class-conditioned 3D shape space.
  2. Retrained the `LatentDragRegressor` (`latent_regressor_best_80.pth`) on the new multi-config latent space $z$ to predict drag coefficients across Fastbacks, Estatebacks, and Notchbacks.
* **Outcome:** Validated latent representation and high-accuracy drag surrogate model capable of guiding gradient-based shape optimization.

---

## 🟢 Completed: Phase 6D: Multi-Category Latent Shape Optimization
**Goal:** Generate optimized low-drag car bodies across specific vehicle body categories.
* **What we did:**
  1. Executed gradient-based latent space optimization (`scripts/optimize_latent_shape.py`) with L2 regularization, latent vector clamping, and volume constraints to minimize predicted drag.
  2. Extracted watertight STL car bodies from decoded occupancy fields using Marching Cubes (`optimization_output/`).
  3. Dynamic reporting and shape generation validated without out-of-distribution mode collapse.
* **Outcome:** Generated physically realistic low-drag vehicle meshes ready for CFD validation.

---

## 🟡 Next Immediate Step: Phase 7: OpenFOAM Ground-Truth CFD Validation & Iterative Refinement

**Goal:** Validate AI-generated low-drag vehicle geometries using OpenFOAM CFD simulations and close the feedback loop to refine the surrogate model.

**Hardware:** HP Pro Tower 280 G9 — Intel i7-12700 (12C/20T), 16 GB RAM, Ubuntu 24.04.
**CFD Budget:** **10–15 total simulations** (~3–5 hrs each on 8 cores, half-car symmetry, ~2M cells, steady RANS $k$-$\omega$ SST).

> **Design principle:** CFD is selective, not exhaustive. Every simulation must be justified.

### Stage 1: Mesh Calibration + Champion Validation (~5–6 runs)
```text
AI optimization (already done)
      ↓
OpenFOAM validation (one-way)
      ↓
error quantification: CdA_CFD vs CdA_surrogate
```
* **What we do:**
  1. Run 2 known DrivAerNet baseline geometries (with published $C_dA$) to calibrate the mesh setup.
  2. Validate 3 AI champion geometries (one per body type: Fastback, Estateback, Notchback).
  3. Quantify surrogate prediction error: $\Delta C_dA = C_dA_{\text{CFD}} - C_dA_{\text{surrogate}}$.
* **Timeline:** ~1.5 weeks (overnight runs)

### Stage 2: Affine Surrogate Correction + Re-Optimization (~3–4 runs)
```text
AI re-optimization (corrected surrogate)
      ↓
OpenFOAM feedback
      ↓
correct surrogate: CdA_true = α·CdA_surr + β
```
* **What we do:**
  1. Fit an affine correction model $C_dA_{\text{true}} = \alpha \cdot C_dA_{\text{surrogate}} + \beta$ using the 3–5 Stage 1 data points.
  2. Re-run latent space optimization with the bias-corrected drag area predictions.
  3. Validate 3 corrected champion v2 geometries via CFD.
* **Timeline:** ~1 week

### Stage 3: Final Closed-Loop Validation (~2–3 runs)
```text
AI optimization
      ↕
OpenFOAM feedback
      ↕
iterative refinement (if residual > 0.005 m²)
```
* **What we do:**
  1. If Stage 2 residual error $|\Delta C_dA| > 0.005\text{ m}^2$: apply one additional correction cycle.
  2. Final publication-quality CFD validation of best champion geometries.
  3. Generate pressure contour and streamline visualizations in ParaView.
* **Timeline:** ~3–5 days

**Total compute:** ~30–65 wall-clock hours over ~3–4 weeks of overnight batch runs.

---

## 🔴 Phase 8: Physics-Informed AI Integration (NVIDIA Modulus)
**Goal:** Upgrade the aerodynamic evaluator from a scalar regressor to a 3D pressure and velocity field predictor.
* **What we do:** Integrate PINN (Physics-Informed Neural Network) and Neural Operator surrogates using NVIDIA Modulus to evaluate surface pressure distributions and volumetric velocity fields during latent shape morphing.
* **Compute Needed:** **Cloud Multi-GPU Cluster**.
