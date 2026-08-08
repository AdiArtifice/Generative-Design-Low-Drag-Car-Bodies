# Project Roadmap: From Multi-Config Dataset to Aerodynamic Optimization

This document outlines the project roadmap following the successful completion of the **Multi-Config Preprocessing Pipeline** and **Conditional Triplane VAE (C-VAE) Refactor**.

---

## 🟢 Completed: Phase 6A: Multi-Config Preprocessing & Dataset Scaling
**Goal:** Expand the dataset from 692 Fastbacks (`F_S_WWC_WM`) to a 7-configuration vehicle dataset covering Fastback, Estateback, and Notchback body types.
* **What we did:** 
  1. Processed all 7 target vehicle configurations (`E_S_WW_WM`, `E_S_WWC_WM`, `F_S_WWC_WM`, `F_S_WWS_WM`, `N_S_WW_WM`, `N_S_WWC_WM`, `N_S_WWS_WM`) using a stream-and-delete parallel pipeline.
  2. Preprocessed **4,857 unique vehicle meshes** into 50,000-point surface point clouds (`.ply`) and 2,048 query point occupancy grids (`.npz`).
  3. Built master metadata file `metadata/metadata.csv` with a deterministic 80/10/10 split (3,885 Train / 485 Val / 487 Test) and class index mappings (`0: Fastback`, `1: Estateback`, `2: Notchback`).
* **Outcome:** Clean, standardized dataset of 4,857 samples ready for multi-class C-VAE training.

---

## 🟢 Completed: Phase 6B: C-VAE Architecture Refactor
**Goal:** Prevent geometric mode collapse when generating diverse vehicle body shapes by conditioning the model on body type class embeddings.
* **What we did:**
  1. Integrated `nn.Embedding(num_classes=3, embed_dim=16)` into `src/models/triplane.py` and `src/models/vae.py`.
  2. Conditioned PointNet Encoder features and Triplane Decoder latents on vehicle category embeddings.
  3. Updated `src/dataset.py` to extract `class_idx` and `scripts/train_triplane.py` to support class-conditioned training.
  4. Verified CPU training via `--smoke_test`.

---

## 🟡 Next Immediate Step: Phase 6C: C-VAE Training & Drag Regressor Retraining on GPU
**Goal:** Train the C-VAE and Latent Drag Regressor on the complete 4,857-sample multi-config dataset.
* **What we do:**
  1. Train the C-VAE model on GPU (`python scripts/train_triplane.py --epochs 20 --batch_size 16 --lr 1e-3 --beta 0.005`) to learn a smooth, class-conditioned 3D shape space.
  2. Retrain the `LatentDragRegressor` on the new multi-config latent space $z$ to predict drag coefficients across Fastbacks, Estatebacks, and Notchbacks.
* **Compute Needed:** **Single GPU (NVIDIA L4 / T4 / Titan Xp)**. 20 epochs will take ~5–12 minutes.

---

## 🟠 Phase 6D: Multi-Category Latent Shape Optimization
**Goal:** Generate optimized low-drag car bodies across specific vehicle body categories.
* **What we do:**
  1. Run gradient-based latent space optimization (`optimize_latent_shape.py`) to minimize predicted drag for target vehicle classes.
  2. Use Marching Cubes to extract watertight STL car bodies from the optimized decoded occupancy fields.
  3. Quantify drag reduction percentages while enforcing volume conservation constraints.

---

## 🔴 Phase 7: Physics-Informed AI Integration (NVIDIA Modulus)
**Goal:** Upgrade the aerodynamic evaluator from a scalar regressor to a 3D pressure and velocity field predictor.
* **What we do:** Integrate PINN (Physics-Informed Neural Network) surrogates to evaluate surface pressure distributions during latent morphing.
* **Compute Needed:** **Cloud Multi-GPU Cluster**.

---

## 🔴 Phase 8: OpenFOAM Ground-Truth CFD Validation
**Goal:** Validate AI-generated low-drag vehicle geometries using industry-standard CFD.
* **What we do:** Run high-fidelity OpenFOAM wind-tunnel simulations on the AI-designed "Champion" car geometries.
* **Compute Needed:** **HPC CPU Cluster**.
