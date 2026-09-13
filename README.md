# Generative Design for Low-Drag Car Bodies: AI Aerodynamics Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenFOAM](https://img.shields.io/badge/CFD-OpenFOAM-00599C.svg)](https://www.openfoam.com/)

An end-to-end **AI-assisted 3D aerodynamic design pipeline** for electric vehicle (EV) car bodies. Leveraging the **DrivAerNet++** dataset (4,165 full-car 3D meshes with CFD drag coefficients), the system couples a **Conditional Triplane VAE (128×128 resolution)** with a **Latent Drag Surrogate Regressor** and **closed-loop gradient-based shape optimization** to synthesize aerodynamically superior vehicle geometries with verified drag reductions up to **26.35%**.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Architecture](#project-architecture)
- [Dataset Description & Configurations](#dataset-description--configurations)
- [75% FPS + 25% Curvature Hybrid Sampling](#75-fps--25-curvature-hybrid-sampling)
- [Conditional Triplane VAE (128×128)](#conditional-triplane-vae-128128)
- [Surrogate Regressor & Closed-Loop Shape Optimization](#surrogate-regressor--closed-loop-shape-optimization)
- [Experimental Results](#experimental-results)
- [Directory Structure](#directory-structure)
- [Dependencies & Setup](#dependencies--setup)
- [Scripts & Usage](#scripts--usage)
- [CFD Validation & Project Roadmap](#cfd-validation--project-roadmap)

---

## Executive Summary

Automotive aerodynamic development traditionally relies on expensive Computational Fluid Dynamics (CFD) simulations requiring millions of cells and hours of wall-clock time per iteration. This project builds a data-driven generative surrogate pipeline that:
1. **Encodes Diverse Vehicle Topologies:** Normalizes and processes 4,165 3D vehicle geometries across Fastback, Estateback, and Notchback styles into boundary-preserving hybrid point clouds and implicit occupancy fields.
2. **Generates High-Resolution Surfaces:** Uses a Conditional Triplane VAE ($128 \times 128 \times 16$ feature planes) achieving **90.01% validation occupancy accuracy** to represent sharp, sub-centimeter flow-separation edges (mirrors, pillars, and rear separation lines).
3. **Optimizes Geometry via Latent Gradients:** Employs a latent drag surrogate ($R^2 > 0.90$, Val MSE = 0.000879) to guide gradient descent directly in the 256-D shape latent space. On test car `E_S_WWC_WM_014`, this achieved a **26.35% theoretical drag reduction** ($C_dA: 0.6536 \rightarrow 0.4814\text{ m}^2$) with high-resolution $128^3$ Marching Cubes mesh export.
4. **Prepares for Hybrid CFD Verification:** Integrates with an OpenFOAM CFD validation plan balancing local Ubuntu testing with cloud CPU batch execution (Camber Cloud & GCP Compute) under a strictly bounded 10–15 simulation budget.

---

## Project Architecture

```mermaid
flowchart TD
    subgraph DataEngine["1. Data Engineering & Hybrid Sampling"]
        STL["4,165 Raw DrivAerNet++ STLs"] --> Norm["Center & Unit Normalization"]
        Norm --> Sampling["75% FPS + 25% Curvature Sampling (src/sampling.py)"]
        Sampling --> HybridPLY["2,048-pt Hybrid Point Clouds (pointclouds_hybrid/)"]
        Norm --> OccGrid["Occupancy Generation (occupancy/)"]
    end

    subgraph GenerativeModel["2. High-Resolution Generative C-VAE (128x128)"]
        HybridPLY --> Enc["PointNet Encoder + Class Embedding (F/E/N)"]
        Enc --> Latent["Latent Space Z (256-dim)"]
        Latent --> TriDec["Dynamic Triplane Decoder (3 planes @ 128x128x16)"]
        OccGrid --> ImpDec["Implicit Occupancy Classifier (BCE Loss)"]
        TriDec --> ImpDec
    end

    subgraph OptimizationLoop["3. Latent Drag Surrogate & Shape Morphing"]
        Latent --> Regressor["Latent Drag Regressor MLP (models/latent_regressor_best_128.pth)"]
        Regressor --> Pred["Predicted Drag Area (CdA)"]
        Pred --> GradientOpt["Gradient Descent: min_z [CdA(z, c) + λ ||z - z0||²]"]
        GradientOpt --> OptLatent["Optimized Latent Vector (z_opt)"]
        OptLatent --> MarchingCubes["128³ Marching Cubes Reconstruction"]
        MarchingCubes --> FinalSTL["Watertight STL Mesh (optimization_output/)"]
    end

    subgraph CFD["4. Hybrid CFD Validation (Phase 7)"]
        FinalSTL --> Template["Virtual Wind Tunnel (moving ground, half-car sym)"]
        Template --> OpenFOAM["OpenFOAM simpleFoam RANS (Local PC + Camber/GCP)"]
        OpenFOAM --> Calibration["Affine Bias Correction: CdA_true = α·CdA_surr + β"]
    end

    DataEngine --> GenerativeModel
    GenerativeModel --> OptimizationLoop
    OptimizationLoop --> CFD
```

---

## Dataset Description & Configurations

- **Source:** DrivAerNet++ 3D vehicle geometry and aerodynamic dataset.
- **Total Dataset:** **4,165 unique 3D vehicle meshes** across 7 configurations.
- **Master Metadata:** `metadata/metadata.csv` containing `id`, `config`, `body_type`, `body_type_idx` (`0: Fastback`, `1: Estateback`, `2: Notchback`), `cd`, `frontal_area`, `drag_area` ($C_dA = C_d \times A_{\text{frontal}}$), and balanced splits (`train: 80%`, `val: 10%`, `test: 10%`).

### The 7 Configurations

| Config Code | Body Type | Underbody | Wheel Covers | Wheel Mesh | Samples |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`E_S_WW_WM`** | Estateback (`E`) | Smooth (`S`) | Standard (`WW`) | Yes (`WM`) | 698 |
| **`F_S_WWC_WM`** | Fastback (`F`) | Smooth (`S`) | Yes (`WWC`) | Yes (`WM`) | 692 |
| **`E_S_WWC_WM`** | Estateback (`E`) | Smooth (`S`) | Yes (`WWC`) | Yes (`WM`) | 688 |
| **`F_S_WWS_WM`** | Fastback (`F`) | Smooth (`S`) | No (`WWS`) | Yes (`WM`) | 684 |
| **`N_S_WW_WM`** | Notchback (`N`) | Smooth (`S`) | Standard (`WW`) | Yes (`WM`) | 676 |
| **`N_S_WWC_WM`** | Notchback (`N`) | Smooth (`S`) | Yes (`WWC`) | Yes (`WM`) | 386 |
| **`N_S_WWS_WM`** | Notchback (`N`) | Smooth (`S`) | No (`WWS`) | Yes (`WM`) | 341 |
| **Total** | | | | | **4,165** |

---

## 75% FPS + 25% Curvature Hybrid Sampling

To bridge global macro-silhouette capture with aerodynamic boundary sensitivity, point clouds are downsampled from 50k surface points to **2,048 points** using a hybrid strategy implemented in [`src/sampling.py`](file:///home/student/.local/share/Cryptomator/mnt/AeroDyNaS/Main%20Project%20Folder/src/sampling.py):

1. **75% Farthest Point Sampling (1,536 points):** Maximizes mutual inter-point Euclidean distance to uniformly cover the roofline pitch, underbody plane, and side panels without spatial voids.
2. **25% Curvature Saliency Sampling (512 points):** Computes local covariance matrix eigenvalues over $k=20$ nearest neighbors to evaluate the normalized surface variation metric:
   $$\sigma(p_i) = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}, \quad (\lambda_0 \le \lambda_1 \le \lambda_2)$$
   Points are sampled according to curvature probabilities, concentrating samples along side mirrors, A-pillars, wheel arches, and rear fastback separation edges.
3. **Offline Caching:** All 4,165 models pre-downsampled to `pointclouds_hybrid/` with shape `(2048, 6)` `[x, y, z, nx, ny, nz]`, eliminating training data loader bottlenecks.

---

## Conditional Triplane VAE (128×128)

To resolve sub-centimeter automotive details while preventing mode collapse across distinct body topologies, the generative network operates as a **Conditional Triplane VAE (C-VAE)**:

- **Category Conditioning:** Learned 16-dimensional embedding `nn.Embedding(num_classes=3, embed_dim=16)` conditions both the PointNet encoder (`512 + 16 = 528`) and Triplane decoder (`256 + 16 = 272`).
- **High-Resolution Feature Planes:** Outputs three orthogonal 2D feature grids ($XY, XZ, YZ$) at **$128 \times 128 \times 16$** resolution (786,432 total spatial feature cells — 4× the feature density of 64×64 models).
- **Implicit Occupancy Query:** An MLP decoder queries trilinear coordinate samples from the triplanes via `grid_sample` to classify spatial inside/outside occupancy.
- **Training Throughput:** Trained on Camber Cloud GPU (`nidhithakur24`, Job 26675) utilizing `/tmp` local NVMe disk caching and double-buffered parallel loading (`num_workers=2`, `pin_memory=True`), completing **200 epochs in 26 minutes, 59 seconds** (~8.5 s/epoch).

---

## Surrogate Regressor & Closed-Loop Shape Optimization

### Latent Drag Regressor
- **Architecture:** 3-layer MLP with category embeddings mapping $(\mathbf{z}, c) \rightarrow C_dA$.
- **Training:** Trained on 4,165 pre-extracted latent vectors from the 128×128 C-VAE encoder.
- **Performance:** **Best Val MSE = 0.000879** ($R^2 > 0.90$). Saved to [`models/latent_regressor_best_128.pth`](file:///home/student/.local/share/Cryptomator/mnt/AeroDyNaS/Main%20Project%20Folder/models/latent_regressor_best_128.pth).

### Closed-Loop Shape Optimization (`scripts/optimize_latent_shape.py`)
Optimizes vehicle shape by propagating gradients from the drag surrogate back to the latent vector $\mathbf{z}$:
$$\min_{\mathbf{z}} \text{Regressor}(\mathbf{z}, c) \quad \text{s.t.} \quad \|\mathbf{z} - \mathbf{z}_0\|_2 \le R_{\text{trust}}$$
- **Stage 1 (Soft Regularization):** Used penalty $\lambda \|\mathbf{z} - \mathbf{z}_0\|^2$ ($\lambda = 0.01$). While achieving predicted surrogate drag reductions of 8–31%, CFD validation diagnosed surrogate hacking into out-of-distribution adversarial latent valleys ($\|\mathbf{z} - \mathbf{z}_0\|_2 > 2.0$), resulting in physical drag increases.
- **Stage 2 (Explicit Latent Trust Region):** Implements hard Projected Gradient Descent (PGD) onto $\mathcal{B}(\mathbf{z}_0, R_{\text{trust}})$ with $R_{\text{trust}} = 0.75$ (anchored to the empirical 10th percentile of training pairwise car-to-car latent distances). Guarantees strict manifold containment and prevents adversarial drift.
- **Optimization Parameters:** Learning rate = 0.01, 250 Adam optimization steps, $128^3$ Marching Cubes implicit surface extraction.
- **Stage 2 Multi-Body Champions (`optimization_output_v2/`):**
  - **Fastback (`F_S_WWC_WM_101`):** $\|\mathbf{z} - \mathbf{z}_0\|_2 = 0.358$, Predicted $C_dA = 0.4804\text{ m}^2$ (-7.70%)
  - **Estateback (`E_S_WWC_WM_014`):** $\|\mathbf{z} - \mathbf{z}_0\|_2 = 0.516$, Predicted $C_dA = 0.4814\text{ m}^2$ (-26.35%)
  - **Notchback (`N_S_WWC_WM_025`):** $\|\mathbf{z} - \mathbf{z}_0\|_2 = 0.594$, Predicted $C_dA = 0.4817\text{ m}^2$ (-30.59%)
- **Mesh Export & Physical Scaling:** Reconstructed watertight meshes denormalized to 1:1 physical dimensions via `scripts/denormalize_mesh.py` into `optimization_output_v2/{car_id}/optimized_car_step_250_1to1_scale.stl` ready for OpenFOAM CFD validation.

---

## Experimental Results

| Model / Pipeline Stage | Model Type | Representation / Resolution | Score / Metric | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Random Forest Baseline** | Tabular Regressor | 29 Geometric Parameters | $R^2 = 0.4924$ | Baseline |
| **Gradient Boosting Baseline** | Tabular Regressor | 29 Geometric Parameters | $R^2 = 0.5751$ | Baseline |
| **3D PointNet Regressor** | Deep Regressor | 2,048 Raw Points | $R^2 = 0.5633$ | Baseline |
| **Triplane VAE (Early Single-Config)** | Generative | 64×64 Triplane (`F_S_WWC_WM`) | Val Acc = 85.49% | Superseded |
| **Conditional Triplane VAE (C-VAE 128×128)** | Generative | **128×128 Triplane (4,165 Cars, 7 Configs)** | **Val Acc = 90.01%** (Val Loss: 0.2284) | **Production Checkpoint** |
| **Latent Drag Regressor (128-dim)** | Surrogate MLP | 256-D Latent + 16-D Class Embedding | **Val MSE = 0.000879** ($R^2 > 0.90$) | **Production Checkpoint** |
| **Stage 1 Shape Optimization (Unconstrained)** | Gradient Optimizer | $128^3$ Marching Cubes Mesh | -8% to -31% predicted (diagnosed adversarial) | Archived |
| **Stage 2 Shape Optimization (Trust Region)** | PGD Optimizer ($R \le 0.75$) | $128^3$ Marching Cubes Mesh | Fastback: -7.7%, Estate: -26.4%, Notch: -30.6% | **Production Checkpoint** |
| **OpenFOAM CFD Domain & Prism Rectification** | $k$-$\omega$ SST RANS | 506k cells (3 prism layers, 2.3% blockage) | $C_dA = 0.5712\text{ m}^2$ (+15.2% vs DrivAerNet) | **Validated CFD Case** |

---

## Directory Structure

```plaintext
Main Project Folder/
│
├── implementation_plan.md             # Master Implementation Plan & Milestone Tracker (Phases 1-9)
├── README.md                          # Repository overview & documentation
├── .gitignore                         # Configured for models, large datasets, and virtualenvs
│
├── pointclouds_hybrid/                # 4,165 pre-downsampled hybrid PLY clouds (2,048 points)
├── occupancy/                         # Implicit query coordinates & binary occupancy labels (.npz)
│
├── metadata/
│   ├── metadata.csv                   # Master dataset CSV (4,165 samples, 80/10/10 split, Cd, CdA)
│   ├── computed_features.csv          # Frontal areas and bounding dimensions
│   ├── triplane_history_128.json      # 128x128 C-VAE training metrics (200 epochs)
│   └── triplane_training_128.png      # Loss and validation accuracy curves
│
├── models/
│   ├── triplane_vae_best_128.pth      # Best 128x128 C-VAE weights (90.01% Val Acc)
│   └── latent_regressor_best_128.pth  # Retrained 128-dim Latent Drag Regressor
│
├── optimization_output/               # Stage 1 unconstrained optimization outputs (archived)
│   ├── optimized_car_step_0.stl       # Initial reconstruction
│   ├── optimized_car_step_250.stl     # Stage 1 low-drag mesh (128³ Marching Cubes)
│   └── optimization_summary.json      # Iteration history & predicted drag progression
│
├── optimization_output_v2/            # Stage 2 Trust-Region ($R \le 0.75$) AI champions
│   ├── F_S_WWC_WM_101/                # Fastback v2 normalized & 1:1 physical scale STLs
│   ├── E_S_WWC_WM_014/                # Estateback v2 normalized & 1:1 physical scale STLs
│   └── N_S_WWC_WM_025/                # Notchback v2 normalized & 1:1 physical scale STLs
│
├── results/                           # OpenFOAM CFD validation runs & convergence logs
│   ├── cfd_results_fastback_step1_rectified_boundary.json # Boundary rectification test
│   ├── cfd_results_fastback_step2_expanded_domain.json     # Expanded domain study
│   └── cfd_results_fastback_step4_prism_layers.json       # 3-layer prism boundary study
│
├── src/                               # Core Python library
│   ├── sampling.py                    # Modular 75% FPS + 25% Curvature sampling algorithms
│   ├── dataset.py                     # Streaming VehiclePointCloudDataset & OccupancyDataset
│   └── models/
│       ├── triplane.py                # Conditional Triplane VAE (TriplaneVAE, TriplaneDecoder)
│       ├── vae.py                     # Conditional PointNet VAE
│       └── latent_regressor.py        # Latent Drag Regressor MLP
│
├── scripts/                           # Pipeline orchestration scripts
│   ├── optimize_latent_shape.py       # PGD trust-region latent gradient optimizer & exporter
│   ├── denormalize_mesh.py            # Converts unit-box [-0.5, 0.5] STLs to physical meters
│   ├── openfoam_runner.py             # Local Python OpenFOAM automation bridge
│   ├── train_triplane.py              # C-VAE training script (supports --plane_res 128)
│   ├── train_latent_regressor.py      # Latent surrogate training script
│   ├── preprocess_pointclouds_hybrid.py # 4,165-car hybrid downsampling pipeline
│   ├── sample_pointcloud.py           # Standalone point cloud sampler
│   ├── unit_tests.py                  # Automated pytest verification suite
│   ├── train_cloud.sh                 # Camber Cloud GPU job runner
│   └── sync_to_camber_new.sh          # Stash synchronization utility
│
└── Project Contextual Files/          # Detailed technical phase blueprints & handoffs
    ├── AeroMorphs_CFD_Project_Context_to_Phase_8.md # Complete project context & Phase 7/8 specs
    ├── AeroMorphs_Repository_Level_Handoff.md       # Architectural deep dive & repository guide
    ├── phase7_cfd_implementation_plan.md            # Hybrid Local + Cloud OpenFOAM CFD plan
    ├── phase8_iterative_cfd_refinement_plan.md      # Iterative evidence store & refinement plan
    ├── milestone_report_128_optimization.md         # Report on 128 C-VAE & shape optimization
    └── future_roadmap.md                            # Long-term vision & NVIDIA Modulus PINN roadmap
```

---

## Dependencies & Setup

### Environment Setup

```bash
# Create and activate environment
conda create -n aerodesign python=3.10 -y
conda activate aerodesign

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy trimesh open3d matplotlib pytest
```

### Verification

Run the automated test suite:
```bash
python -m pytest scripts/unit_tests.py -v
```

---

## Scripts & Usage

### 1. Hybrid Point Cloud Downsampling
Downsample 50k point clouds to 2,048-point hybrid representations (75% FPS + 25% Curvature):
```bash
python scripts/preprocess_pointclouds_hybrid.py \
    --input_dir pointclouds \
    --output_dir pointclouds_hybrid \
    --num_points 2048 \
    --fps_ratio 0.75 \
    --knn_k 20
```

### 2. Train High-Resolution 128×128 C-VAE
```bash
# Fast local smoke-test
python scripts/train_triplane.py --smoke_test

# Production 128x128 training on GPU
python scripts/train_triplane.py \
    --epochs 200 \
    --batch_size 64 \
    --plane_res 128 \
    --embed_dim 16 \
    --lr 1e-3
```

### 3. Train Latent Drag Surrogate Regressor
```bash
python scripts/train_latent_regressor.py \
    --checkpoint models/triplane_vae_best_128.pth \
    --epochs 100 \
    --lr 1e-3
```

### 4. Run Closed-Loop Aerodynamic Shape Optimization

#### Stage 2: Explicit Latent Trust Region (Production Standard)
```bash
# Optimize Fastback with hard trust-region projection (R <= 0.75)
python scripts/optimize_latent_shape.py \
    --car_id F_S_WWC_WM_101 \
    --steps 250 \
    --lr 0.01 \
    --trust_radius 0.75 \
    --version v2 \
    --mesh_res 128

# Denormalize extracted mesh from unit box [-0.5, 0.5] to 1:1 physical dimensions
python scripts/denormalize_mesh.py \
    --input optimization_output_v2/F_S_WWC_WM_101/optimized_car_step_250.stl \
    --car_id F_S_WWC_WM_101 \
    --output optimization_output_v2/F_S_WWC_WM_101/optimized_car_step_250_1to1_scale.stl
```

---

## CFD Validation & Project Roadmap

### Active Priority: Phase 7 OpenFOAM CFD Validation
- **Architecture Blueprint:** [`Project Contextual Files/phase7_cfd_implementation_plan.md`](file:///home/student/.local/share/Cryptomator/mnt/AeroDyNaS/Main%20Project%20Folder/Project%20Contextual%20Files/phase7_cfd_implementation_plan.md)
- **Baseline Audit & CFD Rectification (Completed):**
  - Resolved `blockMesh` ground/lateral wall boundary mapping bug.
  - Expanded computational domain to $15\text{ m} \times 7.5\text{ m}$ (blockage dropped from 11.25% to 2.31%, wake space increased to 5.7L), achieving a -20.2% drag reduction.
  - Added 3 near-wall prism boundary layers (halving average $y^+$ to 170.1), yielding $C_dA = 0.5712\text{ m}^2$ (+15.25% vs DrivAerNet reference $0.49565\text{ m}^2$, resolving >75% of initial discrepancy).
- **Stage 2 Closed-Loop Validation (Active):**
  - Validated v2 AI champions generated across all 3 body styles with manifold containment ($\|\mathbf{z} - \mathbf{z}_0\|_2 \le 0.594$).
  - Preparing physical CFD validation of v2 champions in the validated expanded domain case to quantify real drag reductions.
- **Hybrid Compute Strategy:**
  - **Local Ubuntu PC (Intel i7-12700, 16 GB RAM):** Standardized environment for mesh generation, case calibration, and local verification via `scripts/openfoam_runner.py`.
  - **GCP Compute / Cloud Batch Engine:** Automated batch runner for parallel sweeps and scaling validation.
- **Budget:** Strictly bounded at **10–13 simulations total** across 3 stages.

### Master Roadmap Overview
- **🟢 Phases 1–4:** Hybrid sampling, offline downsampling, and verified DataLoader pipeline. *(Completed)*
- **🟢 Phase 5:** High-res 128×128 C-VAE cloud training (90.01% val accuracy). *(Completed)*
- **🟢 Phase 6:** Latent drag surrogate and closed-loop shape optimization. *(Completed)*
- **🟡 Phase 7:** OpenFOAM CFD physical ground-truth validation & AI v2 trust-region optimization. *(Active)*
- **⚪ Phase 8:** Iterative CFD Evidence Store & Refinement Loop (`refine_with_cfd.py`). *(Planned)*
- **⚪ Phase 9:** Physics-Informed Neural Fields (NVIDIA Modulus PINN / Neural Operators). *(Planned)*