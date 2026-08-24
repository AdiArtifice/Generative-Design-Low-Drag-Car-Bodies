# Executive Summary

This project builds an **AI-assisted aerodynamic design pipeline** for low-drag EV car bodies. It leverages a **3D vehicle geometry dataset** of 4,165 meshes with associated CFD drag coefficients to train machine learning models and generative design tools. The pipeline includes **mesh inspection**, **geometry normalization**, **point-cloud conversion**, **occupancy grid preprocessing**, **metadata integration**, and a **Conditional Triplane VAE (C-VAE)** architecture. We scaled our implementation from an initial single-configuration sandbox (`F_S_WWC_WM`, 692 samples) to a complete **7-configuration streaming preprocessing workflow** (`E_S_WW_WM`, `E_S_WWC_WM`, `F_S_WWC_WM`, `F_S_WWS_WM`, `N_S_WW_WM`, `N_S_WWC_WM`, `N_S_WWS_WM`), yielding **4,165 preprocessed vehicle samples**. The **C-VAE** conditions the PointNet encoder and Triplane decoder on learned label embeddings (`Fastback=0`, `Estateback=1`, `Notchback=2`) to prevent geometric mode collapse across diverse body shapes.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Overview](#project-overview)  
- [Goals and Motivation](#goals-and-motivation)  
- [Dataset Description](#dataset-description)  
- [7-Configuration Preprocessing Strategy](#7-configuration-preprocessing-strategy)  
- [Conditional VAE Architecture (C-VAE)](#conditional-vae-architecture-c-vae)  
- [Hardware & Storage Constraints](#hardware--storage-constraints)  
- [Directory Structure](#directory-structure)  
- [Dependencies and Setup](#dependencies-and-setup)  
- [Preprocessing Pipeline](#preprocessing-pipeline)  
  - [Mesh Inspection](#mesh-inspection)  
  - [Mesh Normalization](#mesh-normalization)  
  - [Point-Cloud Sampling](#point-cloud-sampling)  
  - [Occupancy Grid Generation](#occupancy-grid-generation)
  - [Metadata & Split Linking](#metadata--split-linking)  
- [Scripts and Usage](#scripts-and-usage)  
- [Modeling and Baseline Results](#modeling-and-baseline-results)  
- [Mermaid Diagrams](#mermaid-diagrams)  
- [Future Roadmap](#future-roadmap)  

---

## Project Overview

This project focuses on **learning geometry-aerodynamics relationships** for car bodies. Rather than relying on heavy CFD, we use data-driven generative AI to predict drag (`Cd`) and optimize 3D vehicle geometry. Key points:

- **AI-Assisted Design**: Use ML surrogates to approximate aerodynamic behavior directly from 3D geometry, enabling instant shape iteration.
- **Full 4,857-Sample Dataset**: Scaled across 7 full vehicle configurations covering Fastback, Estateback, and Notchback body types.
- **Parallel Preprocessing**: Stream-and-delete parallel pipeline to normalize, sample point clouds, and extract occupancy fields while maintaining disk usage below 80 GB.
- **Conditional Generative Model**: Conditional Triplane VAE (C-VAE) using learned category embeddings (`F`, `E`, `N`) to generate sharp, category-conditioned 3D vehicle geometries without mode collapse.
- **EV-Oriented Focus**: Focuses on smooth underbodies, wheel covers, and aerodynamic body profiles relevant to electric vehicle design.

---

## Goals and Motivation

- **Long-Term Vision**: A generative system that ingests 3D car designs and synthesizes optimized, low-drag variants while preserving structural volume.
- **Fast Aerodynamic Surrogate**: Train 3D deep learning models to predict drag from shape as a surrogate for computational fluid dynamics (CFD).
- **Category-Conditioned Latent Space**: Learn a smooth, interpolatable latent representation conditioned on vehicle class to guide gradient-based shape optimization.
- **Pipeline Robustness**: Automated, reproducible data engineering in Python for 3D point clouds and implicit occupancy fields.

---

## Dataset Description

- **Source:** DrivAerNet++ 3D vehicle geometry and aerodynamic dataset.
- **Total Preprocessed Dataset:** **4,165 unique 3D vehicle meshes** across 7 configurations.
- **Point Cloud Representation:** 50,000 surface points + normal vectors per mesh (`.ply` format).
- **Occupancy Representation:** 2,048 interior/exterior query points and occupancy labels per mesh (`.npz` format).
- **Metadata:** Master metadata file (`metadata/metadata.csv`) containing `id`, `config`, `body_type`, `body_type_idx` (`0: F`, `1: E`, `2: N`), `cd`, `drag_area`, and split assignment (`train`, `val`, `test`).

### 7 Target Configurations

| Config Code | Body Type | Underbody | Wheel Covers | Wheel Mesh | Sample Count |
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

## Dataset Split Statistics

The master dataset split is deterministically balanced as follows:

- **Train Set (80%):** 3,332 vehicles
- **Validation Set (10%):** 416 vehicles
- **Test Set (10%):** 417 vehicles
- **Point Cloud Inputs:** 2,048 points sampled dynamically per item during training.
- **Occupancy Inputs:** 2,048 query 3D points paired with binary occupancy labels (0/1).

---

## Conditional VAE Architecture (C-VAE)

To prevent geometric blur when generating vastly different car shapes (e.g., Estateback vs. Fastback vs. Notchback), we updated our Triplane VAE to a **Conditional Triplane VAE (C-VAE)**:

1. **Category Embedding:** `nn.Embedding(num_classes=3, embed_dim=16)` maps `class_idx` (`0: Fastback`, `1: Estateback`, `2: Notchback`) into a 16-dimensional vector $c_{emb}$.
2. **Conditioned Encoder:** PointNet encoder concatenates $c_{emb}$ with global max-pooled features (`512 + 16 = 528`) before outputting latent distribution parameters $\mu$ and $\sigma$.
3. **Conditioned Decoder:** Triplane decoder concatenates $c_{emb}$ with the 256-D latent vector $z$ (`256 + 16 = 272`) to generate spatial feature grids ($64 \times 64 \times 16$) for XY, XZ, and YZ planes.
4. **Implicit Occupancy Decoder:** Queries spatial coordinates $(x,y,z)$ from the triplane grids to predict inside/outside occupancy probabilities.

---

## Hardware & Storage Constraints

- **Storage Policy ("Stream-and-Delete"):** Original raw STL files total hundreds of gigabytes. To maintain disk usage under 80 GB on local drives, raw STL files are streamed, processed into `.ply` point clouds and `.npz` occupancies in chunks of 50, and raw STLs are purged immediately.
- **Training Requirements:** 
  - **Local CPU:** Supports `--smoke_test` for rapid code verification.
  - **Cloud/Local GPU:** Model trained using CUDA (`--batch_size 16` to `32`). Runs on GPUs with $\ge 4\text{ GB}$ VRAM (NVIDIA Titan Xp, T4, L4).

---

## Directory Structure

```plaintext
local_subset/
│
├── pointclouds/                       # Output: 50k sampled surface point clouds (PLY)
│   ├── E_S_WWC_WM/
│   ├── E_S_WW_WM/
│   ├── F_S_WWC_WM/
│   ├── F_S_WWS_WM/
│   ├── N_S_WWC_WM/
│   ├── N_S_WWS_WM/
│   └── N_S_WW_WM/
│
├── occupancy/                         # Output: implicit query points & labels (NPZ)
│   ├── E_S_WWC_WM/
│   ├── E_S_WW_WM/
│   ├── F_S_WWC_WM/
│   ├── F_S_WWS_WM/
│   ├── N_S_WWC_WM/
│   ├── N_S_WWS_WM/
│   └── N_S_WW_WM/
│
├── metadata/
│   ├── metadata.csv                   # Master dataset CSV (4,857 rows, split, class_idx, Cd)
│   ├── computed_features.csv          # Bounding box dimensions & frontal areas
│   └── target_scales.json             # Normalization statistics (Cd & drag_area)
│
├── src/                               # Core Python library
│   ├── dataset.py                     # VehicleOccupancyDataset & VehiclePointCloudDataset
│   └── models/
│       ├── triplane.py                # Conditional Triplane VAE (TriplaneVAE)
│       ├── vae.py                     # Conditional PointNet VAE (PointNetVAE)
│       ├── pointnet.py                # 3D PointNet Drag Regressor
│       └── latent_regressor.py        # MLP Latent Drag Regressor
│
└── scripts/                           # Execution & orchestration scripts
    ├── preprocess_config_batch.py     # Orchestrator for stream-and-delete batching
    ├── preprocess_mesh_combined.py    # Parallel worker script (mesh -> PLY + NPZ)
    ├── link_metadata.py               # Metadata builder & 80/10/10 split generator
    └── train_triplane.py              # C-VAE PyTorch training script
```

---

## Dependencies and Setup

```bash
# Create environment
conda create -n aerodesign python=3.10
conda activate aerodesign

# Install dependencies
pip install numpy pandas trimesh open3d torch torchvision matplotlib pytest
```

---

## Preprocessing Pipeline

```mermaid
graph LR
    raw("Raw STL Stream") --> inspect["Mesh Validation"]
    inspect --> normalize["Unit Normalization"]
    normalize --> sample["Point Cloud (50k PLY)"]
    sample --> occ["Occupancy Sampling (NPZ)"]
    occ --> purge["Purge Raw/Norm STLs"]
    purge --> link["Link Metadata & Split"]
    link --> mlready["Master Dataset (4,857 Samples)"]
```

1. **Mesh Validation:** Verifies face orientation and watertightness.
2. **Normalization:** Translates center-of-mass to origin `(0,0,0)` and scales bounding box max dimension to `1.0`.
3. **Point Cloud Sampling:** Samples 50,000 points with normals on the mesh surface.
4. **Occupancy Sampling:** Samples interior/exterior points relative to the mesh surface.
5. **Storage Cleanup:** Deletes raw and normalized STL files to preserve disk space.
6. **Metadata & Split Generation:** Constructs `metadata/metadata.csv` with class indices and 80/10/10 split allocations.

---

## Scripts and Usage

### 1. Preprocess Single Configuration or Full Batch
```bash
python scripts/preprocess_config_batch.py \
    --stl-dir "G:\.shortcut-targets-by-id\1WOsw0v1GPcX8lMXMErBMlQYwLrQKF3pQ\Main Project Resource\3D meshes of EV cars\N_S_WWS_WM" \
    --config-code "N_S_WWS_WM" \
    --chunk-size 50 \
    --num-workers 4
```

### 2. Train Conditional Triplane VAE (C-VAE)
```bash
# Fast CPU Smoke-Test
python scripts/train_triplane.py --smoke_test

# Full GPU Training (e.g. NVIDIA L4 / T4 / Titan Xp)
python scripts/train_triplane.py --epochs 20 --batch_size 16 --lr 1e-3 --beta 0.005
```

---

## Modeling and Baseline Results

| Model Pipeline | Model Type | Input Features | Test / Val Score | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Random Forest** | Regressor | 29 Tabular Parameters | Test $R^2 = 0.4924$ | Baseline |
| **Gradient Boosting** | Regressor | 29 Tabular Parameters | Test $R^2 = 0.5751$ | Baseline |
| **3D PointNet** | Regressor | Raw 3D Point Cloud (2k points) | Test $R^2 = 0.5633$ | Baseline |
| **Triplane VAE** | Generative | Single Config (`F_S_WWC_WM`) | Val Acc = **85.49%** | Completed |
| **Conditional Triplane VAE (C-VAE)** | Generative | **4,165 Vehicles across 7 Configs** | Val Acc = **80%+** / Latent Regressor $R^2 \sim 0.80+$ | **Completed / Active** |

---

## Future Roadmap

1. **🟢 Phase 6C: C-VAE & Drag Regressor Training** *(Completed)* — Trained the C-VAE and Latent Drag Regressor on the multi-config dataset to learn a smooth, category-conditioned shape space and drag surrogate.
2. **🟢 Phase 6D: Multi-Category Latent Shape Optimization** *(Completed)* — Performed gradient-based shape optimization in the conditioned latent space to synthesize low-drag car bodies with volume constraints.
3. **🟡 Phase 7: OpenFOAM Ground-Truth CFD Validation & Iterative Refinement** — Validate AI-designed vehicle geometries with OpenFOAM CFD on desktop hardware (i7-12700, 16 GB RAM). Budget: **10–15 selective simulations** (~2M cells, half-car symmetry, steady RANS $k$-$\omega$ SST, ~3–5 hrs each):
   - **Stage 1 (Mesh Calibration + Validation, ~5–6 runs):** Calibrate against known DrivAerNet $C_dA$ baselines, then validate 3 AI champion geometries. Quantify surrogate error.
   - **Stage 2 (Affine Surrogate Correction, ~3–4 runs):** Fit $C_dA_{\text{true}} = \alpha \cdot C_dA_{\text{surr}} + \beta$, re-optimize with corrected surrogate, validate corrected champions.
   - **Stage 3 (Final Closed-Loop Validation, ~2–3 runs):** Final publication-quality validation. One additional correction cycle only if residual $\Delta C_dA > 0.005\text{ m}^2$.
4. **🔴 Phase 8: Physics-Informed AI Integration (NVIDIA Modulus)** — Integrate 3D pressure field predictions and PINN surrogates to guide fine-grained aerodynamic shape morphing.