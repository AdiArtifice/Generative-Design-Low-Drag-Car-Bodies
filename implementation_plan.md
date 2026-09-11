# Master Implementation Plan & Milestone Status Tracker

> **Project:** Generative Design for Low-Drag Car Bodies (DrivAerNet++ AI Aerodynamics Pipeline)  
> **Target Metric:** $C_dA$ (Drag Area, in m²) = $C_d \times A_{\text{frontal}}$  
> **Last Updated:** September 2026

---

## Executive Status Dashboard

| Phase | Description | Scope / Artifact | Status | Platform |
| :---: | :--- | :--- | :---: | :--- |
| **Phase 1** | Core Sampling Module & Geometric Curvature | `src/sampling.py` (FPS + Curvature $k=20$) | ✅ **COMPLETED** | Local CPU |
| **Phase 2** | Offline 4,165-Car Hybrid Downsampling | `pointclouds_hybrid/` (2,048 pts, 6D coords+normals) | ✅ **COMPLETED** | Local CPU (Parallel) |
| **Phase 3** | PyTorch Dataset Loader Integration | `src/dataset.py` (Streaming, caching, batching) | ✅ **COMPLETED** | Local Python |
| **Phase 4** | Sampling & Loader Unit Test Verification | `scripts/unit_tests.py` (Determinism, geometry) | ✅ **COMPLETED** | Local Pytest |
| **Phase 5** | High-Res 128×128 C-VAE Model Training | `models/triplane_vae_best_128.pth` (Acc: 90.01%) | ✅ **COMPLETED** | Camber Cloud GPU |
| **Phase 6** | Latent Drag Surrogate & Shape Optimization | `models/latent_regressor_best_128.pth`, `optimization_output/` | ✅ **COMPLETED** | Local GPU/CPU |
| **Phase 7** | OpenFOAM CFD Validation (Hybrid Local + Cloud) | `Project Contextual Files/phase7_cfd_implementation_plan.md` | 🟡 **ACTIVE** | Local PC + Camber/GCP |
| **Phase 8** | Iterative CFD-Driven Surrogate Refinement | `Project Contextual Files/phase8_iterative_cfd_refinement_plan.md` | ⚪ **PLANNED** | Local PC + Cloud |
| **Phase 9** | Physics-Informed Neural Fields (NVIDIA Modulus) | Surface pressure & volumetric flow field prediction | ⚪ **PLANNED** | Multi-GPU Cluster |

---

## Completed Milestones (Phases 1–6)

### Phases 1–4: 75% FPS + 25% Curvature Hybrid Point Cloud Pipeline

- **Aerodynamic Rationale:** 
  - **75% FPS (1,536 points):** Preserves macro-silhouette, overall vehicle proportions, roofline pitch, and underbody profiles without spatial voids.
  - **25% Curvature Saliency (512 points):** Concentrates points on sharp separation boundaries (side mirrors, A-pillars, rear fastback trailing edges, and wheel arches) using the local covariance surface variation metric:
    $$\sigma(p_i) = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}, \quad (\lambda_0 \le \lambda_1 \le \lambda_2 \text{ over } k=20 \text{ neighbors})$$
- **Dataset Scaling:**
  - Downsampled all **4,165 vehicles** across 7 configurations from raw 50k-point meshes into `pointclouds_hybrid/` with `(2048, 6)` shape `[x, y, z, nx, ny, nz]`.
- **Validation:** 
  - Verified determinism, dimension correctness, and boundary clustering in `scripts/unit_tests.py`.

---

### Phase 5: High-Resolution 128×128 C-VAE Training (Camber Cloud)

- **Model Architecture:**
  - **Encoder:** PointNet mapping `(6, 2048)` points concatenated with 16-dim learned class embeddings (`Fastback=0`, `Estateback=1`, `Notchback=2`) into latent space $\mathcal{Z} \in \mathbb{R}^{256}$.
  - **Decoder:** Generates three dynamic high-resolution feature planes ($XY, XZ, YZ$) at **128×128×16** feature resolution (4× feature density over previous 64×64 baseline).
  - **Implicit Occupancy Decoder:** Bilinearly interpolates triplane features at query coordinates to predict inside/outside probabilities.
- **Training Execution (Camber Cloud Job `26675`):**
  - **Account / Infrastructure:** Fresh Camber Cloud account (`nidhithakur24`), Stash `stash://nidhithakur24/aerodesign`.
  - **Throughput Optimization:** `/tmp` local caching + double-buffered parallel DataLoader (`num_workers=2`, `pin_memory=True`) achieved **~8.5 seconds per epoch**.
  - **Total Training Duration:** **26 minutes, 59 seconds** for 200 epochs.
  - **Performance Metrics:**
    - **Peak Occupancy Validation Accuracy:** **90.01%** (vs ~85% in older 64×64 models).
    - **Best Validation Loss:** **0.2284** (Epoch 179).
  - **Artifacts:**
    - Model Checkpoint: `models/triplane_vae_best_128.pth`
    - Curves: `metadata/triplane_training_128.png`

---

### Phase 6: Latent Drag Surrogate & Closed-Loop Shape Optimization

- **Surrogate Retraining (`models/latent_regressor_best_128.pth`):**
  - Extracted 256-D latent codes for all 4,165 vehicles using the new 128×128 C-VAE encoder.
  - Retrained the 3-layer MLP Latent Drag Regressor with body-class embeddings to predict target drag area $C_dA = C_d \times A_{\text{frontal}}$.
  - **Validation Performance:** Best Val MSE = **0.000879** (R² > 0.90).
- **Closed-Loop Gradient Optimization (`scripts/optimize_latent_shape.py`):**
  - Formulation: $\min_z \text{Regressor}(z, c) + \lambda \|z - z_0\|^2$ with $\lambda=0.01, \text{LR}=0.01$, 250 Adam steps.
  - Target Vehicle: **`E_S_WWC_WM_014`** (Estateback / SUV configuration).
  - **Optimization Results:**
    - Baseline Ground Truth Drag Area: $0.6296\text{ m}^2$ ($C_d = 0.2576$)
    - Baseline Predicted Drag Area: $0.6536\text{ m}^2$
    - Optimized Predicted Drag Area: **$0.4814\text{ m}^2$**
    - **Theoretical Drag Reduction:** **26.35%**
- **High-Resolution Mesh Extraction:**
  - Exported final geometry via **$128^3$ Marching Cubes** isosurface extraction.
  - Output: `optimization_output/optimized_car_step_250.stl` (20,153 vertices, 40,334 triangles, watertight, clean normals).
  - Detailed report: `Project Contextual Files/milestone_report_128_optimization.md`.

---

## Active Milestone: Phase 7 (OpenFOAM CFD Validation)

> **Detailed Execution Document:** `Project Contextual Files/phase7_cfd_implementation_plan.md`

### Hybrid Local + Cloud Strategy

CFD is structured around a flexible multi-platform strategy rather than a single bottlenecked desktop:
1. **Local Ubuntu PC (Intel i7-12700, 16 GB RAM):** Primary environment for initial OpenFOAM case setup, mesh refinement debugging (`snappyHexMesh`), visual inspection in ParaView, and baseline calibration (Runs 1–2).
2. **Camber Cloud (CPU Engine):** Automated batch runner for validated cases (Runs 3–6 and 7–10) utilizing monthly compute quota and avoiding local desktop lockup.
3. **GCP Compute Engine (e.g., `c2-standard-8`):** Elastic burst compute for running concurrent simulation sweeps when cloud credits are available or turnaround speed is critical.

### Preserved Simulation Budget & 3-Stage Workflow

| Stage | Focus | Runs | Platforms | Deliverables |
| :---: | :--- | :---: | :--- | :--- |
| **Stage 1** | Mesh Calibration & AI Champion Validation | 5–6 | Local PC (Runs 1–2); Local / Camber / GCP (Runs 3–6) | Calibrated CFD setup matching published DrivAerNet baselines; initial AI surrogate error $\Delta C_dA$. |
| **Stage 2** | Affine Bias Correction & Re-Optimization | 3–4 | Local Optimizer + Local / Camber / GCP (Runs 7–10) | Affine correction parameters ($\alpha, \beta$); re-optimized v2 champion geometries. |
| **Stage 3** | Final Closed-Loop Validation | 2–3 | Local / Camber / GCP (Runs 11–13) | Final publication-grade drag area verification; ParaView surface pressure and wake flow visualizations. |
| **Total** | **Strictly Resource-Bounded** | **10–13** | **Hybrid Local + Cloud** | **~30–65 core-hours total** |

### Immediate Next Action Items for Phase 7

1. **Local OpenFOAM Installation:** Verify/install OpenFOAM on the Ubuntu desktop (`sudo apt install openfoam`).
2. **Template Wind Tunnel Case:** Create standard virtual wind tunnel case directory with moving ground ($30\text{ m/s}$), inlet ($30\text{ m/s}$), slip boundaries, and half-car symmetry plane ($y=0$).
3. **Run 1 Mesh Calibration:** Execute OpenFOAM simulation on known Fastback baseline STL to calibrate cell count (~2M cells) and turbulence parameters against published DrivAerNet data.

---

## Upcoming Milestones (Phases 8 & 9)

- **Phase 8: Iterative CFD-Driven Surrogate Refinement**
  - Persist CFD results in an evolving **Evidence Store** (`metadata/cfd_evidence_store.json`).
  - Upgrade from global affine correction to localized Gaussian Process / multi-class residual correction as CFD points accumulate.
  - Implement automated orchestrator `scripts/refine_with_cfd.py`.
  - Full details: `Project Contextual Files/phase8_iterative_cfd_refinement_plan.md`.

- **Phase 9: Physics-Informed AI Integration (NVIDIA Modulus)**
  - Integrate 3D Neural Operators (FNO / DeepONet / MeshGraphNet) to evaluate full surface pressure distributions and volumetric velocity fields during latent shape morphing.
  - Target compute: Cloud multi-GPU cluster.
  - Details: `Project Contextual Files/future_roadmap.md`.
