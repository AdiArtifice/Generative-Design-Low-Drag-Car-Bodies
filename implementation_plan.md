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
| **Phase 7** | OpenFOAM CFD Validation (Hybrid Local + Cloud) | `Project Contextual Files/phase7_cfd_implementation_plan.md` | ✅ **COMPLETED** | Local PC + Camber/GCP |
| **Phase 8** | Iterative CFD-Driven Surrogate Refinement | `Project Contextual Files/phase8_iterative_cfd_refinement_plan.md` | ✅ **COMPLETED** | Local PC (Prism Layers) |
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

### Compute Strategy: GCP Primary + Local Fallback

CFD validation operates across a proven dual-platform architecture:
1. **GCP Compute Engine (`c2-standard-8`) — Primary MVP Backend:**
   - Officially promoted to primary backend following successful Milestone 1 Parity Benchmark.
   - Verified 100.000% numerical parity against local workstation down to 5 decimal places.
   - Autonomous execution: GCS staging → Spot/On-Demand VM launch → 3,000 iterations in ~56.6 min → GCS upload → guaranteed self-teardown and local watchdog deletion (~$0.31/run on-demand, ~$0.05/run spot).
   - Leaves local workstation unencumbered (0% local CPU load).
2. **Local Ubuntu Desktop (Intel i7-12700, 16 GB RAM) — Reference & Fallback:**
   - Kept in `/home/student/AeroMorphs/cfd_automation` on native ext4 as the golden numerical reference and offline fallback.

### Preserved Simulation Budget & 3-Stage Workflow

| Stage | Focus | Runs | Platforms | Deliverables |
| :---: | :--- | :---: | :--- | :--- |
| **Stage 1** | Mesh Calibration & AI Champion Validation | 5–6 | GCP Primary (Local Fallback) | Calibrated CFD setup matching published DrivAerNet baselines; initial AI surrogate error $\Delta C_dA$. |
| **Stage 2** | Affine Bias Correction & Re-Optimization | 3–4 | Local Optimizer + GCP Primary | Affine correction parameters ($\alpha, \beta$); re-optimized v2 champion geometries. |
| **Stage 3** | Final Closed-Loop Validation | 2–3 | GCP Primary (Local Fallback) | Final publication-grade drag area verification; ParaView surface pressure and wake flow visualizations. (Optional single 2M+ fine mesh run on final winner for publication graphics). |
| **Total** | **Strictly Resource-Bounded** | **10–13** | **GCP Primary + Local Fallback** | **~10–15 total lifetime runs preserved (< $3.00 cloud spend)** |

### Completed Milestones in Phase 7
- [x] **Production Mesh Frozen:** Standardized on Medium `level (3 4)` (~414k cells) after 3-tier study proved asymptotic convergence (1.80% delta vs. 1.15M Fine mesh).
- [x] **Local Python CFD Bridge:** Built `scripts/openfoam_runner.py` to execute outside the encrypted Cryptomator FUSE mount.
- [x] **Stage 1 Complete (All 6 CFD Runs Done):** Evaluated all 3 baselines (Fastback, Estateback, Notchback) and all 3 v1 AI champions. Evidence Store initialized at `metadata/cfd_evidence_store.json`.
- [x] **Surrogate Hacking Diagnosed:** Proved all 3 v1 AI champions increased real drag (+18% to +25%) despite predicted drops (-8% to -31%) due to unconstrained optimization into adversarial latent valleys.
- [x] **Deep CFD Baseline Audit (10-Point Analysis):** Identified root causes of the 46–62% baseline discrepancy: `blockMesh` boundary mapping bug (moving ground placed on side wall), severe domain blockage (11.25%), short wake space (1.35L), and high near-wall $y^+ \sim 380$ (no prism layers).
- [x] **CFD Case Rectification & Validation:**
  - *Step 1:* Rectified boundary mapping (ground $Z=0$ moving at 30 m/s; ceiling & sides slip).
  - *Step 2:* Expanded domain to $15\text{ m} \times 7.5\text{ m}$ (blockage dropped from 11.25% to 2.31%, wake extended to 5.7L). Drag dropped by 20.2% ($C_dA: 0.804 \to 0.642\text{ m}^2$), cutting discrepancy vs DrivAerNet from +62.2% to +29.5%.
  - *Step 4:* Extruded 3 near-wall prism layers (506k cells). $y^+$ halved to 170, and $C_dA$ dropped to **$0.5712\text{ m}^2$ (+15.25% vs DrivAerNet)**, resolving over 75% of the discrepancy.
  - *Step 5:* Built `scratch/split_wheels_body.py` and computed rotating wheel kinematics ($\omega = 96.96\text{ rad/s}$).
- [x] **Stage 2 Explicit Latent Trust Region Implemented:** Added Projected Gradient Descent with hard trust radius $R_{\text{trust}} = 0.75$ (anchored to 10th percentile of training car-to-car distance) in `scripts/optimize_latent_shape.py`.
- [x] **Stage 2 Re-Optimization (v2 Champions):** Generated v2 AI champions across all 3 body types (`optimization_output_v2/`) with verified manifold containment ($\|z - z_0\|_2 \le 0.594$) and denormalized to 1:1 physical scale.

### Immediate Next Action Items for Phase 7
1. **Validate v2 AI Champions in CFD:** Run OpenFOAM CFD on the v2 champion geometries using the validated expanded-domain case to measure real physical drag reduction vs baselines.
2. **Update Evidence Store:** Record v2 validation runs into `metadata/cfd_evidence_store.json`.
3. **Stage 3 Closed-Loop Review:** Compare baseline $\rightarrow$ v1 $\rightarrow$ v2 trajectories and assess final publication-ready figures.

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
