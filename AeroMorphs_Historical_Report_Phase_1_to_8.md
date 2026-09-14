# AeroMorphs: Definitive Historical Project Record (Phases 1–8)

> **Project:** Generative Design for Low-Drag Car Bodies (DrivAerNet++ AI Pipeline)
> **Goal:** An end-to-end AI-assisted 3D aerodynamic design pipeline for electric vehicle car bodies.
> **Status:** Phase 8 Completed. 

This document serves as the comprehensive, evidence-based historical record of the AeroMorphs project. It reconstructs the project chronologically from Phases 1 through 8, preserving the actual sequence of events, including failed approaches, numerical discrepancies, and hard-earned aerodynamic lessons. 

---

## 1. Project Objective and End-to-End Architecture

The central objective of AeroMorphs is to synthesize aerodynamically superior vehicle geometries by coupling a generative AI model with a physical Computational Fluid Dynamics (CFD) validation loop. The goal is to drastically reduce the computational expense of evaluating candidate shapes by searching a latent surrogate space, and only invoking expensive OpenFOAM CFD simulations on the final "champion" candidates.

**The End-to-End Pipeline:**
1. **Data Engineering:** Ingest 4,165 3D STLs from the DrivAerNet++ dataset and extract 2,048-point hybrid point clouds and implicit occupancy fields.
2. **Generative Modeling:** Train a Conditional Triplane Variational Autoencoder (C-VAE) to compress car shapes into a 256-dimensional latent space (`z`).
3. **Surrogate Modeling:** Train a Latent Drag Regressor MLP mapping latent vectors (`z`) to predicted Drag Area ($C_dA$).
4. **Optimization:** Use gradient descent on the surrogate to morph the latent vector toward lower drag, then reconstruct the 3D STL via Marching Cubes.
5. **Physical Validation (Closed-Loop):** Run the AI-generated champion mesh through OpenFOAM CFD. Use the ground-truth physical force discrepancy to update the surrogate and re-optimize.

---

## 2. Phases 1–4: Data Engineering & Representation

**What was built:**
The project began by processing 4,165 raw 3D STLs across 7 configuration categories (Fastback, Estateback, Notchback).

**Key Decisions & Reasoning:**
- **Hybrid Point Cloud Sampling:** Rather than purely uniform sampling, the pipeline implemented a **75% Farthest Point Sampling (FPS) + 25% Curvature Sampling** strategy (via `src/sampling.py`). 
- **Reasoning:** FPS ensures global macro-silhouette coverage, while the curvature sampling concentrates points on aerodynamically critical sharp boundaries (side mirrors, A-pillars, rear fastback separation edges). The clouds were downsampled to 2,048 points with 6D coordinates/normals to eliminate training data loader bottlenecks.

---

## 3. Phase 5: High-Resolution Generative Model Training

**What was built:**
A Conditional Triplane VAE trained on Camber Cloud GPUs.

**Iterative Evolution & Improvements:**
- **Initial State:** A coarse $64 \times 64$ triplane representation that suffered from slow training (4h 48m) due to FUSE cloud storage latency.
- **Resolution Upgrade:** The network was upgraded to **$128 \times 128$ spatial resolution** (16 feature planes), capturing sub-centimeter flow features.
- **I/O Breakthrough:** By caching data to local NVMe/RAM disk (`/tmp`) and double-buffering workers, training time for 200 epochs plummeted to **26 minutes, 59 seconds**.
- **Final Metrics:** The 128×128 C-VAE achieved **90.01% validation occupancy accuracy** (checkpoint: `models/triplane_vae_best_128.pth`).

---

## 4. Phase 6: Surrogate Optimization & The Adversarial Pitfall (Round 1)

**What was built:**
A Latent Drag Regressor MLP trained on the 256-D latent vectors ($R^2 > 0.90$, Val MSE = 0.000879). The script `scripts/optimize_latent_shape.py` was used to run gradient descent on the latent vector to minimize predicted $C_dA$.

**The Experiment & The Failure:**
In Round 1 (Stage 1), the optimizer used a soft quadratic penalty ($\lambda = 0.01$) to prevent the shape from straying too far from the baseline. 
- **The Illusion:** The surrogate confidently predicted massive drag reductions (e.g., -26.35% for Estateback `014`, bringing $C_dA$ from $0.6536 \to 0.4814\text{ m}^2$). 
- **The Reality (Surrogate Hacking):** The surrogate was "hacked." Mathematical analysis revealed the soft penalty allowed the latent vector to travel distances of $\|z - z_0\|_2 \approx 4 - 9$, pushing the geometry completely outside the training distribution manifold (where the median car-to-car distance is 2.51). 
- **The Physical Consequence:** When validated in CFD, the Round 1 Fastback champion actually had a **$+9.74\%$ ($+41.00\text{ N}$)** drag penalty. The unguided descent had over-sharpened the backlight curvature, exceeding the critical separation angle and tripping massive flow recirculation. Notchback geometries similarly exploded in drag (+22.59%) due to Marching Cubes discretization artifacts.

---

## 5. Phase 7: OpenFOAM Baseline Setup & The Discrepancy Audit

**What was built:**
An automated OpenFOAM RANS CFD pipeline (`simpleFoam`, $k$-$\omega$ SST) designed to run on local Ubuntu hardware and GCP Compute Engine.

**The Discrepancy:**
Initial CFD tests yielded $C_dA$ values **46% to 62% higher** than the DrivAerNet++ published ground truth. For example, the Fastback baseline read $0.8006\text{ m}^2$ against a reference of $0.4957\text{ m}^2$.

**The Deep Audit & Rectification (A Major Success):**
Rather than artificially scaling the numbers, the team performed a rigorous physical audit and discovered severe numerical flaws in the initial setup:
1. **Domain Blockage (11.25%):** The virtual wind tunnel was too small, causing artificial Venturi flow acceleration. Expanding the domain to $15\text{m} \times 7.5\text{m}$ (2.31% blockage) and lengthening the wake to 5.7L dropped drag by **-20.2%**.
2. **Missing Prism Layers:** The initial mesh lacked boundary layers, leading to high $y^+ \sim 380$ and premature flow separation. Adding 3 prism layers halved $y^+$ and reduced drag by another **-11.0%**.
3. **Boundary Mapping Bug:** The original `blockMeshDict` mistakenly assigned the moving road condition to the left wall, and slip to the ground. This was rectified.

**Current CFD Limitations (Unresolved):**
The rectified baseline $C_dA$ ($0.5712\text{ m}^2$) is still **+15.25% higher** than DrivAerNet++. The remaining discrepancy is mathematically accounted for but computationally unfeasible to fix on current hardware:
- **Static vs. Rotating Wheels:** AeroMorphs models static wheels merged into the body, causing massive stagnation pressure (~4-6% drag penalty). DrivAerNet uses Moving Reference Frames (MRF).
- **RANS vs. LES Mesh:** The 506k-cell RANS mesh cannot capture shear layers as accurately as DrivAerNet's 12M+ cell setup.

---

## 6. Phase 8: Closed-Loop CFD Refinement (Round 2 Breakthrough)

**What was built:**
The final integration of OpenFOAM physical feedback directly into the AI optimization loop (`cfd_evidence_store.json`). 

**The Corrections:**
1. **Explicit Latent Trust Region:** The soft penalty was replaced with hard Projected Gradient Descent. The latent vector was strictly clamped within a sphere of radius $R_{\text{trust}} \le 0.75$ to prevent out-of-distribution adversarial hallucination.
2. **Active Directional Falsification:** The failed CFD results from Round 1 were used to erect a quadratic repulsive barrier in the latent space.

**The Breakthrough Results:**
Deflected by the repulsive barrier ($83.67^\circ$ away from the failed path), the optimizer found alternate aerodynamic paths (e.g., streamlining lateral greenhouse taper instead of steepening the roof). 
High-fidelity OpenFOAM confirmed physical drag reductions:
- **Fastback Round 2 Champion:** **$-7.53\%$ physical drag reduction** vs. Step 0 Baseline ($421.02\text{ N} \to 389.32\text{ N}$).
- **Estateback Round 2 Champion:** **$-13.88\%$ physical drag reduction** vs. Step 0 Baseline ($571.72\text{ N} \to 492.36\text{ N}$).
- **Notchback Round 2 Champion:** Neutral stability (+0.13%), fully recovering from the massive +22% Stage 1 failure via subdivision and Taubin smoothing.

*(Note: The breakthrough report evaluated a specific `025` triad of vehicles (Fastback 025, Estateback 025, Notchback 025). Earlier Phase 6 and Stage 2 planning documents referenced Fastback 101 and Estateback 014. This discrepancy in vehicle ID testing sets is flagged here for future researchers.)*

---

## 7. Definitively Proved vs. Unproven Claims

**Definitively Proved:**
- The 128×128 C-VAE successfully encodes and decodes highly complex automotive geometries with >90% occupancy accuracy.
- Blind surrogate optimization with soft penalties **will fail** by exploiting out-of-distribution latent geometries (Surrogate Hacking).
- A hard latent trust region ($R_{\text{trust}} \le 0.75$) combined with physical CFD repulsive feedback forces the optimizer to find valid, physically verifiable aerodynamic improvements.
- Expanding computational domains and adding prism layers directly resolves >75% of baseline CFD inaccuracies.

**Remains Unproven / Caveats:**
- The **absolute** $C_dA$ values are still inflated by ~15% due to static wheels and coarse 500k-cell RANS meshes. The pipeline is highly accurate at ranking *relative* design improvements ($\Delta C_dA$), but cannot replace high-fidelity 12M-cell wind-tunnel simulations for absolute homologation values.
- The Affine Bias Correction ($C_dA_{\text{true}} = \alpha \cdot C_dA_{\text{surrogate}} + \beta$) is a statistical heuristic. It carries a 5–7% residual uncertainty and should not be treated as a strict physical law.

---

## 8. Current Project Status & Sensible Next Steps

**Status:**
Phase 8 is complete. The system is functional as an end-to-end local/cloud hybrid application. It successfully demonstrated that iterative CFD-driven surrogate refinement yields true physical drag reductions of 7–13%. 

**Sensible Next Steps:**
1. **Phase 9 Execution (Physics-Informed Neural Fields):** Transition from predicting scalar drag to predicting the full 3D volumetric flow field and surface pressure maps using NVIDIA Modulus or Neural Operators.
2. **Wheel Kinematics:** Implement MRF (Moving Reference Frame) rotating boundary conditions for the wheels, separating them from the main vehicle `noSlip` patch, to close the final 15% absolute accuracy gap.
3. **Evidence Store Scaling:** Expand the `cfd_evidence_store.json` using the automated GCP batch runner to establish a k-NN residual correction model instead of a global affine fit.
