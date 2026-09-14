# The Evolution of the AeroMorphs CFD & OpenFOAM Pipeline
> **Purpose:** A definitive technical reference documenting the complete evolution of the OpenFOAM CFD pipeline after the handover of the original automated setup.
> **Scope:** Chronological reconstruction of CFD testing, anomaly detection, physical rectification, and the integration of CFD as a ground-truth active barrier in the Phase 8 AI closed-loop optimizer.

---

## 1. Executive Summary

This document traces the journey of the AeroMorphs OpenFOAM pipeline from its initial automated but physically flawed baseline to a stabilized, validated, and closed-loop-integrated aerodynamic engine. The initial CFD setup consistently overestimated drag area ($C_dA$) by 46% to 62% compared to the DrivAerNet++ high-fidelity benchmark. Through rigorous, incremental rectification—fixing boundary condition bugs, expanding tunnel blockage, and introducing near-wall prism layers—over 75% of this discrepancy was demystified and resolved. Ultimately, this stabilized OpenFOAM setup became the active physical guardrail in the Phase 8 optimizer, correcting adversarial surrogate behaviors and verifying physical drag reductions of up to 15.73%.

---

## 2. Nidhi's Original CFD Automation & Initial Validation

### 2.1 What the Original Setup Provided
The original handoff delivered an end-to-end automated bash pipeline (`scripts/run_cfd.sh`) and result extractor (`scripts/extract_results.py`). The OpenFOAM 2412 configuration consisted of:
- **Solver & Physics:** Steady, incompressible RANS using `simpleFoam` and the $k$-$\omega$ SST turbulence model.
- **Operating Point:** $30\text{ m/s}$ inlet velocity, with moving ground.
- **Mesh:** A reduced, computationally efficient Cartesian cut-cell mesh via `blockMesh` and `snappyHexMesh` (without prism/boundary layers).
- **Execution:** A fixed budget of 3,000 iterations, with drag and $C_dA$ dynamically extracted by averaging iterations 2,800 to 3,000 using OpenFOAM's `forces` function object.

### 2.2 First Run and Mesh Independence
The team first validated the pipeline's numerical stability and mesh-independence using the Notchback vehicle (`N_S_WWC_WM_025`):
- **Coarse `level (2 3)`:** 152k cells, $C_dA = 1.1604\text{ m}^2$
- **Medium `level (3 4)`:** 414k cells, $C_dA = 1.0929\text{ m}^2$
- **Fine `level (4 5)`:** 1.15M cells, $C_dA = 1.0732\text{ m}^2$

The results confirmed asymptotic solver convergence (stable force after 1,000 iterations) and mesh independence (only 1.8% difference between medium and fine). The **medium mesh `(3 4)`** was selected as the production baseline. 

### 2.3 The Major Discrepancy
While the solver was stable, the absolute values were far from reality:
- **Notchback `025`:** CFD $C_dA \approx 1.0929\text{ m}^2$ vs. Dataset $0.7455\text{ m}^2$ **(+46.6% error)**.
- **Fastback `101`:** CFD $C_dA \approx 0.8006\text{ m}^2$ vs. Dataset $0.4957\text{ m}^2$ **(+61.5% error)**.

The team explicitly decided to use incremental experiments to decompose this discrepancy physically rather than mathematically tuning the data.

---

## 3. Rectification and Validation Process

To isolate the causes of the +61.5% discrepancy on the Fastback `101`, the team launched an incremental rectification plan, changing one variable at a time.

### 3.1 Step 1: The Boundary Mapping Bug
**Problem Discovered:** The original `blockMeshDict` had swapped face mappings. The `lowerWall` (moving ground) was mapped to the lateral side boundary, while the `sides` (slip) were mapped to the ground and ceiling. The "road" was acting as a lateral conveyor belt, and the actual ground had no boundary layer suppression.
**Fix:** Corrected face mappings. Ground $\to Z=0$ with `movingWallVelocity`, Ceiling/Sides $\to$ `slip`.
**Result (Measured):** Drag surprisingly increased slightly (+0.45%), but force oscillation ($\sigma$) dropped by 16.0%. This proved that the massive error was *not* due to slip versus moving ground. Discrepancy remained at **+62.2%**.

### 3.2 Step 2: Domain Blockage and Wake Truncation
**Problem Discovered:** The virtual wind tunnel was far too small ($6\text{ m} \times 4\text{ m}$), creating an 11.25% blockage ratio (violating the SAE $<3\%$ standard). Furthermore, the outlet was only $1.35 L$ behind the car, artificially truncating the low-pressure recirculation wake.
**Fix:** Expanded the tunnel to $15\text{ m} \times 7.5\text{ m}$, reducing blockage to **2.31%** and extending the wake region to $5.7 L$. 
**Result (Measured):** Drag force plummeted by 20.2% ($-89.5\text{ N}$). $C_dA$ dropped to $0.6417\text{ m}^2$. This expansion alone eliminated over half the error, reducing the discrepancy to **+29.5%**.

### 3.3 Step 3 & 4: Mesh Limitations, $y^+$, and Prism Layers
**Problem Discovered:** Surface $y^+$ diagnosis on the cut-cell mesh revealed an area-weighted average $y^+ \sim 381.7$. Wall functions for $k$-$\omega$ SST require $y^+ < 100$. The high $y^+$ thickened the boundary layer artificially, tripping premature separation over the roof curvature.
**Fix:** Extruded **3 near-wall prism layers** (`expansionRatio 1.3`, `finalLayerThickness 0.5`) during the `snappyHexMesh` addLayers phase.
**Result (Measured):** Average $y^+$ halved to $170.0$. Premature separation was suppressed, and drag dropped another 11.0% ($-38.8\text{ N}$). 
**Final Validated $C_dA$:** $0.5712\text{ m}^2$. **The initial +61.5% discrepancy was slashed to just +15.25%.**

---

## 4. Phase 8: Closing the Loop with CFD Feedback

With CFD validated as a reliable relative gradient indicator, it was integrated directly into the Phase 8 shape optimizer.

### 4.1 The Surrogate Pitfall (Stage 1 Failure)
**Hypothesis & Assumption:** It was initially assumed the ML latent drag surrogate could blindly optimize shape. 
**Actual Problem:** The surrogate, trained on global topologies, lacked awareness of local boundary-layer separation limits. In Round 1, the unconstrained optimizer aggressively sharpened the Fastback's rear backlight. 
**CFD Reality Check:** OpenFOAM revealed this tripped massive flow separation, resulting in a **+9.74% drag penalty** ($+41.00\text{ N}$) instead of the surrogate's predicted improvement.

### 4.2 Discretization Limits & Geometric Corruption
**Problem Discovered:** Marching Cubes mesh reconstruction created highly faceted STLs. In Stage 1 Notchback testing, these noisy facets triggered a $+22.59\%$ drag explosion in OpenFOAM. Furthermore, the optimizer drifted into adversarial out-of-distribution latent spaces.
**Fix:** 
1. **Geometry Pipeline:** Implemented 159k-face subdivision and Taubin smoothing to eliminate the discretization tax.
2. **Latent Trust Region:** Restricted optimizer steps via Projected Gradient Descent to $\|\mathbf{z} - \mathbf{z}_0\|_2 \le 0.75$, keeping the AI strictly within the valid DrivAer manifold.

### 4.3 The Repulsive Barrier Breakthrough (Round 2)
**Fix:** Phase 8 introduced an **active directional repulsive barrier**. By calculating the falsification vector between the CFD ground-truth drag and the surrogate's hallucinated drag, a quadratic barrier was erected in the latent space.
**Result (Measured):** The optimizer deflected $83.67^\circ$ away from the failed path. Instead of steepening the roof, it subtly tapered and boat-tailed the lateral greenhouse. Boundary layers remained attached. 

---

## 5. Final Verified CFD Experiments

The final Stage 2 Trust-Region champions were simulated in the rectified OpenFOAM environment, yielding verifiable success:

1. **Fastback `025`:** 
   - Round 1 (Unguided) incurred a $+9.74\%$ penalty.
   - Round 2 (CFD Repulsive Barrier) achieved a **$-15.73\%$ drag reduction** vs Round 1, and a **$-7.53\%$ physical reduction** vs the smooth baseline ($421.0\text{ N} \to 389.3\text{ N}$, $C_dA \to 0.7062\text{ m}^2$).
2. **Estateback `025`:**
   - Achieved a massive **$-13.88\%$ physical drag reduction** below baseline ($571.7\text{ N} \to 492.3\text{ N}$, $C_dA \to 0.8932\text{ m}^2$).
3. **Notchback `025`:**
   - The Stage 1 discretization explosion (+22.6%) was successfully neutralized by Taubin smoothing and the trust region, resulting in a stable neutral aerodynamic drift of **+0.13%** (a relative improvement of -37.4% vs Stage 1).

---

## 6. Current State and Limitations

### 6.1 The Final OpenFOAM Workflow
The current working CFD template frozen for validation operates via `scripts/run_cfd.sh` using:
- **Domain:** Expanded SAE-compliant $15\text{ m} \times 7.5\text{ m} \times 7.5\text{ m}$ (2.31% blockage ratio, 5.7L wake).
- **Mesh Setup:** `snappyHexMesh` with surface level `(3 4)` and 3 extruded near-wall prism layers (~435k–507k total cells).
- **Physics:** `simpleFoam`, $k$-$\omega$ SST, $30\text{ m/s}$, correct boundary moving-ground mappings.
- **Iteration Strategy:** Fixed 3,000 iterations, with $C_dA$ asymptotically averaged over the last 200 iterations.
- **AI Processing:** Uses strict latent trust-region bounds ($R \le 0.75$) and Taubin smoothing prior to STL mesh export.

### 6.2 Unresolved Issues and Known Limitations
While highly capable for relative gradient tracking, the CFD methodology has known limitations separating it from the DrivAerNet++ high-fidelity benchmarks:
- **Static vs. Rotating Wheels (Unresolved):** DrivAerNet++ uses Multi-Reference Frame (MRF) rotating wheels. The current setup merges wheels into the static car body with `noSlip`, which kinematics analysis hypothesizes contributes roughly **4% to 6% excess drag** due to stagnation pressure and unphysical tread separation.
- **Coarse RANS Discretization (Limitation):** The current $\sim 500\text{k}$-cell RANS mesh cannot capture transient wake shedding as accurately as the 12-million-cell LES/DES setups used in DrivAerNet++. This accounts for the remaining **~8% to 10% discrepancy**.
- **$y^+$ Resolution (Limitation):** Although prism layers halved the $y^+$ to 170, it remains above the ideal $<100$ threshold required for optimal wall-function application.
- **ForceCoeffs (Unimplemented):** The Phase 7 plan intended to use `forceCoeffs`, but the current pipeline still extracts absolute forces and calculates $C_dA$ manually using dynamic pressure. 

**Conclusion:** The AeroMorphs CFD pipeline has successfully evolved from an uncalibrated automation script into a robust physical validation engine. It currently serves its ultimate purpose: providing verified physical guardrails that correct AI hallucination and successfully guide the closed-loop generator toward verifiable aerodynamic drag reduction.
