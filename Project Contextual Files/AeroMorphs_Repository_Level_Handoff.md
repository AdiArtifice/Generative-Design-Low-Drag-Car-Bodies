# AeroMorphs — Repository-Level Architecture & Handoff Context

> **Purpose:** Secondary handoff document for Aditya and IDE agents.
>
> This document complements `AeroMorphs_CFD_Project_Context_to_Phase_8.md`.
> It focuses on repository architecture, ownership boundaries, integration points, and the
> distinction between **verified/current CFD structure** and **planned future files**.
>
> **Important limitation:** the full AeroMorphs source repository was not available for direct
> inspection when this document was created. Therefore, this document intentionally does **not**
> invent an exhaustive ML/frontend/backend directory tree. Everything not directly established
> by the available project context is marked as **UNVERIFIED / TO INSPECT**.

---

## 1. Executive Summary

AeroMorphs is intended to be an AI-assisted aerodynamic vehicle optimization system.

The high-level product workflow is:

```text
User STL
   |
   v
Select body/model type
   |
   v
ML optimization
   |
   v
Optimized parameters + optimized STL
   |
   v
ML-predicted aerodynamic metric
   |
   v
ONE final / sparse OpenFOAM CFD validation
   |
   v
CFD-validated optimized design candidate
```

The important architectural principle is:

> **Do not place OpenFOAM inside every ML candidate evaluation.**

ML is intended to search cheaply. CFD is expensive and therefore should be used selectively for final validation and, later, for sparse physics feedback.

Phase 8 extends this into a selective closed loop:

```text
AI optimization
      |
      v
champion only
      |
      v
OpenFOAM
      |
      v
CFD evidence
      |
      v
correction update
      |
      v
re-optimization
```

Only the champion from a round should be sent to CFD; intermediate latent candidates should not all be simulated.

---

# 2. Current Architecture — What Is Actually Established

The strongest verified architectural information currently available is the CFD side.

The repository/workspace structure known from the CFD implementation is:

```text
~/AeroMorphs/
├── baselineCar/
├── baselineCar_backup/
└── cfd_automation/
```

These directories have distinct purposes.

### `baselineCar/`

Original baseline OpenFOAM case.

**Status:** existing and must remain untouched.

### `baselineCar_backup/`

Backup of the original baseline case.

**Status:** existing and must remain untouched.

### `cfd_automation/`

Working OpenFOAM automation case used for experiments and reusable CFD execution.

**Status:** existing and actively developed.

---

# 3. Verified CFD Repository Structure

The known current CFD automation structure is approximately:

```text
cfd_automation/
├── 0/
├── constant/
│   ├── triSurface/
│   │   └── vehicle.stl
│   ├── transportProperties
│   └── turbulenceProperties
├── system/
│   ├── controlDict
│   ├── fvSolution
│   └── snappyHexMeshDict
├── scripts/
│   ├── run_cfd.sh
│   └── extract_results.py
├── results/
│   └── cfd_results.json
├── postProcessing/
├── postProcessing_backup/
├── log.simpleFoam
└── <other OpenFOAM-generated case files/directories>
```

The exact complete generated tree should still be verified from the repository itself before treating the above as exhaustive.

---

# 4. CFD Case Layer

The CFD case is a standard OpenFOAM case organized into the usual:

```text
0/
constant/
system/
```

The important distinction is that this is the **CFD engine layer**, not the application/backend layer.

The ML system should not need to know individual OpenFOAM dictionary commands.

The intended contract is closer to:

```text
STL in
  ->
CFD runner
  ->
JSON result out
```

---

# 5. `constant/triSurface/`

Current vehicle geometry path:

```text
constant/triSurface/vehicle.stl
```

The current test geometry is:

```text
N_S_WWC_WM_025.stl
```

It was copied/renamed into the automation case as:

```text
vehicle.stl
```

The current test geometry is a **Notchback** case.

Do not refer to this geometry as Fastback.

The current automation script accepts an arbitrary STL path:

```bash
./scripts/run_cfd.sh <path-to-stl>
```

and copies it into:

```text
constant/triSurface/vehicle.stl
```

when necessary.

---

# 6. `constant/transportProperties`

Current working fluid model:

```text
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] 1.5e-05;
```

Thus the current reduced CFD operating point uses:

```text
kinematic viscosity = 1.5e-05 m²/s
```

This is part of the current local setup and should not be silently replaced with values from another CFD dataset/reference.

---

# 7. `constant/turbulenceProperties`

Current turbulence model:

```text
simulationType RAS;

RAS
{
    RASModel kOmegaSST;
    turbulence on;
    printCoeffs on;
}
```

Therefore:

```text
RANS
k-omega SST
simpleFoam
```

is the current CFD foundation.

---

# 8. `0/` Initial and Boundary Conditions

The current known fields include:

```text
U
p
k
omega
nut
```

The current operating point is:

```text
U = 30 m/s
rho = 1.225 kg/m³
```

The current vehicle wall is:

```text
car
```

with no-slip velocity.

The current domain boundaries are:

```text
inlet
outlet
lowerWall
upperWall
sides
car
```

Important:

The actual current case uses:

```text
sides -> slip
```

It is **not** currently the Phase 7 planned explicit half-car `symmetryPlane` setup.

Do not merge those two descriptions.

---

# 9. Current Domain Geometry

Known current bounding box:

```text
(-3 -3 0)
to
(10 3 4)
```

Known current patches:

```text
inlet
outlet
lowerWall
upperWall
sides
car
```

This describes the current experimental case.

The Phase 7 document proposes a future half-car/symmetry-oriented validation configuration. That plan should not be assumed to describe the current case exactly.

---

# 10. Current `snappyHexMeshDict`

The current mesh pipeline is:

```text
blockMesh
   ->
snappyHexMesh
   ->
checkMesh
   ->
simpleFoam
```

Important current settings include:

```text
castellatedMesh true;
snap true;
addLayers false;
```

Vehicle geometry:

```text
vehicle.stl
{
    type triSurfaceMesh;
    name car;
}
```

Current fine preparation:

```text
refinementSurfaces
{
    car
    {
        level (4 5);
    }
}
```

Other known settings include:

```text
maxLocalCells 200000;
maxGlobalCells 1000000;
nCellsBetweenLevels 2;
resolveFeatureAngle 30;
locationInMesh (8 0 2);
```

Current mesh does not use added prism/boundary layers.

This is one of the important places where the present reduced implementation differs from higher-fidelity automotive CFD references.

---

# 11. Current `system/controlDict`

Current simulation setup:

```text
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         3000;
writeControl    timeStep;
writeInterval   100;
```

Forces are currently collected using the OpenFOAM `forces` function object.

Known force configuration:

```text
forces
{
    type            forces;
    libs            (forces);
    patches         (car);
    rho             rhoInf;
    rhoInf          1.225;
    CofR            (0 0 0);
    writeControl    timeStep;
    writeInterval   10;
}
```

Important:

> `forceCoeffs` is planned in the Phase 7 methodology, but the current implementation uses `forces`.

Do not describe `forceCoeffs` as already implemented.

---

# 12. Current `system/fvSolution`

Known current solver strategy:

```text
p     -> GAMG
U     -> smoothSolver
k     -> smoothSolver
omega -> smoothSolver
```

Known tolerances include:

```text
p     tolerance 1e-7
U     tolerance 1e-8
k     tolerance 1e-8
omega tolerance 1e-8
```

Current SIMPLE settings include:

```text
nNonOrthogonalCorrectors 0;
consistent yes;
```

Current equation relaxation factors include approximately:

```text
U     0.9
k     0.7
omega 0.7
```

This represents the current working configuration, not a universal best-practice configuration.

---

# 13. CFD Automation Layer

The automation layer currently consists primarily of:

```text
scripts/run_cfd.sh
scripts/extract_results.py
```

This is the most important reusable application-facing CFD layer.

---

# 14. `scripts/run_cfd.sh`

Current known script behavior:

```text
input STL
   |
   v
clean previous run
   |
   v
prepare vehicle.stl
   |
   v
blockMesh
   |
   v
snappyHexMesh -overwrite
   |
   v
checkMesh
   |
   v
simpleFoam
   |
   v
extract_results.py
   |
   v
results/cfd_results.json
```

The script requires:

```bash
./scripts/run_cfd.sh <path-to-stl>
```

It validates that the STL exists before execution.

It cleans previous generated time directories without deleting `0/`.

The cleanup pattern was deliberately changed because an earlier pattern accidentally matched `0/`.

The safe current numeric-time cleanup pattern is:

```bash
find . -maxdepth 1 -type d -regex './[1-9][0-9]*' -exec rm -rf {} +
```

The script also avoids copying the STL onto itself by comparing real paths before performing `cp`.

These are important implementation details; do not regress them.

---

# 15. `scripts/extract_results.py`

The current result extraction path is:

```text
postProcessing/forces/0/force.dat
```

The script currently:

1. Reads force data.
2. Selects iterations 2800–3000 inclusive.
3. Computes mean drag force.
4. Computes drag standard deviation.
5. Converts drag to CdA using dynamic pressure.
6. Writes JSON output.

Current formula:

```text
CdA = Drag / (0.5 * rho * V²)
```

with:

```text
rho = 1.225 kg/m³
V   = 30 m/s
```

The output file is:

```text
results/cfd_results.json
```

Current JSON contains:

```text
mean_drag_force_N
std_drag_force_N
cda_m2
velocity_mps
density_kg_m3
averaging_start_iteration
averaging_end_iteration
samples
```

---

# 16. Current CFD Output Contract

The most useful abstraction for backend integration is:

```json
{
  "mean_drag_force_N": <number>,
  "std_drag_force_N": <number>,
  "cda_m2": <number>,
  "velocity_mps": 30.0,
  "density_kg_m3": 1.225,
  "averaging_start_iteration": 2800,
  "averaging_end_iteration": 3000,
  "samples": <integer>
}
```

This is the current result schema.

A future productionized interface may add:

```text
status
model
mesh
iterations
runtime
validation_status
error_message
run_id
```

but those additions are not yet established as implemented.

---

# 17. Current CFD Results

The medium mesh experiment used:

```text
surface refinement = (3 4)
cells              = 414,139
```

The completed medium run used:

```text
3000 iterations
```

and produced:

```text
Mean drag ≈ 602.467 N
Std drag  ≈ 3.206 N
CdA       ≈ 1.09291 m²
Runtime   ≈ 58.8 min
```

The medium mesh reported one failed mesh-quality check.

Known quality numbers:

```text
max aspect ratio          ≈ 5.15975
max non-orthogonality     ≈ 64.5766
average non-orthogonality ≈ 12.0048
max skewness              ≈ 4.53538
```

The mesh contained two highly skew faces.

Therefore:

> The medium mesh should not be described as having passed every mesh check perfectly.

---

# 18. Coarse Mesh Experiment

The coarse mesh used:

```text
surface refinement = (2 3)
cells              = 152,209
```

The completed 3000-iteration run produced approximately:

```text
Mean drag ≈ 639.670 N
Std drag  ≈ 0.610 N
CdA       ≈ 1.16040 m²
```

Compared with the medium mesh:

```text
coarse CdA  = 1.16040 m²
medium CdA  = 1.09291 m²
difference  ≈ 6.18%
```

This successfully demonstrated that the current result is mesh-sensitive.

The lower standard deviation of the coarse result does **not** imply that the coarse mesh is more physically accurate.

---

# 19. Fine Mesh Status

The current fine configuration was:

```text
surface refinement = (4 5)
```

**The fine run has successfully completed.**

Results:
- Cells: 1,151,572
- Mean Drag: 591.64 N
- CdA: 1.07327 m²
- Run time: ~186 minutes

**Conclusion:** Asymptotic convergence was achieved. The difference between Medium (CdA 1.09291 m²) and Fine (CdA 1.07327 m²) is only 1.80%.

The following actions have now been completed:
1. `system/snappyHexMeshDict` reverted and frozen at `level (3 4)` as the production baseline.
2. Built `scripts/openfoam_runner.py` as the Python execution bridge outside the encrypted FUSE mount.
3. Built `scripts/denormalize_mesh.py` to rescale AI unit-box STLs to 1:1 physical dimensions.
4. GCP Compute Engine (`c2-standard-8`) benchmarked and promoted to Primary MVP Backend (100.000% numerical parity, 56.6 min runtime, guaranteed automated teardown).

---

# 20. Current Development Stage

The current CFD work is primarily between these layers:

```text
Layer 1 — CFD case
OpenFOAM dictionaries

Layer 2 — CFD automation
run_cfd.sh
extract_results.py

Layer 3 — CFD validation
convergence
mesh independence
reference comparison
geometry robustness
```

The later layers are architectural goals:

```text
Layer 4 — reusable CFD template
STL -> OpenFOAM -> JSON

Layer 5 — application integration
ML STL -> CFD runner -> validated CdA

Layer 6 — Phase 8 physics refinement
AI champion -> selective CFD -> evidence -> correction -> re-optimization
```

Do not skip directly from the current state to the Phase 8 closed loop.

---

# 21. ML / Optimization Layer — Repository Status

The project context establishes that AeroMorphs has an ML optimization side that generates optimized vehicle parameters/geometry, including an optimized STL.

However, the full ML source tree was **not directly inspected** while preparing this document.

Therefore the exact current filenames, packages, classes, and data-flow implementation on the ML side are:

> **UNVERIFIED — MUST INSPECT REPOSITORY**

Do not invent paths such as:

```text
src/optimizer.py
models/
data/
latent/
```

unless they are actually present in the repository.

---

# 22. Known / Planned ML Integration File

The Phase 8 planning document names:

```text
scripts/optimize_latent_shape.py
```

as a script that may receive a minimal modification to support optional evidence-store integration.

Important:

> This path comes from the Phase 8 plan and must not be assumed to exist until the repository is inspected.

Likewise, do not assume its current API, function names, command-line arguments, or data structures.

---

# 23. Backend / Application Integration

The intended application behavior discussed for AeroMorphs is roughly:

```text
Upload STL
   |
   v
Select model/body type
   |
   v
Optimize Vehicle
   |
   v
Track optimization job
   |
   v
Show optimized result
   |
   v
Run CFD Validation
   |
   v
Track CFD job
   |
   v
Show CFD-validated result
```

A conceptual backend API discussed earlier includes endpoints such as:

```text
POST /api/upload
POST /api/optimize
GET  /api/optimization/{job_id}
POST /api/cfd/{job_id}
GET  /api/cfd/{job_id}
```

These are **conceptual architecture discussed for the MVP**, not verified repository endpoints.

The actual backend files/routes/framework must be inspected before implementing against them.

---

# 24. Responsibility Boundary for Nidhi's CFD Work

Nidhi's responsibility is the OpenFOAM/CFD side.

That includes:

```text
baseline CFD
automation
mesh/solver configuration
mesh independence
CFD validation
STL -> OpenFOAM workflow
result extraction
reusable model templates
CFD-side interface definition
documentation/handoff
```

The backend engineer is expected to handle the application/job-orchestration side.

Therefore Nidhi should not unnecessarily take ownership of:

```text
frontend
API implementation
job queues
backend orchestration
database design
```

unless the team later explicitly changes the division of responsibilities.

---

# 25. Intended CFD Service Boundary

The clean architectural abstraction is:

```text
Backend
   |
   | model + STL
   v
CFD Runner
   |
   | model-specific case/template
   v
OpenFOAM
   |
   v
CFD Result
```

A conceptual input object previously discussed is:

```json
{
  "model": "notchback",
  "stl_path": "optimized.stl"
}
```

A conceptual output object is:

```json
{
  "status": "success",
  "drag_force": <number>,
  "cda": <number>,
  "iterations": <number>
}
```

Again, this is an integration contract concept, not a verified live API implementation.

---

# 26. Why the CFD Interface Should Be Generic

The backend should not contain model-specific OpenFOAM commands.

Instead:

```text
model = notchback
     ->
notchback CFD template

model = fastback
     ->
fastback CFD template

model = estateback
     ->
estateback CFD template
```

The backend can then call a single generic CFD service interface.

This makes the CFD implementation easier to scale across the supported body/model categories.

---

# 27. Model Template Strategy

The project intends to support approximately four to five supported vehicle/body models.

The current detailed CFD implementation is for:

```text
Notchback
```

Other body types are future template work.

The correct progression is:

```text
1. Validate Notchback foundation
2. Freeze reusable Notchback template
3. Introduce genuinely different STL
4. Build additional body templates
5. Standardize shared CFD interface
```

Do not pretend that all vehicle models already have validated OpenFOAM templates.

---

# 28. Phase 7 — Position in Repository Architecture

Phase 7 is the CFD validation/bootstrap phase.

The important conceptual structure is:

```text
ML candidate generation
        |
        v
selected candidate
        |
        v
OpenFOAM validation
        |
        v
CFD evidence
        |
        v
surrogate/reference comparison
        |
        v
correction model
```

The supplied Phase 7 plan treats CFD as a **sparse resource**, not something to run on every ML candidate.

The planned CFD budget is approximately:

```text
10–15 total CFD runs
```

The actual completed work so far is only the CFD implementation/foundation plus initial experiments.

Full Phase 7 validation is **not complete**.

---

# 29. Phase 7 — Completed vs Pending

## Completed / established

```text
OpenFOAM 2412 setup                    ✅
Notchback baseline case                ✅
Automated STL-driven CFD pipeline     ✅
Initial 3000-iteration run             ✅
Coarse mesh experiment                 ✅
Medium mesh experiment                 ✅
Fine mesh result confirmation          ✅
Mesh choice / freeze                   ✅
Force extraction to JSON               ✅
Force stability investigation          ✅
Residual spot-check                    ✅
Mesh sensitivity demonstrated         ✅
```

## Pending

```text
Absolute CFD calibration               ❌
Known/reference case validation        ❌
AI champion validation                 ❌
Surrogate error quantification         ❌
Affine correction                      ❌
Corrected re-optimization              ❌
Final Phase 7 validation               ❌
Python CFD Bridge                      ❌
Different-STL Robustness Test          ❌
```

Therefore:

> Phase 7 is not completed.

---

# 30. Phase 8 — Intended Repository Additions

The Phase 8 plan proposes the following future files:

```text
src/cfd_evidence_store.py
src/surrogate_correction.py
scripts/refine_with_cfd.py
scripts/openfoam_runner.py
```

and a minimal optional modification to:

```text
scripts/optimize_latent_shape.py
```

These paths are **planned**, not verified current files.

The repository agent must inspect the actual tree before creating or modifying them.

---

# 31. Phase 8 — Evidence Store Concept

The planned evidence store links AI predictions with CFD observations.

Conceptual record:

```json
{
  "id": "phase7_fastback_001",
  "source": "phase7_stage1",
  "body_type": "Fastback",
  "latent_vector_path": "cfd_evidence/latents/fb_001.npy",
  "stl_path": "cfd_evidence/stls/fb_001.stl",
  "cda_surrogate": 0.312,
  "cda_cfd": 0.328,
  "frontal_area": 2.15,
  "timestamp": "..."
}
```

The exact values above are example/planned data from the Phase 8 architecture, not evidence that such a record currently exists in the repository.

---

# 32. Phase 8 — Planned Correction Tiers

The planned correction progression is:

```text
0 evidence
    -> identity

1–2 evidence points
    -> constant offset

3–5
    -> global affine

6–15
    -> per-body-type affine

15+
    -> local correction such as k-NN residual or lightweight GP
```

This correction model is future architecture.

It is not currently implemented based on the available repository context.

---

# 33. Phase 8 — One Round

One planned closed-loop round is:

```text
Round N

1. optimize using current correction
2. select champion latent vector
3. decode champion STL
4. run OpenFOAM
5. obtain CFD CdA
6. record:
      latent vector
      surrogate CdA
      CFD CdA
      body type
7. add evidence
8. refit correction
9. check CFD/time budget
10. continue only if budget remains
```

The key optimization is selective CFD:

```text
many AI candidates
       |
       v
cheap surrogate evaluation
       |
       v
one champion
       |
       v
expensive CFD
```

---

# 34. Current vs Future Data Flow

## Current

```text
STL
 |
 v
OpenFOAM automation
 |
 v
force.dat
 |
 v
extract_results.py
 |
 v
cfd_results.json
```

## Intended application integration

```text
ML optimized STL
 |
 v
CFD runner
 |
 v
OpenFOAM
 |
 v
validated CdA
```

## Future Phase 8

```text
AI optimizer
 |
 v
champion
 |
 v
CFD
 |
 v
evidence store
 |
 v
surrogate correction
 |
 v
AI optimizer
```

---

# 35. Important Metric Distinction

There are currently two metric concepts in the broader project:

### Current CFD metric

Current OpenFOAM implementation:

```text
force -> mean drag -> CdA
```

using:

```text
CdA = Drag / (0.5 * rho * V²)
```

### Planned Phase 7 metric path

The Phase 7 plan proposes a more explicit:

```text
forceCoeffs
   ->
Cd
   ->
CdA
```

Do not say that the latter is already implemented.

---

# 36. DrivAerNet++ / Reference Dataset Position

DrivAerNet++ is being treated as a reference/benchmark for methodology and comparison.

The current AeroMorphs setup is a reduced local CFD implementation.

It does **not** reproduce the high-fidelity DrivAerNet++ methodology exactly.

Therefore:

```text
DrivAerNet++
    =
reference / benchmark

AeroMorphs current CFD
    =
reduced working implementation
```

Do not use reference-dataset CdA as automatically equivalent to the current CFD ground truth without investigating methodological differences.

---

# 37. Known Current Dataset/Geometry Numbers

Known geometry reference values associated with the current test context:

```text
Dataset Cd            = 0.27552
Dataset frontal area  = 2.706 m²
Dataset CdA           = 0.745558 m²
```

These numbers should be treated as reference data from the test context, not as proof that the current reduced OpenFOAM case must reproduce them exactly.

---

# 38. Current Major Risks

The main technical risks known so far are:

### Mesh sensitivity

Coarse and medium meshes differ by approximately 6.18% in CdA.

Therefore mesh independence is not yet established.

### Absolute CFD discrepancy

The current CFD CdA is substantially different from the associated dataset reference value.

This requires methodological investigation before using the CFD value as a calibrated truth source.

### Geometry/template generalization

Only the Notchback case has been worked through in detail so far.

### Phase 8 premature integration

The closed-loop correction architecture should not be built on top of unvalidated CFD.

### Automation regressions

The cleanup logic and same-file STL handling already caused real problems and were corrected.

---

# 39. Critical "Do Not" Rules

1. Do not overwrite:

```text
baselineCar
baselineCar_backup
```

2. Do not restore the old cleanup expression that can delete `0/`.

3. Do not call the current `N_S_WWC_WM_025` case Fastback.

4. Do not assume the fine CFD run has completed.

5. Do not claim mesh independence before analyzing the fine result.

6. Do not claim exact reproduction of DrivAerNet++.

7. Do not claim Phase 7 is complete.

8. Do not claim Phase 8 is implemented.

9. Do not claim `forceCoeffs` is currently implemented.

10. Do not put OpenFOAM into every ML optimization step.

11. Do not run CFD for every latent candidate.

12. Do not change multiple variables during a mesh-independence experiment.

13. Do not treat small residuals alone as proof of aerodynamic accuracy.

14. Do not treat low force standard deviation alone as proof of accuracy.

15. Do not treat the reference dataset CdA as unquestioned CFD ground truth.

16. Do not make manufacturing-readiness claims from aerodynamic CFD alone.

---

# 40. Recommended Repository Inspection Order

When the full source repository is available, the next AI agent should inspect it in this order:

```text
1. Top-level directory tree
2. README / project documentation
3. ML / optimization entry points
4. model definitions and checkpoints
5. data/preprocessing paths
6. STL generation/decoding
7. backend/API
8. frontend/application
9. current scripts
10. configuration files
11. test suite
12. deployment/container files
13. Git status / ignored directories
```

The goal is to replace the "UNVERIFIED" sections of this handoff with facts from the actual repository.

---

# 41. What the Repository Agent Should Search For

Useful search terms:

```text
AeroMorphs
OpenFOAM
simpleFoam
snappyHexMesh
STL
optimize
latent
surrogate
CdA
drag
CFD
Notchback
Fastback
Estateback
evidence
correction
API
job_id
upload
```

Useful filename searches:

```text
run_cfd.sh
extract_results.py
optimize_latent_shape.py
cfd_evidence_store.py
surrogate_correction.py
refine_with_cfd.py
openfoam_runner.py
```

The important principle is:

> Verify existence before documenting a file as current.

---

# 42. Expected Integration Ownership

A clean division is:

```text
                    AeroMorphs
                        |
        +---------------+---------------+
        |                               |
        v                               v
   ML / Optimization                CFD
        |                               |
        |                               |
        |                        OpenFOAM case
        |                        run_cfd.sh
        |                        extract_results.py
        |                               |
        +---------- STL ----------------+
                        |
                        v
                    Backend
                        |
                        v
                    Frontend
```

The actual backend/frontend package layout remains repository-dependent and must be inspected.

---

# 43. Definition of a Good CFD Module

A reusable CFD component should eventually satisfy:

```text
Input:
    STL
    model/body type
    optionally run/configuration ID

Process:
    prepare case
    mesh
    solve
    validate
    extract force/CdA

Output:
    structured JSON
```

The caller should not have to understand:

```text
blockMesh
snappyHexMesh
simpleFoam
OpenFOAM dictionaries
postProcessing paths
```

That complexity belongs behind the CFD abstraction.

---

# 44. Future Model-Specific Templates

Once Notchback validation is sufficiently stable:

```text
templates/
    notchback/
    fastback/
    estateback/
    ...
```

may be a useful organizational concept.

However:

> This folder structure is an architectural suggestion only and is not a claim that such a `templates/` directory currently exists.

The team should inspect the current repository before deciding whether to create this structure.

---

# 45. Future Evidence Store Organization

A natural future organization is:

```text
cfd_evidence/
├── latents/
├── stls/
├── results/
└── evidence.json
```

but again this is a **future organization concept**, not verified current repository structure.

---

# 46. Current Known Project State

As of the latest known project context:

```text
OpenFOAM 2412                  WORKING
Notchback baseline             WORKING
CFD automation                 WORKING
Medium mesh                   COMPLETE
Coarse mesh                   COMPLETE
Residual spot-check           COMPLETE
Mesh sensitivity              DEMONSTRATED
Fine mesh                     COMPLETE
Mesh choice                   COMPLETE (Medium)
Absolute CFD validation       PENDING
Different STL test            PENDING
Other templates               PENDING
Phase 7 full validation       PENDING
Phase 8 implementation        PENDING
```

The immediate next decision is:

```text
build Python CFD bridge
   ->
run Different-STL test
   ->
continue validation
```

---

# 47. Practical Handoff for Aditya's AI Agent

The agent should mentally model the project as:

```text
                    AeroMorphs
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
 AI / Geometry                         CFD
 optimization                          |
       |                                |
       v                                v
 optimized STL                    reusable OpenFOAM
       |                                case
       +---------------+                |
                       |                v
                       +----------> CFD result JSON
                                        |
                                        v
                               application layer
```

Then, only after CFD validation matures:

```text
AI optimizer
     |
     v
champion
     |
     v
OpenFOAM
     |
     v
evidence
     |
     v
correction
     |
     v
AI re-optimization
```

The core engineering idea is therefore:

> **Use ML for cheap aerodynamic search and CFD selectively for physical validation and later physics-informed correction.**

---

# 48. Source-of-Truth Rule

For continuing work, use this priority:

```text
1. Actual repository state
2. Actual terminal/log/output state
3. Current OpenFOAM configuration
4. Current handoff document
5. Phase 7 plan for intended future behavior
6. Phase 8 plan for intended future architecture
```

Never reverse this order.

A planning document can tell the agent what the project intends to become; it cannot prove that the implementation already exists.

---

# 49. Companion Document

This document should be read together with:

```text
AeroMorphs_CFD_Project_Context_to_Phase_8.md
```

The companion document contains the detailed CFD implementation history, configurations, run outputs, mesh results, and Phase 7/8 context.

This document is intentionally more focused on:

```text
repository architecture
ownership boundaries
integration interfaces
current vs planned file status
```

---

# 50. Final Status

This is a **repository-level architecture/handoff scaffold**, not a verified complete repository inventory.

The missing step is direct inspection of the actual AeroMorphs repository tree and source files.

Once the repository is available, the ideal final handoff should add:

```text
exact full directory tree
exact file list
file-by-file purpose
entry points
imports/dependencies
ML data flow
STL generation flow
backend route map
frontend component map
configuration files
test files
deployment structure
actual Phase 8 implementation status
```

Until that inspection occurs, unknown areas should remain explicitly marked rather than guessed.
