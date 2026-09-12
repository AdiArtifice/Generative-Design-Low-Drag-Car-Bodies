# AeroMorphs — Comprehensive Project Context, CFD Worklog, Decisions, and Phase 7/8 Handoff

> **Purpose:** Primary context document for continuing AeroMorphs CFD work in an IDE/AI agent.
>
> **Current CFD focus:** OpenFOAM implementation and validation for the Notchback vehicle case.
>
> **Current state:** Automated OpenFOAM pipeline is working end-to-end on the local Ubuntu machine. A 3000-iteration medium baseline and a 3000-iteration coarse mesh-independence run have completed. A fine mesh has been configured but its CFD run has **not yet been confirmed complete in the available context**.
>
> **Important:** This document deliberately separates **confirmed implementation facts**, **project plans**, **assumptions**, and **pending work**. Do not silently treat planned Phase 7/8 architecture as already implemented.

---

## 1. Project Identity and Core Objective

### 1.1 AeroMorphs

AeroMorphs is an AI-assisted aerodynamic vehicle optimization project. The intended system combines:

1. A learned surrogate/AI optimization system.
2. Vehicle/body-type design representations.
3. STL geometry generation.
4. OpenFOAM CFD used as a **physics validation mechanism**.
5. A final engineering-oriented result package.

The central product idea is to avoid repeatedly running expensive CFD throughout the entire design-search process.

### 1.2 Intended high-level workflow

The agreed conceptual workflow is:

```text
User
  |
  | Upload STL + select supported vehicle/body model
  v
AI / ML Optimization
  |
  | searches design space using surrogate
  v
Optimized Candidate
  |
  +--> optimized design parameters
  +--> optimized geometry / STL
  +--> ML-predicted CdA
  |
  v
ONE FINAL CFD VALIDATION
  |
  | OpenFOAM
  |   STL -> mesh -> simpleFoam -> forces -> CdA
  v
Final Engineering-Oriented Result
  |
  +--> CFD-validated CdA
  +--> drag force
  +--> ML prediction vs CFD validation
  +--> percentage improvement
  +--> optimized STL
  +--> optimized parameters
  +--> validation/report artifacts
```

### 1.3 Critical architecture decision

**OpenFOAM is not intended to sit inside the normal surrogate optimization loop for every candidate.**

The intended default product behavior is:

```text
ML explores many candidates cheaply
        |
        v
one optimized/champion candidate
        |
        v
OpenFOAM final validation
```

The Phase 8 plan later introduces an **optional physics-refinement mode** where only selected champions are sent to CFD, not every optimizer step. This is different from running CFD on every latent vector.

### 1.4 Engineering claim boundary

The output should be described as a **CFD-validated optimized design candidate**, not automatically as a manufacturing-ready vehicle design.

Structural safety, crashworthiness, thermal/cooling performance, manufacturability, regulatory requirements, wheel/suspension constraints, etc. are outside the current CFD scope unless separately implemented.

---

# 2. Division of Responsibilities

## 2.1 Nidhi / CFD responsibility

Nidhi's agreed responsibility in the current team workflow is the **OpenFOAM side**, not the web/backend application.

Her CFD deliverable is:

- Build and validate OpenFOAM CFD cases.
- Create reusable case templates for supported vehicle/body types.
- Automate:
  - STL input
  - meshing
  - solver execution
  - force extraction
  - CdA calculation
  - machine-readable result output.
- Define the interface concept from ML-generated STL to CFD.
- Ensure the CFD setup can be reused for different STL geometries.
- Eventually hand the CFD templates/scripts/documentation to the backend/integration teammate.
- Validate the final optimized candidates using the agreed CFD configuration.

The backend engineer is expected to connect the application/API/job workflow to these CFD templates. Nidhi should **not** be pulled into building the backend job queue/API unless the team explicitly changes responsibilities.

## 2.2 Current working style preference

The CFD work has been intentionally handled one concrete step at a time.

For future continuation:

- Explain **what** is being changed.
- Explain **why** it is being changed.
- Explain **what result is expected**.
- Only then provide the command/change.
- Do not make unrelated configuration changes during an experiment.
- Do not modify the untouched original baseline case unless explicitly required.

---

# 3. Current Machine and Environment

## 3.1 Confirmed local environment

- OS: Ubuntu 24.04 LTS.
- CPU: Intel i7-12700, 12 cores / 20 threads.
- RAM: 16 GB.
- GPU: Intel UHD 770 integrated graphics; no CUDA.
- OpenFOAM: **OpenFOAM 2412 (OpenCFD/ESI)**.
- Solver: `simpleFoam`.
- Local working host/user seen in terminal: `student@sjcem`.
- Project root:
  ```text
  ~/AeroMorphs/
  ```

## 3.2 Important directories

Original baseline case:

```text
~/AeroMorphs/baselineCar
```

Backup:

```text
~/AeroMorphs/baselineCar_backup
```

Automation working copy:

```text
~/AeroMorphs/cfd_automation
```

The original `baselineCar` and its backup were intentionally preserved. CFD automation experiments are being performed in `cfd_automation`.

---

# 4. Geometry and Dataset Context

## 4.1 Current CFD geometry

Current test geometry:

```text
N_S_WWC_WM_025.stl
```

Inside the automation case it is used as:

```text
constant/triSurface/vehicle.stl
```

This is a **Notchback** case. Earlier conversation incorrectly referred to it as Fastback; that was corrected. Do not revert to calling this particular current case Fastback.

## 4.2 DrivAerNet++ metadata for this geometry

Confirmed metadata row:

```text
id            = N_S_WWC_WM_025
config        = N_S_WWC_WM
body_type     = N
split         = test
cd            = 0.27552
drag_area     = 0.745558
frontal_area  = 2.706
length_x      = 0.999791
width_y       = 0.485848
height_z      = 0.30456
```

Dataset paths:

```text
raw_stl/N_S_WWC_WM/N_S_WWC_WM_025.stl
normalized_stl/N_S_WWC_WM/N_S_WWC_WM_025_norm.stl
```

The metadata relationship is:

```text
CdA = Cd * frontal_area
     = 0.27552 * 2.706
     ≈ 0.74556 m²
```

This is the dataset's `drag_area`.

## 4.3 Dataset vs current CFD discrepancy

Current OpenFOAM result for the medium mesh:

```text
CFD CdA ≈ 1.09291 m²
```

Dataset CdA:

```text
≈ 0.74556 m²
```

Difference:

```text
≈ 0.34735 m²
```

The CFD value is approximately **46.6% higher** than the dataset value.

This is a major discrepancy and is **not explained by density alone**.

The current team decision was:

> Do not stop the automation work to calibrate this discrepancy immediately. First complete the practical automation/mesh-validation phase, then investigate the absolute CdA discrepancy as part of CFD validation.

Do not claim that the current reduced CFD reproduces DrivAerNet++ high-fidelity CFD.

---

# 5. DrivAerNet++ Reference Context

The project uses DrivAerNet++ as a benchmark/reference source.

Confirmed reference characteristics from the published DrivAerNet++ methodology:

- ~8,000 diverse car designs.
- Fastback, Notchback, Estateback categories.
- 26 design parameters.
- CFD data and ML benchmarks.
- OpenFOAM v11 in the published methodology.
- `simpleFoam`.
- Steady incompressible RANS.
- k-omega SST.
- 30 m/s.
- 1:1 scaled models.
- Approximately 12 million cells per half-domain / approximately 24 million full equivalent.
- Boundary-layer/prism layers.
- Category-dependent surface refinement.
- Wake refinement.
- Moving ground.
- Rotating wheels.
- Slip top/lateral boundaries.
- Published air properties around 25 °C:
  - rho ≈ 1.184 kg/m³
  - nu ≈ 1.56e-5 m²/s.
- Potential-flow initialization.
- Published solver tolerances are stricter than the current reduced automation case.

### Important interpretation

AeroMorphs' current CFD is a **reduced, computationally efficient OpenFOAM RANS configuration inspired by established automotive CFD methodology**.

It should be described as a benchmark/reference-informed setup, **not as an exact reproduction of the DrivAerNet++ high-fidelity simulation setup**.

DrivAerNet++ itself involved much larger meshes and more detailed boundary-layer/wheel/wake treatment than the current working case.

---

# 6. Current OpenFOAM Case Configuration

## 6.1 Solver

```text
simpleFoam
```

`simpleFoam` is being used as a steady, incompressible turbulent SIMPLE solver.

## 6.2 Turbulence

```text
simulationType RAS;

RAS
{
    RASModel kOmegaSST;
    turbulence on;
    printCoeffs on;
}
```

## 6.3 Transport properties

Current:

```text
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] 1.5e-05;
```

## 6.4 Inlet speed

```text
30 m/s
```

This is also hard-coded in the current result-extraction script.

## 6.5 Current density used for CdA extraction

```text
rho = 1.225 kg/m³
```

This differs from the published DrivAerNet++ air-property value (~1.184 kg/m³).

This difference is **not sufficient to explain the large CFD-vs-dataset CdA discrepancy**.

## 6.6 Current force function

The current case uses OpenFOAM's `forces` function object, not yet the Phase 7 planned `forceCoeffs` implementation.

Current configuration:

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

The force data are therefore directly extracted as forces over the `car` patch.

The drag-area calculation is performed afterward:

```text
CdA = mean_drag_force / (0.5 * rho * V²)
```

### Important distinction

The Phase 7 plan says:

```text
forceCoeffs -> Cd -> CdA = Cd * frontal_area
```

but that is **planned architecture**, not the current implementation.

Current implementation:

```text
forces -> mean drag force -> CdA using dynamic pressure
```

Do not claim `forceCoeffs` is already implemented unless it is explicitly changed later.

---

# 7. Boundary Conditions — Current Implemented Case

## 7.1 U field

Current `0/U`:

```text
dimensions [0 1 -1 0 0 0 0];

internalField uniform (30 0 0);

boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform (30 0 0);
    }

    outlet
    {
        type zeroGradient;
    }

    lowerWall
    {
        type movingWallVelocity;
        value uniform (30 0 0);
    }

    upperWall
    {
        type slip;
    }

    sides
    {
        type slip;
    }

    car
    {
        type noSlip;
    }
}
```

This means the ground is modeled as moving at 30 m/s, while the car is stationary in the computational frame.

## 7.2 p field

Current `0/p`:

```text
dimensions [0 2 -2 0 0 0 0];

internalField uniform 0;

boundaryField
{
    inlet
    {
        type zeroGradient;
    }

    outlet
    {
        type fixedValue;
        value uniform 0;
    }

    lowerWall
    {
        type zeroGradient;
    }

    upperWall
    {
        type zeroGradient;
    }

    sides
    {
        type zeroGradient;
    }

    car
    {
        type zeroGradient;
    }
}
```

## 7.3 k

Current setup:

- inlet: fixed value `0.135`
- outlet: zeroGradient
- lower/upper/car: `kqRWallFunction`
- sides: `kqRWallFunction`

## 7.4 omega

Current setup:

- inlet: fixed value `100`
- outlet: zeroGradient
- lower/upper/car: `omegaWallFunction`
- sides: zeroGradient

## 7.5 nut

Current setup:

- inlet/outlet/sides: calculated 0
- lowerWall/car/upperWall: `nutkWallFunction`

---

# 8. Current Domain and Mesh Topology

Current `checkMesh` results for the **medium mesh**:

```text
points:           555,932
faces:            1,370,673
internal faces:   1,250,490
cells:            414,139
```

Domain bounding box:

```text
(-3 -3 0) (10 3 4)
```

Boundary patches:

```text
inlet
outlet
lowerWall
upperWall
sides
car
```

The current mesh is not configured as the Phase 7 plan's explicit half-car `symmetryPlane`; the actual current case has a `sides` patch using slip.

Do not silently assume the current case is a half-car symmetry-plane case just because the Phase 7 plan describes one.

---

# 9. Current snappyHexMesh Configuration

Relevant current configuration:

```text
castellatedMesh true;
snap true;
addLayers false;

geometry
{
    vehicle.stl
    {
        type triSurfaceMesh;
        name car;
    }
}

castellatedMeshControls
{
    maxLocalCells 200000;
    maxGlobalCells 1000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;

    features
    (
    );

    refinementSurfaces
    {
        car
        {
            level (4 5);

            patchInfo
            {
                type wall;
            }
        }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
    }

    locationInMesh (8 0 2);

    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;

    implicitFeatureSnap false;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}

addLayersControls
{
}

meshQualityControls
{
    #include "meshQualityDict"
}

writeFlags
(
    scalarLevels;
);

mergeTolerance 1e-6;
```

### Important current status

The file was most recently changed from:

```text
level (2 3);
```

to:

```text
level (4 5);
```

to prepare the fine mesh.

Therefore the **current file state is fine refinement `(4 5)`**, unless a later step changed it.

---

# 10. Current fvSolution

Current `system/fvSolution`:

```text
solvers
{
    p
    {
        solver GAMG;
        smoother GaussSeidel;
        tolerance 1e-7;
        relTol 0.01;
    }

    Phi { $p; }

    U
    {
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }

    k
    {
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }

    omega
    {
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
}

potentialFlow
{
    nNonOrthogonalCorrectors 10;
}

relaxationFactors
{
    equations
    {
        U 0.9;
        k 0.7;
        omega 0.7;
    }
}

cache
{
    grad(U);
}
```

There is currently **no `residualControl` block** in this configuration.

---

# 11. Current controlDict and Iteration Strategy

Current relevant `controlDict` settings:

```text
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         3000;
writeControl    timeStep;
writeInterval   100;
```

Forces:

```text
writeControl   timeStep;
writeInterval  10;
```

### Decision made during this work

A full 3000-iteration run was deliberately completed before making convergence decisions.

The current working comparison therefore uses:

```text
3000 simpleFoam iterations
```

for each mesh level.

This is currently a **fixed iteration budget for the mesh-independence experiment**, not a claim that 3000 is universally the scientifically correct stopping criterion.

The Phase 7 plan proposes a future dual criterion:

```text
residuals < 1e-4
AND
CdA variation < 0.1% over the last 200 iterations
```

with a maximum of 3000–5000 iterations.

That criterion is **planned**, not yet implemented in the current case.

---

# 12. Automation Script — Current Implemented Version

Current file:

```text
scripts/run_cfd.sh
```

Current behavior:

```bash
#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/run_cfd.sh <path-to-stl>"
    exit 1
fi

STL_FILE="$1"

if [ ! -f "$STL_FILE" ]; then
    echo "Error: STL file not found: $STL_FILE"
    exit 1
fi

echo "======================================"
echo " AeroMorphs CFD - Automated Run"
echo "======================================"

echo "Input STL: $STL_FILE"

echo "[1/7] Cleaning previous CFD run..."

find . -maxdepth 1 -type d -regex './[1-9][0-9]*' -exec rm -rf {} +
rm -rf constant/polyMesh
rm -rf postProcessing
rm -f log.simpleFoam
rm -f results/cfd_results.json

echo "[2/7] Preparing vehicle STL..."
if [ "$(realpath "$STL_FILE")" != "$(realpath constant/triSurface/vehicle.stl)" ]; then
    cp "$STL_FILE" constant/triSurface/vehicle.stl
fi

echo "[3/7] Creating background mesh..."
blockMesh

echo "[4/7] Generating vehicle mesh..."
snappyHexMesh -overwrite

echo "[5/7] Checking mesh..."
checkMesh

echo "[6/7] Running simpleFoam..."
simpleFoam > log.simpleFoam 2>&1

echo "[7/7] Extracting CFD results..."
python3 scripts/extract_results.py

echo "======================================"
echo " CFD pipeline completed successfully."
echo "======================================"
```

### Important script bug that was solved

An attempted run used:

```bash
./scripts/run_cfd.sh constant/triSurface/vehicle.stl
```

The script originally did:

```bash
cp "$STL_FILE" constant/triSurface/vehicle.stl
```

which caused:

```text
cp: 'constant/triSurface/vehicle.stl' and 'constant/triSurface/vehicle.stl' are the same file
```

Because `set -e` was active, the script stopped before `blockMesh`.

This was fixed by conditionally copying only when the source and destination resolve to different paths:

```bash
if [ "$(realpath "$STL_FILE")" != "$(realpath constant/triSurface/vehicle.stl)" ]; then
    cp "$STL_FILE" constant/triSurface/vehicle.stl
fi
```

This fix is confirmed in the working script.

### Important cleanup bug that was also solved

An earlier cleanup command used a regex equivalent to:

```bash
find . -maxdepth 1 -type d -regex './[0-9]+' -exec rm -rf {} +
```

This accidentally removed `0/`.

The cleanup was corrected to:

```bash
find . -maxdepth 1 -type d -regex './[1-9][0-9]*' -exec rm -rf {} +
```

This preserves the initial `0/` directory.

`0/` was restored from:

```text
baselineCar_backup
```

Do not reintroduce the old cleanup regex.

---

# 13. Current Result Extraction Script

Current file:

```text
scripts/extract_results.py
```

Current logic:

```python
import math
import json
import os

FORCE_FILE = "postProcessing/forces/0/force.dat"

RHO = 1.225
VELOCITY = 30.0
START_ITERATION = 2800
END_ITERATION = 3000

drag_forces = []

with open(FORCE_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        iteration = int(float(parts[0]))
        drag_force = float(parts[1])

        if START_ITERATION <= iteration <= END_ITERATION:
            drag_forces.append(drag_force)

if not drag_forces:
    raise RuntimeError("No force data found for the selected averaging range.")

mean_drag = sum(drag_forces) / len(drag_forces)

variance = sum(
    (force - mean_drag) ** 2 for force in drag_forces
) / len(drag_forces)

std_drag = math.sqrt(variance)

cda = mean_drag / (0.5 * RHO * VELOCITY ** 2)

results = {
    "mean_drag_force_N": mean_drag,
    "std_drag_force_N": std_drag,
    "cda_m2": cda,
    "velocity_mps": VELOCITY,
    "density_kg_m3": RHO,
    "averaging_start_iteration": START_ITERATION,
    "averaging_end_iteration": END_ITERATION,
    "samples": len(drag_forces)
}

os.makedirs("results", exist_ok=True)

with open("results/cfd_results.json", "w") as f:
    json.dump(results, f, indent=4)
```

### Current extraction meaning

Force output is written every 10 iterations.

Averaging:

```text
2800 through 3000 inclusive
```

therefore gives:

```text
21 samples
```

The script computes:

- mean drag
- standard deviation
- CdA
- velocity
- density
- averaging interval
- sample count.

---

# 14. Medium Mesh Baseline Run — Completed

The original/medium refinement was:

```text
level (3 4);
```

Mesh:

```text
cells = 414,139
faces = 1,370,673
points = 555,932
```

`checkMesh` results:

```text
Max aspect ratio          = 5.15975
Max non-orthogonality     = 64.5766
Average non-orthogonality = 12.0048
Max skewness              = 4.53538
```

The mesh reported:

```text
Failed 1 mesh checks.
```

The two highly skew faces were noted as a warning worth investigating during later V&V. They were only 2 out of ~1.37 million faces, but the mesh should not be described as having passed every check perfectly.

Topology and patch topology were otherwise acceptable.

### Medium CFD

Completed:

```text
3000 iterations
```

Execution time:

```text
~3525 s
~3530 s clock time
~58.8 minutes
```

Final force summary from log:

```text
Total    : (595.676 16.7708 -14.0649)
Pressure : (565.03 16.6745 -16.5938)
Viscous  : (30.6452 0.0962476 2.52896)
```

Latest force history showed oscillatory drag but a stable mean:

```text
2760 607.5102
2770 593.1300
2780 608.0413
2790 596.4528
2800 608.5106
...
2980 599.3954
2990 605.2266
3000 595.6757
```

Extracted result:

```json
{
    "mean_drag_force_N": 602.4667666666667,
    "std_drag_force_N": 3.2057118877067117,
    "cda_m2": 1.0929102343159487,
    "velocity_mps": 30.0,
    "density_kg_m3": 1.225,
    "averaging_start_iteration": 2800,
    "averaging_end_iteration": 3000,
    "samples": 21
}
```

Therefore record the medium result as:

```text
Mean drag ≈ 602.467 N
Std drag  ≈ 3.206 N
CdA       ≈ 1.09291 m²
```

---

# 15. Medium-Run Force Stability Analysis

A windowed analysis was performed using 200-iteration windows from 1000–3000.

Results:

```text
1000-1200 : Mean Drag = 600.274 N   CdA = 1.08893
1200-1400 : Mean Drag = 600.875 N   CdA = 1.09002
1400-1600 : Mean Drag = 600.553 N   CdA = 1.08944
1600-1800 : Mean Drag = 602.513 N   CdA = 1.09299
1800-2000 : Mean Drag = 599.650 N   CdA = 1.08780
2000-2200 : Mean Drag = 600.399 N   CdA = 1.08916
2200-2400 : Mean Drag = 602.905 N   CdA = 1.09371
2400-2600 : Mean Drag = 600.636 N   CdA = 1.08959
2600-2800 : Mean Drag = 600.257 N   CdA = 1.08890
2800-3000 : Mean Drag = 602.467 N   CdA = 1.09291
```

Interpretation:

- Mean drag is roughly stable around ~600 N after ~1000 iterations.
- Window CdA variation is on the order of ~0.006 m² across the windows.
- This is encouraging for **statistical force stability**.
- It does **not** establish absolute physical accuracy.
- It does **not** establish grid independence.
- It does **not** prove production-level CFD credibility.

---

# 16. Residual Check — Completed Enough for Current Decision

A check was performed to compare beginning and end Ux residuals.

Beginning:

```text
Initial residual = 1
Final residual   = 0.0770062

Initial residual = 0.19428
Final residual   = 0.0160929

Initial residual = 0.0901479
Final residual   = 0.00763773

Initial residual = 0.0636159
Final residual   = 0.00460933

Initial residual = 0.0472556
Final residual   = 0.00339803
```

Near the end:

```text
Initial residual = 0.00229129
Final residual   = 0.000188441

Initial residual = 0.00226481
Final residual   = 0.00018608

Initial residual = 0.00224583
Final residual   = 0.000183687

Initial residual = 0.00222614
Final residual   = 0.000181261

Initial residual = 0.0022375
Final residual   = 0.000180257
```

A count check returned:

```text
3000
```

meaning 3000 Ux solve entries were recorded.

### Decision

The team explicitly decided **not to spend more time analyzing residual history right now**.

The full 3000-iteration run has already been completed, and the next useful validation experiment is mesh independence.

Do not reopen residual analysis unless later evidence requires it.

---

# 17. Coarse Mesh Run — Completed

The surface refinement was changed from:

```text
level (3 4);
```

to:

```text
level (2 3);
```

The same STL was deliberately retained.

### Why the same STL?

This is a controlled mesh-independence experiment.

The question is:

> If only mesh resolution changes, how much does the computed CdA change?

Changing the STL simultaneously would confound:

```text
geometry effect
```

with:

```text
mesh-resolution effect
```

Therefore the same geometry is required for this experiment.

This does **not** mean the eventual ML model will be trained on one STL.

The ML system is a separate stage and should learn/generalize across many designs.

### Coarse mesh check

Completed successfully:

```text
points: 197,385
faces: 497,735
internal faces: 452,118
cells: 152,209
```

Mesh quality:

```text
Max aspect ratio          = 5.20902
Max non-orthogonality     = 62.9989
Average non-orthogonality = 10.7872
Max skewness              = 3.20618
```

Output:

```text
Mesh OK.
```

### Coarse CFD result

The full 3000-iteration CFD run completed.

Extracted:

```text
Averaging range : 2800-3000
Samples         : 21
Mean drag force : 639.66972 N
Std deviation   : 0.61044 N
CdA             : 1.16040 m²
```

Record:

```text
Coarse CdA = 1.16040 m²
```

### Coarse vs medium

```text
Coarse = 1.16040 m²
Medium = 1.09291 m²
```

Relative difference from medium:

```text
≈ 6.18%
```

Therefore:

> The current result is **mesh-sensitive** between the coarse and medium meshes.

The coarse mesh is not broken; `checkMesh` reported `Mesh OK`.

The lower force standard deviation of the coarse case should **not** be interpreted as greater accuracy.

---

# 18. Current Fine Mesh State

The fine mesh setting was changed to:

```text
level (4 5);
```

Verified with:

```text
grep -n -A4 "refinementSurfaces" system/snappyHexMeshDict
```

Output confirmed:

```text
car
{
    level (4 5);
```

### Current status

The fine mesh run was previously thought to be incomplete, but a review of the CFD directory proved it **completed successfully**.

The fine run results are:
- Cells: 1,151,572
- Mean Drag: 591.64 N
- CdA: 1.07327 m²
- Run time: ~186 minutes

**Conclusion:** Asymptotic convergence was achieved. The difference between Medium (CdA 1.09291 m²) and Fine (CdA 1.07327 m²) is only 1.80%. **Medium `(3 4)` is the optimal production mesh.**

When continuing:

1. Revert `system/snappyHexMeshDict` to `level (3 4)`.

---

# 19. Why Mesh Independence Is Being Done

Mesh independence is a **development/validation experiment**, not part of the normal end-user workflow.

The experiment is:

```text
Same STL
   |
   +--> coarse mesh (2 3) --> CdA
   |
   +--> medium mesh (3 4) --> CdA
   |
   +--> fine mesh (4 5) --> CdA
```

The desired question:

> Does CdA approach a stable value as mesh resolution increases?

Possible interpretation:

### Case A — medium ≈ fine

If:

```text
coarse -> materially different
medium -> close to fine
```

then the medium mesh may provide a useful accuracy/cost compromise.

### Case B — medium and fine still differ materially

Then the medium setup is not sufficiently mesh-independent.

### Case C — result continues changing substantially

Then more investigation is needed before freezing the production CFD template.

### Important

A mesh-independence study does **not** prove agreement with experiment or the DrivAerNet++ reference value.

It answers a different question:

> Is the numerical result stable with respect to spatial resolution?

---

# 20. Why the Current CFD Runtime Changed With Mesh

The coarse case:

```text
152,209 cells
```

completed substantially faster than the medium case:

```text
414,139 cells
```

even though both ran:

```text
3000 iterations
```

This is expected.

Each solver iteration operates on the mesh, so fewer cells generally mean less computational work per iteration.

This is also relevant to AeroMorphs' architecture:

- CFD is expensive.
- ML is used to avoid CFD for every candidate.
- Final CFD should be selective.
- Mesh choice is therefore an accuracy-vs-cost engineering decision.

---

# 21. What Has Been Proven So Far

The following are **confirmed achievements**:

### Pipeline correctness

The automated case successfully performs:

```text
STL
 -> blockMesh
 -> snappyHexMesh
 -> checkMesh
 -> simpleFoam
 -> force extraction
 -> JSON result
```

### STL input support

The script accepts an STL path.

### Reusable STL handling

The same case can use:

```text
constant/triSurface/vehicle.stl
```

or copy an external STL into that location.

### Machine-readable output

The pipeline produces:

```text
results/cfd_results.json
```

with drag/CdA data.

### Full 3000-iteration execution

The medium and coarse mesh experiments both completed 3000 iterations.

### Mesh sensitivity demonstrated

Coarse and medium results differ by approximately 6.18% in CdA.

### Important negative result

The current CFD setup **does not yet match the dataset CdA** for the current geometry:

```text
dataset ≈ 0.74556 m²
current medium CFD ≈ 1.09291 m²
```

This is a major validation item still open.

---

# 22. What Has NOT Yet Been Proven

Do not claim any of the following yet:

- Exact reproduction of DrivAerNet++ CFD.
- Experimental validation.
- Production-grade CFD accuracy.
- Grid-independent CdA.
- Universal validity across all vehicle models.
- Robustness across arbitrary STL meshes.
- Agreement between ML predictions and CFD.
- Validated AI optimization improvement.
- Manufacturing readiness.
- Closed-loop Phase 8 operation.
- Automatic cloud dispatch.
- Backend/API integration.
- Evidence-store implementation.
- Affine correction implementation.
- Per-body correction implementation.
- Automatic iterative refinement.

---

# 23. Phase 7 Plan — Intended Architecture

The supplied Phase 7 implementation plan defines a hybrid local/cloud CFD strategy.

It specifies:

- Local Ubuntu PC as primary setup/calibration environment.
- Camber Cloud CPU for batch runs.
- GCP Compute Engine for elastic burst execution.
- Strict total CFD budget of approximately 10–15 simulations.
- CFD as a sparse validation tool.
- Target CdA as the surrogate-aligned metric.
- Planned use of `forceCoeffs` for Cd and conversion to CdA.
- Planned frontal-area calculation for AI-generated champion STLs.
- Planned validation/calibration runs followed by champion validation.
- Planned affine surrogate correction.

The plan's proposed CFD specification includes:

```text
simpleFoam
k-omega SST
30 m/s
moving ground
slip far-field/top/side
wall functions
3000–5000 max iterations
dual convergence criterion
```

and proposes approximately:

```text
~2M cells half-car
```

for the eventual higher-quality template.

The plan estimates approximately:

```text
3–5 hours per simulation
```

on an 8-core environment for its intended higher-resolution setup.

### Important reconciliation with current work

The current local case is **much smaller** than the Phase 7 target and currently uses a full-looking domain with a slip `sides` patch rather than an explicit symmetry plane.

Therefore the Phase 7 document is a **target/implementation plan**, not a statement of current case completion.

---

# 24. Phase 7 Proposed Three Stages

## Stage 1 — Mesh Calibration + Champion Validation

Proposed 5–6 runs:

1. Known DrivAerNet baseline — Fastback.
2. Known DrivAerNet baseline — Estateback.
3. AI Champion — Fastback.
4. AI Champion — Estateback.
5. AI Champion — Notchback.
6. Optional worst-performing AI geometry.

Purpose:

- establish CFD credibility against known reference designs.
- quantify surrogate-vs-CFD error.
- populate initial evidence.

Planned outputs:

```text
ΔCdA = CdA_CFD - CdA_surrogate
mean bias
std
correlation
```

### Current status relative to Stage 1

The current work has **not yet completed this proposed Stage 1**.

The present Notchback geometry is a CFD development/calibration case, but it is not enough by itself to satisfy the full Stage 1 plan.

---

# 25. Phase 7 Proposed Stage 2 — Surrogate Correction

The plan proposes an affine correction:

```text
CdA_corrected = α * CdA_surrogate + β
```

using approximately 3–5 CFD/surrogate data pairs.

Reasoning:

- Constant offset can underfit if slope differs from 1.
- Affine correction is reasonable for sparse data.
- Quadratic models risk overfitting.
- GP/neural-network corrections are not justified with only ~5 points.

Planned implementation:

```python
drag_area_predicted = regressor(z)
drag_area_corrected = alpha * drag_area_predicted + beta
loss = drag_area_corrected + lambda_reg * torch.norm(z)**2
```

This is a **planned Phase 7/8 feature**, not current OpenFOAM implementation.

---

# 26. Phase 7 Proposed Stage 3 — Final Validation

Proposed:

- Final overall champion.
- Final per-class champion.
- Optional second correction cycle if residual error remains high.

Outputs:

- final validated CdA
- pressure contours
- streamlines/wake
- comparison tables
- baseline -> AI v1 -> AI v2 -> final.

Again, this is future work.

---

# 27. Phase 8 Plan — Purpose

The supplied Phase 8 plan changes the project from a one-time calibrated system into a **selective iterative AI–CFD refinement system**.

Its novelty statement is:

> Integrate CFD simulations with AI models to iteratively refine designs.

Phase 8 relationship to Phase 7:

```text
Phase 7 = bootstrap
           |
           v
CFD evidence store
           |
           v
Phase 8 = living refinement system
```

Phase 7's affine calibration becomes the first evidence-informed correction rather than the permanent final state.

---

# 28. Phase 8 Operational Modes

## Fast Mode

Default.

```text
AI + current correction
```

No CFD.

Use for:

- rapid exploration
- generating many candidates
- instant optimization.

## Physics-Refinement Mode

On-demand.

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
update correction
    |
    v
re-optimize
```

Only the converged champion from each round is sent to CFD.

**Never send every intermediate latent candidate to OpenFOAM.**

This is essential to preserve the sparse CFD budget.

---

# 29. Phase 8 One-Round Loop

Planned sequence:

```text
Round N

1. Run optimize_latent_shape.py with current correction
2. Get champion latent vector z_N*
3. Decode to champion_roundN.stl
4. Run OpenFOAM
5. Obtain CdA_CFD_N
6. Record:
   - latent vector
   - surrogate CdA
   - CFD CdA
   - body type
7. Add to evidence store
8. Re-fit correction from all evidence
9. Check CFD/session budget
10. If budget remains, run Round N+1
```

This is the planned future loop.

It should not be confused with the current one-way CFD validation pipeline.

---

# 30. Phase 8 Correction Model Tiers

Planned evidence-dependent strategy:

| Evidence count | Correction |
|---:|---|
| 0 | Identity |
| 1–2 | Constant offset |
| 3–5 | Global affine |
| 6–15 | Per-body-type affine |
| 15+ | Local correction such as k-NN residual or lightweight GP |

The correction should remain differentiable for gradient-based optimization.

This design is **planned**, not currently implemented.

---

# 31. Phase 8 Evidence Store

Planned persistent JSON structure:

```json
{
    "evidence": [
        {
            "id": "phase7_fastback_001",
            "source": "phase7_stage1",
            "body_type": "Fastback",
            "latent_vector_path": "cfd_evidence/latents/fb_001.npy",
            "stl_path": "cfd_evidence/stls/fb_001.stl",
            "cda_surrogate": 0.312,
            "cda_cfd": 0.328,
            "frontal_area": 2.15,
            "timestamp": "2026-09-01T22:30:00"
        }
    ]
}
```

Planned operations:

```text
add_evidence(...)
fit_correction()
predict_corrected(...)
get_evidence_count(...)
nearest_evidence_distance(...)
```

Again, this is a future architecture.

---

# 32. Phase 8 Proposed New Files

The supplied Phase 8 plan proposes:

```text
src/cfd_evidence_store.py
src/surrogate_correction.py
scripts/refine_with_cfd.py
scripts/openfoam_runner.py
```

and a minimal modification to:

```text
scripts/optimize_latent_shape.py
```

to support an optional evidence store.

These files should **not be assumed to exist or be implemented** unless verified in the repository.

---

# 33. Phase 8 Hard Budget

Planned safety bounds include:

```text
--max_rounds N
--max_cfd_hours H
```

These are intended as hard safety limits, not optimization targets.

The exact stopping criteria are intentionally left for empirical determination after Phase 7.

---

# 34. Current CFD-to-ML Metric Mismatch to Keep in Mind

There are two possible CdA calculation pathways in the project:

### Current OpenFOAM implementation

```text
drag force
   |
   v
CdA = F_drag / (0.5 * rho * V²)
```

This implicitly uses a force-based drag-area definition.

### Phase 7 planned pathway

```text
OpenFOAM forceCoeffs
   |
   v
Cd
   |
   + frontal area from geometry
   v
CdA = Cd * A_frontal
```

These should be mathematically consistent **only if the reference area, density, velocity, coordinate directions, and coefficient definitions are aligned**.

Before Phase 7 final validation, the team must settle the authoritative CdA definition and ensure the CFD output and ML target use exactly the same convention.

---

# 35. Frontal Area for AI-Generated Geometry

The Phase 7 plan states that AI-generated champion geometries may not have direct entries in metadata.

Planned method:

```python
import trimesh
import numpy as np
from scipy.spatial import ConvexHull

mesh = trimesh.load("optimization_output/champion_fastback.stl")

yz_points = mesh.vertices[:, 1:3]

hull = ConvexHull(yz_points)

A_frontal = hull.volume
```

Then:

```text
CdA = Cd * A_frontal
```

### Important caution

This is the **planned implementation from the Phase 7 document**, not a verified current repository implementation.

The method also uses a convex-hull projection, which may overestimate frontal area for geometries with concavities. This should be reviewed before being declared the authoritative production method.

---

# 36. STL Preparation — Planned Future Requirements

The Phase 7 plan says AI-generated Marching Cubes STLs should be:

1. Scaled to physical dimensions.
2. Checked/repaired for mesh defects.
3. Given outward-oriented normals.
4. Positioned on the ground plane.
5. Used to calculate frontal area.

Planned utilities include:

```bash
surfaceCheck
surfaceOrient
surfaceTransformPoints
```

Example from the plan:

```bash
surfaceCheck car_champion.stl
surfaceOrient car_champion.stl car_oriented.stl "(0 0 1)"
surfaceTransformPoints -translate '(-0.5 0 0)' car_oriented.stl car_positioned.stl
```

These commands are **not evidence that the current champion STLs have already been processed**.

---

# 37. Important Risks Identified

## 37.1 Mesh memory

16 GB RAM can become limiting for aggressive meshing.

Mitigation:

- keep local target manageable.
- avoid unnecessarily aggressive layers/refinement.
- move large runs to cloud if needed.

## 37.2 Non-convergence

Potential mitigations from the Phase 7 plan:

- first-order upwind initially.
- switch to higher-order discretization later.
- potentialFoam initialization.
- debug locally before cloud batch execution.

These are future/optional recommendations, not necessarily current settings.

## 37.3 STL defects

Marching Cubes geometry may have:

- holes
- non-manifold edges
- bad normals.

Use `surfaceCheck` and repair tools before production meshing.

## 37.4 Surrogate correction nonlinearity

If affine correction leaves large residuals, the Phase 7 plan suggests considering per-body-type correction.

## 37.5 Runtime

If a run becomes too expensive locally, options include:

- lower mesh size/refinement.
- cloud CPU execution.
- GCP compute-optimized VM.
- Camber batch execution.

Cloud compute does **not** increase the CFD simulation budget.

---

# 38. Phase 7/8 Compute Strategy

The supplied plan proposes:

### Local Ubuntu

Use for:

- case setup
- mesh debugging
- calibration
- interactive troubleshooting
- initial runs.

### Camber Cloud CPU

Use for:

- asynchronous batch runs
- freeing local workstation
- extra RAM.

### GCP Compute Engine

Use for:

- elastic burst
- concurrent batch runs
- cases when cloud credits/quota permit.

The Phase 7 plan explicitly preserves the total CFD budget at approximately:

```text
10–15 simulations
```

Phase 8 also says the budget must remain sparse and resource-bounded.

Do not casually add simulations just because cloud compute is available.

---

# 39. What Should Happen Next — Immediate CFD Track

## Immediate next step

**Build the Python CFD Bridge and run the Different-STL test.**

The mesh independence study is complete, and Medium `(3 4)` has been selected as the production baseline.

The next goals are:
1. Revert `system/snappyHexMeshDict` to `level (3 4)`.
2. Build the Python CFD Bridge (`scripts/openfoam_runner.py`) to wrap Nidhi's bash script.
3. Execute the "Different STL" Robustness Test using the CFD Bridge.

Do not restart any mesh independence studies.

---

# 40. Next Validation After Mesh Study

Once a reasonable mesh level is chosen:

## Test a genuinely different STL

This is separate from mesh independence.

Purpose:

> Prove that the CFD automation accepts a new geometry rather than merely reproducing one hard-coded case.

The test should use:

```text
different STL
same CFD template
same automation script
```

Expected pipeline:

```text
new STL
  -> copy/preparation
  -> blockMesh
  -> snappyHexMesh
  -> checkMesh
  -> simpleFoam
  -> forces
  -> JSON
```

This is a critical step toward reusable CFD templates.

---

# 41. After Different-STL Test

Then:

1. Freeze the validated Notchback template.
2. Generalize the case to supported vehicle/body types.
3. Create model-specific templates where geometry/refinement requires it.
4. Ensure each template accepts an external STL.
5. Standardize result JSON.
6. Document the ML-to-CFD interface.
7. Package scripts/templates for GitHub/backend handoff.

---

# 42. Important Architecture Principle for Multi-Model Support

The backend should eventually be able to select:

```text
body_type
```

and invoke the appropriate CFD template.

Conceptually:

```text
body_type = notchback
    -> notchback OpenFOAM template

body_type = fastback
    -> fastback OpenFOAM template

body_type = estateback
    -> estateback OpenFOAM template
```

The CFD side should hide OpenFOAM terminal details from the end user.

The backend/application should ultimately receive structured output rather than scrape terminal logs.

Example desired result contract:

```json
{
    "status": "success",
    "drag_force_N": 0.0,
    "cda_m2": 0.0,
    "iterations": 3000
}
```

Actual final schema is still to be agreed with the backend engineer.

---

# 43. Desired Production-Level User Experience

The user should not have to:

- open a terminal
- edit OpenFOAM dictionaries
- manually run `blockMesh`
- manually run `snappyHexMesh`
- manually run `checkMesh`
- manually run `simpleFoam`
- manually parse `force.dat`.

Instead:

```text
Upload STL
    |
Select body type
    |
Optimize
    |
ML produces candidate
    |
Validate
    |
CFD service runs
    |
Result returned
```

The current shell automation is the technical foundation for that behavior.

---

# 44. Current Project Status Summary

## Completed
 
 - OpenFOAM 2412 working.
 - `simpleFoam` working.
 - Notchback baseline case established.
 - Automation working copy created.
 - Original baseline and backup preserved.
 - STL input argument implemented.
 - Same-file copy bug fixed.
 - Cleanup bug that deleted `0/` fixed.
 - `blockMesh` automated.
 - `snappyHexMesh` automated.
 - `checkMesh` automated.
 - `simpleFoam` automated.
 - force extraction automated.
 - JSON result extraction automated.
 - Medium 3000-iteration run completed.
 - Coarse 3000-iteration run completed.
 - Ux residual beginning/end check completed.
 - Force-history statistical analysis completed.
 - Coarse vs medium mesh sensitivity established.
 - Fine refinement configured.
 - Fine mesh run completed.
 - Final mesh-independence decision completed (Medium `(3 4)` selected).
 
 ## In progress / pending
 
 - Absolute CdA discrepancy investigation.
 - Different-STL robustness test.
 - Python CFD Bridge.
 - Notchback template freeze.
 - Other body-type templates.
 - Final standardized CFD result interface.
 - GitHub handoff packaging.
 - Phase 7 calibration/AI champion validation.
 - Surrogate-CFD error dataset.
 - Affine correction.
 - Phase 8 evidence store.
 - Iterative AI-CFD refinement mode.
 - Cloud batch execution.
 - Final publication-quality validation.

---

# 45. Key Numbers to Keep Handy

### Current geometry metadata

```text
Dataset Cd       = 0.27552
Dataset frontal area = 2.706 m²
Dataset CdA      = 0.745558 m²
```

### Medium mesh

```text
Refinement       = (3 4)
Cells            = 414,139
Mean drag        = 602.46677 N
Std drag         = 3.20571 N
CdA              = 1.09291023 m²
Runtime           ≈ 58.8 min
```

### Coarse mesh

```text
Refinement       = (2 3)
Cells            = 152,209
Mean drag        = 639.66972 N
Std drag         = 0.61044 N
CdA              = 1.16040 m²
```

### Current fine configuration

```text
Refinement       = (4 5)
Run status       = not confirmed complete in this handoff
```

### CFD operating point

```text
Velocity         = 30 m/s
Density          = 1.225 kg/m³
Turbulence       = k-omega SST
Solver           = simpleFoam
Iterations       = 3000 fixed for current experiment
```

---

# 46. Important "Do Not" Rules for the Next Agent

1. **Do not call the current `N_S_WWC_WM_025` case Fastback. It is the current Notchback test case.**
2. **Do not overwrite `baselineCar` or `baselineCar_backup`.**
3. **Do not use the old cleanup regex that can delete `0/`.**
4. **Do not re-run the fine mesh or mesh independence. It is completed.**
5. **Do not claim the current CFD reproduces DrivAerNet++ exactly.**
6. **Do not treat the Phase 7 plan as already implemented.**
7. **Do not claim `forceCoeffs` is implemented; current extraction uses `forces`.**
8. **Do not put OpenFOAM inside every ML optimization step.**
9. **Do not run CFD for every latent candidate.**
10. **Do not change multiple CFD variables during a mesh-independence experiment.**
11. **Do not interpret low residuals alone as proof of aerodynamic accuracy.**
12. **Do not interpret low force standard deviation alone as proof of accuracy.**
13. **Do not use the dataset CdA as if it were automatically the ground truth for the current CFD setup without investigating the methodological differences.**
14. **Do not spend the entire CFD budget on arbitrary experiments; the planned total budget is sparse (~10–15 runs).**
15. **Do not make manufacturing-readiness claims from aerodynamic CFD alone.**

---

# 47. Recommended Decision Tree From Here

```text
CURRENT
  |
  v
Fine mesh run
  |
  v
Compare coarse / medium / fine
  |
  +---- medium ≈ fine ----+
  |                       |
  |                       v
  |                 choose efficient mesh
  |
  +---- medium != fine ---+
                          |
                          v
                    investigate mesh
                          |
                          v
                 choose/further refine
                          |
                          v
                Different STL test
                          |
                          v
                 Freeze Notchback
                          |
                          v
              Build other body templates
                          |
                          v
              Standardize CFD interface
                          |
                          v
                 Phase 7 calibration
                          |
                          v
              CFD evidence collection
                          |
                          v
                Affine correction
                          |
                          v
                 Phase 8 integration
                          |
                          v
              Selective AI → CFD loop
                          |
                          v
                Evidence store grows
                          |
                          v
              Correction model evolves
```

---

# 48. Final Mental Model for the AI Agent

When continuing AeroMorphs CFD, think in these layers:

### Layer 1 — CFD case

```text
OpenFOAM dictionaries
```

### Layer 2 — CFD automation

```text
run_cfd.sh
extract_results.py
```

### Layer 3 — CFD validation

```text
convergence
mesh independence
reference comparison
different-geometry robustness
```

### Layer 4 — Reusable CFD template

```text
STL in
   ->
OpenFOAM
   ->
JSON out
```

### Layer 5 — Application integration

```text
ML optimized STL
   ->
CFD runner
   ->
validated CdA
```

### Layer 6 — Phase 8 physics refinement

```text
AI champion
   ->
selective CFD
   ->
evidence store
   ->
correction update
   ->
AI re-optimization
```

The current work is primarily between **Layers 2 and 3**.

Do not jump to Layer 6 until the CFD foundation is sufficiently validated.

---

# 49. Source/Plan Context Included in This Handoff

Two supplied planning documents are the authoritative planning references for the future phases:

- **Phase 7: OpenFOAM CFD Validation — Hybrid Local & Cloud Implementation Plan**
- **Phase 8: Iterative AI–CFD Closed-Loop Refinement Plan**

They define the intended future architecture, compute strategy, correction model, evidence store, and iterative refinement workflow. Their planned features must be verified against the actual repository before being treated as implemented.

---

# 50. Handoff Status

**As of the latest known terminal state:**

```text
OpenFOAM 2412                 WORKING
Notchback baseline            WORKING
Automation pipeline           WORKING
3000-iteration medium run    COMPLETE (414,139 cells, CdA = 1.0929 m²)
3000-iteration coarse run    COMPLETE (152,209 cells, CdA = 1.1604 m²)
3000-iteration fine run      COMPLETE (1,151,572 cells, CdA = 1.0733 m²)
Mesh sensitivity / study     COMPLETE (Asymptotic convergence: 1.80% delta; Medium frozen)
Different STL test           COMPLETE (Estateback E_S_WWC_WM_014, CdA = 0.9186 m²)
AI STL 1:1 Scale Fix         COMPLETE (scripts/denormalize_mesh.py implemented)
First AI Champion Validation COMPLETE (step_250_1to1_scale: predicted 0.4814 vs CFD 1.0873 m²)
GCP Backend Parity Benchmark COMPLETE (c2-standard-8: exact 0.000% parity, 56.6 min, auto-cleanup)
GCP Compute Engine Role      PROMOTED TO PRIMARY MVP BACKEND
Absolute CdA calibration     READY (Stage 1 Affine Model fitting next)
Phase 7 full validation      IN PROGRESS (Stage 1 calibration batch)
Phase 8 implementation       PENDING (Awaiting Stage 1 affine parameters)
```

The validated next operational step is:

> **Execute the Phase 7 Stage 1 calibration batch on GCP Compute Engine to populate the Evidence Store, compute the Affine Correction parameters ($\alpha, \beta$), and re-optimize the AI surrogate.**
