# GCP Compute Engine CFD Backend: Setup, Architecture, and Integration Plan

> **Project:** AeroMorphs — Generative Design for Low-Drag Car Bodies  
> **Document Role:** Implementation-Ready Execution Plan (Architecture & Setup)  
> **Target Cloud Backend:** Google Cloud Platform (GCP) Compute Engine (Intended Primary for MVP)  
> **Reference & Fallback Platform:** Local Ubuntu 24.04 Desktop (Intel Core i7-12700, 16 GB RAM)  
> **Target Metric:** $C_dA$ (Drag Area, in $\text{m}^2$) $= C_d \times A_{\text{frontal}}$  
> **Preserved CFD Budget:** 10–15 Total Simulations across Phase 7 and Phase 8  
> **Status:** Pending Milestone 1 Parity Benchmark  
> **Last Updated:** September 2026

---

## 1. Epistemic Baseline: Separation of Facts, Plans, and Targets

To maintain strict scientific and engineering rigor, all information in this plan is categorized into three explicit domains:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EPISTEMIC SEPARATION MATRIX                        │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ Domain                            │ Scope & Status                          │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ 1. VERIFIED CURRENT FACTS         │ Measured on local hardware & codebase   │
│ 2. PLANNED CONFIGURATION          │ Engineered design ready for execution   │
│ 3. ESTIMATED TARGETS TO VALIDATE  │ Hypotheses requiring GCP benchmark test │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

### 1.1 Verified Current Facts
* **Local Compute Hardware:** Intel Core i7-12700 (12 cores / 20 threads: 8 Performance cores with hyperthreading + 4 Efficient cores), 16 GB DDR4 RAM, NVMe SSD storage.
* **Current CFD Execution Mode:** Single-process / serial execution (`simpleFoam > log.simpleFoam 2>&1`) running on a single CPU core. No `decomposeParDict` or MPI parallelism is currently configured in the working case.
* **Local Runtime:** A 3,000-iteration medium-mesh run takes **~50–60 minutes** on the local i7-12700 CPU.
* **CFD Solver & Settings:** OpenFOAM 2412 (ESI/OpenCFD), `simpleFoam` (steady incompressible RANS), $k$-$\omega$ SST turbulence, Newtonian air ($\nu = 1.5 \times 10^{-5}\text{ m}^2/\text{s}$, $\rho = 1.225\text{ kg/m}^3$), $V_{\infty} = 30\text{ m/s}$, moving ground ($30\text{ m/s}$), slip side/top boundaries, fixed 3,000 iterations.
* **Mesh Independence & Baselining Status:**
  * Coarse `(2 3)`: 152,209 cells, $C_dA = 1.16040\text{ m}^2$
  * Medium `(3 4)`: 414,139 cells, $C_dA = 1.09291\text{ m}^2$, Mean Drag $= 602.48\text{ N}$ (~59 min) — **Selected Production Mesh**
  * Fine `(4 5)`: 1,151,572 cells, $C_dA = 1.07327\text{ m}^2$, Mean Drag $= 591.64\text{ N}$ (~186 min) — **Asymptotic convergence verified (1.80% delta)**
  * Estateback Baseline (`E_S_WWC_WM_014`): 414,139 cells, $C_dA = 0.91862\text{ m}^2$, Mean Drag $= 506.39\text{ N}$ (~60 min)
* **Local Storage Topology:** The Git repository lives inside an encrypted Cryptomator FUSE mount (`/home/student/.local/share/Cryptomator/mnt/AeroDyNaS/Main Project Folder`). OpenFOAM case execution is strictly quarantined to a native ext4 directory (`/home/student/AeroMorphs/cfd_automation`) to prevent I/O thrashing and vault bloat.
* **Local Tooling State:** The Google Cloud SDK (`gcloud`) is **not** currently installed on the system.

### 1.2 Planned Configuration
* **Target Cloud Service:** GCP Compute Engine instance running Ubuntu 24.04 LTS.
* **Target Instance Family:** `c2-standard-8` (Compute-Optimized, 8 vCPUs, 32 GB RAM, 50 GB `pd-balanced` SSD).
* **Provisioning Model:** Ephemeral **Spot VM** (fallback to On-Demand if Spot capacity is unavailable).
* **OpenFOAM Software:** OpenFOAM 2412 (ESI/OpenCFD) installed to match local binary and numerical behavior identically.
* **Job & Storage Flow:** Cloud Storage (GCS) bucket holds a clean ~250 KB template case bundle and staged 1:1 STL geometries. Worker VMs boot, execute CFD, write results, and self-terminate.
* **Runner Abstraction:** Pluggable `openfoam_runner.py` with `--backend local` and `--backend gcp` flags, outputting identical `results/cfd_results.json`.

### 1.3 Estimated Targets That Must Be Benchmarked / Validated
* **GCP Runtime Target (Estimate):** Estimated **~35–50 minutes** for serial execution on C2 (3.8 GHz all-core turbo), or **~15–25 minutes** if parallel decomposition (`decomposePar`, 4–8 cores) is validated. *This figure is an unconfirmed target until Milestone 1 benchmark executes.*
* **GCP Cost Target (Estimate):** Published list prices indicate `c2-standard-8` Spot pricing is ~$0.07/hr and On-Demand is ~$0.33/hr. Estimated per-run cost is **~$0.04–$0.08 (Spot)** or **~$0.20–$0.30 (On-Demand)**. *Actual billing rates, network egress, and disk charges must be validated from real GCP billing data.*
* **Parity Criterion:** GCP calculated $C_dA$ must match local $C_dA$ within **$\pm 0.5\%$** on the identical STL.
* **Preemption Tolerance:** Spot preemption rate in `us-central1` during a 45-minute window must be observed.

---

## 2. Decision Framework: Primary vs. Fallback

GCP Compute Engine is intended as the **primary CFD backend for the MVP**, while the local Ubuntu desktop remains the **authoritative reference baseline and offline fallback**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DECISION GATEWAY (MILESTONE 1)                     │
│                                                                             │
│               Run 3000-Iteration Medium Mesh on GCP C2 Worker               │
│                                       │                                     │
│                                       ▼                                     │
│                         Check Benchmark Parity:                             │
│                         1. |ΔCdA| < 0.5% vs Local?                          │
│                         2. CheckMesh Clean & No Inverted Faces?             │
│                         3. Wall-Clock Runtime Acceptable?                   │
│                         4. Cost per Simulation within Target?               │
│                                       │                                     │
│                     ┌─────────────────┴─────────────────┐                   │
│                     │ YES                               │ NO                │
│                     ▼                                   ▼                   │
│        ┌──────────────────────────┐       ┌──────────────────────────┐      │
│        │ PROMOTE GCP TO PRIMARY   │       │ RETAIN LOCAL AS PRIMARY  │      │
│        │ - Use GCP for MVP batch  │       │ - Fix GCP discrepancy or │      │
│        │ - Local kept as fallback │       │   keep local execution   │      │
│        └──────────────────────────┘       └──────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Promotion Criteria to Primary Backend
The decision to promote GCP to the primary backend requires passing **all** of the following gates during Milestone 1:
1. **Numerical Parity Gate:** GCP calculated $C_dA$ must be within **$\pm 0.5\%$** of the local Notchback baseline ($1.09291\text{ m}^2 \pm 0.0054\text{ m}^2$).
2. **Mesh Topology Gate:** `checkMesh` on GCP produces zero severe non-orthogonal faces, zero illegal pyramids, and identical patch assignments (`inlet`, `outlet`, `lowerWall`, `upperWall`, `sides`, `car`).
3. **Turnaround Gate:** Total job wall-clock time (boot + mesh + solve + result transfer) must not exceed local runtime (~60 min).
4. **Cost Gate:** Cost per simulation run on Spot VM must not exceed **$0.20**.

If any gate fails, the system automatically falls back to local execution with zero changes to the ML pipeline.

---

## 3. Step-by-Step Implementation Roadmap

```
Stage 1: Setup & Auth ──> Stage 2: Packaging ──> Stage 3: Benchmark (GATE)
                                                         │
                                                         ▼
Stage 6: ML Refinement <── Stage 5: Lifecycle <── Stage 4: Integration
```

---

### Milestone 1: GCP Workspace Setup and Authentication

#### Step 1.1: Install Google Cloud SDK on Local Workstation
Since `gcloud` is not currently present on the workstation, install it via the official Google package repository:

```bash
# Add official Google Cloud package repo
sudo apt update
sudo apt install -y apt-transport-https ca-certificates gnupg curl

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg |   sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" |   sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt update && sudo apt install -y google-cloud-cli
```

#### Step 1.2: Authenticate and Configure Project Defaults
```bash
# 1. Interactive login for developer
gcloud auth login
gcloud auth application-default login

# 2. Select project and target region
# Target region: us-central1 (highest C2 Spot availability, low cost)
gcloud config set project <GCP_PROJECT_ID>
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# 3. Enable required Google Cloud APIs
gcloud services enable   compute.googleapis.com   storage.googleapis.com   logging.googleapis.com   iam.googleapis.com
```

#### Step 1.3: Quota & Budget Enforcement
1. **Quota Check:** Verify available C2 and Spot CPU quotas:
   ```bash
   gcloud compute project-info describe --format="table(quotas.metric,quotas.usage,quotas.limit)" | grep -E 'CPUS|C2'
   ```
   *Requirement:* At least 8 vCPUs under `CPUS` / `C2_CPUS` or `PREEMPTIBLE_CPUS`.
2. **Budget Alert:** Create a **$15.00/month** maximum budget cap in the GCP Billing Console with email alerts configured at 50% ($7.50), 80% ($12.00), and 100% ($15.00).

#### Step 1.4: Cloud Storage Bucket Setup
Create a dedicated bucket for staging CFD cases and results:
```bash
# Create bucket in target region
gsutil mb -p <GCP_PROJECT_ID> -c standard -l us-central1 gs://aeromorphs-cfd-<GCP_PROJECT_ID>/
```

---

### Milestone 2: CFD Case Packaging & VM Environment Strategy

#### Step 2.1: Package Existing OpenFOAM Case into Template Bundle
The local CFD case (`/home/student/AeroMorphs/cfd_automation`) has accumulated gigabytes of transient iteration folders (`100/`, ..., `3000/`) and logs. We extract only the pristine configuration dictionaries into a lightweight bundle (`< 250 KB`):

```
template_case_v1/
├── 0/                      # Initial & boundary conditions (U, p, k, omega, nut)
├── constant/
│   ├── transportProperties # Newtonian (nu = 1.5e-5)
│   ├── turbulenceProperties# RAS kOmegaSST
│   └── triSurface/         # Placeholder directory for vehicle.stl
├── system/
│   ├── blockMeshDict       # Background domain (-3 -3 0) to (10 3 4)
│   ├── controlDict         # endTime 3000, writeInterval 100, forces writeInterval 10
│   ├── fvSchemes           # Standard schemes
│   ├── fvSolution          # GAMG for p, smoothSolver for U, k, omega
│   ├── meshQualityDict     # Mesh quality criteria
│   └── snappyHexMeshDict   # Optimal Medium mesh: refinementSurfaces car level (3 4)
└── scripts/
    ├── run_cfd.sh          # Execution orchestrator
    └── extract_results.py  # Forces parser & JSON generator
```

A packaging utility (`deploy/package_cfd_case.py`) will create `template_case_v1.tar.gz` and upload it to GCS:
```bash
gsutil cp template_case_v1.tar.gz gs://aeromorphs-cfd-<GCP_PROJECT_ID>/templates/
```

#### Step 2.2: VM Provisioning Strategy
* **Machine Type:** `c2-standard-8` (Compute-Optimized, 8 vCPUs, 32 GB RAM, 50 GB `pd-balanced` SSD).
* **OS Image:** `ubuntu-os-cloud/ubuntu-2404-lts-amd64`.
* **Two-Phase VM Deployment:**
  1. **Prototype Phase (Milestone 1 Benchmark):** Provision a standard VM using an automated startup script (`deploy/gcp_bootstrap.sh`) that installs OpenFOAM 2412 via apt (`dl.openfoam.com`).
  2. **Production Golden Image Phase:** After benchmark validation, create a custom disk image (`aeromorphs-openfoam2412-v1`) from the prototype disk. Subsequent runs boot directly from this image in **< 40 seconds**, bypassing package installation.

#### Step 2.3: Worker Startup & Bootstrap Script (`deploy/gcp_bootstrap.sh`)
```bash
#!/bin/bash
set -ex

# 1. Base tools
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y     curl wget gnupg software-properties-common     build-essential python3 python3-pip python3-numpy python3-scipy jq

# 2. Official OpenCFD / ESI OpenFOAM Repository
curl -s https://dl.openfoam.com/add-debian-repo.sh | bash
apt-get update

# 3. Exact matching OpenFOAM version
apt-get install -y openfoam2412-default

# 4. Environment setup
echo "source /usr/lib/openfoam/openfoam2412/etc/bashrc" >> /etc/bash.bashrc
mkdir -p /opt/aeromorphs_cfd
chmod 777 /opt/aeromorphs_cfd
```

---

### Milestone 3: Parity and Performance Benchmark (Crucial Decision Gate)

This is the central verification milestone. It executes the exact same benchmark run used locally to establish ground-truth comparison data.

#### Step 3.1: Benchmark Test Specification
* **Benchmark Geometry:** `N_S_WWC_WM_025.stl` (Notchback reference car from DrivAerNet++).
* **Mesh Setting:** Production Medium mesh `level (3 4)` in `snappyHexMeshDict`.
* **Solver Iterations:** Exactly 3,000 iterations of `simpleFoam`.
* **Force Averaging Window:** Iterations 2,800 to 3,000 (21 samples).
* **Local Baseline Reference (Verified):**
  * Cell Count: **414,139 cells**
  * Mean Drag Force: **602.48 N**
  * Calculated $C_dA$: **1.09291 m²**
  * Force Standard Deviation: **1.84 N**
  * Local Runtime: **~59 minutes**

#### Step 3.2: Benchmark Execution Command
Launch the benchmark VM via `gcloud`:
```bash
gcloud compute instances create cfd-benchmark-worker     --zone=us-central1-a     --machine-type=c2-standard-8     --provisioning-model=SPOT     --instance-termination-action=DELETE     --image-family=ubuntu-2404-lts-amd64     --image-project=ubuntu-os-cloud     --boot-disk-size=50GB     --boot-disk-type=pd-balanced     --scopes=cloud-platform     --metadata=startup-script="$(cat deploy/run_benchmark_job.sh)"
```

#### Step 3.3: Verification Scorecard & Target Tracking

| Metric | Local Baseline (Verified Fact) | GCP Benchmark (Target / Estimate) | Acceptance Criteria |
| :--- | :---: | :---: | :---: |
| **Mesh Cell Count** | 414,139 cells | *To be measured* | $\pm 0.5\%$ ($412	ext{k} - 416	ext{k}$) |
| **Mean Drag Force** | $602.48	ext{ N}$ | *To be measured* | $\pm 0.5\%$ ($599.5 - 605.5	ext{ N}$) |
| **Calculated $C_dA$** | $1.09291	ext{ m}^2$ | *To be measured* | $\pm 0.5\%$ ($1.087 - 1.098	ext{ m}^2$) |
| **Force Stability ($\sigma$)** | $1.84	ext{ N}$ | *To be measured* | $< 3.0	ext{ N}$ |
| **Wall-Clock Time** | ~59 minutes | *Estimated Target: ~35–50 min* | $\le 60	ext{ minutes}$ |
| **Cost per Run** | $0.00 (Electricity ~$0.05) | *Estimated Target: ~$0.04–$0.08* | $\le \$0.20$ |

*Upon passing all acceptance criteria, record the confirmed GCP metrics in this document and proceed to Milestone 4.*

---

### Milestone 4: Unified Local/GCP CFD Runner Abstraction

To ensure the ML surrogate optimization and evaluation loops never depend directly on the execution environment, `scripts/openfoam_runner.py` will implement a provider pattern.

#### Step 4.1: Architecture Interface
```python
# In scripts/openfoam_runner.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class BaseCFDBackend(ABC):
    @abstractmethod
    def run_cfd(self, stl_path: Path, job_id: str) -> Dict[str, Any]:
        """Executes CFD on the given STL and returns standardized metrics."""
        pass

class LocalCFDBackend(BaseCFDBackend):
    """Executes via /home/student/AeroMorphs/cfd_automation/scripts/run_cfd.sh."""
    ...

class GCPComputeBackend(BaseCFDBackend):
    """Stages STL to GCS, launches Spot VM, awaits results, cleans up VM."""
    ...
```

#### Step 4.2: CLI Interface & Standard Output Contract
The runner will accept a `--backend` parameter:
```bash
# Run locally (fallback or interactive debugging)
python3 scripts/openfoam_runner.py --backend local path/to/vehicle.stl

# Run on GCP Compute Engine (primary MVP backend)
python3 scripts/openfoam_runner.py --backend gcp path/to/vehicle.stl
```

Both backends produce an identical, immutable `results/cfd_results.json`:
```json
{
  "job_id": "notchback_benchmark_001",
  "backend": "gcp_c2_standard_8_spot",
  "status": "COMPLETED",
  "mean_drag_force_N": 602.481,
  "std_drag_force_N": 1.842,
  "cda_m2": 1.09291,
  "velocity_mps": 30.0,
  "density_kg_m3": 1.225,
  "averaging_start_iteration": 2800,
  "averaging_end_iteration": 3000,
  "samples": 21,
  "execution_wall_time_seconds": 2340.5,
  "timestamp": "2026-09-12T01:00:00Z"
}
```

---

### Milestone 5: Production Job Lifecycle, Cleanup & Cost Governance

#### Step 5.1: Ephemeral Worker Lifecycle Flow

```
[Local Workspace]                 [Cloud Storage (GCS)]           [GCP Ephemeral Worker]
       │                                   │                                 │
1. Denormalize STL to 1:1                  │                                 │
   (scripts/denormalize_mesh.py)           │                                 │
       │                                   │                                 │
2. Upload vehicle.stl ────────────────────>│ gs://bucket/jobs/ID/vehicle.stl │
       │                                   │                                 │
3. Create Spot VM (gcloud compute create)  │                                 │
   with job metadata ───────────────────────────────────────────────────────>│
       │                                   │                                 │
       │                                   │<── Download template & STL ─────│
       │                                   │                                 │
       │                                   │    Run: blockMesh               │
       │                                   │    Run: snappyHexMesh           │
       │                                   │    Run: checkMesh               │
       │                                   │    Run: simpleFoam (3000 iters) │
       │                                   │    Run: extract_results.py      │
       │                                   │                                 │
       │                                   │<── Upload cfd_results.json ─────│
       │                                   │    Upload log.simpleFoam.gz     │
       │                                   │                                 │
       │                                   │    Self-Terminate (poweroff) ──>│ [Instance Stops]
       │                                   │                                 │
4. Poll GCS for results.json <─────────────│                                 │
       │                                                                     │
5. Delete VM resource (gcloud compute delete --quiet)                        │
       │                                                                     │
6. Ingest into Evidence Store (metadata/cfd_evidence_store.json)             │
```

#### Step 5.2: Cost Governance and Safety Safeguards
To guarantee zero runaway costs and prevent accidental billing:
1. **Linux-Level Hard Solver Timeout:** The worker execution script wraps the solver:
   ```bash
   timeout 75m simpleFoam > log.simpleFoam 2>&1 || touch TIMEOUT_ERROR
   ```
2. **Unconditional VM Self-Termination:** The worker startup script traps all exit states:
   ```bash
   trap 'sudo poweroff' EXIT
   ```
3. **Local Watchdog Cleanup:** If the local Python runner has not detected completion after 90 minutes, it automatically executes:
   ```bash
   gcloud compute instances delete cfd-worker-<JOB_ID> --zone=us-central1-a --quiet
   ```
4. **Intermediate Field Data Pruning:** Volumetric time directories (`100/`, ..., `3000/`) and `polyMesh/` (~2–3 GB) are deleted on the VM before result upload, uploading only `cfd_results.json` and compressed log summaries to GCS.

---

### Milestone 6: ML Integration & Preserved Simulation Budget

#### Step 6.1: Strict Preservation of the 10–15 Simulation Budget
The introduction of GCP Compute Engine does **not** expand the CFD budget. The total budget across Phase 7 and Phase 8 remains strictly **10–15 simulations total**:

| Project Phase | Focus | Planned Runs | Role of GCP Backend |
| :--- | :--- | :---: | :--- |
| **Stage 1 (Calibration & Baseline)** | Benchmark & AI Champion Baseline | 5–6 runs | Run 1 locally; Runs 2–6 on GCP |
| **Stage 2 (Correction & Re-Opt)** | Affine Model Validation | 3–4 runs | Dispatched to GCP Spot VMs |
| **Stage 3 (Final Verification)** | Best-in-class Champion Validation | 2–3 runs | Dispatched to GCP Spot VMs |
| **Total Lifetime Budget** | **Strictly Resource-Bounded** | **10–15 runs** | **Total cloud spend target: < $3.00** |

#### Step 6.2: End-to-End Closed-Loop Refinement Path
1. **Optimization:** `scripts/optimize_latent_shape.py` generates latent vector $z^*$ and extracts Marching Cubes STL.
2. **Physical Scaling:** `scripts/denormalize_mesh.py` scales geometry to 1:1 real-world vehicle dimensions ($L \approx 4.5\text{ m}$).
3. **Validation Dispatch:** `openfoam_runner.py --backend gcp` submits the geometry to an ephemeral GCP Spot VM.
4. **Evidence Store Ingestion:** Validated $C_dA_{\text{CFD}}$ is appended to `metadata/cfd_evidence_store.json`:
   ```json
   {
     "id": "estateback_opt_step250",
     "body_type": "Estateback",
     "cda_surrogate": 0.4814,
     "cda_cfd": 1.0873,
     "delta_cda": 0.6059,
     "timestamp": "2026-09-12T00:01:00Z"
   }
   ```
5. **Surrogate Calibration:** `src/surrogate_correction.py` updates the affine correction parameters:
   $$C_dA_{\text{corrected}} = \alpha \cdot C_dA_{\text{surrogate}} + \beta$$
6. **Re-Optimization:** The updated $(\alpha, \beta)$ informs the next gradient optimization round, steering shape morphing away from surrogate exploitation toward genuine aerodynamic drag reduction.

---

## 4. Execution Readiness Checklist

Follow this checklist during actual implementation:

- [ ] **Milestone 1: GCP Setup & Quotas**
  - [ ] Install `google-cloud-cli` on local workstation.
  - [ ] Authenticate via `gcloud auth login` and set default project/region (`us-central1`).
  - [ ] Enable Compute Engine, Cloud Storage, Logging, and IAM APIs.
  - [ ] Verify C2 and Spot CPU quotas $\ge 8$ vCPUs.
  - [ ] Create GCS bucket `gs://aeromorphs-cfd-<PROJECT_ID>/`.
  - [ ] Set $15.00 budget notification alert in GCP Billing.
- [ ] **Milestone 2: Case Packaging & Bootstrap**
  - [ ] Bundle pristine CFD case into `template_case_v1.tar.gz`.
  - [ ] Upload template case to GCS.
  - [ ] Author `deploy/gcp_bootstrap.sh` with OpenFOAM 2412 installation logic.
- [ ] **Milestone 3: Parity Benchmark (DECISION GATE)**
  - [ ] Stage `N_S_WWC_WM_025.stl` to GCS.
  - [ ] Launch `c2-standard-8` Spot worker for 3,000 iterations.
  - [ ] Evaluate results against Parity Acceptance Criteria ($|\Delta C_dA| < 0.5\%$).
  - [ ] Record confirmed GCP runtime and billing metrics.
  - [ ] Make official backend promotion decision (GCP as Primary vs. Retain Local).
- [ ] **Milestone 4: Backend Integration**
  - [ ] Implement `LocalCFDBackend` and `GCPComputeBackend` in `scripts/openfoam_runner.py`.
  - [ ] Validate `--backend local` and `--backend gcp` CLI execution.
- [ ] **Milestone 5: Production Operationalization**
  - [ ] Snapshot validated benchmark disk into `aeromorphs-openfoam2412-v1` custom image.
  - [ ] Test end-to-end autonomous ephemeral job lifecycle with self-poweroff and deletion.
  - [ ] Connect CFD results output to `metadata/cfd_evidence_store.json`.

---
