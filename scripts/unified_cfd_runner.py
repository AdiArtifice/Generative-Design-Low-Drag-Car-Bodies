#!/usr/bin/env python3
"""
AeroMorphs Unified CFD Runner (Local & GCP)
Provides a single clean API for dispatching CFD jobs to either the local workstation 
or the GCP 4-core MPI backend. Abstracting the infrastructure away from the AI loop.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

# Project Configuration
PROJECT_ID = "aerodesign-cfd-mvp"
REGION = "us-central1"
ZONES = ["us-central1-c", "us-central1-b", "us-central1-f", "us-central1-a"]
BUCKET = f"gs://aeromorphs-cfd-{PROJECT_ID}"
MACHINE_TYPE = "c2-standard-8"
BOOT_DISK_SIZE = "50GB"

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
STARTUP_SCRIPT = DEPLOY_DIR / "run_mpi_benchmark_job.sh"

def run_cmd(cmd, check=True, capture=True):
    env = os.environ.copy()
    env["CLOUDSDK_PYTHON"] = "/usr/bin/python3"
    env["PATH"] = f"/home/student/google-cloud-sdk/bin:{env.get('PATH', '')}"
    
    if capture:
        res = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        res = subprocess.run(cmd, shell=True, env=env)
        
    if check and res.returncode != 0:
        err = res.stderr if capture else "command failed"
        raise RuntimeError(f"Command failed (exit {res.returncode}): {cmd}\nError: {err}")
    return res

def _execute_local(stl_path: Path) -> dict:
    from scripts.openfoam_runner import run_cfd
    print(f"[UnifiedRunner] Dispatching job to LOCAL backend for {stl_path.name}...")
    # run_cfd returns the results dict directly
    return run_cfd(stl_path)

def _execute_gcp(stl_path: Path, cores: int = 4) -> dict:
    print(f"[UnifiedRunner] Dispatching job to GCP backend ({cores} cores) for {stl_path.name}...")
    
    if not STARTUP_SCRIPT.exists():
        raise FileNotFoundError(f"Startup script not found: {STARTUP_SCRIPT}")

    timestamp = int(time.time())
    job_id = f"cfd_job_{timestamp}"
    vm_name = f"cfd-mpi-{cores}c-{timestamp}"
    
    # 1. Upload the STL to GCS
    stl_gcs_path = f"{BUCKET}/jobs/{job_id}/input.stl"
    print(f"[UnifiedRunner] Uploading STL to {stl_gcs_path}...")
    run_cmd(f"gcloud storage cp '{stl_path}' '{stl_gcs_path}'", check=True, capture=False)

    # 2. Provision VM
    staged_startup = Path("/tmp/run_mpi_benchmark_job.sh")
    shutil.copyfile(STARTUP_SCRIPT, staged_startup)
    staged_startup.chmod(0o755)

    metadata_flags = (
        f"job-id={job_id},"
        f"bucket-name={BUCKET},"
        f"stl-path={stl_gcs_path},"
        f"template-path={BUCKET}/templates/template_case_v1.tar.gz,"
        f"mpi-cores={cores}"
    )

    provisioned_zone = None
    for target_zone in ZONES:
        print(f"[UnifiedRunner] Attempting to provision VM in {target_zone}...")
        create_cmd = (
            f"gcloud compute instances create {vm_name} "
            f"--zone={target_zone} "
            f"--machine-type={MACHINE_TYPE} "
            f"--image-family=ubuntu-2404-lts-amd64 "
            f"--image-project=ubuntu-os-cloud "
            f"--boot-disk-size={BOOT_DISK_SIZE} "
            f"--boot-disk-type=pd-balanced "
            f"--scopes=cloud-platform "
            f"--metadata={metadata_flags} "
            f"--metadata-from-file=startup-script={staged_startup}"
        )
        res = run_cmd(create_cmd, check=False, capture=True)
        if res.returncode == 0:
            print(f"[UnifiedRunner] VM '{vm_name}' provisioned successfully in {target_zone}.")
            provisioned_zone = target_zone
            break
        else:
            print(f"[UnifiedRunner] Zone {target_zone} full/unavailable. Error: {res.stderr.strip()}")
            time.sleep(2)

    if not provisioned_zone:
        raise RuntimeError(f"Could not provision GCP VM in any candidate zone ({ZONES}).")

    # 3. Watchdog Polling
    print(f"[UnifiedRunner] Monitoring job {job_id} in GCS...")
    start_time = time.time()
    max_duration_sec = 60 * 60
    job_completed = False
    
    try:
        while True:
            elapsed_min = (time.time() - start_time) / 60.0
            if (time.time() - start_time) > max_duration_sec:
                print(f"\n[UnifiedRunner ERROR] Hard timeout of 60 minutes exceeded. Aborting!")
                break
                
            status_check = run_cmd(f"gcloud storage ls {BUCKET}/jobs/{job_id}/status.json", check=False)
            if status_check.returncode == 0:
                print(f"\n[UnifiedRunner] Detected completion signal 'status.json' at {elapsed_min:.1f} minutes!")
                job_completed = True
                break
                
            vm_check = run_cmd(
                f"gcloud compute instances describe {vm_name} --zone={provisioned_zone} --format='value(status)'", 
                check=False
            )
            vm_status = vm_check.stdout.strip() if vm_check.returncode == 0 else "UNKNOWN"
            
            if vm_status == "TERMINATED":
                print(f"\n[UnifiedRunner] VM entered TERMINATED state unexpectedly. Verifying results...")
                time.sleep(10)
                job_completed = True
                break
                
            print(f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f} min | VM: {vm_status} | Waiting for OpenFOAM MPI...", end="\r", flush=True)
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n[UnifiedRunner] Ctrl+C detected! Terminating cloud execution.")
    finally:
        print(f"\n[UnifiedRunner] Enforcing VM cleanup: {vm_name} in {provisioned_zone}...")
        try:
            run_cmd(f"gcloud compute instances delete {vm_name} --zone={provisioned_zone} --quiet", check=False)
            print("[UnifiedRunner] VM deleted safely.")
        except Exception as e:
            print(f"[UnifiedRunner ERROR] Failed to delete VM: {e}")
        if staged_startup.exists():
            staged_startup.unlink(missing_ok=True)

    if not job_completed:
        raise RuntimeError("GCP CFD job did not complete successfully.")

    # 4. Download Results
    print("[UnifiedRunner] Downloading results...")
    out_results_file = Path(f"/tmp/{job_id}_cfd_results.json")
    dl_res = run_cmd(f"gcloud storage cp '{BUCKET}/jobs/{job_id}/cfd_results.json' '{out_results_file}'", check=False)
    
    if dl_res.returncode != 0 or not out_results_file.exists():
        raise RuntimeError("Failed to retrieve cfd_results.json from GCS. Job may have failed internally.")

    with open(out_results_file, 'r') as f:
        res_data = json.load(f)
        
    return res_data

def execute_cfd(stl_path: Path, backend: str = "gcp", cores: int = 4) -> dict:
    """
    Unified entrypoint for running CFD. 
    Accepts backend='local' or backend='gcp'.
    """
    stl_path = Path(stl_path).resolve()
    if not stl_path.exists():
        raise FileNotFoundError(f"Input STL not found: {stl_path}")
        
    if backend.lower() == "local":
        return _execute_local(stl_path)
    elif backend.lower() == "gcp":
        return _execute_gcp(stl_path, cores=cores)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported: 'local', 'gcp'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified AeroMorphs CFD Runner")
    parser.add_argument("stl", type=str, help="Path to input STL file")
    parser.add_argument("--backend", type=str, choices=["local", "gcp"], default="gcp", help="Execution backend (default: gcp)")
    parser.add_argument("--cores", type=int, default=4, help="Number of MPI cores for GCP backend (default: 4)")
    args = parser.parse_args()
    
    print(f"Starting Unified CFD Runner (backend={args.backend})...")
    res = execute_cfd(Path(args.stl), backend=args.backend, cores=args.cores)
    
    print("\n=== FINAL CFD RESULTS ===")
    print(json.dumps(res, indent=2))
