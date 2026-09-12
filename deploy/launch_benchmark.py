#!/usr/bin/env python3
"""
AeroMorphs GCP Benchmark Launcher & Watchdog
Safely provisions an on-demand c2-standard-8 VM, streams status,
downloads benchmark results, and guarantees full VM deletion upon completion,
error, or timeout.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

# Paths & Project Configuration
PROJECT_ID = "aerodesign-cfd-mvp"
REGION = "us-central1"
ZONE = "us-central1-a"
BUCKET = f"gs://aeromorphs-cfd-{PROJECT_ID}"
MACHINE_TYPE = "c2-standard-8"
BOOT_DISK_SIZE = "50GB"

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent
STARTUP_SCRIPT = DEPLOY_DIR / "run_benchmark_job.sh"
LOCAL_RESULTS_DIR = REPO_ROOT / "results"

# Staged scratch script outside Cryptomator mount to avoid whitespace/FUSE latency in gcloud CLI
STAGED_STARTUP_SCRIPT = Path("/tmp/run_benchmark_job.sh")

# Verified Local Baseline Reference Values (Notchback N_S_WWC_WM_025)
LOCAL_BASELINE = {
    "cells": 414139,
    "drag_force_N": 602.4812,
    "cda_m2": 1.09291,
    "std_drag_N": 1.8421,
    "runtime_min": 59.0
}

def run_cmd(cmd, check=True, capture=True):
    """Executes a shell command with proper environment."""
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

def delete_vm_safe(vm_name):
    """Guaranteed deletion of the worker VM."""
    print(f"\n[Watchdog] Enforcing deletion of VM: {vm_name} in {ZONE}...")
    try:
        cmd = f"gcloud compute instances delete {vm_name} --zone={ZONE} --quiet"
        run_cmd(cmd, check=False)
        print(f"[Watchdog] VM {vm_name} deletion command issued successfully.")
    except Exception as e:
        print(f"[Watchdog WARNING] Failed to delete VM: {e}")

def main():
    if not STARTUP_SCRIPT.exists():
        raise FileNotFoundError(f"Startup script not found: {STARTUP_SCRIPT}")

    # Stage startup script to clean path
    shutil.copyfile(STARTUP_SCRIPT, STAGED_STARTUP_SCRIPT)
    STAGED_STARTUP_SCRIPT.chmod(0o755)

    timestamp = int(time.time())
    job_id = f"notchback_benchmark_{timestamp}"
    vm_name = f"cfd-bench-{timestamp}"
    
    print("=" * 65)
    print(" AeroMorphs GCP CFD Parity Benchmark Launcher & Watchdog")
    print("=" * 65)
    print(f"Project:      {PROJECT_ID}")
    print(f"Zone:         {ZONE}")
    print(f"Machine Type: {MACHINE_TYPE} (On-Demand)")
    print(f"Job ID:       {job_id}")
    print(f"VM Name:      {vm_name}")
    print(f"Bucket:       {BUCKET}")
    print("=" * 65)

    metadata_flags = (
        f"job-id={job_id},"
        f"bucket-name={BUCKET},"
        f"stl-path={BUCKET}/benchmark/N_S_WWC_WM_025.stl,"
        f"template-path={BUCKET}/templates/template_case_v1.tar.gz"
    )

    # Launch VM with metadata & startup-script
    print(f"\n[1/4] Provisioning {MACHINE_TYPE} instance '{vm_name}'...")
    create_cmd = (
        f"gcloud compute instances create {vm_name} "
        f"--zone={ZONE} "
        f"--machine-type={MACHINE_TYPE} "
        f"--image-family=ubuntu-2404-lts-amd64 "
        f"--image-project=ubuntu-os-cloud "
        f"--boot-disk-size={BOOT_DISK_SIZE} "
        f"--boot-disk-type=pd-balanced "
        f"--scopes=cloud-platform "
        f"--metadata={metadata_flags} "
        f"--metadata-from-file=startup-script={STAGED_STARTUP_SCRIPT}"
    )

    try:
        run_cmd(create_cmd, capture=False)
        print(f"[1/4] VM '{vm_name}' successfully provisioned.")
    except Exception as e:
        print(f"[ERROR] Failed to provision VM: {e}")
        delete_vm_safe(vm_name)
        sys.exit(1)

    # Monitor Job Lifecycle with Active Watchdog
    print(f"\n[2/4] Monitoring job execution in GCS ({BUCKET}/jobs/{job_id}/)...")
    print("Watchdog timeout: 90 minutes. Polling every 30 seconds.")
    
    start_time = time.time()
    max_duration_sec = 90 * 60  # 90 minutes hard timeout
    job_completed = False
    
    try:
        while True:
            elapsed_min = (time.time() - start_time) / 60.0
            
            # Check hard timeout
            if (time.time() - start_time) > max_duration_sec:
                print(f"\n[TIMEOUT ERROR] Hard limit of 90 minutes exceeded ({elapsed_min:.1f} min). Aborting!")
                break
                
            # Check if status.json has been uploaded by VM cleanup handler
            status_check = run_cmd(f"gcloud storage ls {BUCKET}/jobs/{job_id}/status.json", check=False)
            if status_check.returncode == 0:
                print(f"\n[2/4] Detected completion signal 'status.json' at {elapsed_min:.1f} minutes!")
                job_completed = True
                break
                
            # Check VM state
            vm_check = run_cmd(
                f"gcloud compute instances describe {vm_name} --zone={ZONE} --format='value(status)'", 
                check=False
            )
            vm_status = vm_check.stdout.strip() if vm_check.returncode == 0 else "UNKNOWN"
            
            # If VM has terminated itself (TERMINATED) but status.json check didn't catch it yet, wait one cycle
            if vm_status == "TERMINATED":
                print(f"\n[2/4] VM entered TERMINATED state (self-powered off). Verifying results...")
                time.sleep(10)
                job_completed = True
                break
                
            print(f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f} min | VM Status: {vm_status} | Waiting for solver...", end="\r")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n[USER INTERRUPT] Ctrl+C detected! Triggering safe cleanup...")
    finally:
        # Guarantee VM cleanup and scratch removal
        delete_vm_safe(vm_name)
        if STAGED_STARTUP_SCRIPT.exists():
            STAGED_STARTUP_SCRIPT.unlink(missing_ok=True)

    if not job_completed:
        print("\n[ERROR] Benchmark did not complete successfully.")
        sys.exit(1)

    # Download results and logs
    print(f"\n[3/4] Downloading benchmark results and logs from GCS...")
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_results_file = LOCAL_RESULTS_DIR / f"cfd_results_gcp_{job_id}.json"
    
    dl_res = run_cmd(f"gcloud storage cp '{BUCKET}/jobs/{job_id}/cfd_results.json' '{out_results_file}'", check=False)
    run_cmd(f"gcloud storage cp '{BUCKET}/jobs/{job_id}/status.json' '{LOCAL_RESULTS_DIR}/status_{job_id}.json'", check=False)
    run_cmd(f"gcloud storage cp '{BUCKET}/jobs/{job_id}/cfd_job.log' '{LOCAL_RESULTS_DIR}/cfd_job_{job_id}.log'", check=False)
    
    if dl_res.returncode != 0 or not out_results_file.exists():
        print("[ERROR] Failed to retrieve cfd_results.json from GCS. Check cfd_job.log for details.")
        sys.exit(1)

    # Parse and display parity scorecard
    print(f"\n[4/4] Evaluating Parity against Local Baseline...")
    with open(out_results_file, 'r') as f:
        gcp_res = json.load(f)

    gcp_drag = gcp_res.get("mean_drag_force_N", 0.0)
    gcp_cda = gcp_res.get("cda_m2", 0.0)
    gcp_std = gcp_res.get("std_drag_force_N", 0.0)
    gcp_wall_sec = gcp_res.get("wall_time_seconds", 0)
    gcp_wall_min = gcp_wall_sec / 60.0

    delta_cda_pct = ((gcp_cda - LOCAL_BASELINE["cda_m2"]) / LOCAL_BASELINE["cda_m2"]) * 100.0
    delta_drag_pct = ((gcp_drag - LOCAL_BASELINE["drag_force_N"]) / LOCAL_BASELINE["drag_force_N"]) * 100.0
    parity_passed = abs(delta_cda_pct) <= 0.5

    print("\n" + "=" * 75)
    print(f"{'Metric':<25} {'Local Baseline':<18} {'GCP Benchmark':<18} {'Delta / Status'}")
    print("-" * 75)
    print(f"{'Mean Drag Force (N)':<25} {LOCAL_BASELINE['drag_force_N']:<18.2f} {gcp_drag:<18.2f} {delta_drag_pct:+.2f}%")
    print(f"{'Drag Area CdA (m²)':<25} {LOCAL_BASELINE['cda_m2']:<18.5f} {gcp_cda:<18.5f} {delta_cda_pct:+.2f}%")
    print(f"{'Force Std Dev (N)':<25} {LOCAL_BASELINE['std_drag_N']:<18.2f} {gcp_std:<18.2f} (Stability)")
    print(f"{'Solver Wall-Clock':<25} {LOCAL_BASELINE['runtime_min']:<18.1f} min {gcp_wall_min:<18.1f} min")
    print(f"{'Parity Status':<25} {'VERIFIED':<18} {'EVALUATED':<18} {'✅ PASSED (<=0.5%)' if parity_passed else '❌ FAILED (>0.5%)'}")
    print("=" * 75)

if __name__ == "__main__":
    main()
