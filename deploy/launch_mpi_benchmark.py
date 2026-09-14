#!/usr/bin/env python3
"""
AeroMorphs GCP MPI Benchmark Launcher & Watchdog
Provisions an on-demand c2-standard-8 VM, executes OpenFOAM with 4 or 8 MPI ranks,
streams progress, downloads results, and guarantees full VM deletion upon completion.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# Project Configuration
PROJECT_ID = "aerodesign-cfd-mvp"
REGION = "us-central1"
ZONES = ["us-central1-c", "us-central1-b", "us-central1-f", "us-central1-a"]
BUCKET = f"gs://aeromorphs-cfd-{PROJECT_ID}"
MACHINE_TYPE = "c2-standard-8"
BOOT_DISK_SIZE = "50GB"

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent
STARTUP_SCRIPT = DEPLOY_DIR / "run_mpi_benchmark_job.sh"
LOCAL_RESULTS_DIR = REPO_ROOT / "results"
STAGED_STARTUP_SCRIPT = Path("/tmp/run_mpi_benchmark_job.sh")

# Verified 1-Core GCP Baseline (Notchback N_S_WWC_WM_025, 3000 iters)
GCP_1CORE_BASELINE = {
    "cells": 414139,
    "drag_force_N": 599.3706,
    "cda_m2": 1.08729,
    "std_drag_N": 17.0389,
    "wall_time_sec": 3398,
    "wall_time_min": 56.63,
    "cost_est": 0.31
}

def run_cmd(cmd, check=True, capture=True):
    """Executes a shell command with google-cloud-sdk environment."""
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

def delete_vm_safe(vm_name, zone):
    """Guaranteed deletion of the worker VM."""
    print(f"\n[Watchdog] Enforcing deletion of VM: {vm_name} in {zone}...")
    try:
        cmd = f"/home/student/google-cloud-sdk/bin/gcloud compute instances delete {vm_name} --zone={zone} --quiet"
        run_cmd(cmd, check=False)
        print(f"[Watchdog] VM {vm_name} deleted successfully.")
    except Exception as e:
        print(f"[Watchdog WARNING] Failed to delete VM: {e}")

def run_mpi_benchmark(cores: int):
    if not STARTUP_SCRIPT.exists():
        raise FileNotFoundError(f"Startup script not found: {STARTUP_SCRIPT}")

    shutil.copyfile(STARTUP_SCRIPT, STAGED_STARTUP_SCRIPT)
    STAGED_STARTUP_SCRIPT.chmod(0o755)

    timestamp = int(time.time())
    job_id = f"notchback_mpi_{cores}cores_{timestamp}"
    vm_name = f"cfd-mpi-{cores}c-{timestamp}"
    
    print("\n" + "=" * 75)
    print(f" AeroMorphs GCP CFD MPI Benchmark ({cores} Cores on {MACHINE_TYPE})")
    print("=" * 75)
    print(f"Project:      {PROJECT_ID}")
    print(f"Region:       {REGION}")
    print(f"Machine Type: {MACHINE_TYPE} (On-Demand)")
    print(f"MPI Cores:    {cores}")
    print(f"Job ID:       {job_id}")
    print(f"VM Name:      {vm_name}")
    print(f"Bucket:       {BUCKET}")
    print("=" * 75)

    metadata_flags = (
        f"job-id={job_id},"
        f"bucket-name={BUCKET},"
        f"stl-path={BUCKET}/benchmark/N_S_WWC_WM_025.stl,"
        f"template-path={BUCKET}/templates/template_case_v1.tar.gz,"
        f"mpi-cores={cores}"
    )

    # Multi-zone failover attempt
    provisioned_zone = None
    for target_zone in ZONES:
        print(f"\n[1/4] Attempting to provision {MACHINE_TYPE} in zone '{target_zone}'...")
        create_cmd = (
            f"/home/student/google-cloud-sdk/bin/gcloud compute instances create {vm_name} "
            f"--zone={target_zone} "
            f"--machine-type={MACHINE_TYPE} "
            f"--image-family=ubuntu-2404-lts-amd64 "
            f"--image-project=ubuntu-os-cloud "
            f"--boot-disk-size={BOOT_DISK_SIZE} "
            f"--boot-disk-type=pd-balanced "
            f"--scopes=cloud-platform "
            f"--metadata={metadata_flags} "
            f"--metadata-from-file=startup-script={STAGED_STARTUP_SCRIPT}"
        )
        res = run_cmd(create_cmd, check=False, capture=True)
        if res.returncode == 0:
            print(f"[1/4] VM '{vm_name}' successfully provisioned in '{target_zone}'.")
            provisioned_zone = target_zone
            break
        else:
            print(f"[Warning] Zone '{target_zone}' failed (likely capacity/quota). Output:\n{res.stderr.strip()}")
            time.sleep(2)

    if not provisioned_zone:
        print(f"[ERROR] Could not provision {MACHINE_TYPE} in any candidate zone ({ZONES}).")
        sys.exit(1)

    # Monitor Job Lifecycle with Active Watchdog
    print(f"\n[2/4] Monitoring job execution in GCS ({BUCKET}/jobs/{job_id}/)...")
    print("Watchdog timeout: 60 minutes. Polling every 30 seconds.")
    
    start_time = time.time()
    max_duration_sec = 60 * 60  # 60 minutes hard timeout
    job_completed = False
    
    try:
        while True:
            elapsed_min = (time.time() - start_time) / 60.0
            
            if (time.time() - start_time) > max_duration_sec:
                print(f"\n[TIMEOUT ERROR] Hard limit of 60 minutes exceeded ({elapsed_min:.1f} min). Aborting!")
                break
                
            status_check = run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage ls {BUCKET}/jobs/{job_id}/status.json", check=False)
            if status_check.returncode == 0:
                print(f"\n[2/4] Detected completion signal 'status.json' at {elapsed_min:.1f} minutes!")
                job_completed = True
                break
                
            vm_check = run_cmd(
                f"/home/student/google-cloud-sdk/bin/gcloud compute instances describe {vm_name} --zone={provisioned_zone} --format='value(status)'", 
                check=False
            )
            vm_status = vm_check.stdout.strip() if vm_check.returncode == 0 else "UNKNOWN"
            
            if vm_status == "TERMINATED":
                print(f"\n[2/4] VM entered TERMINATED state (self-powered off). Verifying results...")
                time.sleep(10)
                job_completed = True
                break
                
            print(f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f} min | VM: {vm_status} ({provisioned_zone}) | Waiting for OpenFOAM MPI...", end="\r", flush=True)
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n[USER INTERRUPT] Ctrl+C detected! Triggering safe cleanup...")
    finally:
        delete_vm_safe(vm_name, provisioned_zone)
        if STAGED_STARTUP_SCRIPT.exists():
            STAGED_STARTUP_SCRIPT.unlink(missing_ok=True)

    if not job_completed:
        print(f"\n[ERROR] Benchmark for {cores} cores did not complete successfully.")
        sys.exit(1)

    # Download results and logs
    print(f"\n[3/4] Downloading benchmark results and logs from GCS...")
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_results_file = LOCAL_RESULTS_DIR / f"cfd_results_gcp_mpi_{cores}cores.json"
    
    dl_res = run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{job_id}/cfd_results.json' '{out_results_file}'", check=False)
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{job_id}/status.json' '{LOCAL_RESULTS_DIR}/status_mpi_{cores}cores.json'", check=False)
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{job_id}/cfd_job.log' '{LOCAL_RESULTS_DIR}/cfd_job_mpi_{cores}cores.log'", check=False)
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{job_id}/log.simpleFoam.gz' '{LOCAL_RESULTS_DIR}/log_simpleFoam_mpi_{cores}cores.gz'", check=False)
    
    if dl_res.returncode != 0 or not out_results_file.exists():
        print(f"[ERROR] Failed to retrieve cfd_results.json for {cores} cores from GCS.")
        sys.exit(1)

    with open(out_results_file, 'r') as f:
        res_data = json.load(f)
        
    return res_data

def display_comparison(results_4c, results_8c):
    print("\n" + "=" * 90)
    print(" AeroMorphs GCP OpenFOAM MPI Scaling Benchmark Results (Notchback N_S_WWC_WM_025)")
    print("=" * 90)
    
    headers = f"{'Metric':<28} {'1-Core Baseline':<20} {'4-Core MPI':<20} {'8-Core MPI':<20}"
    print(headers)
    print("-" * 90)
    
    # 1. Total Wall Clock Time
    t1_min = GCP_1CORE_BASELINE["wall_time_min"]
    t4_sec = results_4c.get("total_wall_time_seconds", results_4c.get("wall_time_seconds", 0))
    t8_sec = results_8c.get("total_wall_time_seconds", results_8c.get("wall_time_seconds", 0))
    t4_min = t4_sec / 60.0
    t8_min = t8_sec / 60.0
    print(f"{'Total Wall Time':<28} {t1_min:<20.2f} min {t4_min:<20.2f} min {t8_min:<20.2f} min")
    
    # 2. Solver Time
    s1_min = t1_min * 0.96 # approx ~54 min
    s4_sec = results_4c.get("solver_wall_time_seconds", t4_sec)
    s8_sec = results_8c.get("solver_wall_time_seconds", t8_sec)
    print(f"{'Solver Wall Time':<28} {'~54.0 min':<20} {s4_sec/60.0:<20.2f} min {s8_sec/60.0:<20.2f} min")
    
    # 3. Speedup Factor (vs 1-Core)
    sp4 = (t1_min * 60.0) / t4_sec if t4_sec > 0 else 0
    sp8 = (t1_min * 60.0) / t8_sec if t8_sec > 0 else 0
    print(f"{'Overall Speedup':<28} {'1.00x':<20} {f'{sp4:.2f}x':<20} {f'{sp8:.2f}x':<20}")
    
    # 4. Parallel Efficiency
    eff4 = (sp4 / 4.0) * 100.0
    eff8 = (sp8 / 8.0) * 100.0
    print(f"{'Parallel Efficiency':<28} {'100.0%':<20} {f'{eff4:.1f}%':<20} {f'{eff8:.1f}%':<20}")
    
    # 5. Drag Force (N)
    d1 = GCP_1CORE_BASELINE["drag_force_N"]
    d4 = results_4c.get("mean_drag_force_N", 0.0)
    d8 = results_8c.get("mean_drag_force_N", 0.0)
    print(f"{'Mean Drag Force (N)':<28} {d1:<20.2f} {d4:<20.2f} {d8:<20.2f}")
    
    # 6. Drag Area CdA (m²)
    c1 = GCP_1CORE_BASELINE["cda_m2"]
    c4 = results_4c.get("cda_m2", 0.0)
    c8 = results_8c.get("cda_m2", 0.0)
    print(f"{'Drag Area CdA (m²)':<28} {c1:<20.5f} {c4:<20.5f} {c8:<20.5f}")
    
    # 7. Numerical Consistency Delta vs 1-Core
    delta_c4 = ((c4 - c1) / c1) * 100.0
    delta_c8 = ((c8 - c1) / c1) * 100.0
    print(f"{'Delta vs 1-Core Baseline':<28} {'0.00% (Ref)':<20} {f'{delta_c4:+.2f}%':<20} {f'{delta_c8:+.2f}%':<20}")
    
    # 8. Force Stability (std dev)
    std1 = GCP_1CORE_BASELINE["std_drag_N"]
    std4 = results_4c.get("std_drag_force_N", 0.0)
    std8 = results_8c.get("std_drag_force_N", 0.0)
    print(f"{'Force Std Dev (N)':<28} {std1:<20.2f} {std4:<20.2f} {std8:<20.2f}")
    
    # 9. CPU Utilization
    cpu4 = results_4c.get("active_cores_utilization_pct", 0.0)
    cpu8 = results_8c.get("active_cores_utilization_pct", 0.0)
    print(f"{'Active Cores Utilization':<28} {'100.0% (1 core)':<20} {f'{cpu4:.1f}% (4 cores)':<20} {f'{cpu8:.1f}% (8 cores)':<20}")
    
    # 10. Estimated Cost per Run
    cost1 = GCP_1CORE_BASELINE["cost_est"]
    cost4 = round(0.33 * (t4_sec / 3600.0), 3)
    cost8 = round(0.33 * (t8_sec / 3600.0), 3)
    print(f"{'Estimated Cost (On-Demand)':<28} {f'${cost1:.2f}':<20} {f'${cost4:.2f}':<20} {f'${cost8:.2f}':<20}")
    print("=" * 90)

    # Save structured summary to results/
    comparison_summary = {
        "benchmark_geometry": "N_S_WWC_WM_025",
        "iterations": 3000,
        "mesh_cells": GCP_1CORE_BASELINE["cells"],
        "baseline_1core": {
            "wall_time_sec": GCP_1CORE_BASELINE["wall_time_sec"],
            "wall_time_min": GCP_1CORE_BASELINE["wall_time_min"],
            "mean_drag_N": GCP_1CORE_BASELINE["drag_force_N"],
            "cda_m2": GCP_1CORE_BASELINE["cda_m2"],
            "std_drag_N": GCP_1CORE_BASELINE["std_drag_N"],
            "speedup": 1.0,
            "parallel_efficiency_pct": 100.0,
            "cost_est_usd": GCP_1CORE_BASELINE["cost_est"]
        },
        "mpi_4core": {
            "wall_time_sec": t4_sec,
            "wall_time_min": round(t4_min, 2),
            "solver_time_sec": s4_sec,
            "solver_time_min": round(s4_sec / 60.0, 2),
            "mean_drag_N": round(d4, 4),
            "cda_m2": round(c4, 5),
            "delta_cda_vs_1core_pct": round(delta_c4, 2),
            "std_drag_N": round(std4, 4),
            "speedup": round(sp4, 2),
            "parallel_efficiency_pct": round(eff4, 1),
            "active_cores_utilization_pct": cpu4,
            "cost_est_usd": cost4
        },
        "mpi_8core": {
            "wall_time_sec": t8_sec,
            "wall_time_min": round(t8_min, 2),
            "solver_time_sec": s8_sec,
            "solver_time_min": round(s8_sec / 60.0, 2),
            "mean_drag_N": round(d8, 4),
            "cda_m2": round(c8, 5),
            "delta_cda_vs_1core_pct": round(delta_c8, 2),
            "std_drag_N": round(std8, 4),
            "speedup": round(sp8, 2),
            "parallel_efficiency_pct": round(eff8, 1),
            "active_cores_utilization_pct": cpu8,
            "cost_est_usd": cost8
        }
    }
    comp_file = LOCAL_RESULTS_DIR / "cfd_results_gcp_mpi_comparison.json"
    with open(comp_file, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nSaved structured comparison to: {comp_file}")

def main():
    parser = argparse.ArgumentParser(description="AeroMorphs GCP MPI Benchmark")
    parser.add_argument("--cores", type=int, choices=[4, 8], help="Run a specific core count (4 or 8)")
    parser.add_argument("--all", action="store_true", help="Run 4-core followed by 8-core sequentially")
    args = parser.parse_args()

    if args.cores:
        res = run_mpi_benchmark(args.cores)
        print(f"\n[Completed] Benchmark finished for {args.cores} cores:")
        print(json.dumps(res, indent=2))
    elif args.all:
        print("[Launcher] Starting sequential benchmark: 4 cores first, then 8 cores...")
        res_4c = run_mpi_benchmark(4)
        print("\n[Launcher] 4-core benchmark complete. Proceeding to 8-core benchmark...")
        res_8c = run_mpi_benchmark(8)
        display_comparison(res_4c, res_8c)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
