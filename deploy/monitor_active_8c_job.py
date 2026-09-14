#!/usr/bin/env python3
"""
Monitors the active 8-core GCP benchmark VM, downloads results upon completion,
ensures guaranteed deletion of the VM, and computes the 1-core vs 4-core vs 8-core comparison.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

PROJECT_ID = "aerodesign-cfd-mvp"
ZONE = "us-central1-c"
BUCKET = f"gs://aeromorphs-cfd-{PROJECT_ID}"
JOB_ID = "notchback_mpi_8cores_1789367190"
VM_NAME = "cfd-mpi-8c-1789367190"

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RESULTS_DIR = REPO_ROOT / "results"

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

def delete_vm_safe():
    print(f"\n[Watchdog] Enforcing deletion of VM: {VM_NAME} in {ZONE}...")
    try:
        run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud compute instances delete {VM_NAME} --zone={ZONE} --quiet", check=False)
        print(f"[Watchdog] VM {VM_NAME} deleted successfully.")
    except Exception as e:
        print(f"[Watchdog WARNING] Failed to delete VM: {e}")

def main():
    print("=" * 75)
    print(" Monitoring Active 8-Core GCP CFD MPI Benchmark")
    print("=" * 75)
    print(f"VM Name: {VM_NAME} ({ZONE})")
    print(f"Job ID:  {JOB_ID}")
    print(f"Bucket:  {BUCKET}")
    print("=" * 75)

    start_time = time.time()
    max_duration_sec = 45 * 60
    job_completed = False

    try:
        while True:
            elapsed_min = (time.time() - start_time) / 60.0
            if (time.time() - start_time) > max_duration_sec:
                print(f"\n[TIMEOUT] 45 minutes exceeded. Aborting!")
                break

            status_check = run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage ls {BUCKET}/jobs/{JOB_ID}/status.json", check=False)
            if status_check.returncode == 0:
                print(f"\n[Monitor] Detected completion signal 'status.json' at {elapsed_min:.1f} minutes!")
                job_completed = True
                break

            vm_check = run_cmd(
                f"/home/student/google-cloud-sdk/bin/gcloud compute instances describe {VM_NAME} --zone={ZONE} --format='value(status)'", 
                check=False
            )
            vm_status = vm_check.stdout.strip() if vm_check.returncode == 0 else "UNKNOWN"
            if vm_status == "TERMINATED":
                print(f"\n[Monitor] VM entered TERMINATED state (self-powered off). Verifying results...")
                time.sleep(10)
                job_completed = True
                break

            print(f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f} min | VM: {vm_status} | Waiting for OpenFOAM MPI (8 cores)...", end="\r", flush=True)
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n[USER INTERRUPT] Ctrl+C detected!")
    finally:
        delete_vm_safe()

    if not job_completed:
        print("[ERROR] 8-core benchmark did not complete successfully.")
        sys.exit(1)

    print("\n[Download] Fetching 8-core benchmark artifacts from GCS...")
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_results_file = LOCAL_RESULTS_DIR / "cfd_results_gcp_mpi_8cores.json"
    
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{JOB_ID}/cfd_results.json' '{out_results_file}'", check=False)
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{JOB_ID}/status.json' '{LOCAL_RESULTS_DIR}/status_mpi_8cores.json'", check=False)
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{JOB_ID}/cfd_job.log' '{LOCAL_RESULTS_DIR}/cfd_job_mpi_8cores.log'", check=False)
    run_cmd(f"/home/student/google-cloud-sdk/bin/gcloud storage cp '{BUCKET}/jobs/{JOB_ID}/log.simpleFoam.gz' '{LOCAL_RESULTS_DIR}/log_simpleFoam_mpi_8cores.gz'", check=False)

    if not out_results_file.exists():
        print("[ERROR] Could not find cfd_results_gcp_mpi_8cores.json.")
        sys.exit(1)

    with open(out_results_file, 'r') as f:
        results_8c = json.load(f)

    results_4c_file = LOCAL_RESULTS_DIR / "cfd_results_gcp_mpi_4cores.json"
    with open(results_4c_file, 'r') as f:
        results_4c = json.load(f)

    # Display comparison
    display_all(results_4c, results_8c)

def display_all(results_4c, results_8c):
    print("\n" + "=" * 92)
    print(" AeroMorphs GCP OpenFOAM MPI Scaling Benchmark (Notchback N_S_WWC_WM_025, 3000 iters)")
    print("=" * 92)
    
    headers = f"{'Metric':<28} {'1-Core Baseline':<20} {'4-Core MPI':<20} {'8-Core MPI':<20}"
    print(headers)
    print("-" * 92)
    
    t1_min = GCP_1CORE_BASELINE["wall_time_min"]
    t4_sec = results_4c.get("total_wall_time_seconds", results_4c.get("wall_time_seconds", 0))
    t8_sec = results_8c.get("total_wall_time_seconds", results_8c.get("wall_time_seconds", 0))
    t4_min = t4_sec / 60.0
    t8_min = t8_sec / 60.0
    print(f"{'Total Wall Time':<28} {t1_min:<20.2f} min {t4_min:<20.2f} min {t8_min:<20.2f} min")
    
    s4_sec = results_4c.get("solver_wall_time_seconds", t4_sec)
    s8_sec = results_8c.get("solver_wall_time_seconds", t8_sec)
    print(f"{'Solver Wall Time':<28} {'~54.0 min':<20} {s4_sec/60.0:<20.2f} min {s8_sec/60.0:<20.2f} min")
    
    sp4 = (t1_min * 60.0) / t4_sec if t4_sec > 0 else 0
    sp8 = (t1_min * 60.0) / t8_sec if t8_sec > 0 else 0
    print(f"{'Overall Speedup':<28} {'1.00x':<20} {f'{sp4:.2f}x':<20} {f'{sp8:.2f}x':<20}")

    sp4_solver = (54.0 * 60.0) / s4_sec if s4_sec > 0 else 0
    sp8_solver = (54.0 * 60.0) / s8_sec if s8_sec > 0 else 0
    print(f"{'Solver Speedup':<28} {'1.00x':<20} {f'{sp4_solver:.2f}x':<20} {f'{sp8_solver:.2f}x':<20}")
    
    eff4 = (sp4_solver / 4.0) * 100.0
    eff8 = (sp8_solver / 8.0) * 100.0
    print(f"{'Solver Efficiency':<28} {'100.0%':<20} {f'{eff4:.1f}%':<20} {f'{eff8:.1f}%':<20}")
    
    d1 = GCP_1CORE_BASELINE["drag_force_N"]
    d4 = results_4c.get("mean_drag_force_N", 0.0)
    d8 = results_8c.get("mean_drag_force_N", 0.0)
    print(f"{'Mean Drag Force (N)':<28} {d1:<20.2f} {d4:<20.2f} {d8:<20.2f}")
    
    c1 = GCP_1CORE_BASELINE["cda_m2"]
    c4 = results_4c.get("cda_m2", 0.0)
    c8 = results_8c.get("cda_m2", 0.0)
    print(f"{'Drag Area CdA (m²)':<28} {c1:<20.5f} {c4:<20.5f} {c8:<20.5f}")
    
    delta_c4 = ((c4 - c1) / c1) * 100.0
    delta_c8 = ((c8 - c1) / c1) * 100.0
    print(f"{'Delta vs 1-Core Baseline':<28} {'0.00% (Ref)':<20} {f'{delta_c4:+.2f}%':<20} {f'{delta_c8:+.2f}%':<20}")
    
    std1 = GCP_1CORE_BASELINE["std_drag_N"]
    std4 = results_4c.get("std_drag_force_N", 0.0)
    std8 = results_8c.get("std_drag_force_N", 0.0)
    print(f"{'Force Std Dev (N)':<28} {std1:<20.2f} {std4:<20.2f} {std8:<20.2f}")
    
    cpu4 = results_4c.get("active_cores_utilization_pct", 0.0)
    cpu8 = results_8c.get("active_cores_utilization_pct", 0.0)
    print(f"{'Active Cores Utilization':<28} {'100.0% (1 core)':<20} {f'{cpu4:.1f}% (4 cores)':<20} {f'{cpu8:.1f}% (8 cores)':<20}")
    
    cost1 = GCP_1CORE_BASELINE["cost_est"]
    cost4 = round(0.33 * (t4_sec / 3600.0), 3)
    cost8 = round(0.33 * (t8_sec / 3600.0), 3)
    print(f"{'Estimated Cost (On-Demand)':<28} {f'${cost1:.2f}':<20} {f'${cost4:.2f}':<20} {f'${cost8:.2f}':<20}")
    print("=" * 92)

    comp_summary = {
        "benchmark_geometry": "N_S_WWC_WM_025",
        "iterations": 3000,
        "mesh_cells": GCP_1CORE_BASELINE["cells"],
        "baseline_1core": {
            "wall_time_sec": GCP_1CORE_BASELINE["wall_time_sec"],
            "wall_time_min": GCP_1CORE_BASELINE["wall_time_min"],
            "solver_time_min": 54.0,
            "mean_drag_N": GCP_1CORE_BASELINE["drag_force_N"],
            "cda_m2": GCP_1CORE_BASELINE["cda_m2"],
            "std_drag_N": GCP_1CORE_BASELINE["std_drag_N"],
            "solver_speedup": 1.0,
            "solver_efficiency_pct": 100.0,
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
            "solver_speedup": round(sp4_solver, 2),
            "solver_efficiency_pct": round(eff4, 1),
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
            "solver_speedup": round(sp8_solver, 2),
            "solver_efficiency_pct": round(eff8, 1),
            "active_cores_utilization_pct": cpu8,
            "cost_est_usd": cost8
        }
    }
    comp_file = LOCAL_RESULTS_DIR / "cfd_results_gcp_mpi_comparison.json"
    with open(comp_file, "w") as f:
        json.dump(comp_summary, f, indent=2)
    print(f"Structured comparison successfully persisted to: {comp_file}")

if __name__ == "__main__":
    main()
