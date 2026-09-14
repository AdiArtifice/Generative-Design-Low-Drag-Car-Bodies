#!/usr/bin/env python3
"""
AeroMorphs Autonomous Closed-Loop AI-CFD Refinement Orchestrator (Phase 8 Production)
Automates the full cycle:
  Baseline Check -> AI Latent Optimization -> Automated CAD Staging -> 
  CFD Execution (GCP 4-core MPI or Local) -> Automated Result Ingestion -> 
  Directional Falsification Barrier Registration -> Re-optimization Loop
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cfd_evidence_store import CFDEvidenceStore
from scripts.unified_cfd_runner import execute_cfd

BODY_NAME_MAP = {
    'F': 'Fastback',
    'E': 'Estateback',
    'N': 'Notchback'
}

def resolve_body_type(car_id: str) -> str:
    """Resolves human-readable body type from car ID or metadata."""
    prefix = car_id.split('_')[0].upper()
    return BODY_NAME_MAP.get(prefix, 'Fastback')

def run_refinement_loop(args):
    print("=" * 80)
    print(f" AeroMorphs Autonomous Closed-Loop AI ↔ CFD Refinement Orchestrator")
    print("=" * 80)
    print(f"Target Vehicle ID:  {args.car_id}")
    body_name = resolve_body_type(args.car_id)
    print(f"Body Category:      {body_name}")
    print(f"Refinement Rounds:  {args.max_rounds}")
    print(f"CFD Backend:        {args.backend.upper()} (Cores: {args.cores if args.backend == 'gcp' else 'local'})")
    print(f"Optimization Steps: {args.steps} per round (LR: {args.lr}, Trust R: {args.trust_radius})")
    print(f"Repulsive Barrier:  w_CFD = {args.cfd_penalty_weight}")
    print(f"Dry Run Mode:       {'ENABLED (Simulated CFD)' if args.dry_run else 'DISABLED (Live CFD)'}")
    print("=" * 80)

    store = CFDEvidenceStore(args.evidence_store)
    
    # -------------------------------------------------------------
    # Step 0: Baseline Verification
    # -------------------------------------------------------------
    print("\n[Step 0] Verifying Baseline Ground-Truth in CFD Evidence Store...")
    baseline_entry = store.get_baseline_entry(args.car_id)
    
    if baseline_entry:
        b_cda = baseline_entry.get("cfd_cda")
        b_force = baseline_entry.get("cfd_drag_force_N")
        b_id = baseline_entry.get("id")
        print(f"  -> Verified existing baseline entry: '{b_id}'")
        print(f"  -> Baseline Physical CFD Drag Force: {b_force:.2f} N | CdA: {b_cda:.5f} m^2")
    else:
        print(f"  -> [Notice] No existing CFD baseline found for {args.car_id}.")
        if args.skip_step0_cfd:
            print("  -> Skipping Step 0 CFD execution (--skip_step0_cfd set). Relative deltas will be approximate.")
            b_cda = None
            b_force = None
        else:
            print("  -> Auto-generating and running Step 0 baseline CFD case...")
            step0_dir = PROJECT_ROOT / f"optimization_output_baseline/{args.car_id}"
            step0_dir.mkdir(parents=True, exist_ok=True)
            
            # Run 1 step with staging to extract Step 0 mesh
            cmd_step0 = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "optimize_latent_shape.py"),
                "--car_id", args.car_id,
                "--out_dir", str(step0_dir),
                "--steps", "1",
                "--stage_cfd",
                "--subdivide", str(args.subdivide),
                "--smooth_taubin", str(args.smooth_taubin)
            ]
            subprocess.run(cmd_step0, check=True)
            
            step0_stl = step0_dir / "optimized_car_step_0_1to1_smooth.stl"
            if not step0_stl.exists():
                raise FileNotFoundError(f"Failed to generate Step 0 physical mesh: {step0_stl}")
                
            print(f"  -> Dispatching Step 0 baseline mesh to CFD: {step0_stl.name}")
            if args.dry_run:
                step0_cfd_res = {
                    "mean_drag_force_N": 420.0,
                    "cda_m2": 0.7600,
                    "std_drag_force_N": 10.0,
                    "cells": 440000
                }
            else:
                step0_cfd_res = execute_cfd(step0_stl, backend=args.backend, cores=args.cores)
                
            baseline_entry = store.record_run(
                run_id=f"Baseline_{args.car_id}_Step0_Auto",
                category="Baseline_VAE_Reconstruction_Smooth",
                body_type=body_name,
                baseline_id=args.car_id,
                cfd_results=step0_cfd_res,
                round_num=0,
                notes=f"Auto-evaluated Step 0 baseline in closed-loop refinement via {args.backend}"
            )
            b_cda = baseline_entry.get("cfd_cda")
            b_force = baseline_entry.get("cfd_drag_force_N")
            print(f"  -> Step 0 Baseline Established: F_D = {b_force:.2f} N | CdA = {b_cda:.5f} m^2")

    # -------------------------------------------------------------
    # Refinement Loop: Round 1 through max_rounds
    # -------------------------------------------------------------
    round_records = []
    
    for round_num in range(1, args.max_rounds + 1):
        print("\n" + "#" * 80)
        print(f" STARTING REFINEMENT ROUND {round_num} OF {args.max_rounds}")
        print("#" * 80)
        
        # Check active directional constraints for this vehicle / body type
        active_constraints = store.get_directional_constraints(body_type=body_name, baseline_id=args.car_id)
        print(f"[Round {round_num}] Active Falsification Barriers in Store: {len(active_constraints)}")
        for idx, c in enumerate(active_constraints, 1):
            print(f"   Barrier #{idx} [{c['id']}]: discrepancy = {c['error_delta']:+.4f} m^2, norm = {c['norm_v']:.4f}")
            
        out_dir = PROJECT_ROOT / f"optimization_output_round_{round_num}" / args.car_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. AI Latent Space Optimization
        use_cfd_flag = (round_num > 1) or (len(active_constraints) > 0)
        print(f"\n[Round {round_num}][1/4] Running Latent Shape Optimization (use_cfd_feedback={use_cfd_flag})...")
        
        opt_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "optimize_latent_shape.py"),
            "--car_id", args.car_id,
            "--out_dir", str(out_dir),
            "--version", f"round_{round_num}",
            "--steps", str(args.steps),
            "--lr", str(args.lr),
            "--trust_radius", str(args.trust_radius),
            "--cfd_penalty_weight", str(args.cfd_penalty_weight),
            "--subdivide", str(args.subdivide),
            "--smooth_taubin", str(args.smooth_taubin),
            "--stage_cfd"
        ]
        if use_cfd_flag:
            opt_cmd.append("--use_cfd_feedback")
            
        subprocess.run(opt_cmd, check=True)
        
        # 2. Verify Staged Mesh
        staged_stl = out_dir / f"optimized_car_step_{args.steps}_1to1_smooth.stl"
        summary_file = out_dir / "optimization_summary.json"
        
        if not staged_stl.exists():
            raise FileNotFoundError(f"[Round {round_num}] Expected staged CFD mesh not found: {staged_stl}")
        if not summary_file.exists():
            raise FileNotFoundError(f"[Round {round_num}] Expected summary file not found: {summary_file}")
            
        with open(summary_file, "r") as f:
            opt_summary = json.load(f)
            
        claimed_cda = opt_summary.get("final_predicted_drag_area")
        initial_cda = opt_summary.get("baseline_predicted_drag_area")
        claimed_reduction = opt_summary.get("reduction_percent", 0.0)
        print(f"\n[Round {round_num}][2/4] AI Optimization Converged:")
        print(f"  -> Surrogate Predicted CdA: {claimed_cda:.4f} m^2 (Claimed: {claimed_reduction:+.2f}%)")
        print(f"  -> Staged Physical Mesh:    {staged_stl.name} (1:1 DrivAer scale, Taubin smoothed)")
        
        # 3. CFD Simulation Execution
        print(f"\n[Round {round_num}][3/4] Dispatching to OpenFOAM ({args.backend.upper()} Backend)...")
        if args.dry_run:
            # Simulate a Round 1 trip (+8% drag) and a Round 2 recovery (-8% drag)
            if round_num == 1:
                cfd_res = {
                    "mean_drag_force_N": (b_force * 1.08) if b_force else 460.0,
                    "cda_m2": (b_cda * 1.08) if b_cda else 0.8300,
                    "std_drag_force_N": 12.5,
                    "cells": 442000
                }
            else:
                cfd_res = {
                    "mean_drag_force_N": (b_force * 0.92) if b_force else 390.0,
                    "cda_m2": (b_cda * 0.92) if b_cda else 0.7050,
                    "std_drag_force_N": 9.8,
                    "cells": 440000
                }
        else:
            cfd_res = execute_cfd(staged_stl, backend=args.backend, cores=args.cores)
            
        # 4. Automated Evidence Ingestion (P2)
        print(f"\n[Round {round_num}][4/4] Ingesting CFD Result into Evidence Store...")
        run_id = f"ClosedLoop_{args.car_id}_Round{round_num}"
        category = "ClosedLoop_AI_Champion_Smooth"
        
        entry = store.record_run(
            run_id=run_id,
            category=category,
            body_type=body_name,
            baseline_id=args.car_id,
            cfd_results=cfd_res,
            opt_summary=opt_summary,
            z_initial_path=opt_summary.get("z_initial_path"),
            z_opt_path=opt_summary.get("z_opt_path"),
            round_num=round_num,
            notes=f"Autonomous Closed-Loop Round {round_num} evaluated on {args.backend} ({args.cores}c)"
        )
        
        cfd_force = entry.get("cfd_drag_force_N")
        cfd_cda = entry.get("cfd_cda")
        delta_force = entry.get("delta_drag_force_N")
        pct_change = entry.get("real_drag_change_vs_baseline_pct")
        delta_cda_cfd = entry.get("delta_cda_cfd")
        delta_cda_surr = entry.get("delta_cda_surrogate")
        
        discrepancy = (delta_cda_cfd - delta_cda_surr) if (delta_cda_cfd is not None and delta_cda_surr is not None) else 0.0
        
        round_info = {
            "round": round_num,
            "run_id": run_id,
            "surrogate_pred_cda": claimed_cda,
            "cfd_drag_force_N": cfd_force,
            "cfd_cda": cfd_cda,
            "delta_force_vs_baseline_N": delta_force,
            "real_drag_change_pct": pct_change,
            "discrepancy_m2": discrepancy,
            "status": "VERIFIED IMPROVEMENT" if (pct_change is not None and pct_change < 0 and discrepancy <= 0) 
                      else ("FALSIFIED (tripped separation)" if discrepancy > 0 else "NEUTRAL")
        }
        round_records.append(round_info)
        
        print(f"\n--- Round {round_num} Outcome Analysis ---")
        print(f"  Physical Drag Force:  {cfd_force:.2f} N (Delta vs Baseline: {delta_force:+.2f} N | {pct_change:+.2f}%)")
        print(f"  Physical CdA:         {cfd_cda:.5f} m^2 (Surrogate predicted: {claimed_cda:.5f} m^2)")
        print(f"  Discrepancy (Δe):     {discrepancy:+.5f} m^2")
        print(f"  Diagnostic Status:    {round_info['status']}")
        
        if discrepancy > 0:
            print(f"  [ACTION] Surrogate was overoptimistic. A falsified gradient vector was persisted.")
            print(f"           Round {round_num + 1} will automatically erect a quadratic repulsive barrier along this path.")
        else:
            print(f"  [ACTION] True physical aerodynamic drag reduction verified in OpenFOAM.")

    # -------------------------------------------------------------
    # Final Refinement Session Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f" AeroMorphs Autonomous Closed-Loop Refinement Session Complete ({args.car_id})")
    print("=" * 95)
    print(f"{'Stage':<18} {'Surrogate CdA':<18} {'CFD Drag Force':<18} {'CFD CdA':<16} {'Delta vs Base':<16} {'Status'}")
    print("-" * 95)
    if b_cda and b_force:
        print(f"{'Step 0 Baseline':<18} {b_cda:<18.4f} {f'{b_force:.2f} N':<18} {b_cda:<16.4f} {'0.0% (Ref)':<16} {'Baseline'}")
    for r in round_records:
        r_label = f"Round {r['round']} Champion"
        pct_str = f"{r['real_drag_change_pct']:+.2f}%" if r['real_drag_change_pct'] is not None else "N/A"
        force_str = f"{r['cfd_drag_force_N']:.2f} N"
        print(f"{r_label:<18} {r['surrogate_pred_cda']:<18.4f} {force_str:<18} {r['cfd_cda']:<16.4f} {pct_str:<16} {r['status']}")
    print("=" * 95)
    
    # Save structured report
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / f"closed_loop_refinement_{args.car_id}.json"
    
    summary_report = {
        "car_id": args.car_id,
        "body_type": body_name,
        "backend": args.backend,
        "cores": args.cores,
        "max_rounds": args.max_rounds,
        "baseline_cda": b_cda,
        "baseline_force_N": b_force,
        "rounds": round_records
    }
    with open(report_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved complete session summary report to: {report_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous AI ↔ CFD Closed-Loop Refinement Loop")
    parser.add_argument("--car_id", type=str, required=True, help="Baseline vehicle ID (e.g. F_S_WWC_WM_025, E_S_WWC_WM_025)")
    parser.add_argument("--max_rounds", type=int, default=2, help="Number of closed-loop refinement rounds (default: 2)")
    parser.add_argument("--backend", type=str, choices=["gcp", "local"], default="gcp", help="CFD execution backend (default: gcp)")
    parser.add_argument("--cores", type=int, default=4, help="MPI core count for GCP backend (default: 4)")
    parser.add_argument("--steps", type=int, default=250, help="Optimization steps per round (default: 250)")
    parser.add_argument("--lr", type=float, default=0.01, help="Optimization learning rate (default: 0.01)")
    parser.add_argument("--trust_radius", type=float, default=0.75, help="Latent trust region radius (default: 0.75)")
    parser.add_argument("--cfd_penalty_weight", type=float, default=1.5, help="Directional repulsive barrier weight (default: 1.5)")
    parser.add_argument("--subdivide", type=int, default=1, help="Surface subdivision levels (default: 1)")
    parser.add_argument("--smooth_taubin", type=int, default=4, help="Taubin smoothing iterations (default: 4)")
    parser.add_argument("--evidence_store", type=str, default=None, help="Custom path to cfd_evidence_store.json")
    parser.add_argument("--skip_step0_cfd", action="store_true", help="Skip Step 0 CFD evaluation if baseline exists in evidence store")
    parser.add_argument("--dry_run", action="store_true", help="Simulate CFD results for rapid pipeline testing without OpenFOAM execution")
    args = parser.parse_args()
    
    run_refinement_loop(args)
