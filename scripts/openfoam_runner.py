import os
import sys
import json
import subprocess
from pathlib import Path

# Paths
NATIVE_CFD_DIR = Path('/home/student/AeroMorphs/cfd_automation')
BASH_SCRIPT = NATIVE_CFD_DIR / 'scripts' / 'run_cfd.sh'
RESULTS_FILE = NATIVE_CFD_DIR / 'results' / 'cfd_results.json'

def run_cfd(stl_path: Path):
    """
    Run the automated CFD pipeline on the provided STL file.
    Executes in the native unencrypted CFD folder to prevent FUSE overhead.
    """
    stl_path = Path(stl_path).resolve()
    
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    print(f"Starting CFD bridge for STL: {stl_path}")
    print(f"Target CFD Directory: {NATIVE_CFD_DIR}")
    
    cmd = [str(BASH_SCRIPT), str(stl_path)]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(NATIVE_CFD_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(f"[OpenFOAM] {line.strip()}")
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: CFD run failed with return code {process.returncode}", file=sys.stderr)
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        print("CFD run interrupted by user.")
        process.terminate()
        sys.exit(1)
        
    print("CFD execution completed. Parsing results...")
    
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Expected results file not found: {RESULTS_FILE}")
        
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
        
    print("\n--- CFD Results ---")
    print(json.dumps(results, indent=2))
    print("-------------------")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 openfoam_runner.py <path_to_stl>")
        sys.exit(1)
        
    stl_input = Path(sys.argv[1])
    run_cfd(stl_input)
