import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

DEFAULT_CFD_DIR = Path('/home/student/AeroMorphs/cfd_automation_prism_layers')

def run_cfd(stl_path: Path, cfd_dir: Path = DEFAULT_CFD_DIR, output_dest: Path = None, tag: str = None):
    """
    Run the automated CFD pipeline on the provided STL file.
    Executes in the native unencrypted CFD folder to prevent FUSE overhead.
    """
    stl_path = Path(stl_path).resolve()
    cfd_dir = Path(cfd_dir).resolve()
    bash_script = cfd_dir / 'scripts' / 'run_cfd.sh'
    results_file = cfd_dir / 'results' / 'cfd_results.json'
    
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    if not bash_script.exists():
        raise FileNotFoundError(f"CFD runner script not found: {bash_script}")
    
    print(f"Starting CFD bridge for STL: {stl_path}")
    print(f"Target CFD Directory: {cfd_dir}")
    
    cmd = [str(bash_script), str(stl_path)]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cfd_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(f"[OpenFOAM] {line.strip()}", flush=True)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: CFD run failed with return code {process.returncode}", file=sys.stderr)
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        print("CFD run interrupted by user.")
        process.terminate()
        sys.exit(1)
        
    print("CFD execution completed. Parsing results...")
    
    if not results_file.exists():
        raise FileNotFoundError(f"Expected results file not found: {results_file}")
        
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    if tag:
        results['model_tag'] = tag
    results['stl_source'] = str(stl_path)
    
    print("\n--- CFD Results ---")
    print(json.dumps(results, indent=2))
    print("-------------------")
    
    if output_dest:
        output_dest = Path(output_dest).resolve()
        output_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(output_dest, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Archived copy saved to: {output_dest}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroMorphs OpenFOAM CFD Bridge Runner")
    parser.add_argument("stl", type=str, help="Path to input STL file")
    parser.add_argument("--cfd_dir", type=str, default=str(DEFAULT_CFD_DIR), help=f"Target OpenFOAM CFD directory (default: {DEFAULT_CFD_DIR})")
    parser.add_argument("--output", type=str, default=None, help="Optional path to archive results JSON")
    parser.add_argument("--tag", type=str, default=None, help="Optional model identifier/tag")
    
    args = parser.parse_args()
    run_cfd(Path(args.stl), Path(args.cfd_dir), Path(args.output) if args.output else None, args.tag)
