#!/bin/bash
# AeroMorphs GCP MPI Worker Startup Script with Guaranteed Self-Cleanup
# Executed automatically on VM boot by Google Guest Agent (runs as root)

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:$PATH"
exec > >(tee -a /var/log/cfd_job.log) 2>&1
echo "=== AeroMorphs CFD MPI Worker Starting: $(date -u +%FT%TZ) ==="

# 1. Fetch Job Metadata from GCP Metadata Server
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
JOB_ID=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/job-id" || echo "mpi_benchmark_$(date +%s)")
BUCKET=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/bucket-name" || echo "gs://aeromorphs-cfd-aerodesign-cfd-mvp")
STL_GCS_PATH=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/stl-path" || echo "${BUCKET}/benchmark/N_S_WWC_WM_025.stl")
TEMPLATE_GCS_PATH=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/template-path" || echo "${BUCKET}/templates/template_case_v1.tar.gz")
MPI_CORES=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/mpi-cores" || echo "4")

echo "Job ID:     $JOB_ID"
echo "Bucket:     $BUCKET"
echo "STL:        $STL_GCS_PATH"
echo "Template:   $TEMPLATE_GCS_PATH"
echo "MPI Cores:  $MPI_CORES"

# 2. Define Guaranteed Cleanup Trap
cleanup() {
    EXIT_CODE=$?
    echo "=== Worker Exit Handler Triggered with Code: $EXIT_CODE at $(date -u +%FT%TZ) ==="
    
    # Upload results JSON if generated
    if [ -f /opt/cfd/results/cfd_results.json ]; then
        echo "Uploading cfd_results.json..."
        /home/student/google-cloud-sdk/bin/gcloud storage cp /opt/cfd/results/cfd_results.json "${BUCKET}/jobs/${JOB_ID}/cfd_results.json" 2>/dev/null || gcloud storage cp /opt/cfd/results/cfd_results.json "${BUCKET}/jobs/${JOB_ID}/cfd_results.json" || true
    fi
    
    # Upload compressed solver log if generated
    if [ -f /opt/cfd/log.simpleFoam ]; then
        echo "Compressing and uploading solver log..."
        gzip -c /opt/cfd/log.simpleFoam > /tmp/log.simpleFoam.gz
        gcloud storage cp /tmp/log.simpleFoam.gz "${BUCKET}/jobs/${JOB_ID}/log.simpleFoam.gz" || true
    fi

    # Upload CPU stat log if generated
    if [ -f /tmp/cpu_stat.log ]; then
        gcloud storage cp /tmp/cpu_stat.log "${BUCKET}/jobs/${JOB_ID}/cpu_stat.log" || true
    fi
    
    # Determine final status
    STATUS="FAILED"
    if [ $EXIT_CODE -eq 0 ] && [ -f /opt/cfd/results/cfd_results.json ]; then
        STATUS="COMPLETED"
    fi
    
    # Record and upload completion status JSON
    cat << STATUS_EOF > /tmp/status.json
{
  "job_id": "${JOB_ID}",
  "status": "${STATUS}",
  "exit_code": ${EXIT_CODE},
  "mpi_cores": ${MPI_CORES},
  "completed_at": "$(date -u +%FT%TZ)"
}
STATUS_EOF
    gcloud storage cp /tmp/status.json "${BUCKET}/jobs/${JOB_ID}/status.json" || true

    # Upload execution log
    gcloud storage cp /var/log/cfd_job.log "${BUCKET}/jobs/${JOB_ID}/cfd_job.log" || true
    
    echo "=== Self-terminating VM instance now ==="
    sync
    poweroff || shutdown -h now
}
trap cleanup EXIT ERR INT TERM

# 3. Wait for Network and Cloud-Init / Apt Locks
echo "[1/6] Waiting for network connectivity..."
for i in $(seq 1 30); do
    if curl -sf --head https://dl.openfoam.com >/dev/null; then
        echo "Network is ready."
        break
    fi
    echo "Waiting for network (attempt $i/30)..."
    sleep 2
done

echo "[2/6] Waiting for background system updates / apt locks to clear..."
while fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; do
    echo "Apt lock held by system; waiting 3s..."
    sleep 3
done

# 4. Install OpenFOAM 2412, OpenMPI, and Dependencies
echo "[3/6] Installing OpenFOAM 2412, OpenMPI, and dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl wget gnupg software-properties-common python3 python3-numpy jq openmpi-bin libopenmpi-dev sysstat

# Add official OpenCFD repository
curl -s https://dl.openfoam.com/add-debian-repo.sh | bash
apt-get update -y
apt-get install -y openfoam2412-default

# Configure persistent environment
echo "source /usr/lib/openfoam/openfoam2412/etc/bashrc" > /etc/profile.d/openfoam.sh
chmod +x /etc/profile.d/openfoam.sh
source /usr/lib/openfoam/openfoam2412/etc/bashrc

echo "OpenFOAM verification:"
which simpleFoam || echo "simpleFoam path not resolved yet"
which decomposePar || echo "decomposePar path not resolved yet"
which mpirun || echo "mpirun path not resolved yet"

# 5. Prepare Workspace and Download Assets
echo "[4/6] Staging CFD case from GCS..."
mkdir -p /opt/cfd
cd /opt/cfd

gcloud storage cp "${TEMPLATE_GCS_PATH}" /tmp/template.tar.gz
tar -xzf /tmp/template.tar.gz -C /opt/cfd --strip-components=1

# Download benchmark vehicle STL into constant/triSurface/
mkdir -p constant/triSurface
gcloud storage cp "${STL_GCS_PATH}" constant/triSurface/vehicle.stl

echo "Verifying staged case:"
ls -lh constant/triSurface/vehicle.stl
grep "level" system/snappyHexMeshDict

# 6. Execute Simulation (Serial for 1 core, MPI for >1 cores)
echo "[5/6] Executing Mesh & Solver Pipeline (${MPI_CORES} cores)..."
JOB_START_TIME=$(date +%s)

echo "--- Mesh Generation ---"
MESH_START=$(date +%s)
blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
checkMesh > log.checkMesh 2>&1
MESH_END=$(date +%s)
MESH_TIME=$((MESH_END - MESH_START))
echo "Mesh generated in ${MESH_TIME}s."

# Start background CPU utilization logger (samples /proc/stat every 5s)
(
    while true; do
        grep 'cpu ' /proc/stat
        sleep 5
    done
) > /tmp/cpu_stat.log &
CPU_PID=$!

if [ "$MPI_CORES" -eq 1 ]; then
    echo "--- Running simpleFoam in Serial (1 core) ---"
    SOLVER_START=$(date +%s)
    timeout 75m simpleFoam > log.simpleFoam 2>&1
    SOLVER_END=$(date +%s)
    SOLVER_TIME=$((SOLVER_END - SOLVER_START))
    echo "Serial solver finished in ${SOLVER_TIME}s."
else
    # Setup decomposeParDict
    cat << DECOMP_EOF > system/decomposeParDict
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}
numberOfSubdomains ${MPI_CORES};
method          scotch;
DECOMP_EOF

    echo "--- Domain Decomposition (${MPI_CORES} domains) ---"
    decomposePar -force > log.decomposePar 2>&1

    echo "--- Running simpleFoam in Parallel (${MPI_CORES} MPI ranks) ---"
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

    SOLVER_START=$(date +%s)
    timeout 75m mpirun --allow-run-as-root --use-hwthread-cpus --oversubscribe -np ${MPI_CORES} simpleFoam -parallel > log.simpleFoam 2>&1
    SOLVER_END=$(date +%s)
    SOLVER_TIME=$((SOLVER_END - SOLVER_START))
    echo "MPI solver finished in ${SOLVER_TIME}s."
fi

# Terminate CPU logger
kill $CPU_PID 2>/dev/null || true

echo "--- Extracting Results ---"
python3 scripts/extract_results.py

JOB_END=$(date +%s)
TOTAL_WALL_TIME=$((JOB_END - JOB_START_TIME))
echo "Total CFD wall time: ${TOTAL_WALL_TIME} seconds"

# 7. Calculate CPU utilization and inject execution metadata
if [ -f results/cfd_results.json ]; then
    echo "[6/6] Injecting execution metrics and CPU stats..."
    python3 -c "
import json

# Compute average CPU utilization from /proc/stat samples
avg_cpu_pct = 0.0
try:
    with open('/tmp/cpu_stat.log', 'r') as f:
        lines = [line.strip().split()[1:] for line in f if line.startswith('cpu ')]
    if len(lines) >= 2:
        deltas = []
        for i in range(len(lines) - 1):
            t1 = [float(x) for x in lines[i]]
            t2 = [float(x) for x in lines[i+1]]
            idle1, idle2 = t1[3] + t1[4], t2[3] + t2[4]
            total1, total2 = sum(t1), sum(t2)
            d_total = total2 - total1
            d_idle = idle2 - idle1
            if d_total > 0:
                deltas.append(100.0 * (1.0 - (d_idle / d_total)))
        if deltas:
            avg_cpu_pct = sum(deltas) / len(deltas)
except Exception as e:
    print(f'CPU calc error: {e}')

with open('results/cfd_results.json', 'r') as f:
    data = json.load(f)

data['job_id'] = '${JOB_ID}'
data['backend'] = 'gcp_c2_standard_8_mpi_${MPI_CORES}'
data['mpi_cores'] = int('${MPI_CORES}')
data['total_wall_time_seconds'] = ${TOTAL_WALL_TIME}
data['solver_wall_time_seconds'] = ${SOLVER_TIME}
data['mesh_wall_time_seconds'] = ${MESH_TIME}
data['instance_type'] = 'c2-standard-8'
data['avg_total_cpu_utilization_pct'] = round(avg_cpu_pct, 2)
data['active_cores_utilization_pct'] = round(min(100.0, avg_cpu_pct * (8.0 / float('${MPI_CORES}'))), 2)

with open('results/cfd_results.json', 'w') as f:
    json.dump(data, f, indent=2)
"
fi

echo "=== MPI Benchmark Job Finished Successfully ==="
exit 0
