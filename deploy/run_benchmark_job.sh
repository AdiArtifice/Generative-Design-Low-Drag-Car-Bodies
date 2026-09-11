#!/bin/bash
# AeroMorphs GCP Worker Startup Script with Guaranteed Self-Cleanup
# Executed automatically on VM boot by Google Guest Agent (runs as root)

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:$PATH"
exec > >(tee -a /var/log/cfd_job.log) 2>&1
echo "=== AeroMorphs CFD Worker Starting: $(date -u +%FT%TZ) ==="

# 1. Fetch Job Metadata from GCP Metadata Server
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
JOB_ID=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/job-id" || echo "benchmark_$(date +%s)")
BUCKET=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/bucket-name" || echo "gs://aeromorphs-cfd-aerodesign-cfd-mvp")
STL_GCS_PATH=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/stl-path" || echo "${BUCKET}/benchmark/N_S_WWC_WM_025.stl")
TEMPLATE_GCS_PATH=$(curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/template-path" || echo "${BUCKET}/templates/template_case_v1.tar.gz")

echo "Job ID:    $JOB_ID"
echo "Bucket:    $BUCKET"
echo "STL:       $STL_GCS_PATH"
echo "Template:  $TEMPLATE_GCS_PATH"

# 2. Define Guaranteed Cleanup Trap
cleanup() {
    EXIT_CODE=$?
    echo "=== Worker Exit Handler Triggered with Code: $EXIT_CODE at $(date -u +%FT%TZ) ==="
    
    # Upload results JSON if generated
    if [ -f /opt/cfd/results/cfd_results.json ]; then
        echo "Uploading cfd_results.json..."
        gcloud storage cp /opt/cfd/results/cfd_results.json "${BUCKET}/jobs/${JOB_ID}/cfd_results.json" || true
    fi
    
    # Upload compressed solver log if generated
    if [ -f /opt/cfd/log.simpleFoam ]; then
        echo "Compressing and uploading solver log..."
        gzip -c /opt/cfd/log.simpleFoam > /tmp/log.simpleFoam.gz
        gcloud storage cp /tmp/log.simpleFoam.gz "${BUCKET}/jobs/${JOB_ID}/log.simpleFoam.gz" || true
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

# 4. Install OpenFOAM 2412 and Dependencies
echo "[3/6] Installing OpenFOAM 2412 and runtime dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl wget gnupg software-properties-common python3 python3-numpy jq

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

# 6. Execute Simulation with 75-Minute Linux Timeout
echo "[5/6] Executing CFD Pipeline (blockMesh -> snappyHexMesh -> checkMesh -> simpleFoam)..."
START_TIME=$(date +%s)

# Execute via timeout to prevent any indefinite hang
timeout 75m bash ./scripts/run_cfd.sh constant/triSurface/vehicle.stl

END_TIME=$(date +%s)
WALL_TIME=$((END_TIME - START_TIME))
echo "CFD Execution wall time: ${WALL_TIME} seconds"

# 7. Append execution metadata to results
if [ -f results/cfd_results.json ]; then
    echo "[6/6] Injecting GCP execution metadata..."
    python3 -c "
import json
with open('results/cfd_results.json', 'r') as f:
    data = json.load(f)
data['job_id'] = '${JOB_ID}'
data['backend'] = 'gcp_c2_standard_8_ondemand'
data['wall_time_seconds'] = ${WALL_TIME}
data['instance_type'] = 'c2-standard-8'
with open('results/cfd_results.json', 'w') as f:
    json.dump(data, f, indent=2)
"
fi

echo "=== Benchmark Job Execution Finished Successfully ==="
exit 0
