#!/bin/bash
#SBATCH --job-name=build_af3
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/build_af3_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/build_af3_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

AF3=/scratch/a2149a01/alphafold3
PODMAN=/usr/bin/podman

LOCAL_BASE=/tmp/a2149a01_af3_${SLURM_JOB_ID}
ROOT=${LOCAL_BASE}/root
RUNROOT=${LOCAL_BASE}/runroot

export XDG_RUNTIME_DIR=${LOCAL_BASE}/xdg

EXPORT=/scratch/a2149a01/alphafold3_image.tar

mkdir -p "$ROOT" "$RUNROOT" "$XDG_RUNTIME_DIR"

cd "$AF3"

echo "=== PODMAN PATHS ==="
echo "ROOT=$ROOT"
echo "RUNROOT=$RUNROOT"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"

echo "=== BUILDING AF3 IMAGE ==="

"$PODMAN" \
  --cgroup-manager=cgroupfs \
  --storage-driver=vfs \
  --root "$ROOT" \
  --runroot "$RUNROOT" \
  build \
  -f docker/Dockerfile \
  -t localhost/alphafold3:latest \
  .

echo "=== EXPORTING IMAGE ==="

"$PODMAN" \
  --cgroup-manager=cgroupfs \
  --storage-driver=vfs \
  --root "$ROOT" \
  --runroot "$RUNROOT" \
  save \
  -o "$EXPORT" \
  localhost/alphafold3:latest

echo "=== BUILD COMPLETE ==="
ls -lh "$EXPORT"

rm -rf "$LOCAL_BASE"
