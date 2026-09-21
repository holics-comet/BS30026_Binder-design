#!/bin/bash
#SBATCH --job-name=af3_sif
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/af3_sif_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/af3_sif_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

SINGULARITY=/apps/applications/singularity/4.3.4/bin/singularity
IMAGE=/scratch/a2149a01/alphafold3_image.tar

OUTDIR=/scratch/a2149a01/alphafold3_singularity
SIF=${OUTDIR}/alphafold3.sif

export SINGULARITY_TMPDIR=/scratch/a2149a01/singularity_tmp_${SLURM_JOB_ID}
export SINGULARITY_CACHEDIR=/scratch/a2149a01/singularity_cache

mkdir -p \
  "$OUTDIR" \
  "$SINGULARITY_TMPDIR" \
  "$SINGULARITY_CACHEDIR"

"$SINGULARITY" build \
  "$SIF" \
  "docker-archive://$IMAGE"

echo "=== SIF BUILD COMPLETE ==="
ls -lh "$SIF"
