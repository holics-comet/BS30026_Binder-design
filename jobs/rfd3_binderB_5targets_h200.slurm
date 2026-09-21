#!/bin/bash
#SBATCH --job-name=rfd3_binderB_5targets
#SBATCH --partition=amd_h200nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --comment="field=sciai;appl=pytorch"
#SBATCH --output=/home01/a2149a01/RFdiffusion3/jobs/rfd3_binderB_5targets_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/jobs/rfd3_binderB_5targets_%j.err

set -euo pipefail

PROJECT=/home01/a2149a01/RFdiffusion3
RFD3=/scratch/a2149a01/rfd3/env/bin/rfd3

export FOUNDRY_CHECKPOINT_DIRS=/scratch/a2149a01/rfd3/checkpoints

TARGETS=(
  1_model_7_b0_d0
  3_model_3_b2_d0
  5_model_1_b0_d0
  6_model_5_b3_d0
  9_model_4_b0_d0
)

for TARGET in "${TARGETS[@]}"; do
    echo "===== $TARGET ====="

    "$RFD3" design \
      inputs="$PROJECT/inputs/binderB_rfd3/${TARGET}_binderB.json" \
      out_dir="$PROJECT/outputs/binderB_rfd3/${TARGET}" \
      n_batches=10 \
      diffusion_batch_size=8
done

echo "===== ALL DONE ====="
