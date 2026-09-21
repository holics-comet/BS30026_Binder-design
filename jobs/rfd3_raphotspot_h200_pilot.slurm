#!/bin/bash
#SBATCH --job-name=rfd3_raphotspot_pilot
#SBATCH --partition=amd_h200nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --comment="field=sciai;appl=pytorch"
#SBATCH --output=/home01/a2149a01/RFdiffusion3/jobs/rfd3_raphotspot_pilot_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/jobs/rfd3_raphotspot_pilot_%j.err

set -euo pipefail

PROJECT=/home01/a2149a01/RFdiffusion3
RFD3=/scratch/a2149a01/rfd3/env/bin/rfd3

export FOUNDRY_CHECKPOINT_DIRS=/scratch/a2149a01/rfd3/checkpoints

"$RFD3" design \
  inputs="$PROJECT/inputs/frb_rap_binder_raphotspot.json" \
  out_dir="$PROJECT/outputs/frb_rap_raphotspot_pilot" \
  n_batches=10 \
  diffusion_batch_size=8