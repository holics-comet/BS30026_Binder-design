#!/bin/bash
#SBATCH --job-name=af3_binderA
#SBATCH --partition=amd_h200nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/af3_binderA_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/af3_binderA_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

SINGULARITY=/apps/applications/singularity/4.3.4/bin/singularity
SIF=/scratch/a2149a01/alphafold3_singularity/alphafold3.sif

INPUT=/home01/a2149a01/RFdiffusion3/inputs/af3_binderA_test/5_model_5_b0_d0_plus_RAP.json
OUTPUT=/home01/a2149a01/RFdiffusion3/outputs/af3_binderA_5_model_5_b0_d0
MODELS=/scratch/a2149a01/alphafold3_models

mkdir -p "$OUTPUT"

"$SINGULARITY" exec --nv \
  --bind /home01:/home01 \
  --bind /scratch:/scratch \
  "$SIF" \
  python /app/alphafold/run_alphafold.py \
    --json_path="$INPUT" \
    --model_dir="$MODELS" \
    --output_dir="$OUTPUT" \
    --run_data_pipeline=false
