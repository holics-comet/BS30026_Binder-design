#!/bin/bash
#SBATCH --job-name=af3_A_noRAP
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/af3_A_noRAP_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/af3_A_noRAP_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

PROJECT=/home01/a2149a01/RFdiffusion3
SINGULARITY=/apps/applications/singularity/4.3.4/bin/singularity
SIF=/scratch/a2149a01/alphafold3_singularity/alphafold3.sif
MODELS=/scratch/a2149a01/alphafold3_models

INPUT_DIR="$PROJECT/inputs/af3_binderA_noRAP"
OUTPUT="$PROJECT/outputs/af3_binderA_noRAP_a100"

mkdir -p "$OUTPUT"

mapfile -t INPUTS < <(
    find "$INPUT_DIR" -maxdepth 1 -name "*.json" | sort
)

echo "Total inputs: ${#INPUTS[@]}"

if [ "${#INPUTS[@]}" -ne 88 ]; then
    echo "ERROR: expected 88 inputs"
    exit 1
fi

for i in "${!INPUTS[@]}"; do
    INPUT="${INPUTS[$i]}"
    NAME=$(basename "$INPUT" .json)

    echo
    echo "===== [$((i+1))/${#INPUTS[@]}] $NAME ====="

    "$SINGULARITY" exec --nv \
      --bind /home01:/home01 \
      --bind /scratch:/scratch \
      "$SIF" \
      python /app/alphafold/run_alphafold.py \
        --json_path="$INPUT" \
        --model_dir="$MODELS" \
        --output_dir="$OUTPUT" \
        --run_data_pipeline=false
done

echo
echo "===== ALL DONE ====="
