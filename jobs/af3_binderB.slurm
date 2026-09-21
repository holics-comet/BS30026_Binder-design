#!/bin/bash
#SBATCH --job-name=af3_binderB
#SBATCH --partition=amd_a100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/af3_binderB_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/af3_binderB_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

PROJECT=/home01/a2149a01/RFdiffusion3

SINGULARITY=/apps/applications/singularity/4.3.4/bin/singularity
SIF=/scratch/a2149a01/alphafold3_singularity/alphafold3.sif
MODELS=/scratch/a2149a01/alphafold3_models

MODE=${1:?Usage: sbatch af3_binderB.slurm MODE}

case "$MODE" in
    with_rap)
        INPUT_DIR="$PROJECT/inputs/binderB/af3/with_rap"
        OUTPUT="$PROJECT/outputs/binderB_af3/with_rap"
        ;;
    without_rap)
        INPUT_DIR="$PROJECT/inputs/binderB/af3/without_rap"
        OUTPUT="$PROJECT/outputs/binderB_af3/without_rap"
        ;;
    *)
        echo "ERROR: MODE must be with_rap or without_rap"
        exit 1
        ;;
esac

mkdir -p "$OUTPUT"
mkdir -p "$PROJECT/logs"

mapfile -t INPUTS < <(
    find "$INPUT_DIR" -maxdepth 1 -name "*.json" | sort
)

echo "Mode: $MODE"
echo "Total inputs: ${#INPUTS[@]}"

if [[ ${#INPUTS[@]} -ne 200 ]]; then
    echo "ERROR: expected 200 inputs"
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
