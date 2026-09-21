#!/bin/bash
#SBATCH --job-name=ligmpnn_binderB
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/ligandmpnn_binderB_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/ligandmpnn_binderB_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

PROJECT=/home01/a2149a01/RFdiffusion3
MPNN=/scratch/a2149a01/rfd3/env/bin/mpnn
CKPT=/scratch/a2149a01/rfd3/checkpoints/ligandmpnn_v_32_010_25.pt

SRC="$PROJECT/outputs/binderB_backbone_selected"
OUT="$PROJECT/outputs/binderB_ligandmpnn_50x4"

export FOUNDRY_CHECKPOINT_DIRS=/scratch/a2149a01/rfd3/checkpoints

mkdir -p "$PROJECT/logs"

rm -rf "$OUT"
mkdir -p "$OUT"

mapfile -t INPUTS < <(find "$SRC" -maxdepth 1 -type f -name "*.cif.gz" | sort)

echo "=========================================="
echo "LigandMPNN Binder B"
echo "Backbones: ${#INPUTS[@]}"
echo "Sequences/backbone: 4"
echo "Expected sequences: $((${#INPUTS[@]} * 4))"
echo "Designed chain: B"
echo "=========================================="

if [[ ${#INPUTS[@]} -ne 50 ]]; then
    echo "ERROR: expected 50 backbones, found ${#INPUTS[@]}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for INPUT in "${INPUTS[@]}"
do
    BASENAME=$(basename "$INPUT" .cif.gz)

    # Short unique name:
    # target__rankXX__original_design -> target__rankXX
    TARGET="${BASENAME%%__rank*}"
    REST="${BASENAME#*__}"
    RANK="${REST%%__*}"
    NAME="${TARGET}__${RANK}"

    DESIGN_OUT="$OUT/$NAME"
    mkdir -p "$DESIGN_OUT"

    echo
    echo "=========================================="
    echo "Processing: $NAME"
    echo "Input: $INPUT"
    echo "=========================================="

    "$MPNN" \
        --model_type ligand_mpnn \
        --checkpoint_path "$CKPT" \
        --is_legacy_weights True \
        --structure_path "$INPUT" \
        --out_directory "$DESIGN_OUT" \
        --name "$NAME" \
        --designed_chains B \
        --batch_size 1 \
        --number_of_batches 4 \
        --temperature 0.1 \
        --write_fasta True \
        --write_structures True

    echo "Finished: $NAME"
done

echo
echo "=========================================="
echo "FINAL OUTPUT CHECK"
echo "=========================================="

N_FASTA=$(find "$OUT" -type f -name "*.fa" | wc -l)
N_CIF=$(find "$OUT" -type f -name "*.cif" | wc -l)

echo "FASTA files : $N_FASTA"
echo "CIF files   : $N_CIF"
echo "Expected    : 50 FASTA / 200 CIF"

echo
echo "LigandMPNN Binder B complete."
