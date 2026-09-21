#!/bin/bash
#SBATCH --job-name=ligmpnn_pilot
#SBATCH --partition=amd_h200nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home01/a2149a01/RFdiffusion3/logs/ligandmpnn_pilot_%j.out
#SBATCH --error=/home01/a2149a01/RFdiffusion3/logs/ligandmpnn_pilot_%j.err
#SBATCH --comment="field=sciai;appl=pytorch"

set -euo pipefail

PROJECT=/home01/a2149a01/RFdiffusion3
MPNN=/scratch/a2149a01/rfd3/env/bin/mpnn
CKPT=/scratch/a2149a01/rfd3/checkpoints/ligandmpnn_v_32_010_25.pt

SRC="$PROJECT/outputs/binderA_ligandmpnn_pilot"
OUT="$PROJECT/outputs/binderA_ligandmpnn_22x4"

PREFIX="frb_rap_binder_raphotspot_frb_rap_binder_raphotspot"

export FOUNDRY_CHECKPOINT_DIRS=/scratch/a2149a01/rfd3/checkpoints

mkdir -p "$PROJECT/logs"

# 새 pilot 결과만 남기기
rm -rf "$OUT"
mkdir -p "$OUT"

DESIGNS=(
1_model_7
4_model_7
9_model_3
9_model_4
9_model_1
0_model_0
0_model_1
0_model_6
5_model_5
5_model_3
5_model_1
6_model_3
8_model_7
8_model_2
8_model_1
6_model_5
2_model_3
2_model_2
7_model_7
7_model_3
3_model_3
3_model_0
)

echo "=========================================="
echo "LigandMPNN Binder A pilot"
echo "Backbones: ${#DESIGNS[@]}"
echo "Sequences/backbone: 4"
echo "Expected sequences: $((${#DESIGNS[@]} * 4))"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for DESIGN in "${DESIGNS[@]}"
do
    INPUT="$SRC/${PREFIX}_${DESIGN}.cif.gz"
    DESIGN_OUT="$OUT/$DESIGN"

    if [[ ! -f "$INPUT" ]]; then
        echo "ERROR: input not found:"
        echo "$INPUT"
        exit 1
    fi

    mkdir -p "$DESIGN_OUT"

    echo
    echo "=========================================="
    echo "Processing: $DESIGN"
    echo "Input: $INPUT"
    echo "=========================================="

    "$MPNN" \
        --model_type ligand_mpnn \
        --checkpoint_path "$CKPT" \
        --is_legacy_weights True \
        --structure_path "$INPUT" \
        --out_directory "$DESIGN_OUT" \
        --name "$DESIGN" \
        --designed_chains A \
        --batch_size 1 \
        --number_of_batches 4 \
        --temperature 0.1 \
        --write_fasta True \
        --write_structures True

    echo "Finished: $DESIGN"
done

echo
echo "=========================================="
echo "FINAL OUTPUT CHECK"
echo "=========================================="

N_FASTA=$(find "$OUT" -type f -name "*.fa" | wc -l)
N_CIF=$(find "$OUT" -type f -name "*.cif" | wc -l)

echo "FASTA files : $N_FASTA"
echo "CIF files   : $N_CIF"
echo "Expected    : 22 FASTA / 88 CIF"

echo
echo "LigandMPNN pilot complete."
