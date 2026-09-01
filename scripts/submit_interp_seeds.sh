#!/bin/bash
# Attention analysis for each seed, so the axis result can be checked for reproducibility.
#
# The first run used one fold of one seed, which describes that model and nothing more. The
# criterion in BPEX_R1/interpretability-plan.md is that the z-axis dominance has to hold in all
# three seeds before it goes in the paper.
#SBATCH --job-name=scg_attn
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

SEED=$SLURM_ARRAY_TASK_ID
CK="$REPO/out/interp_train/task1/fusion/seed$SEED"
if [ ! -d "$CK" ]; then echo "no checkpoints at $CK"; exit 1; fi

echo "=== attention: task1 / fusion / seed$SEED on $(hostname) ==="
$PY -W ignore analysis/interpretability.py \
  --task task1 --model fusion --ckpt-dir "$CK" \
  --n-per-class 60 --out "$REPO/out/interp/seed$SEED"
echo "=== done seed$SEED ==="
