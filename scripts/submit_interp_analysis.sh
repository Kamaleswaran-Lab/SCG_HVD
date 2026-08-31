#!/bin/bash
# Runs the attention and Grad-CAM analysis on trained weights and writes the figures.
#SBATCH --job-name=scg_ia
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
REPO=${SCG_HVD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

MODELS=(1d fusion)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=== attention analysis: task1 / $M on $(hostname) ==="
$PY -W ignore analysis/interpretability.py \
  --task task1 --model "$M" \
  --ckpt-dir "$REPO/out/interp_train/task1/$M/seed0" \
  --n-per-class 60 --out "$REPO/out/interp"
echo "=== done $M ==="
