#!/bin/bash
# Runs the Grad-CAM validation on a GPU node: trained model against the shuffled-label and
# untrained controls.
#SBATCH --job-name=scg_gcv
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
set -euo pipefail
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"
echo "=== Grad-CAM validation on $(hostname) ==="
$PY -W ignore analysis/gradcam_validate.py --task task1 \
  --ckpt "out/interp_train/task1/fusion/seed0/seed0_fold0/model.pt" --shuffled-ckpt "out/interp_shuffled/task1/fusion/seed0/seed0_fold0/model.pt" \
  --n-per-class 30 --out "$REPO/out/gradcam_val"
echo "=== done ==="
