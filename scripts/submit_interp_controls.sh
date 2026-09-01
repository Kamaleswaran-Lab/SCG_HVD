#!/bin/bash
# Trains the shuffled-label control for the Grad-CAM validation.
#
# The attribution question is whether the class pattern belongs to the model. A model trained
# on permuted labels has fitted the data without learning the classes, so running the same
# attribution on it shows what the method produces when there is no class structure to find.
# One fold is enough; this is a null, not a measurement.
#SBATCH --job-name=scg_ctrl
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

echo "=== shuffled-label control: task1 / fusion on $(hostname) ==="
$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model fusion --folds 5 --seeds 0 --epochs 25 \
  --num-workers 8 --save-checkpoint --shuffle-labels \
  --out "$REPO/out/interp_shuffled"
echo "=== done ==="
