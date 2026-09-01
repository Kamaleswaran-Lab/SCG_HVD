#!/bin/bash
# Trains the models the interpretability analysis needs. The cross-validation jobs do not keep
# checkpoints, so this runs separately. One fold is enough: the goal is a trained model to read
# attention weights and Grad-CAM maps out of, not another performance estimate.
#SBATCH --job-name=scg_interp
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
# Under sbatch the script is copied to a spool directory, so BASH_SOURCE is useless here.
# SLURM_SUBMIT_DIR is where it was submitted from; fall back to the script location when
# running this file directly.
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

MODELS=(1d fusion)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
# Seed comes from the environment so the same script can build the several seeds the
# reproducibility check needs, rather than being hardcoded to one.
SEED=${SEED:-0}

echo "=== interp train: task1 / $MODEL / seed$SEED on $(hostname) ==="
$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model "$MODEL" --folds 5 --seeds "$SEED" --epochs 25 \
  --num-workers 8 --save-checkpoint \
  --out "$REPO/out/interp_train"
echo "=== done ==="
