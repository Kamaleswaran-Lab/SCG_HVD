#!/bin/bash
# Submits the patient-level cross-validation.
# The array index selects one of 18 (task, model, seed) combinations; each task runs its five
# folds in sequence.
#SBATCH --job-name=scg_cv
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-17

set -euo pipefail
# Under sbatch the script is copied to a spool directory, so BASH_SOURCE is useless here.
# SLURM_SUBMIT_DIR is where it was submitted from; fall back to the script location when
# running this file directly.
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

# index = task*9 + model*3 + seed
TASKS=(task1 task2)
MODELS=(1d 2d fusion)
SEEDS=(0 1 2)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$(( i / 9 ))]}
MODEL=${MODELS[$(( (i % 9) / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-25}
OUT=${OUT:-$REPO/out/patient_cv}

echo "=== [$i] $TASK / $MODEL / seed$SEED | folds=$FOLDS epochs=$EPOCHS on $(hostname) ==="
$PY -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

$PY -W ignore scripts/run_patient_cv.py \
  --task "$TASK" --model "$MODEL" \
  --folds "$FOLDS" --seeds "$SEED" --epochs "$EPOCHS" \
  --num-workers 8 --out "$OUT"

echo "=== done [$i] $TASK/$MODEL/seed$SEED ==="
