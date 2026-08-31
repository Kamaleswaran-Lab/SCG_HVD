#!/bin/bash
# Capacity-matched baselines. All three are sized to the fusion model's 4,767,374
# parameters, so any difference between them is architectural rather than a matter of
# capacity.
#
# These runs are not reported in the manuscript -- see the note in scg_hvd/models.py -- but
# the runs happened and the code stays here.
#SBATCH --job-name=scg_base
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-8

set -euo pipefail
# Under sbatch the script is copied to a spool directory, so BASH_SOURCE is useless here.
# SLURM_SUBMIT_DIR is where it was submitted from; fall back to the script location when
# running this file directly.
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

# index = model*3 + seed. Task I only, since that is the task carrying classification claims.
MODELS=(temporal_matched resnet1d_matched tcn_matched)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

echo "=== [$i] task1 / $MODEL / seed$SEED on $(hostname) ==="
$PY -c "import torch;print('cuda',torch.cuda.is_available())"

$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model "$MODEL" \
  --folds 5 --seeds "$SEED" --epochs 25 \
  --num-workers 8 --out "$REPO/out/patient_cv"

echo "=== done [$i] task1/$MODEL/seed$SEED ==="
