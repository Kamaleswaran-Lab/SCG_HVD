#!/bin/bash
# The non-overlapping-window analysis.
# Windows are 10 s on a 5 s stride, so keeping every second segment removes the overlap
# entirely. Everything else about the protocol is unchanged.
#SBATCH --job-name=scg_noov
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
REPO=${SCG_HVD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

MODELS=(1d 2d fusion)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

echo "=== [$i] task1 / $MODEL / seed$SEED | non-overlapping on $(hostname) ==="
$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model "$MODEL" --folds 5 --seeds "$SEED" --epochs 25 \
  --nonoverlap --num-workers 8 --out "$REPO/out/patient_cv_nonoverlap"
echo "=== done [$i] ==="
