#!/bin/bash
# 환자 단위 교차검증을 gpu-hp 에 제출한다. 배열 인덱스가 (task, model) 조합을 고른다.
#SBATCH --job-name=scg_cv
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err

set -euo pipefail
REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
cd "$REPO"

TASKS=(task1 task1 task1 task2 task2 task2)
MODELS=(1d 2d fusion 1d 2d fusion)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

FOLDS=${FOLDS:-5}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-25}
OUT=${OUT:-$REPO/out/patient_cv}

echo "=== $TASK / $MODEL | folds=$FOLDS seeds=$SEEDS epochs=$EPOCHS on $(hostname) ==="
$PY -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

$PY -W ignore scripts/run_patient_cv.py \
  --task "$TASK" --model "$MODEL" \
  --folds "$FOLDS" --seeds $SEEDS --epochs "$EPOCHS" \
  --num-workers 8 --out "$OUT"
