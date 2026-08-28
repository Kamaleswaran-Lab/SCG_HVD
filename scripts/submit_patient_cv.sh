#!/bin/bash
# 환자 단위 교차검증을 gpu-hp 에 제출한다.
# 배열 인덱스가 (task, model, seed) 18조합을 고른다. 각 잡이 5 fold 를 순차로 돌린다.
#SBATCH --job-name=scg_cv
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err
#SBATCH --array=0-17

set -euo pipefail
REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
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
