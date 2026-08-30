#!/bin/bash
# R2-M1 이 요구한 non-overlapping window 보조 분석.
# 창 10초 / 이동 5초이므로 짝수 인덱스만 취하면 겹침이 사라진다. 나머지 프로토콜은 동일하다.
#SBATCH --job-name=scg_noov
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err
#SBATCH --array=0-8

set -euo pipefail
REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
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
