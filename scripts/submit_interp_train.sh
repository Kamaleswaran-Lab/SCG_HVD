#!/bin/bash
# R2-m7 해석성 분석용 학습. 진행 중인 CV 잡은 체크포인트를 저장하지 않으므로 따로 돌린다.
# 1 fold 만 쓰며, 목적은 attention 가중치와 Grad-CAM 을 낼 학습된 모델을 얻는 것이다.
#SBATCH --job-name=scg_interp
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
cd "$REPO"

MODELS=(1d fusion)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=== interp train: task1 / $MODEL on $(hostname) ==="
$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model "$MODEL" --folds 5 --seeds 0 --epochs 25 \
  --num-workers 8 --save-checkpoint \
  --out "$REPO/out/interp_train"
echo "=== done ==="
