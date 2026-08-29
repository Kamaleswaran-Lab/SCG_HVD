#!/bin/bash
# R1-M5(강한 베이스라인) 과 R1-M6/R2-M4(파라미터 매칭) 용 실행.
# 세 베이스라인 모두 융합 모델(4,767,374) 에 파라미터를 맞췄으므로 용량이 아니라 아키텍처를 비교한다.
#SBATCH --job-name=scg_base
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

# index = model*3 + seed. Task I 만 돌린다 — 분류 주장이 있는 쪽이다.
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
