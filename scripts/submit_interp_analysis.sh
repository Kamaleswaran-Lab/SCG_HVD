#!/bin/bash
# R2-m7. 학습된 가중치로 attention 분석과 그림을 만든다.
#SBATCH --job-name=scg_ia
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
cd "$REPO"

MODELS=(1d fusion)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=== attention analysis: task1 / $M on $(hostname) ==="
$PY -W ignore analysis/interpretability.py \
  --task task1 --model "$M" \
  --ckpt-dir "$REPO/out/interp_train/task1/$M/seed0" \
  --n-per-class 60 --out "$REPO/out/interp"
echo "=== done $M ==="
