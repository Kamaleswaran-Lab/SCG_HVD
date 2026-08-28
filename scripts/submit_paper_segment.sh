#!/bin/bash
# 논문 세그먼트 단위 재현을 SLURM GPU 노드에 제출한다. 로그인/인터랙티브 노드는 메모리가 부족하다.
#SBATCH --job-name=scg_paper
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
cd "$REPO"

# array index -> (task, model)
TASKS=(task1 task1 task1 task2 task2 task2)
MODELS=(1d 2d fusion 1d 2d fusion)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=== $TASK / $MODEL on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

$PY -W ignore scripts/run_paper_segment.py \
  --task "$TASK" --models "$MODEL" \
  --out "$REPO/out/paper" --num-workers 8

echo "=== done $TASK/$MODEL ==="
