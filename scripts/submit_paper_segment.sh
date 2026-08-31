#!/bin/bash
# Submits the segment-level reproduction to a GPU node. It will not fit on a login or
# interactive node.
#SBATCH --job-name=scg_paper
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

# Under sbatch the script is copied to a spool directory, so BASH_SOURCE is useless here.
# SLURM_SUBMIT_DIR is where it was submitted from; fall back to the script location when
# running this file directly.
REPO=${SCG_HVD_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PY=${SCG_HVD_PYTHON:-python}
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
