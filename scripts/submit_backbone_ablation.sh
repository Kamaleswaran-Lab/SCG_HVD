#!/bin/bash
# Swap the image backbone and see whether the fusion gain survives.
#
# The paper does not claim this architecture beats other architectures; it claims that adding a
# spectrotemporal branch to a temporal encoder helps. Testing that means holding the temporal
# branch fixed, varying the image backbone, and checking that the 2D-only -> fusion step keeps
# its sign.
#
# The step costs exactly +564,111 parameters under every backbone (verified), so the amount
# being added is constant even though the backbone underneath is not.
#
# index = backbone*6 + model*3 + seed  (backbone: resnet18/mobilenet/densenet, model: 2d/fusion)
#SBATCH --job-name=scg_bb
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# DenseNet121 (array indices 12-17) was cancelled. ResNet18 at 11.2M and MobileNetV3-Large at
# 4.2M already span a 2.7x range in backbone size, and a third would have spent roughly 10 more
# GPU-hours reaching the same conclusion. Submit with --array=12-17 to run it after all.
#SBATCH --array=0-11

set -euo pipefail
REPO=${SCG_HVD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${SCG_HVD_PYTHON:-python}
cd "$REPO"

BACKBONES=(resnet18 mobilenet densenet)
KINDS=(2d fusion)
SEEDS=(0 1 2)

i=$SLURM_ARRAY_TASK_ID
BB=${BACKBONES[$(( i / 6 ))]}
KIND=${KINDS[$(( (i % 6) / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}
MODEL="${KIND}_${BB}"

echo "=== [$i] task1 / $MODEL / seed$SEED on $(hostname) ==="
$PY -c "
import sys; sys.path.insert(0,'.')
from scg_hvd.models import build_model, count_parameters
m = build_model('$MODEL', 5)
print('  parameters', f'{count_parameters(m):,}')
"

$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model "$MODEL" --folds 5 --seeds "$SEED" --epochs 25 \
  --num-workers 8 --out "$REPO/out/backbone_ablation"

echo "=== done [$i] $MODEL/seed$SEED ==="
