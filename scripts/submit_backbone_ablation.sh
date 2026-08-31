#!/bin/bash
# 2D 백본 교체 ablation. 논문 주장을 직접 시험한다.
#
# 주장은 "이 아키텍처가 다른 아키텍처보다 낫다" 가 아니라 "시간 인코더에 스펙트로템포럴
# 브랜치를 더하면 좋아진다" 이다. 그 주장을 시험하려면 시간 브랜치를 고정하고 이미지 백본을
# 바꿔 가며 (2D 단독 -> 융합) 이득이 유지되는지를 봐야 한다.
#
# 백본마다 2D 단독 -> 융합 증분이 정확히 +564,111 파라미터로 동일하다(검증됨). 즉 백본이
# 달라도 "추가되는 양" 은 같으므로 비교가 깨끗하다.
#
# index = backbone*6 + model*3 + seed  (backbone: resnet18/mobilenet/densenet, model: 2d/fusion)
#SBATCH --job-name=scg_bb
#SBATCH --partition=gpu-hp
#SBATCH --qos=duke_h200_hp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.out
#SBATCH --error=/hpc/home/jkim1/workspace/SCG_HVD/logs/%x_%A_%a.err
# DenseNet121 (배열 12-17) 은 2026-08-31 에 취소했다. ResNet18(11.2M) 과
# MobileNetV3-Large(4.2M) 만으로도 크기가 2.7배 다른 두 백본을 덮으며, 세 번째 백본은
# 같은 결론에 약 10 GPU-시간을 더 쓰는 것이었다. 다시 돌리려면 --array=12-17 로 제출한다.
#SBATCH --array=0-11

set -euo pipefail
REPO=/hpc/home/jkim1/workspace/SCG_HVD
PY=/hpc/home/jkim1/miniforge3/envs/tccc/bin/python
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
print('  파라미터', f'{count_parameters(m):,}')
"

$PY -W ignore scripts/run_patient_cv.py \
  --task task1 --model "$MODEL" --folds 5 --seeds "$SEED" --epochs 25 \
  --num-workers 8 --out "$REPO/out/backbone_ablation"

echo "=== done [$i] $MODEL/seed$SEED ==="
