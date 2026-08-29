#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/fresh/output-directory" >&2
  exit 2
fi

: "${UAV90K_ROOT:?Set UAV90K_ROOT to the prepared UAV90K directory}"
: "${DINOV2_REPO:?Set DINOV2_REPO to the local official DINOv2 repository}"
: "${DINOV2_WEIGHTS:?Set DINOV2_WEIGHTS to dinov2_vitb14_pretrain.pth}"

OUTPUT_ROOT="$1"
if [[ -e "${OUTPUT_ROOT}/stage1" || -e "${OUTPUT_ROOT}/stage2" ]]; then
  echo "Refusing to reuse an existing stage directory: ${OUTPUT_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONHASHSEED=42

python scripts/fingerprint_dataset.py --dataset "${UAV90K_ROOT}"

python -u scripts/train.py \
  --dataset "${UAV90K_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/stage1" \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 8 \
  --learning-rate 3e-4 \
  --seed 42 \
  --deterministic \
  --device cuda \
  --amp \
  --log-interval 20

python scripts/build_satellite_index.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/stage1/best.pt" \
  --output "${OUTPUT_ROOT}/stage1/satellite_index.pt" \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda \
  --amp

python scripts/mine_hard_negatives.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/stage1/best.pt" \
  --index "${OUTPUT_ROOT}/stage1/satellite_index.pt" \
  --output "${OUTPUT_ROOT}/stage1/hard_negatives.csv" \
  --splits train val \
  --search-k 50 \
  --negatives-per-query 5 \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda \
  --amp

python -u scripts/train.py \
  --dataset "${UAV90K_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/stage2" \
  --init-checkpoint "${OUTPUT_ROOT}/stage1/best.pt" \
  --negative-manifest "${OUTPUT_ROOT}/stage1/hard_negatives.csv" \
  --negative-probability 0.5 \
  --confidence-weight 0.5 \
  --epochs 15 \
  --batch-size 16 \
  --num-workers 8 \
  --learning-rate 1e-4 \
  --seed 42 \
  --deterministic \
  --device cuda \
  --amp \
  --log-interval 20

python scripts/build_satellite_index.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/stage2/best.pt" \
  --output "${OUTPUT_ROOT}/stage2/satellite_index.pt" \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda \
  --amp

python -u scripts/evaluate.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/stage2/best.pt" \
  --index "${OUTPUT_ROOT}/stage2/satellite_index.pt" \
  --output-dir "${OUTPUT_ROOT}/stage2/eval/test_final" \
  --split test \
  --top-k 5 \
  --candidate-batch-size 5 \
  --confidence-weight 0.5 \
  --mle-protocol bearing-compatible \
  --device cuda \
  --amp
