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
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to reuse an existing output directory: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONHASHSEED=42

python scripts/fingerprint_dataset.py --dataset "${UAV90K_ROOT}"

# Deterministic metadata preparation; this has no optimizer or checkpoint.
python scripts/prepare_dino_negatives.py \
  --dataset "${UAV90K_ROOT}" \
  --output "${OUTPUT_ROOT}/fixed_dino_negatives.csv" \
  --splits train val \
  --search-k 200 \
  --negatives-per-query 8 \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda --amp

# The only model-training command in the complete pipeline. These values are
# the frozen single_stage_seed42 experiment settings.
python -u scripts/train.py \
  --dataset "${UAV90K_ROOT}" \
  --negative-manifest "${OUTPUT_ROOT}/fixed_dino_negatives.csv" \
  --output-dir "${OUTPUT_ROOT}/train" \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 8 \
  --learning-rate 3e-4 \
  --seed 42 \
  --deterministic \
  --device cuda --amp \
  --log-interval 20

python scripts/build_satellite_index.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/train/best.pt" \
  --output "${OUTPUT_ROOT}/train/satellite_index.pt" \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda --amp

# Tune reranking on validation only and write locked_evaluation_config.json.
python -u scripts/tune_reranking.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/train/best.pt" \
  --index "${OUTPUT_ROOT}/train/satellite_index.pt" \
  --output-dir "${OUTPUT_ROOT}/val_tuning" \
  --split val \
  --max-top-k 15 \
  --candidate-batch-size 15 \
  --device cuda --amp

python -u scripts/evaluate.py \
  --dataset "${UAV90K_ROOT}" \
  --checkpoint "${OUTPUT_ROOT}/train/best.pt" \
  --index "${OUTPUT_ROOT}/train/satellite_index.pt" \
  --output-dir "${OUTPUT_ROOT}/train/eval/test_final" \
  --split test \
  --top-k 15 \
  --candidate-batch-size 15 \
  --confidence-transform logit \
  --confidence-weight 0.05 \
  --mle-protocol bearing-compatible \
  --device cuda --amp
