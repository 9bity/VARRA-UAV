# UAV single-stage

Single-UVP global localization and heading estimation on UAV90K. The model
keeps the original coarse-to-fine design: DINOv2 global satellite retrieval,
3x3 candidate expansion, and VARRA cross-view local registration. Retrieval,
position, heading, and candidate quality are now learned in one continuous
training run.

The previous two-stage implementation is preserved separately in
`D:\paper\UAV`. This directory contains the independent single-stage redesign.

## Pipeline

```text
offline satellite tiles -> final learned descriptors -> satellite index
                                                       ^
single UVP -> DINOv2 -> cross-view retrieval ----------+
     |
     +-> Top-K centers -> 3x3 expansion -> VARRA -> position + heading + quality
```

Training uses one positive and one fixed DINOv2 hard-negative 3x3 candidate per
query. Fixed negatives are prepared with the public frozen DINOv2 backbone,
not with a previously trained UAV model. Preparing them is deterministic data
preprocessing and does not create a model checkpoint.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for tensor flow and objectives,
[docs/UAV90K.md](docs/UAV90K.md) for the dataset format, and
[docs/EVALUATION.md](docs/EVALUATION.md) for the five-metric protocol.

The dataset itself is not stored in this repository. See
[docs/DATASET_DOWNLOAD.md](docs/DATASET_DOWNLOAD.md) for the official
Bearing-UAV-90K download and the deterministic conversion command.

## One-run training

```bash
python -m pip install -e ".[navigation]"

python scripts/prepare_dino_negatives.py \
  --dataset /path/to/UAV90K \
  --output /path/to/fixed_dino_negatives.csv \
  --splits train val \
  --search-k 200 \
  --negatives-per-query 8

python scripts/train.py \
  --dataset /path/to/UAV90K \
  --negative-manifest /path/to/fixed_dino_negatives.csv \
  --output-dir /path/to/run \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 8 \
  --seed 42 \
  --deterministic
```

`train.py` starts from DINOv2 plus randomly initialized task heads and performs
one optimizer/scheduler run. `--resume` only continues an interrupted run; it
does not define a second training stage.

After training, build the final satellite index, select reranking parameters on
the validation split, and evaluate the untouched test split. The complete
sequence is available in `scripts/run_reference_pipeline.sh`.

```bash
python scripts/build_satellite_index.py \
  --dataset /path/to/UAV90K \
  --checkpoint /path/to/run/best.pt \
  --output /path/to/run/satellite_index.pt

python scripts/tune_reranking.py \
  --dataset /path/to/UAV90K \
  --checkpoint /path/to/run/best.pt \
  --index /path/to/run/satellite_index.pt \
  --output-dir /path/to/run/val_tuning \
  --split val \
  --max-top-k 15 \
  --candidate-batch-size 15

python scripts/evaluate.py \
  --dataset /path/to/UAV90K \
  --checkpoint /path/to/run/best.pt \
  --index /path/to/run/satellite_index.pt \
  --output-dir /path/to/run/eval \
  --split test \
  --top-k 15 \
  --candidate-batch-size 15 \
  --confidence-transform logit \
  --confidence-weight 0.05 \
  --mle-protocol bearing-compatible
```

The frozen reference experiment uses seed 42, 30 epochs, batch size 16, a
frozen DINOv2 ViT-B/14 backbone, and the validation-selected reranking settings
shown above. Exact settings and expected hashes are stored under
[`configs/single_stage_seed42`](configs/single_stage_seed42).

Model weights, satellite indexes, downloaded images, logs, and generated run
directories are deliberately excluded from Git. A reproducer builds the
dataset, trains once, builds the index, and then evaluates the fixed test split.

## Bearing-Naver navigation

The repository also contains the fixed eight-route 2D Bearing-Naver runner:

```bash
python scripts/evaluate_single_stage_navigation.py \
  --dataset /path/to/UAV90K \
  --checkpoint /path/to/run/best.pt \
  --index /path/to/run/satellite_index.pt \
  --evaluation-lock /path/to/run/val_tuning/locked_evaluation_config.json \
  --routes navigation/bearing_naver_routes.json \
  --output-dir /path/to/run/navigation \
  --step-m 25 --arrival-radius-m 20 --max-steps 384 \
  --device cuda --amp
```

This public runner evaluates rotated satellite-map crops in a reproducible 2D
closed loop. It must not be described as continuous Google Earth 3D UAV-view
navigation.
