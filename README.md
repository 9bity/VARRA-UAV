# UAV

Research code for global UAV-to-satellite localization and heading estimation
using Bearing-UAV-90K images and a DINOv2-based coarse-to-fine model.

The project is being built in two stages:

1. Convert the original Bearing-UAV-90K layout into `UAV90K`, with one UVP as
   each query and a deduplicated global satellite-tile database.
2. Build a global retrieval and local 3x3 registration network on top of frozen
   DINOv2 features.

Dataset images are intentionally excluded from Git. See
[`docs/UAV90K.md`](docs/UAV90K.md) for the dataset schema and reconstruction
instructions.

The initial trainable model framework is documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). It includes typed UAV90K data
access, a shared frozen DINOv2 wrapper, multi-positive global retrieval, VARRA
local correspondence, and continuous position/heading/confidence heads.

## Stage-one training

Install the package in the selected CUDA environment, then start the initial
positive-candidate training stage:

```bash
pip install -e .
DINOV2_REPO=/root/autodl-tmp/UAV/pretrained/torch_hub/facebookresearch_dinov2_main \
DINOV2_WEIGHTS=/root/autodl-tmp/UAV/pretrained/dinov2_vitb14_pretrain.pth \
python scripts/train.py \
  --dataset /root/autodl-tmp/UAV/UAV90K \
  --output-dir /root/autodl-tmp/UAV/runs/stage1 \
  --batch-size 2
```

The trainer uses a frozen DINOv2 backbone by default and provides validation,
mixed precision, gradient clipping, cosine learning-rate decay, JSONL history,
and atomic latest/best checkpoints. Retrieval positives are grouped by
`gt_tile_id`, so repeated tiles in one batch are not treated as negatives.

Candidate-confidence loss is deliberately disabled in stage one because the
current dataset supplies positive 3x3 candidates only. It will be enabled after
retrieval-mined negative candidates are added.

The fixed five-metric evaluation protocol is defined in
[`docs/EVALUATION.md`](docs/EVALUATION.md) and implemented by
`uavgeo.metrics`.

After stage one, build the satellite index, mine retrieval hard negatives, and
fine-tune candidate confidence:

```bash
python scripts/build_satellite_index.py \
  --dataset /root/autodl-tmp/UAV/UAV90K \
  --checkpoint /root/autodl-tmp/UAV/runs/stage1/best.pt \
  --output /root/autodl-tmp/UAV/UAV90K/features/index/stage1.pt

python scripts/mine_hard_negatives.py \
  --dataset /root/autodl-tmp/UAV/UAV90K \
  --checkpoint /root/autodl-tmp/UAV/runs/stage1/best.pt \
  --index /root/autodl-tmp/UAV/UAV90K/features/index/stage1.pt \
  --output /root/autodl-tmp/UAV/UAV90K/features/index/hard_negatives.csv

python scripts/train.py \
  --dataset /root/autodl-tmp/UAV/UAV90K \
  --output-dir /root/autodl-tmp/UAV/runs/stage2 \
  --init-checkpoint /root/autodl-tmp/UAV/runs/stage1/best.pt \
  --negative-manifest /root/autodl-tmp/UAV/UAV90K/features/index/hard_negatives.csv \
  --negative-probability 0.5 \
  --confidence-weight 0.5 \
  --batch-size 2
```

`--resume` restores an interrupted run including optimizer, scheduler, scaler,
data-loader generator, and random states. `--init-checkpoint` starts a new
training stage from model weights only.
