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
