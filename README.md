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

## One-run training

```bash
python -m pip install -e .

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
  --batch-size 2
```

`train.py` starts from DINOv2 plus randomly initialized task heads and performs
one optimizer/scheduler run. `--resume` only continues an interrupted run; it
does not define a second training stage.

After training, build the final satellite index and evaluate:

```bash
python scripts/build_satellite_index.py \
  --dataset /path/to/UAV90K \
  --checkpoint /path/to/run/best.pt \
  --output /path/to/run/satellite_index.pt

python scripts/evaluate.py \
  --dataset /path/to/UAV90K \
  --checkpoint /path/to/run/best.pt \
  --index /path/to/run/satellite_index.pt \
  --output-dir /path/to/run/eval \
  --split test \
  --top-k 5 \
  --confidence-weight 0.5 \
  --mle-protocol bearing-compatible
```

No training has been run for this redesign yet, so the former two-stage metrics
must not be treated as results of this code.
