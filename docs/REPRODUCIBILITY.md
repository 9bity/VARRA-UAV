# Reproducibility protocol

The single-stage redesign retains the fixed UAV90K split and environment
fingerprints of the preserved baseline. The reference metadata fingerprint is:

```text
fdaf3f74fed5e8fa953152a7d4c7bfbd02f0becdf1ff89a99dace52c3ad4001a
```

The official `dinov2_vitb14_pretrain.pth` SHA256 is:

```text
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

## Deterministic preparation

`prepare_dino_negatives.py` uses a frozen DINOv2 backbone, ordered datasets,
exact cosine search, and stable tile IDs. It has no optimizer and creates no
trained parameters. The produced CSV must be retained with its SHA256; the
trainer records that hash in `reproducibility.json` and every checkpoint.

```bash
python scripts/prepare_dino_negatives.py \
  --dataset /path/to/UAV90K \
  --output /path/to/fixed_dino_negatives.csv \
  --splits train val \
  --search-k 200 \
  --negatives-per-query 8 \
  --batch-size 128 \
  --device cuda --amp
```

The CSV may be distributed directly so a reproducer does not need to repeat
this preprocessing.

## One training command

```bash
python -u scripts/train.py \
  --dataset /path/to/UAV90K \
  --negative-manifest /path/to/fixed_dino_negatives.csv \
  --output-dir /path/to/run \
  --epochs 30 \
  --batch-size 2 \
  --num-workers 4 \
  --learning-rate 3e-4 \
  --seed 42 \
  --deterministic \
  --device cuda --amp
```

This command starts from the public DINOv2 weights and random task heads. It
does not load a first-stage UAV checkpoint. `--resume` is only fault recovery
for the same run and restores model, optimizer, scheduler, scaler, loader, and
random states.

After training, build the satellite index and evaluate. Neither operation
changes model parameters.

## Result status

No experiment has been launched for the single-stage redesign yet. The former
72.01% Recall@1 / 82.73% LSR@15 / 70.16% HSR@15 / 14.10 m MLE / 14.31 deg MHE
numbers belong only to the preserved two-stage baseline and are not claimed by
this branch.
