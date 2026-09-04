# Single-stage network architecture

## Design boundary

The redesign preserves the original proposal:

1. accept one UVP and no RSB prior;
2. retrieve satellite tiles from a global DINOv2 index;
3. expand each retrieved center to its 3x3 neighborhood;
4. compare UVP and satellite dense features with cross-view attention;
5. predict global position and heading.

Only the candidate-scoring architecture and supervision path change. There is
no trained-model-dependent mining stage.

## Inference

```text
                              OFFLINE DATABASE
satellite tiles -> shared DINOv2 -> satellite adapter -> learned descriptors
                                                        -> cosine index

                                ONLINE QUERY
single UVP -> shared DINOv2 -> UVP adapter -> query descriptor
                                           -> Top-K global retrieval
                                           -> each center expands to 3x3

UVP 18x18 tokens -----------------------------+
                                                -> VARRA -> heatmap
3x3 satellite tiles -> DINOv2 -> 54x54 tokens -+          -> position
                                                           -> heading
                                                           -> quality
```

The final candidate score fuses its global cosine similarity and learned local
quality. The selected local coordinate is converted to a global map coordinate
with the candidate mosaic origin.

## Shared DINOv2 backbone

DINOv2 ViT-B/14 is shared by both views. A 252x252 image produces an 18x18
patch-token grid. Nine satellite tiles are encoded independently and stitched
into a 54x54 feature grid, preserving ground sampling distance while avoiding
the quadratic cost of one 756x756 Transformer input.

## Global cross-view retrieval

UVP and satellite CLS tokens pass through view-specific residual adapters and a
shared semantic projector. L2-normalized descriptors are trained with symmetric
multi-positive InfoNCE. Queries sharing one ground-truth tile are positives,
not false in-batch negatives.

## VARRA local registration

VARRA applies view-specific token adapters, bidirectional UVP-to-satellite and
satellite-to-UVP attention, reciprocal agreement, and a differentiable 2D
similarity transform. Its output contains:

- a satellite localization heatmap;
- a geometric heading vector;
- scale, rotation, and translation;
- mean reciprocal semantic agreement;
- geometric reprojection residual;
- cross-view fused UVP and satellite features.

## Geometric candidate-quality head

The former feature-only confidence head is replaced by a quality head. It uses
pooled cross-view features together with heatmap peak, heatmap entropy,
reciprocal agreement, and geometric residual. This makes reranking depend on
observable semantic and geometric consistency.

For every training query, the head receives one positive and one fixed hard
negative candidate. It learns both calibrated binary quality and a direct
positive-over-negative margin. At inference its logit remains compatible with
the existing global/local score fusion interface.

## One continuous training run

Before training, the frozen official DINOv2 backbone creates a deterministic
nearest-neighbor manifest. Candidates whose 3x3 area contains the query's true
tile are excluded. This operation has no optimizer and produces no checkpoint;
it is equivalent to preparing metadata.

Each training item contains:

```text
UVP
+ exact positive satellite tile       -> retrieval supervision
+ positive 3x3 candidate              -> position/heatmap/heading supervision
+ fixed DINOv2 hard-negative candidate -> quality supervision
```

The model extracts the UVP once, reuses it for both candidates, and optimizes:

```text
L = wr * Lretrieval
  + wp * Lposition
  + wm * Lheatmap
  + wh * Lheading
  + wq * (Lquality-BCE + Lquality-rank)
```

All task heads are active from epoch one. There is one optimizer, one learning
rate schedule, and one checkpoint sequence. `--resume` restores an interrupted
run and is not a second training stage.

## Post-training index

After the single training run, all satellite tiles are encoded once with the
final learned retrieval head. Building this index and running evaluation are
inference operations, not additional training.
