# Network architecture

The initial model is deliberately split into independently testable stages.

```text
single UVP -> shared frozen DINOv2 -> global descriptor -> Top-K retrieval
                                      dense tokens -----------+
                                                               |
Top-K tile -> 3x3 expansion -> shared DINOv2 -> dense tokens --+
                                                               v
                view-adaptive reciprocal geometric attention (VARRA)
                                      |
                      position + heading + confidence
```

## Backbone

The default backbone is DINOv2 ViT-B/14. It is frozen in the first training
stage. A single forward pass exposes both the normalized CLS descriptor and the
dense patch-token field. Inputs are resized to dimensions divisible by 14:

- query UVP: 252 x 252, producing an 18 x 18 token field;
- each satellite tile: 252 x 252, producing an 18 x 18 token field;
- nine tile-token fields are stitched spatially into one 54 x 54 field.

The nine tiles are encoded independently as one batched DINOv2 call. Compared
with sending a single 756 x 756 image through ViT, this preserves the same
ground sampling distance and final token coverage while reducing the quadratic
backbone-attention cost by roughly nine times. It also allows satellite token
fields to be cached per tile.

## Retrieval

The global retrieval head has separate zero-initialized residual adapters for
UVP and satellite descriptors, followed by a shared semantic projection. Its
L2-normalized output supports cosine search. `ExactSatelliteIndex` is the
reference implementation; a FAISS backend can replace it without changing the
model interface.

## Local registration and VARRA

VARRA receives dense DINOv2 token fields from both views:

1. view-specific residual adapters align UVP and satellite distributions;
2. UVP-to-satellite and satellite-to-UVP attentions are computed independently;
3. the geometric mean retains mutually supported correspondences;
4. soft correspondences estimate a differentiable 2D similarity transform;
5. the estimated rotation, scale, and translation gate inconsistent matches;
6. the refined correspondence field produces the satellite heatmap.

The final localizer predicts continuous mosaic-normalized position, a normalized
`(cos(theta), sin(theta))` heading vector, and candidate confidence. Global pixel
coordinates are recovered from the candidate mosaic origin.

## Training behavior of 3x3 candidates

For a query with ground-truth tile `g`, any tile whose 3x3 neighborhood contains
`g` is a valid positive center. `LocalRegistrationDataset` changes that center
deterministically each epoch. This prevents the model from learning the invalid
shortcut that the target is always in the central tile.

Negative and retrieval-mined candidates will be introduced in the training
pipeline after the model-level tensor contracts are verified.

The provided loss module combines multi-positive contrastive retrieval,
continuous position regression, continuous-point heatmap likelihood, circular
heading similarity, and optional candidate-confidence supervision. The
multi-positive mask is important because several UVPs in a batch can correspond
to the same satellite tile and must not be treated as mutual negatives.

During joint training, the complete forward contract is intentionally explicit:

```text
query UVP + positive single satellite tile + candidate 3x3 satellite tile grid
```

The single tile supervises the global retrieval descriptor. The 3x3 tile grid is
used only by VARRA and the local prediction heads. At inference time, only the
UVP is provided by the caller; single tiles and mosaics are fetched from the
offline satellite database after Top-K search.
