# Frozen single-stage experiment

This directory records the completed `single_stage_seed42` configuration.
The model is trained once; index construction and evaluation do not update its
parameters.

- Dataset split seed: 42
- Training epochs: 30
- Batch size: 16
- DINOv2 backbone: ViT-B/14, frozen
- Dataset fingerprint: `fdaf3f74fed5e8fa953152a7d4c7bfbd02f0becdf1ff89a99dace52c3ad4001a`
- Source code revision used by the original run: `10346530186b21dd8ef73a03a5095b869f37861c`

`train.json` records optimization/model settings. `evaluation.json` records the
validation-selected final reranking settings and immutable dataset/weight
hashes. Paths are intentionally omitted because they are machine-specific.
