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

