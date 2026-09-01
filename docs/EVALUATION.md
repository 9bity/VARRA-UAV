# Evaluation protocol

All experiments must evaluate the fixed `UAV90K/metadata/splits/test.txt`
manifest. The primary result row contains:

| Recall@1 up | LSR@15 up | HSR@15 up | MLE down | MHE down |
|---:|---:|---:|---:|---:|

- **Recall@1** is the percentage of queries whose **final** predicted position,
  after Top-K retrieval, 3x3 expansion, VARRA matching, and candidate
  reranking, maps to the same satellite `tile_id` as the ground truth. This is
  the global-tile analogue of BearingUAV's final position-region recall; it is
  not the first-stage retrieval recall.
- **Coarse retrieval Recall@1** is written to the JSON result only as a
  diagnostic field named `coarse_retrieval_recall_at_1`. It is not one of the
  five primary paper metrics.
- **LSR@15** is the percentage of final continuous position predictions whose
  Euclidean map error is at most 15 meters.
- **HSR@15** is the percentage of heading predictions whose shortest circular
  error is at most 15 degrees.
- **MLE** defaults to the Bearing-compatible same-city conditional mean in
  meters. Position errors are averaged among correct-city predictions within
  each city and the four city means are then averaged. Cross-city predictions
  remain failures for LSR and are reported separately as a cross-city failure
  rate. This matches Bearing-UAV's local-map precision setting, where the input
  RSB already fixes the map. Use `--mle-protocol global-geodesic` to report the
  stricter end-to-end mean in which cross-city errors use great-circle distance.
- **MHE** is mean shortest circular heading error in degrees over all queries.

The four UAV90K maps have 0.25 meters per pixel in both axes. Evaluation still
reads or passes the map scale explicitly so the implementation remains correct
if maps with a different ground sampling distance are introduced later.

Pixel coordinate systems are local to each city and are never numerically
compared across cities. LSR always treats a cross-city prediction using its
great-circle error. The primary Bearing-compatible MLE measures conditional
local precision and reports cross-city failures beside it; the optional global
protocol includes those great-circle errors in MLE as well.

The metric implementation is centralized in `uavgeo.metrics`. Training,
validation, ablation, and final test scripts must call this implementation
instead of maintaining separate formulas.
