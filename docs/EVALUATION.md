# Evaluation protocol

All experiments must evaluate the fixed `UAV90K/metadata/splits/test.txt`
manifest. The primary result row contains:

| Recall@1 up | LSR@15 up | HSR@15 up | MLE down | MHE down |
|---:|---:|---:|---:|---:|

- **Recall@1** is the percentage of queries whose first globally retrieved
  satellite tile has the same `tile_id` as the ground-truth tile.
- **LSR@15** is the percentage of final continuous position predictions whose
  Euclidean map error is at most 15 meters.
- **HSR@15** is the percentage of heading predictions whose shortest circular
  error is at most 15 degrees.
- **MLE** is mean location error in meters. To reproduce Bearing-UAV's
  multi-map aggregation, errors are averaged within each city and the four city
  means are then averaged.
- **MHE** is mean shortest circular heading error in degrees over all queries.

The four UAV90K maps have 0.25 meters per pixel in both axes. Evaluation still
reads or passes the map scale explicitly so the implementation remains correct
if maps with a different ground sampling distance are introduced later.

Pixel coordinate systems are local to each city. When the global retriever
selects the wrong city, the evaluator therefore uses the predicted and target
latitude/longitude to compute great-circle distance; it never compares the two
cities' coincident pixel coordinates. Same-city errors retain the original
map-scale computation.

The metric implementation is centralized in `uavgeo.metrics`. Training,
validation, ablation, and final test scripts must call this implementation
instead of maintaining separate formulas.
