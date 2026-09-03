# CMCT-Naver: navigation-only temporal fusion

CMCT-Naver is an independent comparison runner placed after the frozen
single-stage localizer. It does not modify DINOv2, retrieval, 3x3 expansion,
VARRA, prediction heads, weights, or the original Bearing-Naver-compatible
runner in `uavgeo.navigation`.

At frame `t`, the unchanged localizer produces one position, circular heading,
and selected-candidate quality. CMCT predicts a pose prior from the previous
filtered state and the previous commanded displacement, measures position and
heading innovation, and computes a confidence-and-motion consistency score.
Consistent observations update the temporal state; isolated jumps are rejected.
Several mutually consistent high-confidence rejected observations can trigger
hysteretic relocalization.

The controller uses the filtered position to point toward the same waypoint.
The default comparison deliberately retains the baseline's fixed 25 m step,
20 m arrival radius, maximum 384 steps, route definitions, waypoint policy,
observation renderer, checkpoint, index, and validation-locked localizer
settings. This isolates temporal navigation from localization changes.

## Scientific scope

The public evaluator is a deterministic 2D closed loop using rotated satellite
map crops. Commanded displacement is therefore exact in the simulator, whereas
real flight has actuation error. Report this limitation explicitly and do not
describe the result as continuous Google Earth 3D UAV-view navigation.

The fixed defaults in `configs/cmct_naver_default.json` were declared before
the eight navigation test routes were run. Do not tune them on those routes.
