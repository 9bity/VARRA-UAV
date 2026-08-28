# UAV90K dataset layout

`UAV90K` is a non-destructive reorganization of Bearing-UAV-90K for global
retrieval followed by local 3x3 registration. The model query contains only one
real UAV-view patch (UVP). Satellite tiles belong to an offline map database and
are never selected from ground-truth RSB information at inference time.

## Layout

```text
UAV90K/
  dataset_info.json
  images/
    uav/{city}/...jpg
    satellite/{city}/tile_rXX_cYY.jpg
  maps/
    {city}/map.jpg
    {city}/map.json
  metadata/
    maps.csv
    satellite_tiles.csv
    neighborhoods.csv
    samples.csv
    splits/
      train.txt
      val.txt
      test.txt
  features/
    satellite/
    index/
```

The image and map files are created as NTFS hard links by default. They occupy
no second copy of the image data and do not modify the source dataset. Generated
images, feature files, and the 90K-row sample manifest are excluded from Git.

## Satellite database

Each source city contains 15 x 15 RSBs. Every RSB stores four RST files, so the
same physical RST is repeated under neighboring block names. `UAV90K` verifies
those duplicate files and converts them into one unique 16 x 16 grid per city:

- 256 unique satellite tiles per city;
- 1,024 unique satellite tiles across four cities;
- one explicit 3 x 3 neighborhood row for every tile.

The 3 x 3 mosaics are assembled at load time. They are not stored as additional
images.

## Query samples

Only the real 3D-rendered UAV image (`target_patch_3d`, filename suffix
`_v3d.jpg`) is used as the query. The 2D satellite-derived `target_path` image is
not copied into the new query set.

Important `samples.csv` fields:

| Field | Meaning |
|---|---|
| `sample_id` | Stable city-qualified query identifier |
| `split` | Deterministic train/val/test assignment |
| `uav_path` | Query path relative to `UAV90K` |
| `global_x`, `global_y` | Continuous pixel position in the 4096 x 4096 city map |
| `latitude`, `longitude` | Geographic ground truth recovered from the map metadata |
| `gt_tile_id` | Unique satellite tile containing the query center |
| `tile_offset_x`, `tile_offset_y` | Continuous position in the GT tile, in [0, 1] |
| `heading_deg` | Heading normalized into [0, 360) |
| `heading_cos`, `heading_sin` | Circular heading target used by Bearing-UAV metrics |

The original `block_x`, `block_y`, `x_norm`, `y_norm`, and Cartesian-coordinate
labels are retained for reproducibility.

## Rebuild

```powershell
python scripts/build_uav90k.py `
  --source D:\Bearing-UAV\bearinguav\Bearing_UAV_90K `
  --output D:\paper\UAV\UAV90K `
  --link-mode hardlink
```

The conversion is idempotent: an interrupted run can be resumed. Existing
destination files are validated rather than overwritten.

