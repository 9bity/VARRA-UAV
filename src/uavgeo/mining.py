"""Read and write retrieval-mined negative candidate manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence, Union


def write_negative_manifest(
    path: Union[str, Path], candidates: Mapping[str, Sequence[str]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "negative_center_tile_ids")
        )
        writer.writeheader()
        for sample_id in sorted(candidates):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "negative_center_tile_ids": "|".join(candidates[sample_id]),
                }
            )
    temporary.replace(destination)


def read_negative_manifest(path: Union[str, Path]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id in result:
                raise ValueError(f"Duplicate sample in negative manifest: {sample_id}")
            values = tuple(
                item for item in row["negative_center_tile_ids"].split("|") if item
            )
            if values:
                result[sample_id] = values
    return result
