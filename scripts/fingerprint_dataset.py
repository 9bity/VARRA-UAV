"""Print the deterministic fingerprint of a prepared UAV90K dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavgeo.training import dataset_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dataset_fingerprint(args.dataset), indent=2))


if __name__ == "__main__":
    main()
