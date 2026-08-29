"""Training utilities shared by command-line entry points and tests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and configure repeatable CUDA behavior.

    Exact bitwise equality is only expected with the same hardware and software
    stack. Across compatible CUDA GPUs this configuration targets statistically
    equivalent metrics rather than identical checkpoint bytes.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    # Some PyTorch/CUDA releases do not provide a deterministic backward for
    # grid_sample, which is required by the heatmap loss. Warn instead of
    # making an otherwise reproducible run fail on those releases.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _git_revision(path: Optional[Path] = None) -> Optional[str]:
    try:
        command = ["git"]
        if path is not None:
            command.extend(("-C", str(path)))
        command.extend(("rev-parse", "HEAD"))
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_manifest(seed: int, deterministic: bool) -> dict[str, Any]:
    """Capture the software and accelerator state needed to reproduce a run."""

    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "git_revision": _git_revision(),
    }


def repository_revision(path: Path) -> Optional[str]:
    """Return a repository revision when the configured source has Git metadata."""

    return _git_revision(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(root: Path) -> dict[str, Any]:
    """Hash task-defining metadata without hashing the duplicated image corpus."""

    relative_paths = (
        "dataset_info.json",
        "metadata/maps.csv",
        "metadata/satellite_tiles.csv",
        "metadata/neighborhoods.csv",
        "metadata/samples.csv",
        "metadata/splits/train.txt",
        "metadata/splits/val.txt",
        "metadata/splits/test.txt",
    )
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Dataset fingerprint input is missing: {path}")
        digest = _file_sha256(path)
        files[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {"sha256": combined.hexdigest(), "files": files}


def build_multi_positive_mask(
    query_tile_ids: Sequence[str], reference_tile_ids: Sequence[str]
) -> Tensor:
    """Mark every query/reference pair that belongs to the same satellite tile."""

    if not query_tile_ids or not reference_tile_ids:
        raise ValueError("Tile ID collections must not be empty")
    return torch.tensor(
        [
            [query_id == reference_id for reference_id in reference_tile_ids]
            for query_id in query_tile_ids
        ],
        dtype=torch.bool,
    )


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
