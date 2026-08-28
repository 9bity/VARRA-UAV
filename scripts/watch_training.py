"""Display a live progress bar for a running UAVGeo training log."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


TRAIN_PATTERN = re.compile(
    r"train step=(?P<step>\d+) samples=(?P<samples>\d+) loss=(?P<loss>[\d.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "calculating"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    return f"{minutes:d}m {seconds:02d}s"


def render(
    epoch: int,
    step: int,
    loss: float,
    epochs: int,
    steps_per_epoch: int,
    steps_per_second: float | None,
) -> None:
    epoch_fraction = min(step / steps_per_epoch, 1.0)
    completed_steps = (epoch - 1) * steps_per_epoch + step
    total_steps = epochs * steps_per_epoch
    overall_fraction = min(completed_steps / total_steps, 1.0)
    filled = round(30 * epoch_fraction)
    bar = "#" * filled + "-" * (30 - filled)
    remaining = total_steps - completed_steps
    eta = (
        remaining / steps_per_second
        if steps_per_second is not None and steps_per_second > 0
        else float("inf")
    )
    speed = f"{steps_per_second:.2f} step/s" if steps_per_second else "warming up"
    print(
        f"\rEpoch {epoch:02d}/{epochs} [{bar}] "
        f"{step:4d}/{steps_per_epoch} ({epoch_fraction:6.2%}) | "
        f"overall {overall_fraction:6.2%} | loss {loss:.6f} | "
        f"{speed} | ETA {format_duration(eta):>11}",
        end="",
        flush=True,
    )


def scan_history(path: Path) -> tuple[int, int, float]:
    epoch = 1
    step = 0
    loss = float("nan")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRAIN_PATTERN.search(line)
        if match:
            step = int(match.group("step"))
            loss = float(match.group("loss"))
            continue
        if line.startswith('{"epoch"'):
            record = json.loads(line)
            epoch = int(record["epoch"]) + 2
            step = 0
    return epoch, step, loss


def main() -> None:
    args = parse_args()
    while not args.log.exists():
        print(f"Waiting for {args.log} ...", end="\r", flush=True)
        time.sleep(1)

    epoch, step, loss = scan_history(args.log)
    render(epoch, step, loss, args.epochs, args.steps_per_epoch, None)
    previous_step = step
    previous_time = time.monotonic()
    smoothed_speed: float | None = None

    with args.log.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(0, 2)
        while epoch <= args.epochs:
            line = stream.readline()
            if not line:
                time.sleep(0.25)
                continue
            match = TRAIN_PATTERN.search(line)
            if match:
                new_step = int(match.group("step"))
                now = time.monotonic()
                if new_step > previous_step and now > previous_time:
                    instantaneous = (new_step - previous_step) / (now - previous_time)
                    smoothed_speed = (
                        instantaneous
                        if smoothed_speed is None
                        else 0.8 * smoothed_speed + 0.2 * instantaneous
                    )
                step = new_step
                loss = float(match.group("loss"))
                previous_step = step
                previous_time = now
                render(
                    epoch,
                    step,
                    loss,
                    args.epochs,
                    args.steps_per_epoch,
                    smoothed_speed,
                )
            elif line.startswith('{"epoch"'):
                record = json.loads(line)
                completed_epoch = int(record["epoch"]) + 1
                print(f"\nEpoch {completed_epoch:02d} finished; checkpoint saved.")
                epoch = completed_epoch + 1
                step = 0
                previous_step = 0
                previous_time = time.monotonic()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped watching; server training is still running.")
