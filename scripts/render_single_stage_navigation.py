#!/usr/bin/env python3
"""Render reproducible single-stage Bearing-Naver navigation showcases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


BLUE = "#4E87D6"
GREEN = "#35B86B"
WHITE = "#F5F7FA"
MUTED = "#AAB7C8"
PANEL = "#111A2B"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation2")):
        path = root / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_points(route_cfg: dict, episode: dict):
    flat = route_cfg["waypoints_xy_flat"]
    reference = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
    actual = [(float(x["real_x"]), float(x["real_y"])) for x in episode["trajectory"]]
    if actual and actual[0] != reference[0]:
        actual.insert(0, reference[0])
    return reference, actual


def crop_box(image: Image.Image, points, aspect: float, pad_ratio: float = 0.11):
    xs, ys = zip(*points)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0 -= span_x * pad_ratio
    x1 += span_x * pad_ratio
    y0 -= span_y * pad_ratio
    y1 += span_y * pad_ratio
    span_x, span_y = x1 - x0, y1 - y0
    if span_x / span_y < aspect:
        extra = (span_y * aspect - span_x) / 2
        x0 -= extra
        x1 += extra
    else:
        extra = (span_x / aspect - span_y) / 2
        y0 -= extra
        y1 += extra
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    return (int(x0), int(y0), int(x1), int(y1))


def mapped(points, box, origin, size):
    x0, y0, x1, y1 = box
    ox, oy = origin
    w, h = size
    return [(ox + (x - x0) * w / (x1 - x0), oy + (y - y0) * h / (y1 - y0)) for x, y in points]


def render_frame(map_image, box, route_cfg, episode, actual_prefix, width, height):
    canvas = Image.new("RGB", (width, height), PANEL)
    margin = int(width * 0.026)
    panel_w = int(width * 0.265)
    map_w = width - panel_w - margin * 3
    map_h = height - margin * 2
    map_origin = (margin, margin)

    map_view = map_image.crop(box).resize((map_w, map_h), Image.Resampling.LANCZOS)
    canvas.paste(map_view, map_origin)
    draw = ImageDraw.Draw(canvas)
    reference, actual = load_points(route_cfg, episode)
    ref_xy = mapped(reference, box, map_origin, (map_w, map_h))
    act_xy = mapped(actual[:actual_prefix], box, map_origin, (map_w, map_h))

    # Thin dark halos preserve visibility without adding a third route color.
    if len(ref_xy) > 1:
        draw.line(ref_xy, fill="#162033", width=max(8, width // 150), joint="curve")
        draw.line(ref_xy, fill=BLUE, width=max(4, width // 280), joint="curve")
    if len(act_xy) > 1:
        draw.line(act_xy, fill="#162033", width=max(9, width // 135), joint="curve")
        draw.line(act_xy, fill=GREEN, width=max(5, width // 230), joint="curve")

    r = max(6, width // 180)
    for p in ref_xy:
        draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=BLUE, outline=WHITE, width=2)
    if act_xy:
        p = act_xy[0]
        draw.ellipse((p[0]-r-2, p[1]-r-2, p[0]+r+2, p[1]+r+2), fill=GREEN, outline=WHITE, width=2)
        p = act_xy[-1]
        draw.ellipse((p[0]-r-2, p[1]-r-2, p[0]+r+2, p[1]+r+2), fill=GREEN, outline=WHITE, width=3)

    px = map_w + margin * 2
    title = font(max(26, width // 48), True)
    sub = font(max(16, width // 85))
    body = font(max(18, width // 72))
    bold = font(max(20, width // 66), True)
    draw.text((px, margin + 12), "Single-Stage UAV", fill=WHITE, font=title)
    draw.text((px, margin + 12 + title.size + 6), "Global-to-Local", fill=WHITE, font=title)
    draw.text((px, margin + 12 + 2 * (title.size + 6)), "Navigation", fill=WHITE, font=title)
    y = margin + 174
    draw.text((px, y), "Bearing-Naver 2D Closed Loop", fill=MUTED, font=sub)
    y += 64
    draw.text((px, y), f"Route  {episode['route_id']}", fill=WHITE, font=bold)
    y += 38
    draw.text((px, y), f"Map    {route_cfg['city']}", fill=MUTED, font=body)
    y += 72
    draw.line((px, y, px + 58, y), fill=BLUE, width=7)
    draw.text((px + 76, y - 14), "Reference route", fill=WHITE, font=body)
    y += 52
    draw.line((px, y, px + 58, y), fill=GREEN, width=8)
    draw.text((px + 76, y - 14), "Actual trajectory", fill=WHITE, font=body)
    y += 84
    status = "SUCCESS" if episode["success"] else "FAILED"
    draw.text((px, y), status, fill=GREEN if episode["success"] else WHITE, font=title)
    y += 64
    for label, value in (
        ("Navigation error", f"{episode['navigation_error_m']:.2f} m"),
        ("SPL", f"{episode['spl']:.2f}%"),
        ("Steps", str(episode["steps"])),
        ("Reached waypoints", str(episode["reached_waypoints"])),
    ):
        draw.text((px, y), label, fill=MUTED, font=sub)
        draw.text((px, y + 28), value, fill=WHITE, font=bold)
        y += 78
    draw.text((px, height - margin - 54), "Trained single-stage DINOv2 localizer", fill=MUTED, font=sub)
    return canvas


def render_one(args, route_cfg, episode):
    route_id = episode["route_id"]
    map_image = Image.open(args.dataset / "maps" / route_cfg["city"] / "map.jpg").convert("RGB")
    reference, actual = load_points(route_cfg, episode)
    box = crop_box(map_image, reference + actual, aspect=(1920 - 509 - 3 * 50) / (1080 - 2 * 50))
    out = args.output / route_id
    out.mkdir(parents=True, exist_ok=True)

    overview = render_frame(map_image, box, route_cfg, episode, len(actual), 1920, 1080)
    overview.save(out / f"route_{route_id}_overview.png", quality=95)

    video_path = out / f"route_{route_id}_navigation.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (1280, 720))
    gif_frames = []
    n = len(actual)
    frame_indices = [1] * 5 + list(range(2, n + 1)) + [n] * 12
    for idx in frame_indices:
        frame = render_frame(map_image, box, route_cfg, episode, idx, 1280, 720)
        rgb = cv2.cvtColor(__import__("numpy").array(frame), cv2.COLOR_RGB2BGR)
        writer.write(rgb)
        gif_frames.append(frame.resize((960, 540), Image.Resampling.LANCZOS))
    writer.release()
    gif_frames[0].save(out / f"route_{route_id}_navigation.gif", save_all=True,
                       append_images=gif_frames[1:], duration=200, loop=0, optimize=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--route-ids", nargs="+", required=True)
    args = p.parse_args()
    routes = {r["route_id"]: r for r in json.loads(args.routes.read_text())}
    for route_id in args.route_ids:
        episode = json.loads((args.episodes / f"{route_id}.json").read_text())
        render_one(args, routes[route_id], episode)
        print(f"rendered {route_id}", flush=True)


if __name__ == "__main__":
    main()
