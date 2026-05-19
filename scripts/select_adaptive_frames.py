#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(path):
    names = [
        name for name in os.listdir(path)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    ]
    return sorted(names)


def load_thumb(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    thumb = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return thumb.astype(np.float32) / 255.0


def moving_average(values, window):
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_motion_scores(rgb_dir, names, thumb_width, thumb_height, smooth_window):
    prev = None
    raw = np.zeros(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        current = load_thumb(os.path.join(rgb_dir, name), (thumb_width, thumb_height))
        if prev is not None:
            diff = np.abs(current - prev)
            third = max(1, diff.shape[1] // 3)
            left = float(diff[:, :third].mean())
            right = float(diff[:, -third:].mean())
            center = float(diff[:, third:-third].mean()) if diff.shape[1] > 2 * third else float(diff.mean())
            asymmetry = abs(left - right)
            raw[i] = float(diff.mean()) + 0.55 * asymmetry + 0.20 * abs(center - 0.5 * (left + right))
        prev = current
    return raw, moving_average(raw, smooth_window)


def choose_peaks(smoothed, peak_count, min_peak_gap, percentile):
    if len(smoothed) == 0:
        return []
    threshold = float(np.percentile(smoothed, percentile))
    ranked = np.argsort(smoothed)[::-1]
    peaks = []
    for idx in ranked:
        idx = int(idx)
        if smoothed[idx] < threshold:
            break
        if all(abs(idx - peak) >= min_peak_gap for peak in peaks):
            peaks.append(idx)
            if len(peaks) >= peak_count:
                break
    return sorted(peaks)


def add_interval_frames(candidates, reasons, start, end, step, reason, priority):
    start = max(0, int(start))
    end = int(end)
    step = max(1, int(step))
    first = start - (start % step)
    if first < start:
        first += step
    for idx in range(first, end + 1, step):
        candidates[idx] = max(candidates.get(idx, 0.0), priority)
        reasons[idx].add(reason)


def build_selection(total, smoothed, args):
    candidates = {}
    reasons = defaultdict(set)

    for idx in range(0, total, args.base_interval):
        candidates[idx] = max(candidates.get(idx, 0.0), 1.0)
        reasons[idx].add("base")
    candidates[0] = max(candidates.get(0, 0.0), 1.0)
    candidates[total - 1] = max(candidates.get(total - 1, 0.0), 1.0)
    reasons[0].add("endpoint")
    reasons[total - 1].add("endpoint")

    peaks = choose_peaks(
        smoothed,
        peak_count=args.peak_count,
        min_peak_gap=args.min_peak_gap,
        percentile=args.peak_percentile,
    )
    if len(smoothed) > 0:
        score_min = float(smoothed.min())
        score_span = max(float(smoothed.max()) - score_min, 1e-8)
    else:
        score_min = 0.0
        score_span = 1.0

    for peak in peaks:
        peak_norm = (float(smoothed[peak]) - score_min) / score_span
        add_interval_frames(
            candidates,
            reasons,
            peak - args.dense_radius,
            peak + args.dense_radius,
            args.dense_interval,
            "dense_motion",
            2.0 + peak_norm,
        )
        add_interval_frames(
            candidates,
            reasons,
            peak - args.very_dense_radius,
            peak + args.very_dense_radius,
            args.very_dense_interval,
            "very_dense_motion",
            3.0 + peak_norm,
        )

    always_keep = {idx for idx, why in reasons.items() if "base" in why or "endpoint" in why}
    selected = set(always_keep)
    remaining = [
        idx for idx in candidates
        if idx not in selected and 0 <= idx < total
    ]
    remaining.sort(key=lambda idx: (candidates[idx], smoothed[idx], -idx), reverse=True)
    for idx in remaining:
        if len(selected) >= args.target_count:
            break
        selected.add(idx)

    return sorted(selected), peaks, reasons


def prepare_dir(path, overwrite):
    if os.path.exists(path):
        if not overwrite:
            raise RuntimeError(f"Output path exists; pass --overwrite to replace: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def symlink_selected(src_dir, dst_dir, names, selected):
    os.makedirs(dst_dir, exist_ok=True)
    missing = []
    for idx in selected:
        name = names[idx]
        src = os.path.abspath(os.path.join(src_dir, name))
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        os.symlink(src, dst)
    return missing


def write_metadata(out_dir, names, selected, peaks, raw, smoothed, reasons):
    with open(os.path.join(out_dir, "selected_indices.txt"), "w") as f:
        for idx in selected:
            f.write(f"{idx}\n")

    with open(os.path.join(out_dir, "selected_frames.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "frame_index", "filename", "raw_score", "smooth_score", "reasons"],
        )
        writer.writeheader()
        for rank, idx in enumerate(selected):
            writer.writerow({
                "rank": rank,
                "frame_index": idx,
                "filename": names[idx],
                "raw_score": f"{float(raw[idx]):.8f}",
                "smooth_score": f"{float(smoothed[idx]):.8f}",
                "reasons": "|".join(sorted(reasons[idx])),
            })

    with open(os.path.join(out_dir, "motion_peaks.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["peak_index", "filename", "raw_score", "smooth_score"])
        writer.writeheader()
        for idx in peaks:
            writer.writerow({
                "peak_index": idx,
                "filename": names[idx],
                "raw_score": f"{float(raw[idx]):.8f}",
                "smooth_score": f"{float(smoothed[idx]):.8f}",
            })


def parse_args():
    parser = argparse.ArgumentParser(description="Build an adaptive symlink frame subset for single-scene 3DGS runs.")
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--depth_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_interval", type=int, default=40)
    parser.add_argument("--target_count", type=int, default=240)
    parser.add_argument("--peak_count", type=int, default=12)
    parser.add_argument("--peak_percentile", type=float, default=82.0)
    parser.add_argument("--min_peak_gap", type=int, default=180)
    parser.add_argument("--dense_radius", type=int, default=130)
    parser.add_argument("--dense_interval", type=int, default=10)
    parser.add_argument("--very_dense_radius", type=int, default=45)
    parser.add_argument("--very_dense_interval", type=int, default=5)
    parser.add_argument("--smooth_window", type=int, default=31)
    parser.add_argument("--thumb_width", type=int, default=96)
    parser.add_argument("--thumb_height", type=int, default=72)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    names = list_images(args.rgb_dir)
    if not names:
        raise RuntimeError(f"No images found in {args.rgb_dir}")

    raw, smoothed = compute_motion_scores(
        args.rgb_dir,
        names,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        smooth_window=args.smooth_window,
    )
    selected, peaks, reasons = build_selection(len(names), smoothed, args)

    prepare_dir(args.out_dir, args.overwrite)
    rgb_out = os.path.join(args.out_dir, "rgb")
    depth_out = os.path.join(args.out_dir, "depth")
    symlink_selected(args.rgb_dir, rgb_out, names, selected)
    if args.depth_dir:
        missing_depth = symlink_selected(args.depth_dir, depth_out, names, selected)
    else:
        missing_depth = []
    write_metadata(args.out_dir, names, selected, peaks, raw, smoothed, reasons)

    print(f"Input frames: {len(names)}")
    print(f"Selected frames: {len(selected)}")
    print(f"Base interval: {args.base_interval}")
    print(f"Target count: {args.target_count}")
    print(f"Motion peaks: {', '.join(str(p) for p in peaks)}")
    print(f"RGB symlink dir: {rgb_out}")
    if args.depth_dir:
        print(f"Depth symlink dir: {depth_out}")
    if missing_depth:
        print(f"Missing depth files: {len(missing_depth)}")


if __name__ == "__main__":
    main()
