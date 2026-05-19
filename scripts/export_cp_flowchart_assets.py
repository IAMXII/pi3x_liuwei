import argparse
import csv
import gc
import math
import os
import re
import sys
from contextlib import nullcontext

import torch
_CUDA_PREFLIGHT_AVAILABLE = torch.cuda.is_available()
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from export_cp_local_competition_paper_figs import (
    alpha_vis,
    gather_gaussians,
    select_indices_by_opacity,
)
from export_cp_quadtree_gaussian_paper_figs import (
    axis_limits,
    blend_heatmap,
    colorize_scalar,
    compute_target_size,
    gaussian_density_from_support,
    load_frame_range as load_numeric_frame_range,
    render_alpha_maps,
    resolve_checkpoint_path,
    robust_limits,
    safe_name,
    save_contact_sheet,
    save_panel,
    save_rgb,
)
from pi3.models.pi3_3dgs_8 import Pi3_3DGS


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_image_files(image_dir):
    if not os.path.isdir(image_dir):
        return []
    files = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    ]
    return sorted(files, key=natural_sort_key)


def load_indexed_image_range(image_dir, frame_start, frame_end, frame_step, pixel_limit):
    if frame_step <= 0:
        raise ValueError(f"frame_step must be positive, got {frame_step}")

    files = list_image_files(image_dir)
    if not files:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    total = len(files)
    if frame_start < 0:
        raise ValueError(f"frame_start must be non-negative for indexed image folders, got {frame_start}")
    if frame_start >= total:
        raise ValueError(f"frame_start {frame_start} is outside indexed image folder with {total} images")

    last = min(frame_end, total - 1)
    if last < frame_start:
        raise ValueError(f"frame_end {frame_end} is before frame_start {frame_start}")

    selected_indices = list(range(frame_start, last + 1, frame_step))
    sources = []
    items = []
    for source_index in selected_indices:
        path = files[source_index]
        img = Image.open(path).convert("RGB")
        sources.append(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        items.append(
            {
                "frame_id": source_index,
                "stem": stem,
                "path": path,
                "source_index": source_index,
                "source_name": os.path.basename(path),
            }
        )

    orig_w, orig_h = sources[0].size
    target_w, target_h = compute_target_size(orig_w, orig_h, pixel_limit)
    tensors = []
    resized_rgb = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        resized_rgb.append((arr * 255.0).round().astype(np.uint8))

    return torch.stack(tensors, dim=0), resized_rgb, items, (orig_h, orig_w), (target_h, target_w)


def load_flowchart_frame_range(data_root, frame_start, frame_end, frame_step, pixel_limit):
    rgb_dir = os.path.join(data_root, "rgb")
    if os.path.isdir(rgb_dir):
        return load_numeric_frame_range(data_root, frame_start, frame_end, frame_step, pixel_limit)

    if list_image_files(data_root):
        return load_indexed_image_range(data_root, frame_start, frame_end, frame_step, pixel_limit)

    images_dir = os.path.join(data_root, "images")
    if list_image_files(images_dir):
        return load_indexed_image_range(images_dir, frame_start, frame_end, frame_step, pixel_limit)

    raise FileNotFoundError(f"No supported image layout found under {data_root}")


def frame_file_tag(item):
    if "source_index" in item:
        return safe_name(f"idx{int(item['source_index']):04d}_{item['stem']}")
    return safe_name(item["stem"])


def frame_display_label(item):
    if "source_index" in item:
        return f"idx{int(item['source_index']):04d} {item['stem']}"
    return f"{int(item['frame_id']):06d}"


def crop_array(arr, box):
    y0, y1, x0, x1 = box
    return arr[y0:y1, x0:x1]


def choose_crop_box(score_map, crop_size_ratio=0.38):
    score = np.asarray(score_map, dtype=np.float32)
    h, w = score.shape
    crop_h = max(96, min(h, int(round(h * crop_size_ratio))))
    crop_w = max(96, min(w, int(round(w * crop_size_ratio))))
    blur_kh = max(9, (crop_h // 4) | 1)
    blur_kw = max(9, (crop_w // 4) | 1)
    smooth = cv2.blur(score, (blur_kw, blur_kh))
    cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    y0 = int(np.clip(cy - crop_h // 2, 0, max(h - crop_h, 0)))
    x0 = int(np.clip(cx - crop_w // 2, 0, max(w - crop_w, 0)))
    return y0, y0 + crop_h, x0, x0 + crop_w


def draw_crop_box(image_rgb, box, color=(255, 236, 80), thickness=3):
    y0, y1, x0, x1 = box
    out = np.asarray(image_rgb, dtype=np.uint8).copy()
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
    return out


def normalize01(arr, low=1, high=99):
    lo, hi = robust_limits(arr, low=low, high=high)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def compute_complexity_maps(frame_rgb, depth):
    rgb_float = np.asarray(frame_rgb, dtype=np.float32) / 255.0
    local_mean = cv2.blur(rgb_float, (7, 7), borderType=cv2.BORDER_REPLICATE)
    color_complexity = np.abs(rgb_float - local_mean).mean(axis=2)
    color_complexity = color_complexity / (float(color_complexity.max()) + 1e-8)

    depth_f = np.asarray(depth, dtype=np.float32)
    depth_mean = cv2.blur(depth_f, (9, 9), borderType=cv2.BORDER_REPLICATE)
    plane_residual = np.abs(depth_f - depth_mean) / np.clip(np.abs(depth_mean), 1e-4, None)
    curvature = np.abs(cv2.Laplacian(depth_f, cv2.CV_32F, ksize=3)) / np.clip(np.abs(depth_f), 1e-4, None)
    clip_val = max(float(np.percentile(np.maximum(plane_residual, curvature), 98)), 1e-6)
    plane_residual = np.clip(plane_residual, 0.0, clip_val) / clip_val
    curvature = np.clip(curvature, 0.0, clip_val) / clip_val
    geometry_complexity = np.maximum(plane_residual, curvature)
    return local_mean, color_complexity, plane_residual, curvature, geometry_complexity


def draw_quadtree_partition(base_rgb, scale_map, quad_mask, min_block=2, crop_box=None):
    if crop_box is not None:
        base_rgb = crop_array(base_rgb, crop_box)
        scale_map = crop_array(scale_map, crop_box)
        quad_mask = crop_array(quad_mask, crop_box)
        y_shift, _, x_shift, _ = crop_box
    else:
        y_shift = 0
        x_shift = 0

    out = np.asarray(base_rgb, dtype=np.float32).copy()
    out = (out * 0.70).astype(np.uint8)
    h, w = scale_map.shape
    line_colors = {
        32: (255, 255, 255),
        16: (230, 238, 255),
        8: (180, 224, 255),
        4: (120, 210, 255),
        2: (80, 180, 255),
        1: (250, 168, 64),
    }
    for size in (32, 16, 8, 4, 2, 1):
        anchors = np.argwhere((quad_mask > 0) & (np.rint(scale_map).astype(np.int32) == size))
        if size < min_block:
            continue
        for y, x in anchors:
            x_global = x + x_shift
            y_global = y + y_shift
            half = max(1, size // 2)
            x0 = int(np.clip(x_global - half - x_shift, 0, w - 1))
            y0 = int(np.clip(y_global - half - y_shift, 0, h - 1))
            x1 = int(np.clip(x_global + half - x_shift, 0, w - 1))
            y1 = int(np.clip(y_global + half - y_shift, 0, h - 1))
            cv2.rectangle(out, (x0, y0), (x1, y1), line_colors[size], 1, lineType=cv2.LINE_AA)
    return out


def draw_anchor_retention(base_rgb, scale_map, quad_mask, low_conf_mask=None, crop_box=None, max_points=4500):
    out = draw_quadtree_partition(base_rgb, scale_map, quad_mask, min_block=2, crop_box=crop_box)
    if crop_box is not None:
        scale_map = crop_array(scale_map, crop_box)
        quad_mask = crop_array(quad_mask, crop_box)
        if low_conf_mask is not None:
            low_conf_mask = crop_array(low_conf_mask, crop_box)
    anchors = np.argwhere(quad_mask > 0)
    if anchors.shape[0] > max_points:
        rng = np.random.default_rng(7)
        anchors = anchors[rng.choice(anchors.shape[0], size=max_points, replace=False)]
    for y, x in anchors:
        s = int(max(1, round(float(scale_map[y, x]))))
        radius = max(1, min(5, s // 5 + 1))
        cv2.circle(out, (int(x), int(y)), radius + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    if low_conf_mask is not None:
        lows = np.argwhere(low_conf_mask > 0)
        if lows.shape[0] > 1400:
            rng = np.random.default_rng(11)
            lows = lows[rng.choice(lows.shape[0], size=1400, replace=False)]
        for y, x in lows:
            cv2.circle(out, (int(x), int(y)), 2, (230, 68, 68), -1, lineType=cv2.LINE_AA)
    return out


def draw_dense_gaussian_overlay(base_rgb, score_map=None, crop_box=None, max_points=18000):
    if crop_box is not None:
        base_rgb = crop_array(base_rgb, crop_box)
        if score_map is not None:
            score_map = crop_array(score_map, crop_box)

    base = (np.asarray(base_rgb, dtype=np.float32) * 0.58).astype(np.uint8)
    overlay = base.copy()
    h, w = base.shape[:2]
    total = max(1, h * w)
    stride = max(1, int(math.ceil(math.sqrt(total / max(max_points, 1)))))
    y_coords = np.arange(stride // 2, h, stride, dtype=np.int32)
    x_coords = np.arange(stride // 2, w, stride, dtype=np.int32)

    if score_map is not None:
        score_norm = normalize01(score_map, low=1, high=99)
    else:
        score_norm = np.zeros((h, w), dtype=np.float32)

    # Draw a decimated dense lattice; rendering every pixel would collapse into a flat wash.
    radius = 1 if stride <= 3 else 2
    rng = np.random.default_rng(17)
    for y in y_coords:
        for x in x_coords:
            t = float(score_norm[y, x])
            major = int(np.clip(radius + 1 + 2.0 * t, 2, 5))
            minor = int(np.clip(radius + 0.8 * t, 1, 3))
            angle = float(rng.uniform(-65, 65))
            color = (
                int(round(82 + 176 * t)),
                int(round(192 - 54 * t)),
                int(round(224 - 146 * t)),
            )
            cv2.ellipse(overlay, (int(x), int(y)), (major, minor), angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.74, base, 0.26, 0)


def draw_gaussian_ellipses(base_rgb, scale_map, support_mask, crop_box=None, max_ellipses=1400):
    if crop_box is not None:
        base_rgb = crop_array(base_rgb, crop_box)
        scale_map = crop_array(scale_map, crop_box)
        support_mask = crop_array(support_mask, crop_box)
    out = (np.asarray(base_rgb, dtype=np.float32) * 0.58).astype(np.uint8)
    overlay = out.copy()
    anchors = np.argwhere(support_mask > 0)
    if anchors.shape[0] > max_ellipses:
        scales = scale_map[support_mask > 0].reshape(-1)
        weights = np.clip(scales, 1, 32).astype(np.float64)
        weights = weights / weights.sum()
        rng = np.random.default_rng(13)
        chosen = rng.choice(anchors.shape[0], size=max_ellipses, replace=False, p=weights)
        anchors = anchors[chosen]
    rng = np.random.default_rng(23)
    for y, x in anchors:
        s = float(max(1.0, scale_map[y, x]))
        major = int(np.clip(2.0 + 0.55 * s, 2, 22))
        minor = int(np.clip(1.5 + 0.34 * s, 1, 14))
        angle = float(rng.uniform(-70, 70))
        color = (222, 161, 90) if s > 4 else (247, 182, 80)
        cv2.ellipse(overlay, (int(x), int(y)), (major, minor), angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (int(x), int(y)), (major, minor), angle, 0, 360, (72, 42, 25), 1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.78, out, 0.22, 0)


def draw_candidate_ellipses_on_crop(frame_crop, support_crop, scale_crop, color, max_ellipses=48):
    out = np.asarray(frame_crop, dtype=np.uint8).copy()
    overlay = out.copy()
    anchors = np.argwhere(support_crop > 0)
    if anchors.shape[0] == 0:
        return out
    scales = scale_crop[support_crop > 0].reshape(-1)
    weights = np.clip(scales, 1, 32).astype(np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(37)
    n = min(max_ellipses, anchors.shape[0])
    anchors = anchors[rng.choice(anchors.shape[0], size=n, replace=False, p=weights)]
    for y, x in anchors:
        s = max(1.0, float(scale_crop[y, x]))
        major = int(np.clip(3 + 0.40 * s, 3, 16))
        minor = int(np.clip(2 + 0.24 * s, 2, 10))
        angle = float(rng.uniform(-75, 75))
        cv2.ellipse(overlay, (int(x), int(y)), (major, minor), angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (int(x), int(y)), (major, minor), angle, 0, 360, (20, 20, 20), 1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.64, out, 0.36, 0)


def save_multi_view_candidate_panel(view_rows, path):
    fig, axes = plt.subplots(len(view_rows), 2, figsize=(5.2, 2.15 * len(view_rows)))
    if len(view_rows) == 1:
        axes = axes.reshape(1, -1)
    for r, row in enumerate(view_rows):
        label, crop, candidates = row
        axes[r, 0].imshow(crop)
        axes[r, 0].set_title(f"{label} image patch", fontsize=9)
        axes[r, 0].axis("off")
        axes[r, 1].imshow(candidates)
        axes[r, 1].set_title("candidate Gaussians", fontsize=9)
        axes[r, 1].axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_common_3d_space(gaussians, path, max_points=70000, seed=0):
    xyz = gaussians["xyz"][0].numpy()
    opacity = gaussians["opacity"][0, :, 0].numpy()
    color = gaussians["color"][0].numpy()
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity) & (opacity > 0.01)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return False
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    points = xyz[idx]
    colors = np.clip(color[idx], 0, 1)
    sizes = 0.5 + 5.0 * normalize01(opacity[idx], low=5, high=95)

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=sizes, c=colors, alpha=0.28, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(axis_limits(points, 0))
    ax.set_ylim(axis_limits(points, 1))
    ax.set_zlim(axis_limits(points, 2))
    ax.view_init(elev=24, azim=-52)
    ax.set_title("Unified 3D Gaussian candidates", fontsize=11)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def select_competition_cluster(gauss_before, gate_proxy, max_members=14):
    xyz = gauss_before["xyz"][0].numpy()
    color = gauss_before["color"][0].numpy()
    opacity = gauss_before["opacity"][0, :, 0].numpy()
    suppression = 1.0 - np.clip(gate_proxy, 0, 1)
    valid = np.isfinite(xyz).all(axis=1) & (opacity > 0.02)
    if not np.any(valid):
        valid = np.isfinite(xyz).all(axis=1)
    center_idx = np.argmax(np.where(valid, suppression, -1.0))
    center = xyz[center_idx]
    center_color = color[center_idx]
    dist = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)
    color_dist = np.linalg.norm(color - center_color.reshape(1, 3), axis=1)
    candidates = np.nonzero(valid & (color_dist < 0.35))[0]
    if candidates.size == 0:
        candidates = np.nonzero(valid)[0]
    for radius in (5, 10, 20, 40, 80, 160):
        idx = candidates[dist[candidates] < radius]
        if idx.size >= 5:
            break
    if idx.size == 0:
        idx = candidates[np.argsort(dist[candidates])[:max_members]]
    rank = np.lexsort((dist[idx], -suppression[idx]))
    return idx[rank[:max_members]], center_idx


def save_cluster_3d(gauss_before, gate_proxy, cluster_idx, path):
    xyz = gauss_before["xyz"][0].numpy()
    suppression = 1.0 - np.clip(gate_proxy, 0, 1)
    cluster = xyz[cluster_idx]
    center = cluster.mean(axis=0)
    dist = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)
    near_idx = np.argsort(dist)[: min(700, xyz.shape[0])]
    near = xyz[near_idx]

    fig = plt.figure(figsize=(5.8, 5.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(near[:, 0], near[:, 1], near[:, 2], s=3, c="#9AA1AA", alpha=0.16, linewidths=0)
    sc = ax.scatter(
        cluster[:, 0], cluster[:, 1], cluster[:, 2],
        s=42 + 70 * suppression[cluster_idx],
        c=suppression[cluster_idx],
        cmap="inferno",
        vmin=0,
        vmax=1,
        alpha=0.88,
        linewidths=0.4,
        edgecolors="#202020",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(axis_limits(near, 0))
    ax.set_ylim(axis_limits(near, 1))
    ax.set_zlim(axis_limits(near, 2))
    ax.view_init(elev=25, azim=-48)
    ax.set_title("Strict local competition group", fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.05)
    cbar.set_label("suppression")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_opacity_redistribution_group(gauss_before, gauss_after, gate_proxy, cluster_idx, path):
    opacity_before = gauss_before["opacity"][0, cluster_idx, 0].numpy()
    opacity_after = gauss_after["opacity"][0, cluster_idx, 0].numpy()
    suppression = 1.0 - np.clip(gate_proxy[cluster_idx], 0, 1)
    order = np.argsort(-opacity_before)[: min(8, cluster_idx.size)]
    opacity_before = opacity_before[order]
    opacity_after = opacity_after[order]
    suppression = suppression[order]
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(order)))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    for ax, values, title in zip(axes, (opacity_before, opacity_after), ("Before competition", "After competition")):
        y = np.arange(len(values))[::-1]
        ax.barh(y, values, color=colors, alpha=0.84)
        for yy, value in zip(y, values):
            ax.text(min(value + 0.02, 0.96), yy, f"{value:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("opacity")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.35)
    axes[0].set_yticks(np.arange(len(opacity_before))[::-1])
    axes[0].set_yticklabels([f"g{i+1}" for i in range(len(opacity_before))])
    axes[1].set_yticks(np.arange(len(opacity_before))[::-1])
    fig.suptitle(f"Group-wise opacity redistribution | mean suppression={suppression.mean():.2f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def stats_to_cpu(stats):
    out = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            flat = value.detach().float().cpu().reshape(-1)
            if flat.numel() > 0:
                out[key] = float(flat.mean().item())
    return out


def gather_gaussians_with_metadata(gaussians, idx):
    keys = (
        "xyz",
        "rotation",
        "scale",
        "opacity",
        "color",
        "conf",
        "competition_color",
        "source_view",
        "redundancy_score",
        "redundancy_coef",
        "competition_gate",
        "competition_active",
    )
    out = {}
    for key in keys:
        value = gaussians.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = value[:, idx].contiguous()
    return out


def as_vector(values, length, fill=0.0):
    if values is None:
        return np.full((length,), fill, dtype=np.float32)
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[0, :, 0]
    elif arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.shape[0] != length:
        out = np.full((length,), fill, dtype=np.float32)
        n = min(length, arr.shape[0])
        out[:n] = arr[:n]
        return out
    return arr.astype(np.float32, copy=False)


def project_gaussians_to_frame(gaussians, camera_poses, intrinsics, frame_index, height, width):
    xyz = gaussians["xyz"][0].numpy()
    c2w = camera_poses[0, frame_index].detach().float().cpu().numpy()
    K = intrinsics[0, frame_index].detach().float().cpu().numpy()
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_cam = (xyz - t.reshape(1, 3)) @ R
    z = pts_cam[:, 2]
    u = (pts_cam[:, 0] / np.clip(z, 1e-6, None)) * K[0, 0] + K[0, 2]
    v = (pts_cam[:, 1] / np.clip(z, 1e-6, None)) * K[1, 1] + K[1, 2]
    uv = np.stack([u, v], axis=-1)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-5)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return uv, z, valid


def crop_box_from_points(uv, indices, height, width, margin=44, min_size=148):
    indices = np.asarray(indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < uv.shape[0])]
    points = uv[indices]
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        cy, cx = height // 2, width // 2
        half_h = min(height // 2, min_size // 2)
        half_w = min(width // 2, min_size // 2)
        return max(0, cy - half_h), min(height, cy + half_h), max(0, cx - half_w), min(width, cx + half_w)

    x0 = float(points[:, 0].min()) - margin
    x1 = float(points[:, 0].max()) + margin
    y0 = float(points[:, 1].min()) - margin
    y1 = float(points[:, 1].max()) + margin
    box_w = max(x1 - x0, float(min_size))
    box_h = max(y1 - y0, float(min_size))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    x0 = int(round(np.clip(cx - box_w * 0.5, 0, max(width - box_w, 0))))
    y0 = int(round(np.clip(cy - box_h * 0.5, 0, max(height - box_h, 0))))
    x1 = int(round(min(width, x0 + box_w)))
    y1 = int(round(min(height, y0 + box_h)))
    return y0, y1, x0, x1


def draw_dashed_line(image, p0, p1, color, thickness=1, dash=8, gap=6):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    step = dash + gap
    vx = (x1 - x0) / length
    vy = (y1 - y0) / length
    distance = 0.0
    while distance < length:
        end = min(distance + dash, length)
        a = (int(round(x0 + vx * distance)), int(round(y0 + vy * distance)))
        b = (int(round(x0 + vx * end)), int(round(y0 + vy * end)))
        cv2.line(image, a, b, color, thickness, lineType=cv2.LINE_AA)
        distance += step


def draw_dashed_ellipse(image, center, axes, color, thickness=2, segments=56):
    cx, cy = center
    rx, ry = max(2, axes[0]), max(2, axes[1])
    points = []
    for i in range(segments + 1):
        angle = 2.0 * math.pi * i / segments
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    for i in range(segments):
        if i % 2 == 0:
            p0 = (int(round(points[i][0])), int(round(points[i][1])))
            p1 = (int(round(points[i + 1][0])), int(round(points[i + 1][1])))
            cv2.line(image, p0, p1, color, thickness, lineType=cv2.LINE_AA)


def draw_gaussian_marker(image, point, color, radius=6, outline=(28, 28, 28), alpha=0.82):
    x, y = int(round(point[0])), int(round(point[1]))
    overlay = image.copy()
    cv2.ellipse(overlay, (x, y), (radius + 3, max(2, radius - 1)), -18, 0, 360, outline, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(overlay, (x, y), (radius + 2, max(2, radius - 2)), -18, 0, 360, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)


def normalize_values(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = robust_limits(values[np.isfinite(values)], low=5, high=95) if np.isfinite(values).any() else (0.0, 1.0)
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def select_redundancy_cluster(
    gauss_before,
    gauss_after,
    uv,
    projected_valid,
    voxel_size,
    color_threshold,
    max_counted=12,
    max_ignored=5,
):
    xyz = gauss_before["xyz"][0].numpy()
    color = gauss_before.get("competition_color", gauss_before["color"])[0].numpy()
    opacity = gauss_before["opacity"][0, :, 0].numpy()
    after_opacity = gauss_after["opacity"][0, :, 0].numpy()
    n = xyz.shape[0]
    coef = as_vector(gauss_after.get("redundancy_coef"), n, fill=0.0)
    active = as_vector(gauss_after.get("competition_active"), n, fill=0.0) > 0.5
    gate = as_vector(gauss_after.get("competition_gate"), n, fill=1.0)
    gate_proxy = np.clip(after_opacity / np.clip(opacity, 1e-6, None), 0.0, 1.0)
    suppression = np.maximum(1.0 - gate, 1.0 - gate_proxy)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity) & (opacity > 0.02)
    center_candidates = finite & projected_valid & (active | (coef > 0))
    if not np.any(center_candidates):
        center_candidates = finite & projected_valid

    priority = 0.48 * normalize_values(coef) + 0.34 * normalize_values(suppression) + 0.18 * normalize_values(opacity)
    center_idx = int(np.argmax(np.where(center_candidates, priority, -1.0)))
    center = xyz[center_idx]
    center_color = color[center_idx]
    dist = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)
    color_dist = np.linalg.norm(color - center_color.reshape(1, 3), axis=1)
    local_radius = max(float(voxel_size), float(np.mean(gauss_before["scale"][0, :, :].numpy())) * 1.5, 1e-5)

    counted = np.array([], dtype=np.int64)
    ignored = np.array([], dtype=np.int64)
    near = np.array([center_idx], dtype=np.int64)
    for radius_mul, color_mul in ((1.0, 1.0), (1.5, 1.25), (2.5, 1.8), (4.0, 2.6), (8.0, 4.0)):
        radius = local_radius * radius_mul
        threshold = max(color_threshold * color_mul, color_threshold + 1e-6)
        local_mask = finite & projected_valid & (dist <= radius)
        similar_mask = local_mask & (color_dist <= threshold)
        different_mask = local_mask & (color_dist > threshold)
        counted = np.nonzero(similar_mask & (np.arange(n) != center_idx))[0]
        ignored = np.nonzero(different_mask & projected_valid)[0]
        near = np.nonzero(local_mask)[0]
        if counted.size >= 4 and ignored.size >= 1:
            break

    if counted.size > max_counted:
        order = np.lexsort((-coef[counted], dist[counted]))
        counted = counted[order[:max_counted]]
    if ignored.size > max_ignored:
        order = np.lexsort((color_dist[ignored], dist[ignored]))
        ignored = ignored[order[:max_ignored]]
    if near.size > max_counted + max_ignored + 8:
        near_order = np.argsort(dist[near])
        near = near[near_order[: max_counted + max_ignored + 8]]

    return {
        "center_idx": center_idx,
        "counted_idx": counted,
        "ignored_idx": ignored,
        "near_idx": near,
        "local_radius": local_radius,
        "color_distance": color_dist,
        "suppression": suppression,
        "gate": gate,
        "gate_proxy": gate_proxy,
        "coef": coef,
    }


def shifted_uv(uv, crop_box):
    y0, _, x0, _ = crop_box
    return uv - np.array([[x0, y0]], dtype=np.float32)


def draw_local_region_overlay(frame_rgb, uv, projected_valid, cluster, crop_box=None):
    if crop_box is not None:
        base = crop_array(frame_rgb, crop_box)
        local_uv = shifted_uv(uv, crop_box)
        y0, y1, x0, x1 = crop_box
        in_crop = projected_valid & (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    else:
        base = np.asarray(frame_rgb, dtype=np.uint8).copy()
        local_uv = uv
        in_crop = projected_valid

    out = (base.astype(np.float32) * 0.68).astype(np.uint8)
    bg_idx = np.nonzero(in_crop)[0]
    if bg_idx.size > 220:
        rng = np.random.default_rng(41)
        bg_idx = rng.choice(bg_idx, size=220, replace=False)
    for idx in bg_idx:
        draw_gaussian_marker(out, local_uv[idx], (241, 166, 73), radius=3, alpha=0.42)

    member_idx = np.unique(np.concatenate([
        np.array([cluster["center_idx"]], dtype=np.int64),
        cluster["near_idx"],
        cluster["counted_idx"],
        cluster["ignored_idx"],
    ]))
    member_points = local_uv[member_idx[np.isfinite(local_uv[member_idx]).all(axis=1)]]
    center = local_uv[cluster["center_idx"]]
    if member_points.shape[0] > 1:
        dx = np.abs(member_points[:, 0] - center[0])
        dy = np.abs(member_points[:, 1] - center[1])
        rx = int(np.clip(np.percentile(dx, 92) + 28, 34, max(out.shape[1] // 2, 36)))
        ry = int(np.clip(np.percentile(dy, 92) + 28, 34, max(out.shape[0] // 2, 36)))
    else:
        rx = max(36, out.shape[1] // 5)
        ry = max(36, out.shape[0] // 5)
    draw_dashed_ellipse(out, center, (rx, ry), (255, 255, 255), thickness=2)

    for idx in cluster["near_idx"]:
        draw_gaussian_marker(out, local_uv[idx], (245, 178, 75), radius=5, alpha=0.70)
    for idx in cluster["ignored_idx"]:
        draw_gaussian_marker(out, local_uv[idx], (96, 126, 214), radius=6, alpha=0.76)
    draw_gaussian_marker(out, center, (245, 72, 57), radius=8, alpha=0.92)
    cv2.putText(out, "G_i", (int(center[0]) + 9, int(center[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 55, 45), 2, cv2.LINE_AA)
    return out


def draw_color_counting_overlay(frame_rgb, uv, cluster, crop_box):
    base = (crop_array(frame_rgb, crop_box).astype(np.float32) * 0.74).astype(np.uint8)
    local_uv = shifted_uv(uv, crop_box)
    center = local_uv[cluster["center_idx"]]
    member_idx = np.unique(np.concatenate([
        np.array([cluster["center_idx"]], dtype=np.int64),
        cluster["counted_idx"],
        cluster["ignored_idx"],
    ]))
    member_points = local_uv[member_idx[np.isfinite(local_uv[member_idx]).all(axis=1)]]
    if member_points.shape[0] > 1:
        dx = np.abs(member_points[:, 0] - center[0])
        dy = np.abs(member_points[:, 1] - center[1])
        rx = int(np.clip(np.percentile(dx, 95) + 30, 38, max(base.shape[1] // 2, 40)))
        ry = int(np.clip(np.percentile(dy, 95) + 30, 38, max(base.shape[0] // 2, 40)))
    else:
        rx = max(40, base.shape[1] // 5)
        ry = max(40, base.shape[0] // 5)

    display_uv = local_uv.copy()
    counted = np.asarray(cluster["counted_idx"], dtype=np.int64)
    ignored = np.asarray(cluster["ignored_idx"], dtype=np.int64)
    if counted.size > 0:
        angles = np.linspace(-2.55, 2.55, counted.size, endpoint=True)
        for k, idx in enumerate(counted):
            px = center[0] + 0.68 * rx * math.cos(float(angles[k]))
            py = center[1] + 0.56 * ry * math.sin(float(angles[k]))
            display_uv[idx] = (
                np.clip(px, 14, base.shape[1] - 15),
                np.clip(py, 14, base.shape[0] - 15),
            )
    if ignored.size > 0:
        angles = np.linspace(-0.65, 0.65, ignored.size, endpoint=True)
        for k, idx in enumerate(ignored):
            px = center[0] + 0.88 * rx
            py = center[1] + 0.62 * ry * math.sin(float(angles[k]))
            display_uv[idx] = (
                np.clip(px, 14, base.shape[1] - 15),
                np.clip(py, 14, base.shape[0] - 15),
            )

    draw_dashed_ellipse(base, center, (rx, ry), (34, 34, 34), thickness=2)

    for idx in cluster["counted_idx"]:
        p = display_uv[idx]
        draw_dashed_line(base, center, p, (235, 86, 58), thickness=1, dash=6, gap=5)
    for idx in cluster["ignored_idx"]:
        p = display_uv[idx]
        draw_dashed_line(base, center, p, (68, 100, 205), thickness=1, dash=4, gap=8)

    for idx in cluster["counted_idx"]:
        draw_gaussian_marker(base, display_uv[idx], (248, 174, 71), radius=7, alpha=0.84)
    for idx in cluster["ignored_idx"]:
        draw_gaussian_marker(base, display_uv[idx], (96, 125, 215), radius=8, alpha=0.86)
    draw_gaussian_marker(base, center, (244, 72, 58), radius=10, alpha=0.94)
    return base


def save_redundancy_curve(redundancy_score, redundancy_coef, threshold, selected_indices, path):
    valid = np.isfinite(redundancy_score) & np.isfinite(redundancy_coef)
    x = redundancy_score[valid]
    y = redundancy_coef[valid]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float32)
        y = np.array([0.0], dtype=np.float32)
    max_score = max(float(np.max(x)), float(threshold) + 1.0)
    xs = np.linspace(0.0, max_score, 512)
    ys = np.clip((xs - float(threshold)) / (max_score - float(threshold) + 1e-6), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    if x.size > 6000:
        rng = np.random.default_rng(59)
        sample = rng.choice(x.size, size=6000, replace=False)
        ax.scatter(x[sample], y[sample], s=4, c="#8A95A5", alpha=0.20, linewidths=0)
    else:
        ax.scatter(x, y, s=5, c="#8A95A5", alpha=0.24, linewidths=0)
    ax.plot(xs, ys, color="#F05A3B", linewidth=2.4, label="current code: linear normalized rho")
    ax.axvline(float(threshold), color="#222222", linestyle="--", linewidth=1.3, label="T = mean + lambda sigma")
    ax.set_xlim(0, max_score)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("redundancy score R_i")
    ax.set_ylabel("redundancy coefficient rho_i")
    ax.grid(True, linewidth=0.35, alpha=0.35)

    colors = {"center": "#E53935", "counted": "#F39C34", "ignored": "#4B6DD9"}
    for label, indices in selected_indices.items():
        for idx in np.asarray(indices, dtype=np.int64).reshape(-1)[:4]:
            if 0 <= idx < redundancy_score.shape[0]:
                ax.scatter([redundancy_score[idx]], [redundancy_coef[idx]], s=48, color=colors.get(label, "#333333"),
                           edgecolors="#202020", linewidths=0.6, zorder=5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_redundancy_histogram(redundancy_score, redundancy_coef, threshold, path):
    valid = np.isfinite(redundancy_score) & np.isfinite(redundancy_coef)
    score = redundancy_score[valid]
    coef = redundancy_coef[valid]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    axes[0].hist(score, bins=80, color="#637083", alpha=0.76)
    axes[0].axvline(float(threshold), color="#E23B2E", linestyle="--", linewidth=1.2)
    axes[0].set_title("R_i distribution", fontsize=9)
    axes[0].set_xlabel("R_i")
    axes[0].set_ylabel("count")
    axes[1].hist(coef, bins=np.linspace(0, 1, 50), color="#F08A3E", alpha=0.78)
    axes[1].set_title("rho_i distribution", fontsize=9)
    axes[1].set_xlabel("rho_i")
    axes[1].set_ylabel("count")
    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_suppression_rows(gauss_before, gauss_after, cluster, max_rows=10):
    opacity_before = gauss_before["opacity"][0, :, 0].numpy()
    opacity_after = gauss_after["opacity"][0, :, 0].numpy()
    redundancy = as_vector(gauss_after.get("redundancy_score"), opacity_before.shape[0], fill=0.0)
    coef = as_vector(gauss_after.get("redundancy_coef"), opacity_before.shape[0], fill=0.0)
    gate = as_vector(gauss_after.get("competition_gate"), opacity_before.shape[0], fill=1.0)
    source_view = as_vector(gauss_before.get("source_view"), opacity_before.shape[0], fill=-1).astype(np.int64)

    ordered = [("main", cluster["center_idx"])]
    counted_order = sorted(cluster["counted_idx"].tolist(), key=lambda i: (-coef[i], -opacity_before[i]))
    ignored_order = sorted(cluster["ignored_idx"].tolist(), key=lambda i: cluster["color_distance"][i])
    remaining = max(0, max_rows - 1)
    ignored_take = min(len(ignored_order), 3, max(0, remaining // 3))
    if ignored_order and ignored_take == 0 and remaining > 0:
        ignored_take = 1
    counted_take = max(0, remaining - ignored_take)
    for idx in counted_order[:counted_take]:
        ordered.append(("duplicate", idx))
    for idx in ignored_order[:ignored_take]:
        ordered.append(("color-different", idx))
    ordered = ordered[:max_rows]

    rows = []
    for rank, (kind, idx) in enumerate(ordered):
        rows.append(
            {
                "label": f"G{rank + 1}" if kind != "main" else "G_i",
                "index": int(idx),
                "type": kind,
                "source_view": int(source_view[idx]),
                "R_i": f"{float(redundancy[idx]):.3f}",
                "rho_i": f"{float(coef[idx]):.3f}",
                "gate": f"{float(gate[idx]):.3f}",
                "opacity_before": f"{float(opacity_before[idx]):.3f}",
                "opacity_after": f"{float(opacity_after[idx]):.3f}",
            }
        )
    return rows


def save_suppression_table(rows, path):
    if not rows:
        return
    columns = ["label", "type", "source_view", "R_i", "rho_i", "gate", "opacity_before", "opacity_after"]
    cell_text = [[row[col] for col in columns] for row in rows]
    fig_h = max(2.4, 0.36 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.22)
    type_colors = {
        "main": "#FAD3CD",
        "duplicate": "#FCE6C4",
        "color-different": "#DDE4FF",
    }
    for r, row in enumerate(rows, start=1):
        color = type_colors.get(row["type"], "#F3F4F6")
        for c in range(len(columns)):
            table[(r, c)].set_facecolor(color)
    for c in range(len(columns)):
        table[(0, c)].set_facecolor("#F0F2F5")
        table[(0, c)].set_text_props(weight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_image_explanations(rows, path):
    with open(path, "w") as f:
        f.write("# CP Flowchart Asset Explanations\n\n")
        for row in rows:
            f.write(f"## {row['file']}\n\n")
            f.write(f"{row['explanation']}\n\n")


def write_manifest(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_model(args, device):
    ckpt = resolve_checkpoint_path(args.ckpt)
    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
        gs_view_stride=args.gs_view_stride,
        enable_local_competition=False,
        max_dense_gaussians=0,
    ).to(device).eval()
    if ckpt.endswith(".safetensors"):
        weight = load_file(ckpt, device="cpu")
    else:
        weight = torch.load(ckpt, map_location="cpu", weights_only=False)
    result = model.load_state_dict(weight, strict=False)
    print(f"Checkpoint: {ckpt}")
    print(f"load_state_dict: {result}")
    del weight
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Export CP real-data assets for the quadtree and local-competition flowcharts.")
    parser.add_argument("--data_root", type=str, default="/data/liuwei/dataset/ntu_seq/cp")
    parser.add_argument("--output_dir", type=str, default="outputs/cp_flowchart_assets")
    parser.add_argument("--ckpt", type=str, default="outputs/pi3_highres_0506_v6_local_comp/ckpts/best_model/model.safetensors")
    parser.add_argument("--frame_start", type=int, default=800)
    parser.add_argument("--frame_end", type=int, default=900)
    parser.add_argument("--frame_step", type=int, default=10)
    parser.add_argument("--focus_frame", type=int, default=850)
    parser.add_argument("--context_frames", type=str, default="830,850,870")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--render_min_opacity", type=float, default=0.01)
    parser.add_argument("--max_render_gaussians", type=int, default=180000)
    parser.add_argument("--spatial_max_points", type=int, default=80000)
    args = parser.parse_args()

    quadtree_dir = os.path.join(args.output_dir, "quadtree")
    local_dir = os.path.join(args.output_dir, "local_competition")
    new_method_dir = os.path.join(args.output_dir, "new_local_competition")
    overview_dir = os.path.join(args.output_dir, "overview")
    for directory in (quadtree_dir, local_dir, new_method_dir, overview_dir):
        os.makedirs(directory, exist_ok=True)

    device = torch.device(args.device)
    imgs_cpu, input_rgbs, frame_items, orig_hw, target_hw = load_flowchart_frame_range(
        args.data_root,
        args.frame_start,
        args.frame_end,
        args.frame_step,
        args.pixel_limit,
    )
    height, width = target_hw
    frame_ids = [item["frame_id"] for item in frame_items]
    if args.focus_frame not in frame_ids:
        raise ValueError(f"focus_frame {args.focus_frame} is not in loaded frames {frame_ids}")
    focus_idx = frame_ids.index(args.focus_frame)
    context_frames = [int(x.strip()) for x in args.context_frames.split(",") if x.strip()]
    context_indices = [frame_ids.index(fid) for fid in context_frames if fid in frame_ids]
    if len(context_indices) == 0:
        context_indices = [max(0, focus_idx - 2), focus_idx, min(len(frame_ids) - 1, focus_idx + 2)]
    focus_item = frame_items[focus_idx]
    context_items = [frame_items[i] for i in context_indices]

    print(f"Frames: {frame_ids}")
    print(f"Focus frame: {args.focus_frame} ({frame_display_label(focus_item)})")
    print(f"Focus source: {focus_item['path']}")
    print(f"Original resolution: {orig_hw[0]}x{orig_hw[1]} | model resolution: {height}x{width}")

    if device.type == "cuda":
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        print(f"CUDA available: {cuda_available} | device_count: {cuda_count}")
        if not cuda_available:
            raise RuntimeError("CUDA was requested, but PyTorch cannot see any CUDA GPU in this process.")

    model, ckpt = load_model(args, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    imgs_batch = imgs_cpu.unsqueeze(0).to(device)

    print("Running baseline inference without local competition...")
    model.enable_local_competition = False
    with torch.no_grad():
        with amp_context():
            before_result = model(imgs_batch, return_viz=False)
    camera_poses_cpu = before_result["camera_poses"].detach().float().cpu()
    intrinsics_cpu = before_result["intrinsics"].detach().float().cpu()
    stats_before = stats_to_cpu(before_result.get("gaussian_stats", {}))
    idx = select_indices_by_opacity(
        before_result["gaussians"],
        min_opacity=args.render_min_opacity,
        max_gaussians=args.max_render_gaussians,
    )
    gauss_before_gpu = gather_gaussians_with_metadata(before_result["gaussians"], idx.to(device))
    alpha_before = render_alpha_maps(gauss_before_gpu, before_result["camera_poses"], before_result["intrinsics"], height, width)
    gauss_before = {
        key: value.detach().float().cpu()
        for key, value in gauss_before_gpu.items()
        if isinstance(value, torch.Tensor)
    }
    del before_result, gauss_before_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print("Running inference with local competition and routing visualizations...")
    model.enable_local_competition = True
    with torch.no_grad():
        with amp_context():
            after_result = model(imgs_batch, return_viz=True)
    stats_after = stats_to_cpu(after_result.get("gaussian_stats", {}))
    gauss_after_gpu = gather_gaussians_with_metadata(after_result["gaussians"], idx.to(device))
    alpha_after = render_alpha_maps(gauss_after_gpu, after_result["camera_poses"], after_result["intrinsics"], height, width)
    gauss_after = {
        key: value.detach().float().cpu()
        for key, value in gauss_after_gpu.items()
        if isinstance(value, torch.Tensor)
    }

    viz = after_result["routing_viz"]
    sub_idx = viz["sub_idx"].detach().cpu().numpy().astype(int).tolist()
    focus_viz_idx = sub_idx.index(focus_idx)
    frame_rgb = input_rgbs[focus_idx]
    stem = frame_file_tag(focus_item)
    score_map = viz["score_map"][0, focus_viz_idx].detach().float().cpu().numpy()
    color_score = viz["color_score"][0, focus_viz_idx].detach().float().cpu().numpy()
    depth_score = viz["depth_score"][0, focus_viz_idx].detach().float().cpu().numpy()
    scale_map = viz["scale_map"][0, focus_viz_idx].detach().float().cpu().numpy()
    quad_mask = viz["quad_keep_mask"][0, focus_viz_idx].detach().bool().cpu().numpy()
    support_mask = viz["support_mask"][0, focus_viz_idx].detach().bool().cpu().numpy()
    low_conf_mask = viz["low_conf_mask"][0, focus_viz_idx].detach().bool().cpu().numpy()
    depth = viz["local_points_before_routing"][0, focus_idx, :, :, 2].detach().float().cpu().numpy()

    local_mean, color_complexity, plane_residual, curvature, geometry_complexity = compute_complexity_maps(frame_rgb, depth)
    fused_score = score_map
    crop_box = choose_crop_box(fused_score, crop_size_ratio=0.38)

    score_hi = max(robust_limits(fused_score, low=1, high=99)[1], 1e-4)
    score_img = colorize_scalar(fused_score, min_val=0.0, max_val=score_hi, cmap=cv2.COLORMAP_INFERNO)
    score_overlay = blend_heatmap(frame_rgb, score_img, np.clip(fused_score / score_hi, 0, 1), max_alpha=0.78)
    color_img = colorize_scalar(color_complexity, min_val=0.0, max_val=max(robust_limits(color_complexity, 1, 99)[1], 1e-4), cmap=cv2.COLORMAP_INFERNO)
    plane_img = colorize_scalar(plane_residual, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_INFERNO)
    curv_img = colorize_scalar(curvature, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_INFERNO)
    geo_img = colorize_scalar(geometry_complexity, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_INFERNO)
    depth_img = colorize_scalar(depth, min_val=robust_limits(depth, 2, 98)[0], max_val=robust_limits(depth, 2, 98)[1], cmap=cv2.COLORMAP_INFERNO)
    local_mean_img = np.clip(local_mean * 255, 0, 255).astype(np.uint8)
    density = gaussian_density_from_support(support_mask, sigma=3.2)
    density_img = colorize_scalar(density, min_val=0, max_val=1, cmap=cv2.COLORMAP_INFERNO)
    density_overlay = blend_heatmap(frame_rgb, density_img, density, max_alpha=0.78, base_dim=0.56)

    dense_gaussian_img = draw_dense_gaussian_overlay(frame_rgb, score_map=fused_score, crop_box=None, max_points=18000)
    dense_gaussian_crop = draw_dense_gaussian_overlay(frame_rgb, score_map=fused_score, crop_box=crop_box, max_points=7000)
    quadtree_img = draw_quadtree_partition(score_overlay, scale_map, quad_mask, crop_box=None)
    anchor_img = draw_anchor_retention(score_overlay, scale_map, quad_mask, low_conf_mask=None, crop_box=None, max_points=5000)
    gaussian_img = draw_gaussian_ellipses(score_overlay, scale_map, support_mask, crop_box=None, max_ellipses=2400)
    quadtree_crop = draw_quadtree_partition(score_overlay, scale_map, quad_mask, crop_box=crop_box)
    anchor_crop = draw_anchor_retention(score_overlay, scale_map, quad_mask, low_conf_mask=None, crop_box=crop_box, max_points=1000)
    gaussian_crop = draw_gaussian_ellipses(score_overlay, scale_map, support_mask, crop_box=crop_box, max_ellipses=1300)

    quadtree_assets = {
        "01_input_view_full": frame_rgb,
        "01_input_view_with_crop_box": draw_crop_box(frame_rgb, crop_box),
        "01_input_patch": crop_array(frame_rgb, crop_box),
        "01_predicted_depth_map": depth_img,
        "01_predicted_depth_patch": crop_array(depth_img, crop_box),
        "02_local_mean_full": local_mean_img,
        "02_local_mean_patch": crop_array(local_mean_img, crop_box),
        "02_color_complexity_full": color_img,
        "02_color_complexity_patch": crop_array(color_img, crop_box),
        "02_plane_residual_full": plane_img,
        "02_plane_residual_patch": crop_array(plane_img, crop_box),
        "02_depth_curvature_full": curv_img,
        "02_depth_curvature_patch": crop_array(curv_img, crop_box),
        "02_geometry_complexity_full": geo_img,
        "02_geometry_complexity_patch": crop_array(geo_img, crop_box),
        "03_fused_complexity_map_full": score_img,
        "03_fused_complexity_overlay_full": score_overlay,
        "03_fused_complexity_patch": crop_array(score_img, crop_box),
        "03_dense_gaussian_overlay_before_quadtree_full": dense_gaussian_img,
        "03_dense_gaussian_overlay_before_quadtree_patch": dense_gaussian_crop,
        "04_quadtree_partition_full": quadtree_img,
        "04_quadtree_partition_patch": quadtree_crop,
        "05_center_anchor_retention_full": anchor_img,
        "05_center_anchor_retention_patch": anchor_crop,
        "06_gaussian_instantiation_full": gaussian_img,
        "06_gaussian_instantiation_patch": gaussian_crop,
        "06_gaussian_support_density_full": density_overlay,
        "06_gaussian_support_density_patch": crop_array(density_overlay, crop_box),
    }
    for name, image in quadtree_assets.items():
        save_rgb(os.path.join(quadtree_dir, f"{name}_{stem}.png"), image)

    save_panel(
        [
            crop_array(frame_rgb, crop_box),
            dense_gaussian_crop,
            crop_array(color_img, crop_box),
            crop_array(geo_img, crop_box),
            crop_array(score_img, crop_box),
            quadtree_crop,
            anchor_crop,
            gaussian_crop,
        ],
        ["Input", "Dense Gaussians", "Color complexity", "Geometry complexity", "Fused score", "Quadtree", "Center anchors", "Gaussians"],
        os.path.join(quadtree_dir, f"quadtree_flow_asset_strip_{stem}.png"),
        figsize_per_image=2.05,
        dpi=320,
    )

    opacity_before = gauss_before["opacity"][0, :, 0].numpy()
    opacity_after = gauss_after["opacity"][0, :, 0].numpy()
    gate_proxy = np.clip(opacity_after / np.clip(opacity_before, 1e-6, None), 0.0, 1.0)
    cluster_idx, center_idx = select_competition_cluster(gauss_before, gate_proxy, max_members=14)

    before = np.clip(alpha_before[focus_idx], 0.0, 1.0)
    after = np.clip(alpha_after[focus_idx], 0.0, 1.0)
    suppressed = np.clip(before - after, 0.0, 1.0)
    shared_lo, shared_hi = robust_limits(np.concatenate([before.reshape(-1), after.reshape(-1)]), low=2, high=99.5)
    before_img, _ = alpha_vis(before, shared_limits=(shared_lo, shared_hi), cmap=cv2.COLORMAP_MAGMA)
    after_img, _ = alpha_vis(after, shared_limits=(shared_lo, shared_hi), cmap=cv2.COLORMAP_MAGMA)
    supp_hi = max(robust_limits(suppressed, low=1, high=99.5)[1], 1e-4)
    supp_norm = np.clip(suppressed / supp_hi, 0.0, 1.0)
    supp_img = colorize_scalar(supp_norm, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_INFERNO)
    supp_overlay = blend_heatmap(frame_rgb, supp_img, supp_norm, max_alpha=0.80, base_dim=0.56)
    supp_crop_box = choose_crop_box(suppressed, crop_size_ratio=0.36)

    view_colors = [(74, 183, 194), (119, 181, 83), (139, 110, 203), (238, 139, 45)]
    view_rows = []
    for row_idx, frame_index in enumerate(context_indices):
        if frame_index not in sub_idx:
            continue
        viz_idx = sub_idx.index(frame_index)
        view_rgb = input_rgbs[frame_index]
        view_scale = viz["scale_map"][0, viz_idx].detach().float().cpu().numpy()
        view_support = viz["support_mask"][0, viz_idx].detach().bool().cpu().numpy()
        crop = crop_array(view_rgb, crop_box)
        candidates = draw_candidate_ellipses_on_crop(
            crop,
            crop_array(view_support, crop_box),
            crop_array(view_scale, crop_box),
            color=view_colors[row_idx % len(view_colors)],
            max_ellipses=52,
        )
        frame_item = frame_items[frame_index]
        frame_tag = frame_file_tag(frame_item)
        label = f"V{row_idx + 1} / {frame_display_label(frame_item)}"
        view_rows.append((label, crop, candidates))
        save_rgb(os.path.join(local_dir, f"01_multiview_patch_{frame_tag}.png"), crop)
        save_rgb(os.path.join(local_dir, f"01_multiview_candidates_{frame_tag}.png"), candidates)
    save_multi_view_candidate_panel(view_rows, os.path.join(local_dir, "01_multiview_candidate_generation_panel.png"))

    save_common_3d_space(gauss_before, os.path.join(local_dir, "02_unified_3d_space_candidates.png"), max_points=args.spatial_max_points)
    save_cluster_3d(gauss_before, gate_proxy, cluster_idx, os.path.join(local_dir, "03_strict_local_group_3d.png"))
    save_opacity_redistribution_group(
        gauss_before,
        gauss_after,
        gate_proxy,
        cluster_idx,
        os.path.join(local_dir, "04_groupwise_opacity_redistribution.png"),
    )

    local_assets = {
        "05_opacity_before_competition_full": before_img,
        "05_opacity_after_competition_full": after_img,
        "05_suppressed_opacity_map_full": supp_img,
        "05_suppressed_opacity_overlay_full": supp_overlay,
        "05_focus_region_input_crop": crop_array(frame_rgb, crop_box),
        "05_focus_region_opacity_before_crop": crop_array(before_img, crop_box),
        "05_focus_region_opacity_after_crop": crop_array(after_img, crop_box),
        "05_focus_region_suppressed_overlay_crop": crop_array(supp_overlay, crop_box),
        "05_input_suppression_crop": crop_array(frame_rgb, supp_crop_box),
        "05_opacity_before_competition_crop": crop_array(before_img, supp_crop_box),
        "05_opacity_after_competition_crop": crop_array(after_img, supp_crop_box),
        "05_suppressed_opacity_overlay_crop": crop_array(supp_overlay, supp_crop_box),
        "06_rendering_effect_input_crop": crop_array(frame_rgb, supp_crop_box),
        "06_rendering_effect_before_crop": crop_array(before_img, supp_crop_box),
        "06_rendering_effect_after_crop": crop_array(after_img, supp_crop_box),
        "06_rendering_effect_suppression_crop": crop_array(supp_overlay, supp_crop_box),
    }
    for name, image in local_assets.items():
        save_rgb(os.path.join(local_dir, f"{name}_{stem}.png"), image)

    save_panel(
        [
            crop_array(frame_rgb, supp_crop_box),
            crop_array(before_img, supp_crop_box),
            crop_array(after_img, supp_crop_box),
            crop_array(supp_overlay, supp_crop_box),
        ],
        ["Input crop", "Before competition", "After competition", "Suppressed opacity"],
        os.path.join(local_dir, f"local_competition_before_after_strip_{stem}.png"),
        figsize_per_image=2.6,
        dpi=320,
    )

    save_panel(
        [
            crop_array(frame_rgb, crop_box),
            crop_array(before_img, crop_box),
            crop_array(after_img, crop_box),
            crop_array(supp_overlay, crop_box),
        ],
        ["Input crop", "Before competition", "After competition", "Suppressed opacity"],
        os.path.join(local_dir, f"local_competition_focus_region_strip_{stem}.png"),
        figsize_per_image=2.6,
        dpi=320,
    )

    n_gaussians = gauss_before["xyz"].shape[1]
    redundancy_score = as_vector(gauss_after.get("redundancy_score"), n_gaussians, fill=0.0)
    redundancy_coef = as_vector(gauss_after.get("redundancy_coef"), n_gaussians, fill=0.0)
    competition_gate = as_vector(gauss_after.get("competition_gate"), n_gaussians, fill=1.0)
    if not np.any(redundancy_score > 0):
        redundancy_score = np.clip(1.0 - gate_proxy, 0.0, 1.0)
        redundancy_coef = redundancy_score.copy()
        competition_gate = np.clip(gate_proxy, 0.0, 1.0)

    redundancy_threshold = stats_after.get(
        "redundancy_threshold",
        stats_after.get("threshold", float(np.percentile(redundancy_score, 90))),
    )
    competition_voxel_size = stats_after.get(
        "voxel_size",
        max(float(np.mean(gauss_before["scale"][0].numpy())) * max(model.competition_radius_scale, 1e-6), 1e-5),
    )
    uv, camera_depth, projected_valid = project_gaussians_to_frame(
        gauss_before,
        camera_poses_cpu,
        intrinsics_cpu,
        focus_idx,
        height,
        width,
    )
    method_cluster = select_redundancy_cluster(
        gauss_before,
        gauss_after,
        uv,
        projected_valid,
        competition_voxel_size,
        model.redundancy_color_threshold,
    )
    method_indices = np.unique(np.concatenate([
        np.array([method_cluster["center_idx"]], dtype=np.int64),
        method_cluster["near_idx"],
        method_cluster["counted_idx"],
        method_cluster["ignored_idx"],
    ]))
    method_crop_box = crop_box_from_points(uv, method_indices, height, width, margin=52, min_size=172)

    region_full = draw_local_region_overlay(frame_rgb, uv, projected_valid, method_cluster, crop_box=None)
    region_crop = draw_local_region_overlay(frame_rgb, uv, projected_valid, method_cluster, crop_box=method_crop_box)
    count_crop = draw_color_counting_overlay(frame_rgb, uv, method_cluster, method_crop_box)
    before_method_crop = crop_array(before_img, method_crop_box)
    after_method_crop = crop_array(after_img, method_crop_box)
    suppression_method_crop = crop_array(supp_overlay, method_crop_box)

    method_assets = []
    def save_method_asset(name, image, explanation):
        path = os.path.join(new_method_dir, f"{name}_{stem}.png")
        save_rgb(path, image)
        method_assets.append({"file": os.path.basename(path), "explanation": explanation})
        return path

    save_method_asset(
        "01_resolution_aware_local_region_full",
        region_full,
        "Full focus view with the selected Gaussian G_i, its projected neighbors, and the local region used to explain resolution-aware competition.",
    )
    save_method_asset(
        "01_resolution_aware_local_region_crop",
        region_crop,
        "Zoomed crop around G_i. Orange ellipses are nearby Gaussian candidates; the dashed ellipse marks the local neighborhood Omega_i.",
    )
    save_method_asset(
        "02_color_aware_redundancy_counting_crop",
        count_crop,
        "Color-aware redundancy example. Candidate positions are spread for legibility: orange candidates are spatially close and color-similar enough to be counted; blue candidates are spatially close but color-different and ignored.",
    )
    save_method_asset(
        "04_opacity_before_local_crop",
        before_method_crop,
        "Rendered opacity before applying the new color-aware local competition gate, cropped around the selected local region.",
    )
    save_method_asset(
        "04_opacity_after_local_crop",
        after_method_crop,
        "Rendered opacity after applying redundancy-coefficient-guided soft suppression.",
    )
    save_method_asset(
        "04_suppressed_opacity_overlay_local_crop",
        suppression_method_crop,
        "Input crop overlaid with the opacity mass reduced by local competition; warmer regions indicate stronger suppression.",
    )

    curve_path = os.path.join(new_method_dir, "03_redundancy_coefficient_curve.png")
    save_redundancy_curve(
        redundancy_score,
        redundancy_coef,
        redundancy_threshold,
        {
            "center": [method_cluster["center_idx"]],
            "counted": method_cluster["counted_idx"],
            "ignored": method_cluster["ignored_idx"],
        },
        curve_path,
    )
    method_assets.append({
        "file": os.path.basename(curve_path),
        "explanation": "Actual redundancy-score to redundancy-coefficient curve from pi3_3dgs_8.py. The current code uses a linear normalized rho above threshold T, and selected local examples are highlighted.",
    })

    hist_path = os.path.join(new_method_dir, "03_redundancy_score_and_coefficient_histograms.png")
    save_redundancy_histogram(redundancy_score, redundancy_coef, redundancy_threshold, hist_path)
    method_assets.append({
        "file": os.path.basename(hist_path),
        "explanation": "Dataset-level distribution for the gathered CP 800-900 Gaussians: left is redundancy score R_i, right is coefficient rho_i.",
    })

    suppression_rows = build_suppression_rows(gauss_before, gauss_after, method_cluster, max_rows=10)
    table_csv = os.path.join(new_method_dir, "04_redundancy_guided_suppression_table.csv")
    write_manifest(suppression_rows, table_csv)
    table_path = os.path.join(new_method_dir, "04_redundancy_guided_suppression_table.png")
    save_suppression_table(suppression_rows, table_path)
    method_assets.append({
        "file": os.path.basename(table_path),
        "explanation": "Selected Gaussian-level table showing type, source view, redundancy score, coefficient, gate, and before/after opacity.",
    })

    suppression_panel_path = os.path.join(new_method_dir, f"04_redundancy_guided_opacity_suppression_panel_{stem}.png")
    save_panel(
        [
            crop_array(frame_rgb, method_crop_box),
            before_method_crop,
            after_method_crop,
            suppression_method_crop,
        ],
        ["Input", "Before", "After", "Suppression"],
        suppression_panel_path,
        figsize_per_image=2.6,
        dpi=320,
    )
    method_assets.append({
        "file": os.path.basename(suppression_panel_path),
        "explanation": "Four-image before/after panel for the paper flowchart's opacity-suppression step.",
    })

    curve_img = np.asarray(Image.open(curve_path).convert("RGB"))
    flow_strip_path = os.path.join(new_method_dir, f"new_local_competition_flowchart_asset_strip_{stem}.png")
    save_panel(
        [region_crop, count_crop, curve_img, suppression_method_crop],
        ["Local region", "Color-aware count", "rho curve", "Suppression"],
        flow_strip_path,
        figsize_per_image=2.65,
        dpi=300,
    )
    method_assets.append({
        "file": os.path.basename(flow_strip_path),
        "explanation": "Compact contact strip mapping the four subfigures: local region, color-aware redundancy counting, redundancy coefficient, and opacity suppression.",
    })
    write_image_explanations(method_assets, os.path.join(new_method_dir, "image_explanations.md"))

    save_contact_sheet(
        [
            [quadtree_assets["01_input_patch"], quadtree_assets["03_fused_complexity_patch"], quadtree_crop, gaussian_crop],
            [
                crop_array(frame_rgb, supp_crop_box),
                crop_array(before_img, supp_crop_box),
                crop_array(after_img, supp_crop_box),
                crop_array(supp_overlay, supp_crop_box),
            ],
        ],
        ["Input", "Score / before", "Sampling / after", "Gaussian / suppression"],
        os.path.join(overview_dir, "flowchart_asset_contact_sheet.png"),
        dpi=260,
    )

    rows = [
        {
            "figure": "quadtree",
            "focus_frame": args.focus_frame,
            "focus_source": focus_item["path"],
            "crop_box_y0_y1_x0_x1": str(tuple(int(x) for x in crop_box)),
            "score_mean": f"{float(score_map.mean()):.8f}",
            "score_p99": f"{float(np.percentile(score_map, 99)):.8f}",
            "support_points": int(np.count_nonzero(support_mask)),
            "quad_points": int(np.count_nonzero(quad_mask)),
        },
        {
            "figure": "local_competition",
            "focus_frame": args.focus_frame,
            "focus_source": focus_item["path"],
            "crop_box_y0_y1_x0_x1": str(tuple(int(x) for x in supp_crop_box)),
            "score_mean": "",
            "score_p99": "",
            "support_points": int(cluster_idx.size),
            "quad_points": "",
        },
        {
            "figure": "new_local_competition",
            "focus_frame": args.focus_frame,
            "focus_source": focus_item["path"],
            "crop_box_y0_y1_x0_x1": str(tuple(int(x) for x in method_crop_box)),
            "score_mean": f"{float(np.mean(redundancy_score)):.8f}",
            "score_p99": f"{float(np.percentile(redundancy_score, 99)):.8f}",
            "support_points": int(1 + method_cluster["counted_idx"].size + method_cluster["ignored_idx"].size),
            "quad_points": int(np.count_nonzero(redundancy_coef > 0)),
        },
    ]
    write_manifest(rows, os.path.join(args.output_dir, "manifest.csv"))

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("Flowchart assets\n")
        f.write("=" * 64 + "\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"frames: {frame_ids}\n")
        f.write(f"frame_sources: {[os.path.basename(item['path']) for item in frame_items]}\n")
        f.write(f"focus_frame: {args.focus_frame}\n")
        f.write(f"focus_source: {focus_item['path']}\n")
        f.write(f"context_frames: {[frame_ids[i] for i in context_indices]}\n")
        f.write(f"context_sources: {[item['path'] for item in context_items]}\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"model_resolution: {height}x{width}\n")
        f.write(f"quadtree_dir: {quadtree_dir}\n")
        f.write(f"local_competition_dir: {local_dir}\n")
        f.write(f"new_local_competition_dir: {new_method_dir}\n")
        f.write(f"overview_dir: {overview_dir}\n")
        f.write(f"compared_gaussians: {int(idx.numel())}\n")
        f.write(f"local_competition_gate_proxy_mean: {float(gate_proxy.mean()):.8f}\n")
        f.write(f"local_competition_suppression_mean: {float((1.0 - gate_proxy).mean()):.8f}\n")
        f.write(f"new_method_center_index: {int(method_cluster['center_idx'])}\n")
        f.write(f"new_method_counted_neighbors: {int(method_cluster['counted_idx'].size)}\n")
        f.write(f"new_method_color_different_ignored: {int(method_cluster['ignored_idx'].size)}\n")
        f.write(f"new_method_redundancy_threshold: {float(redundancy_threshold):.8f}\n")
        f.write(f"new_method_competition_voxel_size: {float(competition_voxel_size):.8f}\n")
        f.write(f"new_method_redundancy_coef_mean: {float(redundancy_coef.mean()):.8f}\n")
        f.write(f"new_method_redundancy_coef_max: {float(redundancy_coef.max()):.8f}\n")
        f.write("\nwithout_local_competition_stats:\n")
        for key in sorted(stats_before):
            f.write(f"  {key}: {stats_before[key]:.8f}\n")
        f.write("\nwith_new_local_competition_stats:\n")
        for key in sorted(stats_after):
            f.write(f"  {key}: {stats_after[key]:.8f}\n")

    del after_result, gauss_after_gpu, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Saved quadtree assets: {quadtree_dir}")
    print(f"Saved local competition assets: {local_dir}")
    print(f"Saved new local competition assets: {new_method_dir}")
    print(f"Saved overview: {overview_dir}")


if __name__ == "__main__":
    main()
