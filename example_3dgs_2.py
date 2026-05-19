import argparse
import csv
import gc
import math
import os
from contextlib import nullcontext

import cv2
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

from pi3.models.pi3_3dgs_1 import Pi3_3DGS


RGB_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def save_heatmap(tensor, path):
    alpha_np = tensor.squeeze().detach().cpu().numpy()
    alpha_np = np.clip(alpha_np, 0, 1)
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap_color)


def save_ply_binary(gaussians, path):
    xyz = gaussians["xyz"].detach().cpu().float().numpy().squeeze()
    rot = gaussians["rotation"].detach().cpu().float().numpy().squeeze()
    scale = gaussians["scale"].detach().cpu().float().numpy().squeeze()
    opacity = gaussians["opacity"].detach().cpu().float().numpy().squeeze()
    color = gaussians["color"].detach().cpu().float().numpy().squeeze()

    if xyz.ndim == 1:
        xyz = xyz[None]
        rot = rot[None]
        scale = scale[None]
        opacity = opacity[None]
        color = color[None]

    scale_ply = np.log(np.clip(scale, 1e-10, None))
    opacity_clipped = np.clip(opacity, 1e-6, 1 - 1e-6)
    opacity_ply = np.log(opacity_clipped / (1 - opacity_clipped))

    sh_c0 = 0.28209479177387814
    f_dc = (color - 0.5) / sh_c0
    normals = np.zeros_like(xyz)

    attributes = np.concatenate((
        xyz, normals, f_dc, opacity_ply[..., np.newaxis], scale_ply, rot
    ), axis=-1).astype(np.float32)

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n".encode("utf-8"))
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write(b"end_header\n")
        f.write(attributes.tobytes())


def se3_inverse(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:4] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def save_image(tensor, path):
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_np)


def save_depth(tensor, path):
    depth_np = tensor.squeeze().detach().cpu().numpy()
    valid_mask = depth_np > 1e-5
    if valid_mask.sum() > 0:
        d_min = np.percentile(depth_np[valid_mask], 2)
        d_max = np.percentile(depth_np[valid_mask], 98)
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
        depth_norm = np.clip(depth_norm, 0, 1)
    else:
        depth_norm = np.zeros_like(depth_np, dtype=np.float32)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(path, depth_gray)


def render_frame(gaussians, w2c, K, H, W, num_gaussians=None, conf_threshold=0.1):
    means = gaussians["xyz"]
    quats = gaussians["rotation"]
    scales = gaussians["scale"]
    opacities = gaussians["opacity"]
    colors = gaussians["color"]
    conf = gaussians.get("conf", None)

    B = means.shape[0]

    if num_gaussians is not None:
        if isinstance(num_gaussians, torch.Tensor):
            max_N = int(num_gaussians.max().item())
            means = means[:, :max_N]
            quats = quats[:, :max_N]
            scales = scales[:, :max_N]
            opacities = opacities[:, :max_N].clone()
            colors = colors[:, :max_N]
            if isinstance(conf, torch.Tensor):
                conf = conf[:, :max_N]

            range_seq = torch.arange(max_N, device=means.device).expand(B, max_N)
            valid_mask = range_seq < num_gaussians.unsqueeze(1)
            opacities[~valid_mask] = 0.0
        else:
            limit = int(num_gaussians)
            means = means[:, :limit]
            quats = quats[:, :limit]
            scales = scales[:, :limit]
            opacities = opacities[:, :limit]
            colors = colors[:, :limit]
            if isinstance(conf, torch.Tensor):
                conf = conf[:, :limit]

    rgb, alpha, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W,
        height=H,
        render_mode="RGB",
        packed=False,
    )

    if isinstance(conf, torch.Tensor):
        conf_prob = torch.sigmoid(conf)
        if conf_prob.ndim == 2:
            conf_prob = conf_prob.unsqueeze(-1)
        opacities_depth = torch.where(
            conf_prob < conf_threshold,
            torch.zeros_like(opacities),
            opacities,
        )
    else:
        opacities_depth = opacities

    depth, _, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities_depth.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W,
        height=H,
        render_mode="ED",
        packed=False,
    )

    return rgb, depth, alpha


def list_sorted_files(directory, extensions):
    filenames = [
        x for x in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, x)) and x.lower().endswith(extensions)
    ]
    return sorted(filenames)


def build_selected_indices(total_count, interval=1, subset_start=None, subset_end=None, subset_step=1):
    base_indices = list(range(0, total_count, interval))
    return base_indices[slice(subset_start, subset_end, subset_step)]


def compute_target_size(width, height, pixel_limit):
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    w_target = width * scale
    h_target = height * scale
    k = round(w_target / 14)
    m = round(h_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(m, 1) > w_target / max(h_target, 1e-8):
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def load_rgb_sequence(path, interval=1, subset_start=None, subset_end=None, subset_step=1, pixel_limit=255000):
    frame_items = []
    sources = []

    if os.path.isdir(path):
        filenames = list_sorted_files(path, RGB_EXTS)
        selected_indices = build_selected_indices(
            len(filenames), interval=interval, subset_start=subset_start, subset_end=subset_end, subset_step=subset_step
        )
        for source_index in selected_indices:
            filename = filenames[source_index]
            full_path = os.path.join(path, filename)
            img = Image.open(full_path).convert("RGB")
            sources.append(img)
            frame_items.append({
                "stem": os.path.splitext(filename)[0],
                "source_index": source_index,
                "path": full_path,
            })
    elif path.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        raw_sources = []
        raw_items = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_sources.append(Image.fromarray(rgb_frame))
                raw_items.append({
                    "stem": f"{frame_idx:06d}",
                    "source_index": frame_idx,
                    "path": None,
                })
            frame_idx += 1
        cap.release()

        subset = slice(subset_start, subset_end, subset_step)
        sources = raw_sources[subset]
        frame_items = raw_items[subset]
    else:
        raise ValueError(f"Unsupported data_path. Must be a directory or .mp4: {path}")

    if not sources:
        return torch.empty(0), [], None, None

    orig_w, orig_h = sources[0].size
    target_w, target_h = compute_target_size(orig_w, orig_h, pixel_limit)

    tensor_list = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.asarray(resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        tensor_list.append(img_tensor)

    imgs = torch.stack(tensor_list, dim=0)
    return imgs, frame_items, (orig_h, orig_w), (target_h, target_w)


def maybe_cuda_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_retryable_inference_error(exc):
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "32-bit index math" in message
        or "input tensor must fit into 32-bit index math" in message
    )


def select_render_gaussians(gaussians, batch_index=0):
    render_keys = {"xyz", "rotation", "scale", "opacity", "color"}
    current = {
        k: v[batch_index:batch_index + 1]
        for k, v in gaussians.items()
        if isinstance(v, torch.Tensor) and k in render_keys
    }

    conf_tensor = gaussians.get("conf")
    if isinstance(conf_tensor, torch.Tensor):
        xyz_tensor = current["xyz"]
        if (
            conf_tensor.ndim in (2, 3)
            and conf_tensor.shape[0] > batch_index
            and conf_tensor.shape[1] == xyz_tensor.shape[1]
        ):
            current["conf"] = conf_tensor[batch_index:batch_index + 1]

    return current


def safe_stem(value):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return safe[:80] if safe else "frame"


def tensor_rgb_to_uint8(tensor):
    img_np = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)


def save_rgb(path, image_rgb):
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(path)


def robust_limits(values, mask=None, low=2.0, high=98.0, fallback=(0.0, 1.0)):
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= mask
    valid = arr[finite]
    if valid.size == 0:
        return fallback
    lo = float(np.percentile(valid, low))
    hi = float(np.percentile(valid, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(valid.mean()) if valid.size > 0 else 0.0
        return center - 0.5, center + 0.5
    return lo, hi


def colorize_scalar(values, path=None, min_val=None, max_val=None, cmap=cv2.COLORMAP_TURBO, mask=None):
    arr = np.asarray(values, dtype=np.float32)
    if min_val is None or max_val is None:
        min_val, max_val = robust_limits(arr, mask=mask)
    norm = (arr - min_val) / (max_val - min_val + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    color_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    if mask is not None:
        color_rgb = color_rgb.copy()
        color_rgb[~mask] = 0
    if path is not None:
        save_rgb(path, color_rgb)
    return color_rgb


def dilate_mask(mask, radius=1):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)


def blend_mask(image_rgb, mask, color, alpha=0.65, dim_background=False):
    base = np.asarray(image_rgb, dtype=np.float32).copy()
    if dim_background:
        base *= 0.55
    mask = np.asarray(mask, dtype=bool)
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    base[mask] = (1.0 - alpha) * base[mask] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def support_overlay(image_rgb, quad_mask=None, low_conf_mask=None, support_mask=None):
    overlay = np.asarray(image_rgb, dtype=np.float32).copy() * 0.55
    if support_mask is not None:
        support_show = dilate_mask(support_mask, radius=1)
        overlay[support_show] = 0.35 * overlay[support_show] + 0.65 * np.array([230, 230, 230], dtype=np.float32)
    if quad_mask is not None:
        quad_show = dilate_mask(quad_mask, radius=2)
        overlay[quad_show] = 0.15 * overlay[quad_show] + 0.85 * np.array([40, 220, 120], dtype=np.float32)
    if low_conf_mask is not None:
        low_show = np.asarray(low_conf_mask, dtype=bool)
        overlay[low_show] = 0.25 * overlay[low_show] + 0.75 * np.array([235, 70, 70], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_panel(images, titles, path, figsize_per_image=3.6):
    fig, axes = plt.subplots(1, len(images), figsize=(figsize_per_image * len(images), figsize_per_image))
    if len(images) == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


SCALE_LEVELS = [1, 2, 4, 8, 16, 32]
SCALE_COLORS = {
    1: (36, 123, 242),
    2: (46, 177, 100),
    4: (250, 190, 60),
    8: (239, 108, 68),
    16: (145, 80, 190),
    32: (88, 88, 88),
}


def discrete_scale_map_image(scale_map):
    scale_arr = np.asarray(scale_map, dtype=np.float32)
    levels = np.asarray(SCALE_LEVELS, dtype=np.float32)
    nearest = levels[np.argmin(np.abs(scale_arr[..., None] - levels.reshape(1, 1, -1)), axis=-1)]
    image = np.zeros((*scale_arr.shape, 3), dtype=np.uint8)
    for level in SCALE_LEVELS:
        image[nearest == level] = SCALE_COLORS[level]
    return image


def save_scale_map_with_legend(scale_img, path):
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))
    ax.imshow(scale_img)
    ax.axis("off")
    handles = [
        Patch(facecolor=np.asarray(SCALE_COLORS[level]) / 255.0, edgecolor="none", label=f"{level}x")
        for level in SCALE_LEVELS
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=True, fontsize=8,
              bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def support_binary_image(support_mask, quad_mask=None, low_conf_mask=None):
    image = np.zeros((*support_mask.shape, 3), dtype=np.uint8)
    support = np.asarray(support_mask, dtype=bool)
    image[support] = (235, 235, 235)
    if quad_mask is not None:
        image[dilate_mask(quad_mask, radius=1)] = (40, 220, 120)
    if low_conf_mask is not None:
        image[np.asarray(low_conf_mask, dtype=bool)] = (235, 70, 70)
    return image


def draw_crop_box(image_rgb, box, color=(255, 255, 255), thickness=2):
    y0, y1, x0, x1 = box
    out = np.asarray(image_rgb, dtype=np.uint8).copy()
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
    return out


def crop_array(arr, box):
    y0, y1, x0, x1 = box
    return arr[y0:y1, x0:x1]


def choose_crop_box(score_map, mode="high", crop_size_ratio=0.28):
    score = np.asarray(score_map, dtype=np.float32)
    h, w = score.shape
    crop_h = max(32, min(h, int(round(h * crop_size_ratio))))
    crop_w = max(32, min(w, int(round(w * crop_size_ratio))))
    blur_kh = max(3, (crop_h // 5) | 1)
    blur_kw = max(3, (crop_w // 5) | 1)
    smooth = cv2.blur(score, (blur_kw, blur_kh))
    if mode == "low":
        cy, cx = np.unravel_index(np.argmin(smooth), smooth.shape)
    else:
        cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    y0 = int(np.clip(cy - crop_h // 2, 0, max(h - crop_h, 0)))
    x0 = int(np.clip(cx - crop_w // 2, 0, max(w - crop_w, 0)))
    return y0, y0 + crop_h, x0, x0 + crop_w


def save_crop_panel(frame_rgb, score_img, scale_img, support_img, binary_img, box, title, path):
    save_panel(
        [
            crop_array(frame_rgb, box),
            crop_array(score_img, box),
            crop_array(scale_img, box),
            crop_array(support_img, box),
            crop_array(binary_img, box),
        ],
        ["Input crop", "Score crop", "Scale crop", "Support overlay", "Binary support"],
        path,
        figsize_per_image=2.4,
    )


def save_support_scale_visuals(frame_rgb, support_mask, quad_mask, low_conf_mask, scale_map,
                               score_map, color_score, depth_score, out_dir, frame_idx,
                               frame_name, crop_size_ratio=0.28):
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{frame_idx:04d}_{safe_stem(frame_name)}"

    support_img = support_overlay(
        frame_rgb,
        quad_mask=quad_mask,
        low_conf_mask=low_conf_mask,
        support_mask=support_mask,
    )
    score_max = robust_limits(score_map, low=1, high=99)[1]
    score_img = colorize_scalar(score_map, min_val=0.0, max_val=max(score_max, 1e-4), cmap=cv2.COLORMAP_INFERNO)
    color_img = colorize_scalar(color_score, min_val=0.0, max_val=max(robust_limits(color_score, low=1, high=99)[1], 1e-4),
                                cmap=cv2.COLORMAP_VIRIDIS)
    depth_img = colorize_scalar(depth_score, min_val=0.0, max_val=max(robust_limits(depth_score, low=1, high=99)[1], 1e-4),
                                cmap=cv2.COLORMAP_MAGMA)
    scale_img = discrete_scale_map_image(scale_map)
    binary_img = support_binary_image(support_mask, quad_mask=quad_mask, low_conf_mask=low_conf_mask)

    save_rgb(os.path.join(out_dir, f"support_mask_{stem}.png"), support_img)
    save_rgb(os.path.join(out_dir, f"support_binary_{stem}.png"), binary_img)
    save_rgb(os.path.join(out_dir, f"score_map_{stem}.png"), score_img)
    save_rgb(os.path.join(out_dir, f"color_score_{stem}.png"), color_img)
    save_rgb(os.path.join(out_dir, f"depth_score_{stem}.png"), depth_img)
    save_rgb(os.path.join(out_dir, f"scale_map_{stem}.png"), scale_img)
    save_scale_map_with_legend(scale_img, os.path.join(out_dir, f"scale_map_legend_{stem}.png"))
    save_panel(
        [frame_rgb, color_img, depth_img, score_img, scale_img, support_img],
        ["Input", "Color score", "Plane residual", "Score map", "Scale map", "Selected support"],
        os.path.join(out_dir, f"allocation_pipeline_panel_{stem}.png"),
        figsize_per_image=2.8,
    )
    save_panel(
        [frame_rgb, score_img, scale_img, support_img, binary_img],
        ["Input", "Score map", "Scale map", "Support overlay", "Binary support"],
        os.path.join(out_dir, f"support_scale_panel_{stem}.png"),
        figsize_per_image=3.0,
    )

    high_box = choose_crop_box(score_map, mode="high", crop_size_ratio=crop_size_ratio)
    low_box = choose_crop_box(score_map, mode="low", crop_size_ratio=crop_size_ratio)
    save_rgb(os.path.join(out_dir, f"crop_boxes_{stem}.png"),
             draw_crop_box(draw_crop_box(frame_rgb, high_box, color=(255, 245, 90)), low_box, color=(90, 210, 255)))
    save_crop_panel(
        frame_rgb, score_img, scale_img, support_img, binary_img, high_box,
        "Complex crop",
        os.path.join(out_dir, f"allocation_crop_complex_{stem}.png"),
    )
    save_crop_panel(
        frame_rgb, score_img, scale_img, support_img, binary_img, low_box,
        "Flat crop",
        os.path.join(out_dir, f"allocation_crop_flat_{stem}.png"),
    )
    return support_img


def save_low_conf_routing_visuals(frame_rgb, conf_prob, low_conf_mask, routed_alpha,
                                  out_dir, frame_idx, frame_name, support_img=None,
                                  routed_mask=None):
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{frame_idx:04d}_{safe_stem(frame_name)}"

    low_overlay = blend_mask(frame_rgb, low_conf_mask, color=(235, 70, 70), alpha=0.70, dim_background=True)
    low_score_img = colorize_scalar(1.0 - conf_prob, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_INFERNO)
    alpha_img = colorize_scalar(routed_alpha, min_val=0.0, max_val=max(float(np.max(routed_alpha)), 1e-4),
                                cmap=cv2.COLORMAP_MAGMA)
    if routed_mask is not None:
        routed_overlay = blend_mask(frame_rgb, routed_mask, color=(255, 155, 45), alpha=0.72, dim_background=True)
    else:
        routed_overlay = low_overlay

    save_rgb(os.path.join(out_dir, f"low_conf_mask_{stem}.png"), low_overlay)
    save_rgb(os.path.join(out_dir, f"low_conf_score_{stem}.png"), low_score_img)
    save_rgb(os.path.join(out_dir, f"low_conf_routed_mask_{stem}.png"), routed_overlay)
    save_rgb(os.path.join(out_dir, f"low_conf_routed_alpha_{stem}.png"), alpha_img)

    route_context = support_img if support_img is not None else low_overlay
    save_panel(
        [frame_rgb, low_score_img, low_overlay, routed_overlay, route_context, alpha_img],
        ["Input", "Low-conf score", "Raw low-conf", "Actually routed", "Routed support", "Low-conf alpha"],
        os.path.join(out_dir, f"low_conf_routing_panel_{stem}.png"),
        figsize_per_image=2.7,
    )


def gaussian_stats(gaussians, conf_threshold=0.1, min_opacity=0.05):
    xyz = gaussians["xyz"][0].detach().float().cpu().numpy()
    scale = gaussians["scale"][0].detach().float().cpu().numpy()
    opacity = gaussians["opacity"][0, :, 0].detach().float().cpu().numpy()
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1) & (opacity > min_opacity)

    conf = gaussians.get("conf")
    if isinstance(conf, torch.Tensor):
        conf_prob = torch.sigmoid(conf[0, :, 0]).detach().float().cpu().numpy()
        low_conf = conf_prob < conf_threshold
    else:
        conf_prob = np.ones(xyz.shape[0], dtype=np.float32)
        low_conf = np.zeros(xyz.shape[0], dtype=bool)

    scale_radius = np.cbrt(np.prod(np.clip(scale, 1e-8, None), axis=1))
    active = valid
    active_low = active & low_conf
    active_normal = active & (~low_conf)

    def mean_or_nan(values, mask):
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    return {
        "num_total": int(xyz.shape[0]),
        "num_active": int(active.sum()),
        "num_low_conf_active": int(active_low.sum()),
        "num_normal_active": int(active_normal.sum()),
        "low_conf_active_ratio": float(active_low.sum() / max(active.sum(), 1)),
        "mean_scale_active": mean_or_nan(scale_radius, active),
        "mean_scale_low_conf": mean_or_nan(scale_radius, active_low),
        "mean_scale_normal": mean_or_nan(scale_radius, active_normal),
        "mean_conf_active": mean_or_nan(conf_prob, active),
    }


def axis_limits(points, idx):
    lo, hi = robust_limits(points[:, idx], low=0.5, high=99.5)
    pad = 0.05 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def scatter_projection(ax, points, low_conf, sizes, axis_a, axis_b, labels):
    normal = ~low_conf
    if np.any(normal):
        ax.scatter(points[normal, axis_a], points[normal, axis_b], s=sizes[normal],
                   c="#2A6FBB", alpha=0.28, linewidths=0, rasterized=True)
    if np.any(low_conf):
        ax.scatter(points[low_conf, axis_a], points[low_conf, axis_b], s=sizes[low_conf],
                   c="#D94A38", alpha=0.45, linewidths=0, rasterized=True)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim(axis_limits(points, axis_a))
    ax.set_ylim(axis_limits(points, axis_b))
    ax.grid(True, linewidth=0.3, alpha=0.35)


def save_gaussian_spatial_distribution(gaussians, path, max_points=80000, min_opacity=0.05,
                                       conf_threshold=0.1, seed=0, title=None):
    xyz = gaussians["xyz"][0].detach().float().cpu().numpy()
    scale = gaussians["scale"][0].detach().float().cpu().numpy()
    opacity = gaussians["opacity"][0, :, 0].detach().float().cpu().numpy()
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1) & (opacity > min_opacity)
    if not np.any(valid):
        valid = np.isfinite(xyz).all(axis=1)

    conf = gaussians.get("conf")
    if isinstance(conf, torch.Tensor):
        conf_prob = torch.sigmoid(conf[0, :, 0]).detach().float().cpu().numpy()
        low_conf = conf_prob < conf_threshold
    else:
        low_conf = np.zeros(xyz.shape[0], dtype=bool)

    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return False
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)

    points = xyz[idx]
    low = low_conf[idx]
    scale_radius = np.cbrt(np.prod(np.clip(scale[idx], 1e-8, None), axis=1))
    s_lo, s_hi = robust_limits(scale_radius, low=5, high=95)
    sizes = 1.0 + 7.0 * np.clip((scale_radius - s_lo) / (s_hi - s_lo + 1e-8), 0.0, 1.0)

    fig = plt.figure(figsize=(11.5, 9.5))
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")

    scatter_projection(ax_xy, points, low, sizes, 0, 1, ("X", "Y"))
    ax_xy.set_title("XY distribution")
    scatter_projection(ax_xz, points, low, sizes, 0, 2, ("X", "Z"))
    ax_xz.set_title("XZ distribution")
    scatter_projection(ax_yz, points, low, sizes, 1, 2, ("Y", "Z"))
    ax_yz.set_title("YZ distribution")

    normal = ~low
    if np.any(normal):
        ax_3d.scatter(points[normal, 0], points[normal, 1], points[normal, 2], s=sizes[normal],
                      c="#2A6FBB", alpha=0.18, linewidths=0)
    if np.any(low):
        ax_3d.scatter(points[low, 0], points[low, 1], points[low, 2], s=sizes[low],
                      c="#D94A38", alpha=0.38, linewidths=0)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_xlim(axis_limits(points, 0))
    ax_3d.set_ylim(axis_limits(points, 1))
    ax_3d.set_zlim(axis_limits(points, 2))
    ax_3d.view_init(elev=23, azim=-55)
    ax_3d.set_title("3D view")

    handles = [
        Line2D([0], [0], marker="o", color="w", label="normal / confident",
               markerfacecolor="#2A6FBB", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="low-confidence routed",
               markerfacecolor="#D94A38", markersize=7),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    if title is None:
        title = "Gaussian spatial distribution"
    low_count = int(low.sum())
    fig.suptitle(f"{title} | active={points.shape[0]} | low-conf={low_count}", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def filter_gaussians_by_conf(gaussians, threshold=0.1, keep_low=True, color=None, min_opacity=0.0):
    conf = gaussians.get("conf")
    if not isinstance(conf, torch.Tensor):
        return None
    prob = torch.sigmoid(conf[0, :, 0])
    selected = prob < threshold if keep_low else prob >= threshold
    selected = selected & (gaussians["opacity"][0, :, 0] > min_opacity)
    if not bool(selected.any()):
        return None

    out = {}
    for key in ("xyz", "rotation", "scale", "opacity", "color", "conf"):
        if key in gaussians and isinstance(gaussians[key], torch.Tensor):
            out[key] = gaussians[key][:, selected].clone()

    if color is not None and "color" in out:
        color_tensor = torch.tensor(color, dtype=out["color"].dtype, device=out["color"].device)
        out["color"] = color_tensor.view(1, 1, 3).expand_as(out["color"]).clone()
    return out


def transform_local_to_world(local_points, c2w):
    points = np.asarray(local_points, dtype=np.float32).reshape(-1, 3)
    R = np.asarray(c2w, dtype=np.float32)[:3, :3]
    t = np.asarray(c2w, dtype=np.float32)[:3, 3]
    return points @ R.T + t.reshape(1, 3)


def sample_indices(mask, max_points, rng):
    indices = np.nonzero(mask.reshape(-1))[0]
    if indices.size > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)
    return indices


def set_equal_limits_3d(ax, points):
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return
    pts = points[finite]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_routing_points(ax, points, route_flags, title):
    normal = ~route_flags
    if np.any(normal):
        ax.scatter(points[normal, 0], points[normal, 1], points[normal, 2],
                   s=1.2, c="#2A6FBB", alpha=0.18, linewidths=0, rasterized=True)
    if np.any(route_flags):
        ax.scatter(points[route_flags, 0], points[route_flags, 1], points[route_flags, 2],
                   s=5.0, c="#D94A38", alpha=0.62, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=-55)


def save_routing_before_after_3d(before_local, after_local, routing_mask, c2w, path,
                                 max_normal_points=20000, max_routed_points=12000, seed=0):
    routing_mask = np.asarray(routing_mask, dtype=bool)
    valid_before = np.isfinite(before_local).all(axis=-1)
    valid_after = np.isfinite(after_local).all(axis=-1)
    valid = valid_before & valid_after
    routed = routing_mask & valid
    normal = (~routing_mask) & valid
    if not np.any(valid):
        return False

    rng = np.random.default_rng(seed)
    normal_idx = sample_indices(normal, max_normal_points, rng)
    routed_idx = sample_indices(routed, max_routed_points, rng)
    if routed_idx.size == 0:
        return False

    selected_idx = np.concatenate([normal_idx, routed_idx])
    route_flags = np.concatenate([
        np.zeros(normal_idx.shape[0], dtype=bool),
        np.ones(routed_idx.shape[0], dtype=bool),
    ])

    before_world = transform_local_to_world(before_local, c2w)[selected_idx]
    after_world = transform_local_to_world(after_local, c2w)[selected_idx]
    all_points = np.concatenate([before_world, after_world], axis=0)

    fig = plt.figure(figsize=(10.5, 4.8))
    ax_before = fig.add_subplot(1, 2, 1, projection="3d")
    ax_after = fig.add_subplot(1, 2, 2, projection="3d")
    plot_routing_points(ax_before, before_world, route_flags, "Before routing")
    plot_routing_points(ax_after, after_world, route_flags, "After routing")
    set_equal_limits_3d(ax_before, all_points)
    set_equal_limits_3d(ax_after, all_points)
    handles = [
        Line2D([0], [0], marker="o", color="w", label="normal points",
               markerfacecolor="#2A6FBB", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="routed low-conf points",
               markerfacecolor="#D94A38", markersize=7),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def write_summary_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Pi3_3DGS qualitative visualizations for representation allocation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_qualitative_3dgs")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=1, help="Number of frames per model forward")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_end", type=int, default=None)
    parser.add_argument("--subset_step", type=int, default=3)
    parser.add_argument("--max_viz_frames", type=int, default=12,
                        help="Maximum per-frame qualitative panels to save. Use -1 for all selected frames.")
    parser.add_argument("--viz_frame_stride", type=int, default=1,
                        help="Save per-frame qualitative panels every K loaded frames.")
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--spatial_max_points", type=int, default=80000)
    parser.add_argument("--spatial_min_opacity", type=float, default=0.05)
    parser.add_argument("--crop_size_ratio", type=float, default=0.28,
                        help="Relative crop size for allocation detail panels.")
    parser.add_argument("--routing_3d_max_normal_points", type=int, default=20000)
    parser.add_argument("--routing_3d_max_routed_points", type=int, default=12000)
    parser.add_argument("--skip_routing_3d", action="store_true",
                        help="Skip before/after routing 3D point visualizations.")
    parser.add_argument("--save_render", action="store_true",
                        help="Also save RGB/depth/opacity render outputs for the qualitative frames.")
    parser.add_argument("--save_ply", action="store_true",
                        help="Also save the chunk Gaussian PLY files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    qualitative_dir = os.path.join(args.output_dir, "qualitative")
    spatial_dir = os.path.join(qualitative_dir, "gaussian_spatial")
    support_dir = os.path.join(qualitative_dir, "support_scale")
    routing_dir = os.path.join(qualitative_dir, "low_conf_routing")
    render_dir = os.path.join(args.output_dir, "renders")
    for directory in (qualitative_dir, spatial_dir, support_dir, routing_dir):
        os.makedirs(directory, exist_ok=True)
    if args.save_render:
        os.makedirs(render_dir, exist_ok=True)

    device = torch.device(args.device)
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith(".mp4") else 1

    print(f"Loading Pi3_3DGS model from {args.ckpt}...")
    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
    ).to(device).eval()

    if args.ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(args.ckpt)
        model.load_state_dict(weight, strict=False)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)

    print(f"Loading RGB frames from {args.data_path}...")
    imgs_cpu, frame_items, orig_hw, target_hw = load_rgb_sequence(
        args.data_path,
        interval=args.interval,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        subset_step=args.subset_step,
        pixel_limit=args.pixel_limit,
    )
    if imgs_cpu.numel() == 0:
        raise RuntimeError("No RGB frames loaded.")

    total_frames = imgs_cpu.shape[0]
    H, W = imgs_cpu.shape[2], imgs_cpu.shape[3]
    print(f"Loaded {total_frames} RGB frames. Original resolution: {orig_hw[0]}x{orig_hw[1]}, model resolution: {H}x{W}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    active_chunk_size = max(1, args.chunk_size)
    start_idx = 0
    viz_saved = 0
    summary_rows = []

    while start_idx < total_frames:
        end_idx = min(start_idx + active_chunk_size, total_frames)
        current_batch_size = end_idx - start_idx
        print(f"\nProcessing chunk: frames {start_idx} to {end_idx - 1} ({current_batch_size} frames)...")

        imgs_batch = imgs_cpu[start_idx:end_idx].unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                with amp_context():
                    res = model(imgs_batch, return_viz=True)
        except RuntimeError as e:
            if is_retryable_inference_error(e):
                maybe_cuda_empty_cache(device)
                gc.collect()
                if active_chunk_size <= 1:
                    print(f"ERROR: inference failed even with chunk_size=1 on chunk {start_idx}-{end_idx - 1}.")
                    raise

                new_chunk_size = max(1, active_chunk_size // 2)
                print(
                    f"Chunk {start_idx}-{end_idx - 1} failed with chunk_size={active_chunk_size}: {e}\n"
                    f"Retrying from frame {start_idx} with chunk_size={new_chunk_size}."
                )
                active_chunk_size = new_chunk_size
                del imgs_batch
                continue
            raise

        gaussians = res["gaussians"]
        pred_c2w = res["camera_poses"]
        pred_K = res["intrinsics"]
        pred_w2c = se3_inverse(pred_c2w)
        current_gaussians = select_render_gaussians(gaussians, batch_index=0)

        if args.save_ply:
            ply_filename = os.path.join(args.output_dir, f"gaussians_chunk_{start_idx:04d}_to_{end_idx - 1:04d}.ply")
            save_ply_binary(current_gaussians, ply_filename)
            print(f"Saved PLY point cloud: {ply_filename}")

        spatial_path = os.path.join(
            spatial_dir,
            f"gaussian_spatial_chunk_{start_idx:04d}_to_{end_idx - 1:04d}.png",
        )
        save_gaussian_spatial_distribution(
            current_gaussians,
            spatial_path,
            max_points=args.spatial_max_points,
            min_opacity=args.spatial_min_opacity,
            conf_threshold=args.conf_threshold,
            seed=start_idx,
            title=f"Chunk {start_idx:04d}-{end_idx - 1:04d}",
        )
        stats = gaussian_stats(
            current_gaussians,
            conf_threshold=args.conf_threshold,
            min_opacity=args.spatial_min_opacity,
        )
        stats.update({"chunk_start": start_idx, "chunk_end": end_idx - 1})
        summary_rows.append(stats)
        print(f"Saved Gaussian spatial distribution: {spatial_path}")

        routing_viz = res.get("routing_viz", {})
        sub_idx = routing_viz.get("sub_idx")
        if isinstance(sub_idx, torch.Tensor):
            sub_idx_list = [int(x) for x in sub_idx.detach().cpu().tolist()]
            sub_lookup = {local_idx: j for j, local_idx in enumerate(sub_idx_list)}
        else:
            sub_lookup = {}

        low_conf_gaussians = filter_gaussians_by_conf(
            current_gaussians,
            threshold=args.conf_threshold,
            keep_low=True,
            color=(1.0, 0.05, 0.02),
            min_opacity=args.spatial_min_opacity,
        )

        for i in range(current_batch_size):
            global_frame_idx = start_idx + i
            if args.viz_frame_stride > 1 and (global_frame_idx % args.viz_frame_stride) != 0:
                continue
            if args.max_viz_frames >= 0 and viz_saved >= args.max_viz_frames:
                continue

            frame_name = frame_items[global_frame_idx]["stem"]
            frame_rgb = tensor_rgb_to_uint8(imgs_batch[0, i])
            conf_prob = torch.sigmoid(res["conf"][0, i, ..., 0]).detach().float().cpu().numpy()
            dense_low_conf = conf_prob < args.conf_threshold

            support_img = None
            routed_mask_frame = None
            if i in sub_lookup:
                j = sub_lookup[i]
                scale_map = routing_viz["scale_map"][0, j].detach().float().cpu().numpy()
                score_map = routing_viz["score_map"][0, j].detach().float().cpu().numpy()
                color_score = routing_viz["color_score"][0, j].detach().float().cpu().numpy()
                depth_score = routing_viz["depth_score"][0, j].detach().float().cpu().numpy()
                support_mask = routing_viz["support_mask"][0, j].detach().bool().cpu().numpy()
                quad_mask = routing_viz["quad_keep_mask"][0, j].detach().bool().cpu().numpy()
                low_conf_mask = routing_viz["low_conf_mask"][0, j].detach().bool().cpu().numpy()
                routed_mask_frame = low_conf_mask
                support_img = save_support_scale_visuals(
                    frame_rgb,
                    support_mask=support_mask,
                    quad_mask=quad_mask,
                    low_conf_mask=low_conf_mask,
                    scale_map=scale_map,
                    score_map=score_map,
                    color_score=color_score,
                    depth_score=depth_score,
                    out_dir=support_dir,
                    frame_idx=global_frame_idx,
                    frame_name=frame_name,
                    crop_size_ratio=args.crop_size_ratio,
                )

            view_w2c = pred_w2c[0:1, i:i + 1]
            view_K = pred_K[0:1, i:i + 1]

            if low_conf_gaussians is not None:
                _, _, low_alpha_tensor = render_frame(
                    low_conf_gaussians,
                    view_w2c,
                    view_K,
                    H,
                    W,
                    num_gaussians=None,
                    conf_threshold=args.conf_threshold,
                )
                low_alpha = low_alpha_tensor[0, 0, ..., 0].detach().float().cpu().numpy()
                del low_alpha_tensor
            else:
                low_alpha = np.zeros((H, W), dtype=np.float32)

            save_low_conf_routing_visuals(
                frame_rgb,
                conf_prob=conf_prob,
                low_conf_mask=dense_low_conf,
                routed_alpha=low_alpha,
                out_dir=routing_dir,
                frame_idx=global_frame_idx,
                frame_name=frame_name,
                support_img=support_img,
                routed_mask=routed_mask_frame,
            )

            routing_mask_full = routing_viz.get("routing_mask")
            before_routing = routing_viz.get("local_points_before_routing")
            if (
                not args.skip_routing_3d
                and isinstance(routing_mask_full, torch.Tensor)
                and isinstance(before_routing, torch.Tensor)
            ):
                before_local = before_routing[0, i].detach().float().cpu().numpy()
                after_local = res["local_points"][0, i].detach().float().cpu().numpy()
                route_mask = routing_mask_full[0, i].detach().bool().cpu().numpy()
                c2w_np = pred_c2w[0, i].detach().float().cpu().numpy()
                save_routing_before_after_3d(
                    before_local,
                    after_local,
                    route_mask,
                    c2w_np,
                    os.path.join(routing_dir, f"routing_before_after_3d_{global_frame_idx:04d}_{safe_stem(frame_name)}.png"),
                    max_normal_points=args.routing_3d_max_normal_points,
                    max_routed_points=args.routing_3d_max_routed_points,
                    seed=global_frame_idx,
                )

            if args.save_render:
                rgb_tensor, depth_tensor, alpha_tensor = render_frame(
                    current_gaussians,
                    view_w2c,
                    view_K,
                    H,
                    W,
                    num_gaussians=None,
                    conf_threshold=args.conf_threshold,
                )
                rgb_out = rgb_tensor[0, 0].permute(2, 0, 1)
                depth_out = depth_tensor[0, 0].permute(2, 0, 1)
                alpha_out = alpha_tensor[0, 0].permute(2, 0, 1)
                save_image(rgb_out, os.path.join(render_dir, f"rgb_{global_frame_idx:04d}.png"))
                save_depth(depth_out, os.path.join(render_dir, f"depth_{global_frame_idx:04d}.png"))
                save_heatmap(alpha_out, os.path.join(render_dir, f"opacity_heatmap_{global_frame_idx:04d}.png"))
                del rgb_tensor, depth_tensor, alpha_tensor, rgb_out, depth_out, alpha_out

            viz_saved += 1

        print(f"Saved qualitative panels so far: {viz_saved}")

        del gaussians, pred_c2w, pred_w2c, pred_K, current_gaussians, res, imgs_batch
        del low_conf_gaussians
        maybe_cuda_empty_cache(device)
        gc.collect()
        start_idx = end_idx

    summary_path = os.path.join(qualitative_dir, "qualitative_gaussian_summary.csv")
    write_summary_csv(summary_rows, summary_path)
    print("\nQualitative visualization complete.")
    print(f"Gaussian spatial plots: {spatial_dir}")
    print(f"Support mask / scale map: {support_dir}")
    print(f"Low-confidence routing: {routing_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
