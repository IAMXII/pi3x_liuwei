import argparse
import csv
import gc
import importlib
import inspect
import math
import os
import sys
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    LearnedPerceptualImagePatchSimilarity

from pi3.utils.alignment import align_depth_affine, align_depth_scale


RGB_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEPTH_EXTS = RGB_EXTS + (".npy", ".npz")
DEFAULT_MODEL_IMPL = "pi3.models.pi3_3dgs_9.Pi3_3DGS"
MODEL_IMPL_ALIASES = {
    "_8": "pi3.models.pi3_3dgs_8.Pi3_3DGS",
    "8": "pi3.models.pi3_3dgs_8.Pi3_3DGS",
    "_9": "pi3.models.pi3_3dgs_9.Pi3_3DGS",
    "9": "pi3.models.pi3_3dgs_9.Pi3_3DGS",
    "_10": "pi3.models.pi3_3dgs_10.Pi3_3DGS",
    "10": "pi3.models.pi3_3dgs_10.Pi3_3DGS",
    "_11": "pi3.models.pi3_3dgs_11.Pi3_3DGS",
    "11": "pi3.models.pi3_3dgs_11.Pi3_3DGS",
    "_12": "pi3.models.pi3_3dgs_12.Pi3_3DGS",
    "12": "pi3.models.pi3_3dgs_12.Pi3_3DGS",
}


def save_heatmap(tensor, path):
    alpha_np = tensor.squeeze().detach().cpu().numpy()
    alpha_np = np.clip(alpha_np, 0, 1)
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap_color)


def _write_ply_binary_arrays(xyz, rot, scale, opacity, color, path):
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


def _gaussian_ply_arrays(gaussians):
    xyz = gaussians["xyz"].detach().cpu().float().numpy().reshape(-1, 3)
    rot = gaussians["rotation"].detach().cpu().float().numpy().reshape(-1, 4)
    scale = gaussians["scale"].detach().cpu().float().numpy().reshape(-1, 3)
    opacity = gaussians["opacity"].detach().cpu().float().numpy().reshape(-1)
    color = gaussians["color"].detach().cpu().float().numpy().reshape(-1, 3)
    return xyz, rot, scale, opacity, color


def save_ply_binary(gaussians, path, opacity_threshold=0.05):
    xyz, rot, scale, opacity, color = _gaussian_ply_arrays(gaussians)
    total_count = xyz.shape[0]
    keep_mask = opacity > opacity_threshold
    xyz = xyz[keep_mask]
    rot = rot[keep_mask]
    scale = scale[keep_mask]
    opacity = opacity[keep_mask]
    color = color[keep_mask]

    _write_ply_binary_arrays(xyz, rot, scale, opacity, color, path)

    return xyz.shape[0], total_count


def _gaussian_optional_vector_tensor(gaussians, key, length, device, fill, dtype=torch.float32):
    value = gaussians.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.full((length,), fill, device=device, dtype=dtype), False
    value = value.detach()
    if value.ndim >= 3:
        value = value[0, :, 0]
    elif value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    else:
        value = value.reshape(-1)
    value = value.reshape(-1).to(device=device, dtype=dtype)
    out = torch.full((length,), fill, device=device, dtype=dtype)
    n = min(length, value.numel())
    if n > 0:
        out[:n] = value[:n]
    return out, True


def _redundancy_suppression_mask_tensor(
    gaussians,
    length,
    device,
    redundancy_coef_threshold=0.0,
    gate_epsilon=1e-6,
    require_multiview_support=False,
):
    active, has_active = _gaussian_optional_vector_tensor(gaussians, "competition_active", length, device, 0.0)
    gate, has_gate = _gaussian_optional_vector_tensor(gaussians, "competition_gate", length, device, 1.0)
    coef, has_coef = _gaussian_optional_vector_tensor(gaussians, "redundancy_coef", length, device, 0.0)
    multiview, has_multiview = _gaussian_optional_vector_tensor(
        gaussians,
        "redundancy_multiview_support",
        length,
        device,
        0.0,
    )

    mask = torch.zeros((length,), device=device, dtype=torch.bool)
    if has_active:
        mask |= active > 0.5
    if has_gate:
        mask |= gate < (1.0 - float(gate_epsilon))
    if not has_active and not has_gate and has_coef:
        mask |= coef > float(redundancy_coef_threshold)
    if require_multiview_support and has_multiview:
        mask &= multiview > 0.5
    return mask


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


def render_frame(gaussians, w2c, K, H, W, num_gaussians=None):
    means = gaussians["xyz"]
    quats = gaussians["rotation"]
    scales = gaussians["scale"]
    opacities = gaussians["opacity"]
    colors = gaussians["color"]
    conf = gaussians.get("conf", None)
    pushed = gaussians.get("pushed", None)

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
            if isinstance(pushed, torch.Tensor):
                pushed = pushed[:, :max_N]

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
            if isinstance(pushed, torch.Tensor):
                pushed = pushed[:, :limit]

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

    if isinstance(pushed, torch.Tensor):
        if pushed.ndim == 2:
            pushed = pushed.unsqueeze(-1)
        if pushed.shape[0] == opacities.shape[0] and pushed.shape[1] == opacities.shape[1]:
            opacities_depth = torch.where(
                pushed.to(device=opacities.device) > 0.5,
                torch.zeros_like(opacities),
                opacities,
            )
        else:
            raise RuntimeError(
                "gaussians['pushed'] must align with opacity for depth rendering. "
                f"Got pushed shape {tuple(pushed.shape)} and opacity shape {tuple(opacities.shape)}."
            )
    elif isinstance(conf, torch.Tensor):
        conf_prob = torch.sigmoid(conf)
        if conf_prob.ndim == 2:
            conf_prob = conf_prob.unsqueeze(-1)
        opacities_depth = torch.where(
            conf_prob < 0.1,
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


def parse_rgb_triplet(value):
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).replace(";", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"RGB color must have three comma-separated values, got {value!r}")
    rgb = [float(part) for part in parts]
    if max(rgb) > 1.0:
        rgb = [channel / 255.0 for channel in rgb]
    return tuple(float(np.clip(channel, 0.0, 1.0)) for channel in rgb)


def _select_gaussian_masked_subset(gaussians, keep_mask):
    filtered = {}
    n = keep_mask.numel()
    for key, value in gaussians.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[0] == 1
            and value.shape[1] == n
        ):
            filtered[key] = value[:, keep_mask]
        else:
            filtered[key] = value
    return filtered


def _redundancy_appearance_multiview_mask(
    gaussians,
    source_images,
    all_w2c,
    all_K,
    H,
    W,
    color_threshold=0.12,
    min_views=2,
    match_radius=2,
    chunk_size=200000,
):
    device = gaussians["opacity"].device
    xyz = gaussians["xyz"].detach().float()[0]
    n = xyz.shape[0]
    if n == 0:
        return torch.zeros((0,), device=device, dtype=torch.bool)

    source_images = source_images.detach().to(device=device, dtype=torch.float32)
    if source_images.ndim == 5:
        source_images = source_images.reshape(-1, *source_images.shape[-3:])
    all_w2c = all_w2c.detach().reshape(-1, 4, 4).to(device=device, dtype=torch.float32)
    all_K = all_K.detach().reshape(-1, 3, 3).to(device=device, dtype=torch.float32)
    view_count = min(source_images.shape[0], all_w2c.shape[0], all_K.shape[0])
    min_views = max(1, int(min_views))
    if view_count < min_views:
        return torch.zeros((n,), device=device, dtype=torch.bool)
    source_images = source_images[:view_count]
    all_w2c = all_w2c[:view_count]
    all_K = all_K[:view_count]

    color = gaussians.get("competition_color")
    if not isinstance(color, torch.Tensor):
        color = gaussians["color"]
    ref_color = color.detach().float()[0].to(device=device).clamp(0.0, 1.0)

    R = all_w2c[:, :3, :3]
    t = all_w2c[:, :3, 3]
    fx = all_K[:, 0, 0].abs().clamp_min(1e-6)
    fy = all_K[:, 1, 1].abs().clamp_min(1e-6)
    cx = all_K[:, 0, 2]
    cy = all_K[:, 1, 2]
    out = torch.zeros((n,), device=device, dtype=torch.bool)
    match_radius = max(0, int(match_radius))
    offsets = [
        (dx, dy)
        for dy in range(-match_radius, match_radius + 1)
        for dx in range(-match_radius, match_radius + 1)
    ]
    chunk_size = max(1, int(chunk_size))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pts = xyz[start:end]
        cam = torch.matmul(pts.unsqueeze(0), R.transpose(1, 2)) + t[:, None, :]
        z = cam[..., 2]
        z_safe = z.clamp_min(1e-6)
        px = fx[:, None] * cam[..., 0] / z_safe + cx[:, None]
        py = fy[:, None] * cam[..., 1] / z_safe + cy[:, None]
        projectable = (
            (z > 1e-6)
            & torch.isfinite(px)
            & torch.isfinite(py)
        )

        min_diff = torch.full_like(px, float("inf"), dtype=torch.float32)
        for dx, dy in offsets:
            px_off = px + float(dx)
            py_off = py + float(dy)
            offset_valid = (
                projectable
                & (px_off >= 0)
                & (px_off <= W - 1)
                & (py_off >= 0)
                & (py_off <= H - 1)
            )
            gx = 2.0 * px_off / max(W - 1, 1) - 1.0
            gy = 2.0 * py_off / max(H - 1, 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).reshape(view_count, end - start, 1, 2)
            sampled = F.grid_sample(
                source_images,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(-1).permute(0, 2, 1)
            diff = torch.linalg.norm(sampled - ref_color[start:end].unsqueeze(0), dim=-1)
            diff = torch.where(offset_valid, diff, torch.full_like(diff, float("inf")))
            min_diff = torch.minimum(min_diff, diff)
        support = min_diff <= float(color_threshold)
        out[start:end] = support.sum(dim=0) >= min_views

    return out


def render_gaussian_alpha(gaussians, w2c, K, H, W):
    means = gaussians["xyz"].detach()
    if means.shape[1] == 0:
        return torch.zeros((1, 1, H, W, 1), device=means.device, dtype=torch.float32)
    if means.device.type != "cuda":
        _, alpha = render_gaussian_scene_frame(
            gaussians,
            w2c,
            K,
            H,
            W,
            ellipsoid_alpha_threshold=0.0,
            depth_test=False,
        )
        return alpha

    ones_color = torch.ones_like(gaussians["color"].detach()).contiguous().float()
    _, alpha, _ = rasterization(
        means=means.contiguous().float(),
        quats=gaussians["rotation"].detach().contiguous().float(),
        scales=gaussians["scale"].detach().contiguous().float(),
        opacities=gaussians["opacity"].detach().squeeze(-1).contiguous().float(),
        colors=ones_color,
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W,
        height=H,
        render_mode="RGB",
        packed=False,
    )
    return alpha


def _dilate_binary_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    return F.max_pool2d(
        mask.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ).squeeze(0)


def _erode_binary_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    return 1.0 - _dilate_binary_mask(1.0 - mask, radius)


def _close_binary_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    return _erode_binary_mask(_dilate_binary_mask(mask, radius), radius)


def _binary_mask_boundary(mask, width):
    width = int(width)
    if width <= 0:
        return torch.zeros_like(mask)
    dilated = _dilate_binary_mask(mask, width)
    eroded = _erode_binary_mask(mask, width)
    return (dilated - eroded).clamp(0.0, 1.0)


def build_redundancy_suppression_overlay(
    gaussians,
    base_image,
    w2c,
    K,
    H,
    W,
    overlay_color=(1.0, 0.15, 0.0),
    overlay_alpha=0.55,
    min_gaussian_opacity=0.2,
    redundancy_coef_threshold=0.0,
    overlay_mask=None,
    mask_threshold=0.02,
    connect_radius=4,
    boundary_width=2,
    boundary_alpha=0.95,
    boundary_color=None,
):
    device = gaussians["opacity"].device
    opacity_flat = gaussians["opacity"].detach().float()[0].reshape(-1)
    if overlay_mask is None:
        suppression_mask = _redundancy_suppression_mask_tensor(
            gaussians,
            opacity_flat.numel(),
            device,
            redundancy_coef_threshold=redundancy_coef_threshold,
        )
    else:
        suppression_mask = overlay_mask.detach().to(device=device, dtype=torch.bool).reshape(-1)
    suppressed_count = int(suppression_mask.sum().item())
    if suppressed_count == 0:
        mask_alpha = torch.zeros((1, H, W), device=base_image.device, dtype=torch.float32)
        return base_image.clamp(0.0, 1.0), mask_alpha, suppressed_count

    overlay_gaussians = _select_gaussian_masked_subset(gaussians, suppression_mask)
    overlay_gaussians = dict(overlay_gaussians)
    overlay_opacity = overlay_gaussians["opacity"].detach().clone().float()
    if min_gaussian_opacity > 0:
        overlay_opacity = torch.maximum(
            overlay_opacity,
            torch.full_like(overlay_opacity, float(min_gaussian_opacity)),
        )
    overlay_gaussians["opacity"] = overlay_opacity.clamp(0.0, 0.99)

    alpha = render_gaussian_alpha(overlay_gaussians, w2c, K, H, W)
    mask_alpha = alpha[0, 0].permute(2, 0, 1).to(device=base_image.device).clamp(0.0, 1.0)
    region_mask = (mask_alpha >= float(mask_threshold)).to(dtype=base_image.dtype)
    region_mask = _close_binary_mask(region_mask, connect_radius)
    boundary_mask = _binary_mask_boundary(region_mask, boundary_width)

    overlay_weight = (region_mask * float(overlay_alpha)).clamp(0.0, 1.0)
    color = torch.tensor(overlay_color, device=base_image.device, dtype=base_image.dtype).view(3, 1, 1)
    overlay = base_image.clamp(0.0, 1.0) * (1.0 - overlay_weight) + color * overlay_weight

    if boundary_alpha > 0 and boundary_mask.any():
        if boundary_color is None:
            boundary_color = overlay_color
        boundary_weight = (boundary_mask * float(boundary_alpha)).clamp(0.0, 1.0)
        boundary_rgb = torch.tensor(boundary_color, device=base_image.device, dtype=base_image.dtype).view(3, 1, 1)
        overlay = overlay * (1.0 - boundary_weight) + boundary_rgb * boundary_weight

    return overlay.clamp(0.0, 1.0), region_mask, suppressed_count


def _quat_to_rotation_matrix(quats):
    quats = F.normalize(quats.float(), dim=-1)
    r, x, y, z = quats.unbind(dim=-1)
    one = torch.ones_like(r)
    two = 2.0
    return torch.stack(
        [
            one - two * (y * y + z * z),
            two * (x * y - r * z),
            two * (x * z + r * y),
            two * (x * y + r * z),
            one - two * (x * x + z * z),
            two * (y * z - r * x),
            two * (x * z - r * y),
            two * (y * z + r * x),
            one - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quats.shape[:-1] + (3, 3))


def _select_gaussian_scene_subset(
    opacities,
    px,
    py,
    H,
    W,
    max_gaussians,
    selection_mode="screen_tile",
    tile_size=16,
):
    if max_gaussians is None or int(max_gaussians) <= 0:
        return None
    max_gaussians = int(max_gaussians)
    if opacities.numel() <= max_gaussians:
        return None

    if selection_mode == "opacity":
        return torch.topk(opacities, k=max_gaussians, largest=True, sorted=False).indices

    tile_size = max(1, int(tile_size))
    tiles_x = max(1, int(math.ceil(float(W) / float(tile_size))))
    tiles_y = max(1, int(math.ceil(float(H) / float(tile_size))))
    tile_x = torch.floor(px / float(tile_size)).long().clamp(0, tiles_x - 1)
    tile_y = torch.floor(py / float(tile_size)).long().clamp(0, tiles_y - 1)
    tile_id = tile_y * tiles_x + tile_x

    nonempty_tiles = max(1, int(torch.unique(tile_id).numel()))
    per_tile = max(1, max_gaussians // nonempty_tiles)
    # Sort by tile first and opacity second so each visible image region keeps
    # candidates. Global top-k alone creates empty chunks in large scenes.
    sort_key = tile_id.to(torch.float64) * 2.0 - opacities.clamp(0.0, 1.0).to(torch.float64)
    sorted_idx = torch.argsort(sort_key)
    sorted_tile = tile_id.index_select(0, sorted_idx)

    tile_start = torch.ones(sorted_tile.shape[0], device=sorted_tile.device, dtype=torch.bool)
    tile_start[1:] = sorted_tile[1:] != sorted_tile[:-1]
    group_ids = torch.cumsum(tile_start.to(torch.long), dim=0) - 1
    start_positions = torch.nonzero(tile_start, as_tuple=False).flatten()
    ranks = torch.arange(sorted_tile.shape[0], device=sorted_tile.device) - start_positions.index_select(0, group_ids)
    selected = sorted_idx.index_select(0, torch.nonzero(ranks < per_tile, as_tuple=False).flatten())

    if selected.numel() > max_gaussians:
        rel = torch.topk(opacities.index_select(0, selected), k=max_gaussians, largest=True, sorted=False).indices
        return selected.index_select(0, rel)

    if selected.numel() < max_gaussians:
        remaining = max_gaussians - int(selected.numel())
        keep_mask = torch.ones(opacities.shape[0], device=opacities.device, dtype=torch.bool)
        keep_mask[selected] = False
        rest_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        if rest_idx.numel() > 0:
            k = min(remaining, int(rest_idx.numel()))
            rel = torch.topk(opacities.index_select(0, rest_idx), k=k, largest=True, sorted=False).indices
            selected = torch.cat([selected, rest_idx.index_select(0, rel)], dim=0)

    return selected


def render_gaussian_scene_frame(
    gaussians,
    w2c,
    K,
    H,
    W,
    num_gaussians=None,
    scale_modifier=1.0,
    opacity_multiplier=1.0,
    opacity_power=1.0,
    opacity_threshold=None,
    ellipsoid_alpha_threshold=0.05,
    max_gaussians=0,
    selection_mode="screen_tile",
    selection_tile_size=16,
    exclude_pushed=False,
    depth_test=True,
    max_screen_radius=0.0,
):
    means = gaussians["xyz"].detach()
    quats = gaussians["rotation"].detach()
    scales = gaussians["scale"].detach()
    opacities = gaussians["opacity"].detach()
    colors = gaussians["color"].detach()
    pushed = gaussians.get("pushed", None)
    if isinstance(pushed, torch.Tensor):
        pushed = pushed.detach()
    device = means.device

    B = means.shape[0]

    if num_gaussians is not None:
        if isinstance(num_gaussians, torch.Tensor):
            max_N = int(num_gaussians.max().item())
            means = means[:, :max_N]
            quats = quats[:, :max_N]
            scales = scales[:, :max_N]
            opacities = opacities[:, :max_N].clone()
            colors = colors[:, :max_N]
            if isinstance(pushed, torch.Tensor):
                pushed = pushed[:, :max_N]

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
            if isinstance(pushed, torch.Tensor):
                pushed = pushed[:, :limit]

    means = means[0].float()
    quats = quats[0].float()
    scales = scales[0].float() * float(scale_modifier)
    opacities = opacities[0].reshape(-1).float().clone()
    colors = colors[0].float().clamp(0.0, 1.0)
    if isinstance(pushed, torch.Tensor):
        pushed = pushed[0].reshape(-1).to(device=device)

    if opacity_threshold is not None:
        opacities = torch.where(
            opacities > float(opacity_threshold),
            opacities,
            torch.zeros_like(opacities),
        )
    if opacity_power != 1.0:
        opacities = opacities.clamp(0.0, 1.0).pow(float(opacity_power))
    if opacity_multiplier != 1.0:
        opacities = opacities * float(opacity_multiplier)
    opacities = opacities.clamp(0.0, 0.99)

    w2c_view = w2c.reshape(-1, 4, 4)[0].to(device=device, dtype=torch.float32)
    K_view = K.reshape(-1, 3, 3)[0].to(device=device, dtype=torch.float32)
    R_view = w2c_view[:3, :3]
    t_view = w2c_view[:3, 3]
    xyz_cam = means @ R_view.transpose(0, 1) + t_view
    z_cam = xyz_cam[:, 2]

    fx = K_view[0, 0].abs().clamp_min(1e-6)
    fy = K_view[1, 1].abs().clamp_min(1e-6)
    cx = K_view[0, 2]
    cy = K_view[1, 2]
    z_safe = z_cam.clamp_min(1e-6)
    px = fx * xyz_cam[:, 0] / z_safe + cx
    py = fy * xyz_cam[:, 1] / z_safe + cy
    ndc_x = 2.0 * px / max(float(W), 1.0) - 1.0
    ndc_y = 2.0 * py / max(float(H), 1.0) - 1.0

    finite = (
        torch.isfinite(means).all(dim=-1)
        & torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(quats).all(dim=-1)
        & torch.isfinite(colors).all(dim=-1)
        & torch.isfinite(opacities)
        & torch.isfinite(px)
        & torch.isfinite(py)
        & (opacities > 0)
        & (z_cam > 1e-6)
        & (ndc_x.abs() <= 1.3)
        & (ndc_y.abs() <= 1.3)
    )
    if exclude_pushed and isinstance(pushed, torch.Tensor) and pushed.numel() == finite.numel():
        finite &= pushed <= 0.5

    if not finite.any():
        rgb = torch.zeros((1, 1, H, W, 3), device=device, dtype=torch.float32)
        alpha = torch.zeros((1, 1, H, W, 1), device=device, dtype=torch.float32)
        return rgb, alpha

    def keep(mask):
        return (
            means[mask],
            quats[mask],
            scales[mask],
            opacities[mask],
            colors[mask],
            xyz_cam[mask],
            z_cam[mask],
            px[mask],
            py[mask],
        )

    means, quats, scales, opacities, colors, xyz_cam, z_cam, px, py = keep(finite)

    top_idx = _select_gaussian_scene_subset(
        opacities,
        px,
        py,
        H,
        W,
        max_gaussians,
        selection_mode=selection_mode,
        tile_size=selection_tile_size,
    )
    if top_idx is not None:
        means = means.index_select(0, top_idx)
        quats = quats.index_select(0, top_idx)
        scales = scales.index_select(0, top_idx)
        opacities = opacities.index_select(0, top_idx)
        colors = colors.index_select(0, top_idx)
        xyz_cam = xyz_cam.index_select(0, top_idx)
        z_cam = z_cam.index_select(0, top_idx)
        px = px.index_select(0, top_idx)
        py = py.index_select(0, top_idx)

    rotations = _quat_to_rotation_matrix(quats)
    scale_mats = torch.diag_embed(scales)
    # MonoGS gau_vert.glsl uses M = S * R and Sigma = transpose(M) * M.
    m = torch.matmul(scale_mats, rotations)
    cov_world = torch.matmul(m.transpose(1, 2), m)
    cov_cam = torch.matmul(
        R_view.unsqueeze(0),
        torch.matmul(cov_world, R_view.transpose(0, 1).unsqueeze(0)),
    )

    z_safe = z_cam.clamp_min(1e-6)
    tan_fovx = (0.5 * float(W)) / fx
    tan_fovy = (0.5 * float(H)) / fy
    x_cam = (xyz_cam[:, 0] / z_safe).clamp(-1.3 * tan_fovx, 1.3 * tan_fovx) * z_safe
    y_cam = (xyz_cam[:, 1] / z_safe).clamp(-1.3 * tan_fovy, 1.3 * tan_fovy) * z_safe
    zeros = torch.zeros_like(z_safe)
    jac = torch.stack(
        [
            fx / z_safe,
            zeros,
            -(fx * x_cam) / (z_safe * z_safe),
            zeros,
            fy / z_safe,
            -(fy * y_cam) / (z_safe * z_safe),
        ],
        dim=-1,
    ).reshape(-1, 2, 3)
    cov2d = torch.matmul(jac, torch.matmul(cov_cam, jac.transpose(1, 2)))
    cov00 = cov2d[:, 0, 0] + 0.3
    cov01 = cov2d[:, 0, 1]
    cov11 = cov2d[:, 1, 1] + 0.3
    det = cov00 * cov11 - cov01 * cov01

    valid_cov = torch.isfinite(det) & (det > 1e-8) & (cov00 > 0) & (cov11 > 0)
    if not valid_cov.any():
        rgb = torch.zeros((1, 1, H, W, 3), device=device, dtype=torch.float32)
        alpha = torch.zeros((1, 1, H, W, 1), device=device, dtype=torch.float32)
        return rgb, alpha

    colors = colors[valid_cov]
    opacities = opacities[valid_cov]
    z_cam = z_cam[valid_cov]
    px = px[valid_cov]
    py = py[valid_cov]
    cov00 = cov00[valid_cov]
    cov01 = cov01[valid_cov]
    cov11 = cov11[valid_cov]
    det = det[valid_cov]

    conic00 = cov11 / det
    conic01 = -cov01 / det
    conic11 = cov00 / det
    radius_x = 3.0 * torch.sqrt(cov00.clamp_min(1e-8))
    radius_y = 3.0 * torch.sqrt(cov11.clamp_min(1e-8))
    if max_screen_radius is not None and float(max_screen_radius) > 0:
        radius_limit = float(max_screen_radius)
        radius_valid = (radius_x <= radius_limit) & (radius_y <= radius_limit)
        if not radius_valid.any():
            rgb = torch.zeros((1, 1, H, W, 3), device=device, dtype=torch.float32)
            alpha = torch.zeros((1, 1, H, W, 1), device=device, dtype=torch.float32)
            return rgb, alpha
        colors = colors[radius_valid]
        opacities = opacities[radius_valid]
        z_cam = z_cam[radius_valid]
        px = px[radius_valid]
        py = py[radius_valid]
        conic00 = conic00[radius_valid]
        conic01 = conic01[radius_valid]
        conic11 = conic11[radius_valid]
        radius_x = radius_x[radius_valid]
        radius_y = radius_y[radius_valid]

    x0 = torch.floor(px - radius_x).long().clamp(0, W - 1)
    x1 = torch.ceil(px + radius_x).long().clamp(0, W - 1)
    y0 = torch.floor(py - radius_y).long().clamp(0, H - 1)
    y1 = torch.ceil(py + radius_y).long().clamp(0, H - 1)
    visible = (
        (x1 >= x0)
        & (y1 >= y0)
        & (px + radius_x >= 0)
        & (px - radius_x < W)
        & (py + radius_y >= 0)
        & (py - radius_y < H)
    )
    if not visible.any():
        rgb = torch.zeros((1, 1, H, W, 3), device=device, dtype=torch.float32)
        alpha = torch.zeros((1, 1, H, W, 1), device=device, dtype=torch.float32)
        return rgb, alpha

    colors = colors[visible]
    opacities = opacities[visible]
    z_cam = z_cam[visible]
    px = px[visible]
    py = py[visible]
    conic00 = conic00[visible]
    conic01 = conic01[visible]
    conic11 = conic11[visible]
    x0 = x0[visible]
    x1 = x1[visible]
    y0 = y0[visible]
    y1 = y1[visible]

    order = torch.argsort(z_cam, descending=True)
    colors = colors.index_select(0, order).cpu().numpy()
    opacities = opacities.index_select(0, order).cpu().numpy()
    z_cam = z_cam.index_select(0, order).cpu().numpy()
    px = px.index_select(0, order).cpu().numpy()
    py = py.index_select(0, order).cpu().numpy()
    conic00 = conic00.index_select(0, order).cpu().numpy()
    conic01 = conic01.index_select(0, order).cpu().numpy()
    conic11 = conic11.index_select(0, order).cpu().numpy()
    x0 = x0.index_select(0, order).cpu().numpy()
    x1 = x1.index_select(0, order).cpu().numpy()
    y0 = y0.index_select(0, order).cpu().numpy()
    y1 = y1.index_select(0, order).cpu().numpy()

    image = np.zeros((H, W, 3), dtype=np.float32)
    alpha_acc = np.zeros((H, W), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    threshold = float(ellipsoid_alpha_threshold)

    for idx in range(colors.shape[0]):
        xa, xb = int(x0[idx]), int(x1[idx])
        ya, yb = int(y0[idx]), int(y1[idx])
        if xa > xb or ya > yb:
            continue

        xs = np.arange(xa, xb + 1, dtype=np.float32)[None, :] - float(px[idx])
        ys = np.arange(ya, yb + 1, dtype=np.float32)[:, None] - float(py[idx])
        power = (
            -0.5 * (float(conic00[idx]) * xs * xs + float(conic11[idx]) * ys * ys)
            - float(conic01[idx]) * xs * ys
        )
        exp_power = np.exp(np.minimum(power, 0.0)).astype(np.float32)
        opacity = np.minimum(0.99, float(opacities[idx]) * exp_power)
        src_alpha = ((power <= 0.0) & (opacity >= (1.0 / 255.0)) & (opacity > threshold)).astype(np.float32)
        src_mask = src_alpha > 0.0
        if bool(depth_test):
            depth_roi = depth_buffer[ya:yb + 1, xa:xb + 1]
            src_mask = src_mask & (float(z_cam[idx]) < depth_roi)
        if not np.any(src_mask):
            continue

        src_rgb = colors[idx][None, None, :] * exp_power[:, :, None]
        roi = image[ya:yb + 1, xa:xb + 1]
        alpha_roi = alpha_acc[ya:yb + 1, xa:xb + 1]
        if bool(depth_test):
            depth_roi[src_mask] = float(z_cam[idx])
            roi[src_mask] = src_rgb[src_mask]
            alpha_roi[src_mask] = 1.0
        else:
            inv_alpha = 1.0 - src_alpha
            roi[:] = src_rgb * src_alpha[:, :, None] + roi * inv_alpha[:, :, None]
            alpha_roi[:] = src_alpha + alpha_roi * inv_alpha

    rgb = torch.from_numpy(image).to(device=device, dtype=torch.float32).reshape(1, 1, H, W, 3)
    alpha = torch.from_numpy(alpha_acc[..., None]).to(device=device, dtype=torch.float32).reshape(1, 1, H, W, 1)
    return rgb, alpha


def list_sorted_files(directory, extensions):
    filenames = [
        x for x in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, x)) and x.lower().endswith(extensions)
    ]
    return sorted(filenames)


def build_selected_indices(total_count, interval=1, subset_start=None, subset_end=None, subset_step=1):
    base_indices = list(range(0, total_count, interval))
    return base_indices[slice(subset_start, subset_end, subset_step)]


def build_evenly_spaced_indices(total_count, target_count):
    if total_count <= 0:
        return []
    target_count = max(1, int(target_count))
    if target_count >= total_count:
        return list(range(total_count))

    raw = np.linspace(0, total_count - 1, num=target_count)
    indices = sorted({int(round(x)) for x in raw})
    return indices


def build_gaussian_input_indices(total_count, stride=2):
    if total_count <= 0:
        return []
    stride = max(1, int(stride))
    indices = list(range(0, total_count, stride))
    if indices[-1] != total_count - 1:
        indices.append(total_count - 1)
    return indices


def format_index_preview(indices, max_items=24):
    if len(indices) <= max_items:
        return ", ".join(str(i) for i in indices)
    head_count = max_items // 2
    tail_count = max_items - head_count
    head = ", ".join(str(i) for i in indices[:head_count])
    tail = ", ".join(str(i) for i in indices[-tail_count:])
    return f"{head}, ..., {tail}"


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


def parse_resolution_pair(value):
    if isinstance(value, (list, tuple)):
        width, height = value
        return int(width), int(height)
    if "x" in value:
        width, height = value.lower().split("x", 1)
    elif "," in value:
        width, height = value.split(",", 1)
    else:
        raise ValueError(f"Resolution must look like 518x336 or 518,336, got {value!r}")
    return int(width), int(height)


def format_frame_name_preview(frame_items, max_items=120):
    names = [item.get("frame_name") or os.path.basename(item.get("path") or item["stem"]) for item in frame_items]
    if len(names) <= max_items:
        return ";".join(names)
    head_count = max_items // 2
    tail_count = max_items - head_count
    return ";".join(names[:head_count] + ["..."] + names[-tail_count:])


def load_rgb_sequence(path, interval=1, subset_start=None, subset_end=None, subset_step=1,
                      pixel_limit=255000, target_frame_count=None):
    frame_items = []
    sources = []

    if os.path.isdir(path):
        filenames = list_sorted_files(path, RGB_EXTS)
        if target_frame_count is not None:
            selected_indices = build_evenly_spaced_indices(len(filenames), target_frame_count)
        else:
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
    to_tensor = torch.from_numpy

    tensor_list = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.asarray(resized, dtype=np.float32) / 255.0
        img_tensor = to_tensor(img_np).permute(2, 0, 1)
        tensor_list.append(img_tensor)

    imgs = torch.stack(tensor_list, dim=0)
    return imgs, frame_items, (orig_h, orig_w), (target_h, target_w)


def load_three_sixty_v2_sequence(args):
    from datasets.base.transforms import ImgToTensor
    from datasets.three_sixty_v2_dataset import ThreeSixtyV2Dataset

    target_w, target_h = parse_resolution_pair(args.dataset_resolution)
    dataset = ThreeSixtyV2Dataset(
        data_root=args.dataset_root,
        mode=args.dataset_split,
        image_dir_name=args.dataset_image_dir_name,
        hold_every=args.dataset_hold_every,
        frame_num=args.dataset_frame_num,
        resolution=[[target_w, target_h]],
        transform=ImgToTensor,
        scene_names=args.dataset_scene,
        z_far=0,
        shuffle=args.dataset_shuffle_views,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No 360_v2 frames found for scene={args.dataset_scene}, split={args.dataset_split}, "
            f"root={args.dataset_root}, image_dir={args.dataset_image_dir_name}."
        )

    dataset._rng = np.random.default_rng(args.dataset_seed)
    dataset_index = args.dataset_index % len(dataset)
    views = dataset[dataset_index]
    imgs = torch.stack([view["img"] for view in views], dim=0)
    target_h, target_w = imgs.shape[2], imgs.shape[3]
    frame_items = []
    for source_index, view in enumerate(views):
        frame_name = str(view.get("instance", f"{source_index:06d}"))
        frame_items.append({
            "stem": os.path.splitext(frame_name)[0],
            "source_index": source_index,
            "path": None,
            "frame_name": frame_name,
            "dataset_scene": str(view.get("label", args.dataset_scene or "")),
            "dataset_index": dataset_index,
        })
    return imgs, frame_items, (target_h, target_w), (target_h, target_w)


def select_npz_array(npz_data):
    for key in ("depth", "depths", "arr_0"):
        if key in npz_data:
            return npz_data[key]
    for key in npz_data.files:
        return npz_data[key]
    raise ValueError("No array found in depth npz file.")


def load_raw_depth_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in RGB_EXTS:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Failed to read depth file: {path}")
    elif ext == ".npy":
        depth = np.load(path)
    elif ext == ".npz":
        with np.load(path) as npz_data:
            depth = select_npz_array(npz_data)
    else:
        raise ValueError(f"Unsupported depth file extension: {ext}")

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth file must be single-channel, got shape {depth.shape} from {path}")
    return depth


def resolve_depth_path(data_path, depth_path):
    if depth_path is not None:
        return depth_path

    if os.path.isdir(data_path):
        data_path_norm = os.path.normpath(data_path)
        parent_dir = os.path.dirname(data_path_norm)
        sibling_depth = os.path.join(parent_dir, "depth")
        if os.path.basename(data_path_norm).lower() == "rgb" and os.path.isdir(sibling_depth):
            print(f"Auto depth_path resolved to sibling directory: {sibling_depth}")
            return sibling_depth

    return None


def infer_depth_unit_scale(raw_depth, configured_scale):
    if configured_scale is not None:
        return configured_scale

    if np.issubdtype(raw_depth.dtype, np.integer):
        depth_max = float(np.max(raw_depth)) if raw_depth.size > 0 else 0.0
        if depth_max > 1000:
            print("Auto depth_unit_scale=0.001 (detected integer depth values, assuming millimeters).")
            return 0.001

    print("Auto depth_unit_scale=1.0")
    return 1.0


def resize_sparse_depth(depth, target_h, target_w):
    src_h, src_w = depth.shape
    if src_h == target_h and src_w == target_w:
        return depth.copy()

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((target_h, target_w), dtype=np.float32)

    ys, xs = np.nonzero(valid)
    values = depth[ys, xs].astype(np.float32)

    new_x = np.rint((xs + 0.5) * target_w / src_w - 0.5).astype(np.int64)
    new_y = np.rint((ys + 0.5) * target_h / src_h - 0.5).astype(np.int64)
    new_x = np.clip(new_x, 0, target_w - 1)
    new_y = np.clip(new_y, 0, target_h - 1)

    out = np.full((target_h, target_w), np.inf, dtype=np.float32)
    np.minimum.at(out.reshape(-1), new_y * target_w + new_x, values)
    out[~np.isfinite(out)] = 0.0
    return out


def resize_depth(depth, target_h, target_w, mode="sparse"):
    if mode == "sparse":
        return resize_sparse_depth(depth, target_h, target_w)
    if depth.shape == (target_h, target_w):
        return depth.copy()
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def load_depth_sequence(depth_path, frame_items, target_hw, resize_mode="sparse", depth_unit_scale=None,
                        max_eval_depth=None):
    target_h, target_w = target_hw

    if os.path.isdir(depth_path):
        depth_filenames = list_sorted_files(depth_path, DEPTH_EXTS)
        depth_by_stem = {os.path.splitext(name)[0]: os.path.join(depth_path, name) for name in depth_filenames}

        if all(item["stem"] in depth_by_stem for item in frame_items):
            selected_depths = [depth_by_stem[item["stem"]] for item in frame_items]
            match_mode = "stem"
        else:
            if len(depth_filenames) < len(frame_items):
                raise ValueError(
                    f"Depth directory has fewer files than RGB selection: {len(depth_filenames)} < {len(frame_items)}"
                )
            selected_depths = [os.path.join(depth_path, depth_filenames[i]) for i in range(len(frame_items))]
            match_mode = "sorted_order"

        sample_depth = load_raw_depth_file(selected_depths[0])
        used_depth_unit_scale = infer_depth_unit_scale(sample_depth, depth_unit_scale)

        depth_list = []
        for path in selected_depths:
            raw_depth = load_raw_depth_file(path).astype(np.float32)
            raw_depth *= used_depth_unit_scale
            raw_depth[~np.isfinite(raw_depth)] = 0.0
            raw_depth[raw_depth <= 0] = 0.0
            if max_eval_depth is not None:
                raw_depth[raw_depth > max_eval_depth] = 0.0
            depth_resized = resize_depth(raw_depth, target_h, target_w, mode=resize_mode)
            depth_list.append(torch.from_numpy(depth_resized).float())
        return torch.stack(depth_list, dim=0), match_mode, used_depth_unit_scale

    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth_stack = np.load(depth_path)
    elif ext == ".npz":
        with np.load(depth_path) as npz_data:
            depth_stack = select_npz_array(npz_data)
    else:
        raise ValueError("depth_path must be a directory, .npy, or .npz")

    if depth_stack.ndim != 3:
        raise ValueError(f"Depth stack must have shape [N, H, W], got {depth_stack.shape}")

    used_depth_unit_scale = infer_depth_unit_scale(depth_stack[0], depth_unit_scale)
    depth_list = []
    for item in frame_items:
        source_index = item["source_index"]
        if source_index >= depth_stack.shape[0]:
            raise IndexError(f"Depth stack index {source_index} out of range for shape {depth_stack.shape}")
        depth = depth_stack[source_index].astype(np.float32) * used_depth_unit_scale
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0
        if max_eval_depth is not None:
            depth[depth > max_eval_depth] = 0.0
        depth_resized = resize_depth(depth, target_h, target_w, mode=resize_mode)
        depth_list.append(torch.from_numpy(depth_resized).float())
    return torch.stack(depth_list, dim=0), "stack_index", used_depth_unit_scale


def gather_valid_pairs(pred_depths, gt_depths, min_eval_depth=1e-5, max_eval_depth=None):
    pred_flat_list = []
    gt_flat_list = []
    valid_per_frame = []

    for pred_depth, gt_depth in zip(pred_depths, gt_depths):
        mask = torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
        mask &= pred_depth > min_eval_depth
        mask &= gt_depth > min_eval_depth
        if max_eval_depth is not None:
            mask &= gt_depth <= max_eval_depth
        valid_per_frame.append(mask)
        if mask.any():
            pred_flat_list.append(pred_depth[mask].double())
            gt_flat_list.append(gt_depth[mask].double())

    if not pred_flat_list:
        return None, None, valid_per_frame

    return torch.cat(pred_flat_list), torch.cat(gt_flat_list), valid_per_frame


def subsample_pairs(src, tgt, max_points=200000, seed=0):
    if src.numel() <= max_points:
        return src, tgt
    generator = torch.Generator(device=src.device)
    generator.manual_seed(seed)
    indices = torch.randperm(src.numel(), generator=generator, device=src.device)[:max_points]
    return src[indices], tgt[indices]


def solve_alignment(src, tgt, mode, eps=1e-8):
    src = src.reshape(-1).double()
    tgt = tgt.reshape(-1).double()

    if src.numel() == 0:
        raise ValueError("No valid points for alignment.")

    if mode == "median_scale":
        scale = torch.median(tgt) / torch.median(src).clamp_min(eps)
        shift = torch.zeros_like(scale)
    elif mode == "l2_scale":
        scale = torch.sum(src * tgt) / torch.sum(src * src).clamp_min(eps)
        shift = torch.zeros_like(scale)
    elif mode == "robust_scale":
        weight = torch.ones_like(src)
        scale = align_depth_scale(src[None], tgt[None], weight[None]).squeeze(0)
        shift = torch.zeros_like(scale)
    elif mode == "robust_rel_scale":
        weight = 1.0 / tgt.clamp_min(eps)
        scale = align_depth_scale(src[None], tgt[None], weight[None]).squeeze(0)
        shift = torch.zeros_like(scale)
    elif mode == "l2_affine":
        A = torch.stack([src, torch.ones_like(src)], dim=-1)
        beta = torch.linalg.lstsq(A, tgt[:, None]).solution[:, 0]
        scale, shift = beta[0], beta[1]
    elif mode == "robust_affine":
        weight = torch.ones_like(src)
        scale, shift = align_depth_affine(src[None], tgt[None], weight[None])
        scale = scale.squeeze(0)
        shift = shift.squeeze(0)
    else:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    return float(scale.item()), float(shift.item())


def compute_depth_metrics(pred_depth, gt_depth, mask, eps=1e-8):
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return None

    pred_valid = pred_depth[mask]
    gt_valid = gt_depth[mask]
    abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid.clamp_min(eps))
    d_rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2))
    return {
        "abs_rel": float(abs_rel.item()),
        "d_rmse": float(d_rmse.item()),
        "valid_pixels": valid_count,
    }


SPARSE_STRUCTURE_KEYS = [
    "local_pair_count",
    "sparse_order_acc",
    "sparse_order_pairs",
    "local_residual_consistency",
    "local_residual_pairs",
    "sparse_lidar_dq",
]


SIGMA_FILTER_KEYS = [
    "sigma_status",
    "sigma_input_pixels",
    "sigma_kept_pixels",
    "sigma_keep_ratio",
    "sigma_residual_mean",
    "sigma_residual_std",
]


def empty_sparse_structure_metrics():
    return {
        "local_pair_count": 0,
        "sparse_order_acc": None,
        "sparse_order_pairs": 0,
        "local_residual_consistency": None,
        "local_residual_pairs": 0,
        "sparse_lidar_dq": None,
    }


def empty_sigma_filter_report(status="disabled"):
    return {
        "sigma_status": status,
        "sigma_input_pixels": 0,
        "sigma_kept_pixels": 0,
        "sigma_keep_ratio": None,
        "sigma_residual_mean": None,
        "sigma_residual_std": None,
    }


def gather_pairs_from_masks(pred_depths, gt_depths, masks):
    pred_flat_list = []
    gt_flat_list = []
    for pred_depth, gt_depth, mask in zip(pred_depths, gt_depths, masks):
        if mask.any():
            pred_flat_list.append(pred_depth[mask].double())
            gt_flat_list.append(gt_depth[mask].double())

    if not pred_flat_list:
        return None, None

    return torch.cat(pred_flat_list), torch.cat(gt_flat_list)


def estimate_depth_sigma_stats(src, tgt, scale, shift, args, eps=1e-8):
    aligned = src.reshape(-1).double() * scale + shift
    tgt = tgt.reshape(-1).double()

    valid = torch.isfinite(aligned) & torch.isfinite(tgt)
    valid &= aligned > args.min_eval_depth
    valid &= tgt > args.min_eval_depth
    if args.max_eval_depth is not None:
        valid &= tgt <= args.max_eval_depth

    stats = empty_sigma_filter_report(status="ok")
    input_count = int(valid.sum().item())
    stats["sigma_input_pixels"] = input_count
    if input_count < 2:
        stats["sigma_status"] = "too_few_points"
        return stats

    residual = torch.log(aligned[valid].clamp_min(eps)) - torch.log(tgt[valid].clamp_min(eps))
    residual = residual[torch.isfinite(residual)]
    input_count = int(residual.numel())
    stats["sigma_input_pixels"] = input_count
    if input_count < 2:
        stats["sigma_status"] = "too_few_finite_residuals"
        return stats

    mean = residual.mean()
    std = residual.std(unbiased=False)
    stats["sigma_residual_mean"] = float(mean.item())
    stats["sigma_residual_std"] = float(std.item())

    if not torch.isfinite(std) or float(std.item()) <= eps:
        stats["sigma_status"] = "zero_std_keep_all"
        stats["sigma_kept_pixels"] = input_count
        stats["sigma_keep_ratio"] = 1.0
        return stats

    keep = torch.abs(residual - mean) <= args.depth_sigma * std
    kept_count = int(keep.sum().item())
    stats["sigma_kept_pixels"] = kept_count
    stats["sigma_keep_ratio"] = kept_count / input_count if input_count > 0 else None
    if kept_count < args.min_sigma_points:
        stats["sigma_status"] = "too_few_kept_points"
    return stats


def apply_depth_sigma_filter(pred_depth, gt_depth, base_mask, scale, shift, stats, args, eps=1e-8):
    candidate = base_mask.clone()
    aligned = pred_depth.double() * scale + shift
    gt = gt_depth.double()

    candidate &= torch.isfinite(aligned) & torch.isfinite(gt)
    candidate &= aligned > args.min_eval_depth
    candidate &= gt > args.min_eval_depth
    if args.max_eval_depth is not None:
        candidate &= gt <= args.max_eval_depth

    frame_report = empty_sigma_filter_report(status=stats.get("sigma_status", "disabled"))
    frame_report["sigma_input_pixels"] = int(candidate.sum().item())
    frame_report["sigma_residual_mean"] = stats.get("sigma_residual_mean")
    frame_report["sigma_residual_std"] = stats.get("sigma_residual_std")

    if stats.get("sigma_status") == "disabled":
        frame_report["sigma_kept_pixels"] = frame_report["sigma_input_pixels"]
        frame_report["sigma_keep_ratio"] = 1.0 if frame_report["sigma_input_pixels"] > 0 else None
        return candidate, frame_report

    std = stats.get("sigma_residual_std")
    mean = stats.get("sigma_residual_mean")
    if std is None or mean is None or std <= eps:
        frame_report["sigma_kept_pixels"] = frame_report["sigma_input_pixels"]
        frame_report["sigma_keep_ratio"] = 1.0 if frame_report["sigma_input_pixels"] > 0 else None
        return candidate, frame_report

    residual = torch.log(aligned.clamp_min(eps)) - torch.log(gt.clamp_min(eps))
    keep = candidate & torch.isfinite(residual)
    keep &= torch.abs(residual - mean) <= args.depth_sigma * std

    kept_count = int(keep.sum().item())
    frame_report["sigma_kept_pixels"] = kept_count
    frame_report["sigma_keep_ratio"] = (
        kept_count / frame_report["sigma_input_pixels"]
        if frame_report["sigma_input_pixels"] > 0 else None
    )
    if kept_count < args.min_sigma_points:
        frame_report["sigma_status"] = "too_few_kept_points"

    return keep, frame_report


def build_sparse_local_pairs(valid_mask, max_radius=16, min_pixel_distance=2, max_pairs=20000,
                             max_anchors=4096, max_offsets=512, seed=0):
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    h, w = valid_np.shape
    coords = np.column_stack(np.nonzero(valid_np))
    num_valid = coords.shape[0]
    if num_valid < 2:
        return coords, np.empty((0, 2), dtype=np.int64)

    rng = np.random.default_rng(seed)
    index_map = np.full((h, w), -1, dtype=np.int64)
    index_map[coords[:, 0], coords[:, 1]] = np.arange(num_valid, dtype=np.int64)

    anchor_count = min(num_valid, max(1, int(max_anchors)))
    if anchor_count < num_valid:
        anchor_indices = rng.choice(num_valid, size=anchor_count, replace=False)
    else:
        anchor_indices = np.arange(num_valid, dtype=np.int64)

    radius = max(1, int(max_radius))
    min_distance = max(1.0, float(min_pixel_distance))
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            distance = math.hypot(dy, dx)
            if min_distance <= distance <= radius:
                offsets.append((dy, dx))

    if not offsets:
        return coords, np.empty((0, 2), dtype=np.int64)

    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.shape[0] > max_offsets:
        keep = rng.choice(offsets.shape[0], size=max_offsets, replace=False)
        offsets = offsets[keep]

    anchor_coords = coords[anchor_indices]
    cand_y = anchor_coords[:, 0:1] + offsets[None, :, 0]
    cand_x = anchor_coords[:, 1:2] + offsets[None, :, 1]
    inside = (cand_y >= 0) & (cand_y < h) & (cand_x >= 0) & (cand_x < w)

    flat_neighbor = np.full(cand_y.shape, -1, dtype=np.int64)
    if np.any(inside):
        flat_neighbor[inside] = index_map[cand_y[inside], cand_x[inside]]

    anchor_grid = np.repeat(anchor_indices[:, None], offsets.shape[0], axis=1)
    pair_mask = (flat_neighbor >= 0) & (anchor_grid < flat_neighbor)
    if not np.any(pair_mask):
        return coords, np.empty((0, 2), dtype=np.int64)

    pairs = np.stack([anchor_grid[pair_mask], flat_neighbor[pair_mask]], axis=1)
    pre_unique_cap = max_pairs * 4
    if pairs.shape[0] > pre_unique_cap:
        keep = rng.choice(pairs.shape[0], size=pre_unique_cap, replace=False)
        pairs = pairs[keep]

    pairs = np.unique(pairs, axis=0)
    if pairs.shape[0] > max_pairs:
        keep = rng.choice(pairs.shape[0], size=max_pairs, replace=False)
        pairs = pairs[keep]
    return coords, pairs


def compute_sparse_structure_metrics(pred_depth, gt_depth, eval_mask, args, seed=0, eps=1e-8):
    coords, pairs = build_sparse_local_pairs(
        eval_mask,
        max_radius=args.sparse_metric_max_radius,
        min_pixel_distance=args.sparse_metric_min_pixel_distance,
        max_pairs=args.sparse_metric_max_pairs,
        max_anchors=args.sparse_metric_max_anchors,
        max_offsets=args.sparse_metric_max_offsets,
        seed=seed,
    )
    if pairs.shape[0] == 0:
        return empty_sparse_structure_metrics()

    pred_np = pred_depth.detach().cpu().numpy().astype(np.float64)
    gt_np = gt_depth.detach().cpu().numpy().astype(np.float64)
    pred_values = pred_np[coords[:, 0], coords[:, 1]]
    gt_values = gt_np[coords[:, 0], coords[:, 1]]

    i = pairs[:, 0]
    j = pairs[:, 1]
    pred_i, pred_j = pred_values[i], pred_values[j]
    gt_i, gt_j = gt_values[i], gt_values[j]
    gt_delta = gt_i - gt_j
    pred_delta = pred_i - pred_j
    ref_depth = np.minimum(gt_i, gt_j)

    result = empty_sparse_structure_metrics()
    result["local_pair_count"] = int(pairs.shape[0])

    order_threshold = np.maximum(args.sro_min_depth_delta, args.sro_min_rel_delta * ref_depth)
    order_mask = np.abs(gt_delta) >= order_threshold
    if np.any(order_mask):
        order_correct = gt_delta[order_mask] * pred_delta[order_mask] > 0
        result["sparse_order_acc"] = float(np.mean(order_correct))
        result["sparse_order_pairs"] = int(np.sum(order_mask))

    residual_threshold = np.maximum(args.lrc_max_depth_delta, args.lrc_max_rel_delta * ref_depth)
    residual_mask = np.abs(gt_delta) <= residual_threshold
    if np.any(residual_mask):
        residual = np.log(np.maximum(pred_values, eps)) - np.log(np.maximum(gt_values, eps))
        residual_diff = np.abs(residual[i[residual_mask]] - residual[j[residual_mask]])
        result["local_residual_consistency"] = float(np.mean(residual_diff))
        result["local_residual_pairs"] = int(np.sum(residual_mask))

    if result["sparse_order_acc"] is not None and result["local_residual_consistency"] is not None:
        result["sparse_lidar_dq"] = float(
            result["sparse_order_acc"] * math.exp(-result["local_residual_consistency"])
        )

    return result


def evaluate_alignment_candidate(full_src, full_tgt, scale, shift, min_eval_depth=1e-5, max_eval_depth=None):
    aligned = full_src * scale + shift
    mask = torch.isfinite(aligned) & torch.isfinite(full_tgt)
    mask &= aligned > min_eval_depth
    mask &= full_tgt > min_eval_depth
    if max_eval_depth is not None:
        mask &= full_tgt <= max_eval_depth
    metrics = compute_depth_metrics(aligned.float(), full_tgt.float(), mask)
    if metrics is None:
        return None
    metrics["positive_ratio"] = float((aligned > min_eval_depth).double().mean().item())
    return metrics


def estimate_best_alignment(full_src, full_tgt, args):
    candidate_modes = [args.alignment_mode]
    if args.alignment_mode == "auto":
        candidate_modes = [
            "median_scale",
            "l2_scale",
            "robust_scale",
            "robust_rel_scale",
        ]
        if args.allow_affine:
            candidate_modes.extend(["l2_affine", "robust_affine"])

    solve_src, solve_tgt = subsample_pairs(
        full_src, full_tgt, max_points=args.max_alignment_points, seed=args.alignment_seed
    )

    candidate_results = []
    for mode in candidate_modes:
        try:
            scale, shift = solve_alignment(solve_src, solve_tgt, mode)
        except Exception as exc:
            candidate_results.append({
                "mode": mode,
                "status": f"failed: {exc}",
            })
            continue

        if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
            candidate_results.append({
                "mode": mode,
                "status": "invalid_params",
                "scale": scale,
                "shift": shift,
            })
            continue

        eval_metrics = evaluate_alignment_candidate(
            full_src, full_tgt, scale, shift,
            min_eval_depth=args.min_eval_depth,
            max_eval_depth=args.max_eval_depth,
        )
        if eval_metrics is None or eval_metrics["positive_ratio"] < 0.95:
            candidate_results.append({
                "mode": mode,
                "status": "invalid_eval",
                "scale": scale,
                "shift": shift,
            })
            continue

        candidate_results.append({
            "mode": mode,
            "status": "ok",
            "scale": scale,
            "shift": shift,
            **eval_metrics,
        })

    valid_results = [x for x in candidate_results if x.get("status") == "ok"]
    if not valid_results:
        raise RuntimeError(f"All alignment candidates failed: {candidate_results}")

    metric_key = "d_rmse" if args.alignment_select_metric == "drmse" else "abs_rel"
    best_result = min(valid_results, key=lambda x: (x[metric_key], x["abs_rel"], x["d_rmse"]))
    return best_result, candidate_results


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
    render_keys = {
        "xyz",
        "rotation",
        "scale",
        "opacity",
        "color",
        "competition_color",
        "source_view",
        "redundancy_score",
        "redundancy_coef",
        "competition_gate",
        "competition_active",
        "pushed",
    }
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


def filter_gaussians_by_opacity(gaussians, opacity_threshold=None):
    if opacity_threshold is None:
        return gaussians

    threshold = float(opacity_threshold)
    opacity = gaussians["opacity"].detach().float()
    keep_mask = opacity[0].reshape(-1) > threshold
    if not keep_mask.any() and keep_mask.numel() > 0:
        keep_mask[int(torch.argmax(opacity[0].reshape(-1)).item())] = True

    filtered = {}
    for key, value in gaussians.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[0] == 1
            and value.shape[1] == keep_mask.shape[0]
        ):
            filtered[key] = value[:, keep_mask]
        else:
            filtered[key] = value
    return filtered


def _gaussian_vector(gaussians, key, length, fill=0.0, dtype=torch.float32):
    value = gaussians.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.full((length,), fill, device=gaussians["opacity"].device, dtype=dtype)
    value = value.detach()
    if value.ndim == 3:
        value = value[0, :, 0]
    elif value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    value = value.reshape(-1).to(device=gaussians["opacity"].device, dtype=dtype)
    if value.numel() == length:
        return value
    out = torch.full((length,), fill, device=gaussians["opacity"].device, dtype=dtype)
    n = min(length, value.numel())
    if n > 0:
        out[:n] = value[:n]
    return out


def _gaussian_matrix(gaussians, key, length, channels, fill=0.0):
    value = gaussians.get(key)
    device = gaussians["opacity"].device
    if not isinstance(value, torch.Tensor):
        return torch.full((length, channels), fill, device=device, dtype=torch.float32)
    value = value.detach()
    if value.ndim == 3:
        value = value[0]
    value = value.reshape(value.shape[0], -1).to(device=device, dtype=torch.float32)
    if value.shape[0] == length and value.shape[1] >= channels:
        return value[:, :channels]
    out = torch.full((length, channels), fill, device=device, dtype=torch.float32)
    n = min(length, value.shape[0])
    c = min(channels, value.shape[1])
    if n > 0 and c > 0:
        out[:n, :c] = value[:n, :c]
    return out


def _select_diagnostic_indices(opacity_pre, opacity_post, redundancy_score, redundancy_coef, gate, max_rows):
    n = int(opacity_post.numel())
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return torch.arange(n, device=opacity_post.device)

    priority = torch.maximum(opacity_post, opacity_pre)
    priority = priority + redundancy_coef + 0.05 * redundancy_score.clamp_min(0.0)
    priority = priority + (1.0 - gate).clamp_min(0.0)
    k = min(int(max_rows), n)
    return torch.topk(priority, k=k, largest=True, sorted=False).indices


def save_opacity_cdf_plot(opacity_pre, opacity_post, gate, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = opacity_pre.detach().float().cpu().numpy()
    post = opacity_post.detach().float().cpu().numpy()
    gate_np = gate.detach().float().cpu().numpy()
    bins = np.linspace(0.0, 1.0, 401)

    def cumulative(values):
        values = np.clip(values[np.isfinite(values)], 0.0, 1.0)
        if values.size == 0:
            return np.zeros((bins.size - 1,), dtype=np.float32)
        hist, _ = np.histogram(values, bins=bins)
        return np.cumsum(hist).astype(np.float64) / max(1, int(hist.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    x = bins[1:]
    axes[0].plot(x, cumulative(pre), color="#3B6FB6", linewidth=2.0, label="pre-gate opacity")
    axes[0].plot(x, cumulative(post), color="#D4513C", linewidth=2.0, label="post-gate opacity")
    axes[0].axvline(0.05, color="#202020", linestyle="--", linewidth=1.1, label="0.05 threshold")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("opacity")
    axes[0].set_ylabel("CDF")
    axes[0].grid(True, linewidth=0.3, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=8)

    suppression = 1.0 - np.clip(gate_np[np.isfinite(gate_np)], 0.0, 1.0)
    axes[1].hist(suppression, bins=np.linspace(0, 1, 80), color="#E08B2D", alpha=0.78)
    axes[1].set_xlabel("1 - competition gate")
    axes[1].set_ylabel("count")
    axes[1].set_title("opacity suppression")
    axes[1].grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_gaussian_diagnostics(gaussians, scene_label, output_dir, max_rows=250000, opacity_threshold=0.05):
    diag_dir = os.path.join(output_dir, "gaussian_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    opacity_post = gaussians["opacity"].detach().float()[0, :, 0]
    n = int(opacity_post.numel())
    redundancy_score = _gaussian_vector(gaussians, "redundancy_score", n, fill=0.0)
    redundancy_coef = _gaussian_vector(gaussians, "redundancy_coef", n, fill=0.0)
    gate = _gaussian_vector(gaussians, "competition_gate", n, fill=1.0).clamp(1e-6, 1.0)
    competition_active = _gaussian_vector(gaussians, "competition_active", n, fill=0.0)
    source_view = _gaussian_vector(gaussians, "source_view", n, fill=-1, dtype=torch.float32)
    opacity_pre = (opacity_post / gate).clamp(0.0, 1.0)

    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scene_label)
    cdf_path = os.path.join(diag_dir, f"opacity_cdf_{stem}.png")
    save_opacity_cdf_plot(opacity_pre, opacity_post, gate, cdf_path)

    idx = _select_diagnostic_indices(
        opacity_pre,
        opacity_post,
        redundancy_score,
        redundancy_coef,
        gate,
        max_rows=max_rows,
    )
    xyz = _gaussian_matrix(gaussians, "xyz", n, 3).index_select(0, idx).cpu().numpy()
    color = _gaussian_matrix(gaussians, "color", n, 3).index_select(0, idx).cpu().numpy()
    comp_color = _gaussian_matrix(gaussians, "competition_color", n, 3).index_select(0, idx).cpu().numpy()

    idx_cpu = idx.detach().cpu().numpy()
    columns = {
        "redundancy_score": redundancy_score.index_select(0, idx).cpu().numpy(),
        "redundancy_coef": redundancy_coef.index_select(0, idx).cpu().numpy(),
        "competition_gate": gate.index_select(0, idx).cpu().numpy(),
        "competition_active": competition_active.index_select(0, idx).cpu().numpy(),
        "opacity_pre": opacity_pre.index_select(0, idx).cpu().numpy(),
        "opacity_post": opacity_post.index_select(0, idx).cpu().numpy(),
        "source_view": source_view.index_select(0, idx).cpu().numpy(),
    }

    csv_path = os.path.join(diag_dir, f"per_gaussian_diagnostics_{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "gaussian_index",
                "redundancy_score",
                "redundancy_coef",
                "competition_gate",
                "competition_active",
                "opacity_pre",
                "opacity_post",
                "active_005",
                "source_view",
                "xyz_x",
                "xyz_y",
                "xyz_z",
                "color_r",
                "color_g",
                "color_b",
                "competition_color_r",
                "competition_color_g",
                "competition_color_b",
            ],
        )
        writer.writeheader()
        for row_i, gaussian_index in enumerate(idx_cpu):
            opacity_value = float(columns["opacity_post"][row_i])
            writer.writerow(
                {
                    "scene": scene_label,
                    "gaussian_index": int(gaussian_index),
                    "redundancy_score": f"{float(columns['redundancy_score'][row_i]):.8f}",
                    "redundancy_coef": f"{float(columns['redundancy_coef'][row_i]):.8f}",
                    "competition_gate": f"{float(columns['competition_gate'][row_i]):.8f}",
                    "competition_active": int(float(columns["competition_active"][row_i]) > 0.5),
                    "opacity_pre": f"{float(columns['opacity_pre'][row_i]):.8f}",
                    "opacity_post": f"{opacity_value:.8f}",
                    "active_005": int(opacity_value > float(opacity_threshold)),
                    "source_view": int(round(float(columns["source_view"][row_i]))),
                    "xyz_x": f"{float(xyz[row_i, 0]):.8f}",
                    "xyz_y": f"{float(xyz[row_i, 1]):.8f}",
                    "xyz_z": f"{float(xyz[row_i, 2]):.8f}",
                    "color_r": f"{float(color[row_i, 0]):.8f}",
                    "color_g": f"{float(color[row_i, 1]):.8f}",
                    "color_b": f"{float(color[row_i, 2]):.8f}",
                    "competition_color_r": f"{float(comp_color[row_i, 0]):.8f}",
                    "competition_color_g": f"{float(comp_color[row_i, 1]):.8f}",
                    "competition_color_b": f"{float(comp_color[row_i, 2]):.8f}",
                }
            )

    return {
        "diagnostic_csv": csv_path,
        "diagnostic_rows": int(idx.numel()),
        "diagnostic_total_gaussians": n,
        "opacity_cdf_path": cdf_path,
    }


def _scatter_weighted_average(values, weights, inverse_indices, num_voxels):
    weighted = values * weights.unsqueeze(-1)
    out = torch.zeros((num_voxels, values.shape[-1]), device=values.device, dtype=values.dtype)
    out.scatter_add_(0, inverse_indices[:, None].expand_as(weighted), weighted)
    weight_sums = torch.zeros((num_voxels,), device=values.device, dtype=values.dtype)
    weight_sums.scatter_add_(0, inverse_indices, weights)
    return out / weight_sums.clamp_min(1e-8).unsqueeze(-1), weight_sums


def prune_dense_gaussians_like_hunyuan(gaussians, voxel_size):
    if voxel_size <= 0:
        return gaussians

    batch_items = []
    max_count = 0
    B = gaussians["xyz"].shape[0]
    for b in range(B):
        opacity = gaussians["opacity"][b].squeeze(-1)
        valid = opacity > 0
        if not valid.any():
            batch_item = {
                "xyz": gaussians["xyz"][b, :1],
                "rotation": gaussians["rotation"][b, :1],
                "scale": gaussians["scale"][b, :1],
                "opacity": torch.zeros_like(gaussians["opacity"][b, :1]),
                "color": gaussians["color"][b, :1],
            }
            batch_items.append(batch_item)
            max_count = max(max_count, 1)
            continue

        xyz = gaussians["xyz"][b][valid]
        rotation = gaussians["rotation"][b][valid]
        scale = gaussians["scale"][b][valid]
        color = gaussians["color"][b][valid]
        weights = opacity[valid].clamp_min(1e-6)

        voxel_indices = torch.floor(xyz / voxel_size).long()
        voxel_indices = voxel_indices - voxel_indices.min(dim=0, keepdim=True).values
        max_dims = voxel_indices.max(dim=0).values + 1
        flat_indices = (
            voxel_indices[:, 0] * max_dims[1] * max_dims[2]
            + voxel_indices[:, 1] * max_dims[2]
            + voxel_indices[:, 2]
        )
        _, inverse_indices = torch.unique(flat_indices, return_inverse=True)
        num_voxels = int(inverse_indices.max().item()) + 1

        xyz_merged, weight_sums = _scatter_weighted_average(xyz, weights, inverse_indices, num_voxels)
        scale_merged, _ = _scatter_weighted_average(scale, weights, inverse_indices, num_voxels)
        color_merged, _ = _scatter_weighted_average(color, weights, inverse_indices, num_voxels)
        rot_merged, _ = _scatter_weighted_average(rotation, weights, inverse_indices, num_voxels)
        rot_merged = F.normalize(rot_merged, dim=-1)
        opacity_merged = torch.zeros((num_voxels,), device=xyz.device, dtype=xyz.dtype)
        opacity_merged.scatter_add_(0, inverse_indices, weights * weights)
        opacity_merged = (opacity_merged / weight_sums.clamp_min(1e-8)).unsqueeze(-1)

        batch_item = {
            "xyz": xyz_merged,
            "rotation": rot_merged,
            "scale": scale_merged,
            "opacity": opacity_merged,
            "color": color_merged,
        }
        batch_items.append(batch_item)
        max_count = max(max_count, num_voxels)

    padded = {key: [] for key in ("xyz", "rotation", "scale", "opacity", "color")}
    for item in batch_items:
        pad_len = max_count - item["xyz"].shape[0]
        padded["xyz"].append(F.pad(item["xyz"], (0, 0, 0, pad_len), value=0.0))
        padded["rotation"].append(F.pad(item["rotation"], (0, 0, 0, pad_len), value=1.0))
        padded["scale"].append(F.pad(item["scale"], (0, 0, 0, pad_len), value=1e-5))
        padded["opacity"].append(F.pad(item["opacity"], (0, 0, 0, pad_len), value=0.0))
        padded["color"].append(F.pad(item["color"], (0, 0, 0, pad_len), value=0.0))
    return {key: torch.stack(value, dim=0) for key, value in padded.items()}


def build_hunyuan_like_dense_gaussians(res, imgs_batch, source_indices, args):
    local_points = res["local_points"][:, source_indices]
    camera_poses = res["camera_poses"][:, source_indices]
    intrinsics = res["intrinsics"][:, source_indices]
    src_imgs = imgs_batch[:, source_indices]

    B, S, _, H, W = src_imgs.shape
    local_flat = local_points.reshape(B, S, H * W, 3)
    cam_rot = camera_poses[:, :, :3, :3]
    cam_trans = camera_poses[:, :, :3, 3]
    xyz = torch.matmul(cam_rot[:, :, None], local_flat[..., None]).squeeze(-1) + cam_trans[:, :, None]
    xyz = xyz.reshape(B, S * H * W, 3)

    colors = src_imgs.permute(0, 1, 3, 4, 2).reshape(B, S * H * W, 3).contiguous()
    depth = local_flat[..., 2].abs().reshape(B, S * H * W)
    fx = intrinsics[:, :, 0, 0].reshape(B, S, 1).expand(B, S, H * W).reshape(B, S * H * W).abs()
    fy = intrinsics[:, :, 1, 1].reshape(B, S, 1).expand(B, S, H * W).reshape(B, S * H * W).abs()
    pixel_footprint = depth * 0.5 * (fx.clamp_min(1e-6).reciprocal() + fy.clamp_min(1e-6).reciprocal())
    scale = pixel_footprint * args.hunyuan_like_pixel_scale
    scale = scale.clamp(min=args.hunyuan_like_scale_min, max=args.hunyuan_like_scale_max)
    scale = scale.unsqueeze(-1).expand(-1, -1, 3).contiguous()

    rotation = torch.zeros((B, S * H * W, 4), device=xyz.device, dtype=xyz.dtype)
    rotation[..., 0] = 1.0
    opacity = torch.full((B, S * H * W, 1), args.hunyuan_like_opacity, device=xyz.device, dtype=xyz.dtype)

    finite = torch.isfinite(xyz).all(dim=-1) & torch.isfinite(scale).all(dim=-1)
    valid_depth = depth > args.hunyuan_like_min_depth
    valid = finite & valid_depth
    opacity = torch.where(valid.unsqueeze(-1), opacity, torch.zeros_like(opacity))

    gaussians = {
        "xyz": xyz,
        "rotation": rotation,
        "scale": scale,
        "opacity": opacity,
        "color": colors,
    }
    if args.hunyuan_like_prune:
        gaussians = prune_dense_gaussians_like_hunyuan(gaussians, args.hunyuan_like_voxel_size)
    return gaussians


def format_optional(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def resolve_checkpoint_path(ckpt_path):
    if os.path.isdir(ckpt_path):
        candidates = [
            os.path.join(ckpt_path, "model.safetensors"),
            os.path.join(ckpt_path, "pytorch_model.bin"),
            os.path.join(ckpt_path, "model.pt"),
            os.path.join(ckpt_path, "model.pth"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Checkpoint directory does not contain a supported model file: {ckpt_path}"
        )
    return ckpt_path


def infer_hydra_config_path(ckpt_path):
    current = os.path.abspath(os.path.dirname(ckpt_path))
    while True:
        candidate = os.path.join(current, ".hydra", "config.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def import_model_class(model_impl):
    model_impl = MODEL_IMPL_ALIASES.get(model_impl, model_impl)
    module_name, class_name = model_impl.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_hydra_config(config_path):
    if not config_path:
        return {}, None
    try:
        import yaml
    except ImportError:
        print("PyYAML is not available; using built-in model defaults.")
        return {}, None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, config_path


def load_model_kwargs_from_config(cfg, model_cls):
    model_cfg = cfg.get("model") or {}
    valid_keys = set(inspect.signature(model_cls.__init__).parameters)
    valid_keys.discard("self")
    kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key in valid_keys and key != "ckpt"
    }
    return kwargs


def main():
    parser = argparse.ArgumentParser(description="Pi3_3DGS inference with RGB/depth evaluation")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--eval_source", type=str, choices=["raw_folder", "three_sixty_v2_dataset"], default="raw_folder")
    parser.add_argument("--dataset_root", type=str, default="/data/liuwei/dataset/360_v2")
    parser.add_argument("--dataset_scene", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, choices=["train", "test", "all"], default="test")
    parser.add_argument("--dataset_image_dir_name", type=str, default="images_4")
    parser.add_argument("--dataset_hold_every", type=int, default=8)
    parser.add_argument("--dataset_frame_num", type=int, default=8)
    parser.add_argument("--dataset_resolution", type=str, default="518x336")
    parser.add_argument("--dataset_seed", type=int, default=2024)
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--dataset_shuffle_views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth_path", type=str, default=None,
                        help="Directory or stack file of depth maps. If omitted and not auto-detected, depth eval is skipped.")
    parser.add_argument("--output_dir", type=str, default="output_render_campus_0425")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--target_frame_count", type=int, default=None,
                        help="For raw RGB directories, select this many frames evenly over the full sorted sequence.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Number of frames per inference chunk when --per_chunk_scene is enabled.")
    parser.add_argument("--per_chunk_scene", action="store_true",
                        help="Use the old behavior: build one Gaussian space per chunk instead of one for all frames.")
    parser.add_argument("--gs_view_stride", type=int, default=2,
                        help="View stride used by the model Gaussian branch. 1 uses all input views.")
    parser.add_argument("--geometry_head_view_chunk_size", type=int, default=None,
                        help="Override model point/conf dense head view chunk size for long single-scene inference.")
    parser.add_argument("--gs_decoder_view_chunk_size", type=int, default=None,
                        help="Override model GS decoder view chunk size for long single-scene inference.")
    parser.add_argument("--gs_head_view_chunk_size", type=int, default=None,
                        help="Override model GS head view chunk size for long single-scene inference.")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Hydra config.yaml used to restore model construction args. Default: auto-detect near ckpt.")
    parser.add_argument("--model_impl", type=str, default=None,
                        help="Import path for the model class. Default: use model._target_ from --model_config, then fallback to pi3.models.pi3_3dgs_9.Pi3_3DGS.")
    parser.add_argument("--ignore_model_config", action="store_true",
                        help="Use hard-coded model defaults instead of ckpt-side Hydra config.")
    parser.add_argument("--disable_quadtree", action="store_true",
                        help="Ablation: use dense Gaussian candidates instead of quadtree-selected candidates.")
    parser.add_argument("--disable_local_competition", action="store_true",
                        help="Ablation: disable local competition opacity suppression.")
    parser.add_argument("--hard_prune_redundant_gaussians_eval", action="store_true",
                        help="Physically remove Gaussians above the local redundancy threshold during eval.")
    parser.add_argument("--disable_learnable_sampling", action="store_true",
                        help="Ablation: disable learned density extras; only proposal candidates are used.")
    parser.add_argument("--disable_density_opacity_gate", action="store_true",
                        help="Ablation: keep learned support sampling but stop density logits from reducing opacity.")
    parser.add_argument("--proposal_sampling_mode", type=str, choices=["quadtree", "random_equal"], default=None,
                        help="Override Pi3_3DGS_10 proposal sampling. random_equal keeps quadtree's per-view count but samples random pixels.")
    parser.add_argument("--random_sampling_seed", type=int, default=None,
                        help="Seed used by --proposal_sampling_mode random_equal.")
    parser.add_argument("--color_source", choices=["predicted", "input"], default="predicted",
                        help="Ablation: render with predicted RGB or the source-view input RGB attached to each Gaussian.")
    parser.add_argument("--gaussian_mode", choices=["model", "hunyuan_like"], default="model",
                        help="Render the model-produced Gaussians or a Hunyuan-style dense RGBDC splat baseline from the same Pi3 geometry.")
    parser.add_argument("--hunyuan_like_pixel_scale", type=float, default=1.0,
                        help="Pixel-footprint multiplier used by --gaussian_mode hunyuan_like.")
    parser.add_argument("--hunyuan_like_opacity", type=float, default=0.95,
                        help="Constant opacity used by --gaussian_mode hunyuan_like.")
    parser.add_argument("--hunyuan_like_scale_min", type=float, default=1e-5,
                        help="Minimum scale for --gaussian_mode hunyuan_like.")
    parser.add_argument("--hunyuan_like_scale_max", type=float, default=0.3,
                        help="Maximum scale for --gaussian_mode hunyuan_like, matching Hunyuan's clamp_max.")
    parser.add_argument("--hunyuan_like_min_depth", type=float, default=1e-5,
                        help="Minimum positive camera depth kept by --gaussian_mode hunyuan_like.")
    parser.add_argument("--hunyuan_like_prune", action="store_true",
                        help="Apply a Hunyuan-style voxel merge to the dense RGBDC splats.")
    parser.add_argument("--hunyuan_like_voxel_size", type=float, default=0.002,
                        help="Voxel size used when --hunyuan_like_prune is enabled.")
    parser.add_argument("--density_gate_min_prob", type=float, default=None,
                        help="Override Pi3_3DGS density_gate_min_prob for ablations.")
    parser.add_argument("--density_gate_opacity_power", type=float, default=None,
                        help="Override Pi3_3DGS density_gate_opacity_power for ablations.")
    parser.add_argument("--opacity_filter_threshold", type=float, default=None,
                        help="Override Pi3_3DGS opacity_filter_threshold for ablations.")
    parser.add_argument("--render_opacity_threshold", type=float, default=None,
                        help="If set, render only Gaussians with opacity above this threshold.")
    parser.add_argument("--ply_opacity_threshold", type=float, default=0.05,
                        help="Opacity threshold used when saving the PLY point cloud.")
    parser.add_argument("--save_redundancy_overlay", action=argparse.BooleanOptionalAction, default=True,
                        help="Save an input-image overlay showing where redundant Gaussians had opacity suppressed.")
    parser.add_argument("--redundancy_overlay_color", type=str, default="1.0,0.15,0.0",
                        help="RGB color for the redundancy overlay, as 0-1 or 0-255 comma values.")
    parser.add_argument("--redundancy_overlay_alpha", type=float, default=0.55,
                        help="Blend opacity for the redundancy overlay on top of the input image.")
    parser.add_argument("--redundancy_overlay_min_gaussian_opacity", type=float, default=0.3,
                        help="Minimum visualization opacity used when rendering suppressed Gaussians into the overlay mask.")
    parser.add_argument("--redundancy_overlay_mask_threshold", type=float, default=0.02,
                        help="Rendered alpha threshold used to convert redundancy overlays into solid regions.")
    parser.add_argument("--redundancy_overlay_connect_radius", type=int, default=4,
                        help="Pixel radius for closing small gaps in redundancy overlay regions.")
    parser.add_argument("--redundancy_overlay_boundary_width", type=float, default=0.5,
                        help="Pixel width of the redundancy overlay region boundary.")
    parser.add_argument("--redundancy_overlay_boundary_alpha", type=float, default=0.95,
                        help="Blend opacity for the redundancy overlay region boundary.")
    parser.add_argument("--redundancy_overlay_boundary_color", type=str, default=None,
                        help="Optional RGB color for overlay boundaries. Defaults to --redundancy_overlay_color.")
    parser.add_argument("--redundancy_overlay_coef_threshold", type=float, default=0.0,
                        help="Fallback redundancy_coef threshold used for overlays when competition_active/gate are unavailable.")
    parser.add_argument("--redundancy_overlay_multiview_color_threshold", type=float, default=0.2,
                        help="Max RGB distance for counting a suppressed Gaussian as visible in another input view.")
    parser.add_argument("--redundancy_overlay_match_radius", type=int, default=4,
                        help="Pixel radius around projected Gaussian centers used for multi-view appearance matching.")
    parser.add_argument("--redundancy_overlay_min_views", type=int, default=2,
                        help="Minimum number of input views with matching appearance before a suppressed Gaussian is overlaid.")
    parser.add_argument("--redundancy_overlay_chunk_size", type=int, default=200000,
                        help="Chunk size for appearance-based multi-view overlay filtering.")
    parser.add_argument("--scale_bias_strength", type=float, default=None,
                        help="Override Pi3_3DGS scale_bias_strength for ablations.")
    parser.add_argument("--scale_activation_multiplier", type=float, default=None,
                        help="Override Pi3_3DGS scale activation multiplier for ablations.")
    parser.add_argument("--low_conf_scale_boost", type=float, default=None,
                        help="Override Pi3_3DGS low_conf_scale_boost for ablations.")
    parser.add_argument("--ablation_name", type=str, default="full",
                        help="Name written to reports, e.g. full/no_quadtree/no_local_competition/pure_gaussian.")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--subset_start", type=int, default=None, help="Start index after interval sampling")
    parser.add_argument("--subset_end", type=int, default=None, help="End index after interval sampling")
    parser.add_argument("--subset_step", type=int, default=1, help="Step after interval sampling")
    parser.add_argument("--depth_unit_scale", type=float, default=None,
                        help="Scale factor applied to raw depth values. Default: auto infer (uint16 mm -> 0.001).")
    parser.add_argument("--depth_resize_mode", type=str, choices=["sparse", "nearest"], default="sparse")
    parser.add_argument("--rgb_only", action="store_true",
                        help="Skip depth loading, alignment, and depth metrics even if depth_path can be inferred.")
    parser.add_argument("--metric_lpips_net", type=str, choices=["alex", "vgg", "squeeze"], default="alex",
                        help="LPIPS backbone for RGB metrics. Default alex matches Pi3LossGS training/validation loss.")
    parser.add_argument("--alignment_mode", type=str,
                        choices=["auto", "median_scale", "l2_scale", "robust_scale", "robust_rel_scale",
                                 "l2_affine", "robust_affine"],
                        default="auto")
    parser.add_argument("--allow_affine", action="store_true",
                        help="When alignment_mode=auto, also try affine scale+shift candidates.")
    parser.add_argument("--alignment_scope", type=str, choices=["global", "per_frame"], default="global")
    parser.add_argument("--alignment_select_metric", type=str, choices=["drmse", "abs_rel"], default="drmse")
    parser.add_argument("--max_alignment_points", type=int, default=200000)
    parser.add_argument("--alignment_seed", type=int, default=0)
    parser.add_argument("--min_eval_depth", type=float, default=1e-5)
    parser.add_argument("--max_eval_depth", type=float, default=None)
    parser.add_argument("--min_valid_pixels", type=int, default=64)
    parser.add_argument("--disable_depth_sigma_filter", action="store_true",
                        help="Disable the default 3-sigma log-depth residual filter for alignment and metrics.")
    parser.add_argument("--depth_sigma", type=float, default=3.0,
                        help="Sigma threshold for selecting depth points after first-pass alignment.")
    parser.add_argument("--min_sigma_points", type=int, default=64,
                        help="Minimum kept points required to use sigma-filtered alignment.")
    parser.add_argument("--save_aligned_depth", action="store_true")
    parser.add_argument("--skip_save_ply", action="store_true",
                        help="Skip writing Gaussian .ply files; useful for metric-only sweeps.")
    parser.add_argument("--skip_save_frames", action="store_true",
                        help="Skip writing rendered RGB/depth/opacity images; metrics are still computed.")
    parser.add_argument("--skip_save_gaussian_scene", action="store_true",
                        help="Skip writing MonoGS-style Gaussian scene visualization images.")
    parser.add_argument("--gaussian_scene_scale_modifier", type=float, default=1.0,
                        help="Scale multiplier for Gaussian scene visualization, similar to MonoGS GUI's Gaussian Scale slider.")
    parser.add_argument("--gaussian_scene_opacity_multiplier", type=float, default=1.0,
                        help="Opacity multiplier for Gaussian scene visualization only.")
    parser.add_argument("--gaussian_scene_opacity_power", type=float, default=1.0,
                        help="Opacity power for Gaussian scene visualization only. Values below 1 boost faint Gaussians.")
    parser.add_argument("--gaussian_scene_opacity_threshold", type=float, default=None,
                        help="If set, zero Gaussians below this opacity only for Gaussian scene visualization.")
    parser.add_argument("--gaussian_scene_ellipsoid_alpha_threshold", type=float, default=0.05,
                        help="MonoGS Elipsoid Shader fragment alpha threshold for render_mod=-4.")
    parser.add_argument("--gaussian_scene_max_gaussians", type=int, default=0,
                        help="Max Gaussians used for offline ellipsoid visualization. Use 0 to render every visible Gaussian.")
    parser.add_argument("--gaussian_scene_selection_mode", type=str, choices=["screen_tile", "opacity"], default="screen_tile",
                        help="How to downselect Gaussians when --gaussian_scene_max_gaussians is positive.")
    parser.add_argument("--gaussian_scene_selection_tile_size", type=int, default=16,
                        help="Screen tile size used by --gaussian_scene_selection_mode screen_tile.")
    parser.add_argument("--gaussian_scene_exclude_pushed", action=argparse.BooleanOptionalAction, default=False,
                        help="Exclude model pushed/far support Gaussians from MonoGS-style ellipsoid visualization.")
    parser.add_argument("--gaussian_scene_depth_test", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a z-buffer for MonoGS-style ellipsoid visualization occlusion.")
    parser.add_argument("--gaussian_scene_max_screen_radius", type=float, default=0.0,
                        help="Optional screen-space radius cap for ellipsoid visualization. 0 disables the cap.")
    parser.add_argument("--save_gaussian_diagnostics", action="store_true",
                        help="Write per-Gaussian redundancy/opacity diagnostics and opacity CDF plots.")
    parser.add_argument("--gaussian_diagnostic_max_rows", type=int, default=250000,
                        help="Max rows per diagnostic CSV. Use 0 to write every Gaussian.")
    parser.add_argument("--sparse_metric_max_radius", type=int, default=16,
                        help="Max pixel radius for local sparse LiDAR point pairs.")
    parser.add_argument("--sparse_metric_min_pixel_distance", type=int, default=2,
                        help="Min pixel distance for local sparse LiDAR point pairs.")
    parser.add_argument("--sparse_metric_max_pairs", type=int, default=20000,
                        help="Max local point pairs sampled per frame for sparse structure metrics.")
    parser.add_argument("--sparse_metric_max_anchors", type=int, default=4096,
                        help="Max valid LiDAR anchors sampled per frame for sparse structure metrics.")
    parser.add_argument("--sparse_metric_max_offsets", type=int, default=512,
                        help="Max local pixel offsets sampled per frame for sparse structure metrics.")
    parser.add_argument("--sro_min_depth_delta", type=float, default=0.25,
                        help="Sparse relative ordering ignores LiDAR pairs with smaller absolute depth gaps.")
    parser.add_argument("--sro_min_rel_delta", type=float, default=0.03,
                        help="Sparse relative ordering also requires this relative depth gap.")
    parser.add_argument("--lrc_max_depth_delta", type=float, default=0.25,
                        help="Local residual consistency only uses pairs with depth gaps below this value.")
    parser.add_argument("--lrc_max_rel_delta", type=float, default=0.05,
                        help="Local residual consistency also limits relative depth gaps.")
    args = parser.parse_args()
    args.use_depth_sigma_filter = not args.disable_depth_sigma_filter
    args.redundancy_overlay_color_rgb = parse_rgb_triplet(args.redundancy_overlay_color)
    args.redundancy_overlay_boundary_color_rgb = (
        args.redundancy_overlay_color_rgb
        if args.redundancy_overlay_boundary_color is None
        else parse_rgb_triplet(args.redundancy_overlay_boundary_color)
    )
    if not (0.0 <= args.redundancy_overlay_alpha <= 1.0):
        raise ValueError("--redundancy_overlay_alpha must be in [0, 1].")
    if not (0.0 <= args.redundancy_overlay_mask_threshold <= 1.0):
        raise ValueError("--redundancy_overlay_mask_threshold must be in [0, 1].")
    if args.redundancy_overlay_connect_radius < 0:
        raise ValueError("--redundancy_overlay_connect_radius must be non-negative.")
    if args.redundancy_overlay_boundary_width < 0:
        raise ValueError("--redundancy_overlay_boundary_width must be non-negative.")
    if not (0.0 <= args.redundancy_overlay_boundary_alpha <= 1.0):
        raise ValueError("--redundancy_overlay_boundary_alpha must be in [0, 1].")
    if args.redundancy_overlay_min_gaussian_opacity < 0:
        raise ValueError("--redundancy_overlay_min_gaussian_opacity must be non-negative.")
    if args.redundancy_overlay_coef_threshold < 0:
        raise ValueError("--redundancy_overlay_coef_threshold must be non-negative.")
    if args.redundancy_overlay_multiview_color_threshold < 0:
        raise ValueError("--redundancy_overlay_multiview_color_threshold must be non-negative.")
    if args.redundancy_overlay_match_radius < 0:
        raise ValueError("--redundancy_overlay_match_radius must be non-negative.")
    if args.redundancy_overlay_min_views <= 0:
        raise ValueError("--redundancy_overlay_min_views must be positive.")
    if args.redundancy_overlay_chunk_size <= 0:
        raise ValueError("--redundancy_overlay_chunk_size must be positive.")
    if args.gaussian_scene_scale_modifier <= 0:
        raise ValueError("--gaussian_scene_scale_modifier must be positive.")
    if args.gaussian_scene_opacity_multiplier < 0:
        raise ValueError("--gaussian_scene_opacity_multiplier must be non-negative.")
    if args.gaussian_scene_opacity_power <= 0:
        raise ValueError("--gaussian_scene_opacity_power must be positive.")
    if args.gaussian_scene_opacity_threshold is not None and args.gaussian_scene_opacity_threshold < 0:
        raise ValueError("--gaussian_scene_opacity_threshold must be non-negative when set.")
    if args.gaussian_scene_ellipsoid_alpha_threshold < 0:
        raise ValueError("--gaussian_scene_ellipsoid_alpha_threshold must be non-negative.")
    if args.gaussian_scene_max_gaussians < 0:
        raise ValueError("--gaussian_scene_max_gaussians must be non-negative.")
    if args.gaussian_scene_selection_tile_size <= 0:
        raise ValueError("--gaussian_scene_selection_tile_size must be positive.")
    if args.gaussian_scene_max_screen_radius < 0:
        raise ValueError("--gaussian_scene_max_screen_radius must be non-negative.")
    if args.target_frame_count is not None and args.target_frame_count <= 0:
        raise ValueError("--target_frame_count must be positive when set.")
    if args.gs_decoder_view_chunk_size is not None and args.gs_decoder_view_chunk_size <= 0:
        raise ValueError("--gs_decoder_view_chunk_size must be positive when set.")
    if args.gs_head_view_chunk_size is not None and args.gs_head_view_chunk_size <= 0:
        raise ValueError("--gs_head_view_chunk_size must be positive when set.")
    if args.geometry_head_view_chunk_size is not None and args.geometry_head_view_chunk_size <= 0:
        raise ValueError("--geometry_head_view_chunk_size must be positive when set.")
    if args.eval_source == "raw_folder" and not args.data_path:
        raise ValueError("--data_path is required when --eval_source raw_folder.")
    if args.eval_source == "three_sixty_v2_dataset" and not args.dataset_scene:
        raise ValueError("--dataset_scene is required when --eval_source three_sixty_v2_dataset.")

    os.makedirs(args.output_dir, exist_ok=True)
    aligned_dir = os.path.join(args.output_dir, "aligned_depth")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False. "
            "Please fix the NVIDIA driver/CUDA runtime first or pass --device cpu for a tiny smoke test."
        )
    if args.interval < 0:
        args.interval = 10 if args.data_path and args.data_path.endswith(".mp4") else 1

    args.ckpt = resolve_checkpoint_path(args.ckpt)
    print(f"Loading Pi3_3DGS model from {args.ckpt}...")
    print(f"Ablation: {args.ablation_name}")
    print(f"Quadtree enabled: {not args.disable_quadtree}")
    print(f"Local competition enabled: {not args.disable_local_competition}")
    print(f"Local competition hard redundancy prune: {args.hard_prune_redundant_gaussians_eval}")
    print("Local competition hard cap enabled: False")
    print(f"Gaussian render mode: {args.gaussian_mode}")
    print(f"Gaussian branch view stride: {args.gs_view_stride}")
    config_path = None if args.ignore_model_config else (args.model_config or infer_hydra_config_path(args.ckpt))
    cfg, loaded_config_path = load_hydra_config(config_path)
    model_cfg = cfg.get("model") or {}
    model_impl = args.model_impl or model_cfg.get("_target_") or DEFAULT_MODEL_IMPL
    model_impl = MODEL_IMPL_ALIASES.get(model_impl, model_impl)
    model_cls = import_model_class(model_impl)
    print(f"Using Pi3_3DGS implementation: {model_impl}")
    config_kwargs = load_model_kwargs_from_config(cfg, model_cls)
    if loaded_config_path:
        print(f"Loaded model construction args from: {loaded_config_path}")

    model_kwargs = {
        "pos_type": "rope100",
        "decoder_size": "large",
        "ckpt": None,
        "debug_mem": False,
    }
    model_kwargs.update(config_kwargs)
    model_kwargs.update({
        "ckpt": None,
        "debug_mem": False,
        "gs_view_stride": args.gs_view_stride,
        "enable_quadtree": not args.disable_quadtree,
        "enable_local_competition": not args.disable_local_competition,
    })
    valid_model_keys = set(inspect.signature(model_cls.__init__).parameters)
    if args.hard_prune_redundant_gaussians_eval and "hard_prune_redundant_gaussians_eval" in valid_model_keys:
        model_kwargs["hard_prune_redundant_gaussians_eval"] = True
    if args.disable_learnable_sampling:
        model_kwargs["enable_learnable_sampling"] = False
    if args.disable_density_opacity_gate:
        model_kwargs["density_gate_opacity_power"] = 0.0
    for arg_name in (
        "density_gate_min_prob",
        "density_gate_opacity_power",
        "opacity_filter_threshold",
        "scale_bias_strength",
        "scale_activation_multiplier",
        "low_conf_scale_boost",
        "geometry_head_view_chunk_size",
        "gs_decoder_view_chunk_size",
        "gs_head_view_chunk_size",
    ):
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            model_kwargs[arg_name] = arg_value
    for arg_name in ("proposal_sampling_mode", "random_sampling_seed"):
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            if arg_name in valid_model_keys:
                model_kwargs[arg_name] = arg_value
            else:
                print(f"Warning: {model_impl} does not accept {arg_name}; ignoring override.")
    model = model_cls(**model_kwargs).to(device).eval()

    if args.ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(args.ckpt)
        if hasattr(model, "_load_state_dict_flexible"):
            model._load_state_dict_flexible(weight)
        else:
            model.load_state_dict(weight, strict=False)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        if hasattr(model, "_load_state_dict_flexible"):
            model._load_state_dict_flexible(weight)
        else:
            model.load_state_dict(weight, strict=False)

    if args.eval_source == "three_sixty_v2_dataset":
        print(
            "Loading RGB frames from ThreeSixtyV2Dataset "
            f"scene={args.dataset_scene}, split={args.dataset_split}..."
        )
        imgs_cpu, frame_items, orig_hw, target_hw = load_three_sixty_v2_sequence(args)
    else:
        print(f"Loading RGB frames from {args.data_path}...")
        imgs_cpu, frame_items, orig_hw, target_hw = load_rgb_sequence(
            args.data_path,
            interval=args.interval,
            subset_start=args.subset_start,
            subset_end=args.subset_end,
            subset_step=args.subset_step,
            pixel_limit=args.pixel_limit,
            target_frame_count=args.target_frame_count,
        )
    if imgs_cpu.numel() == 0:
        raise RuntimeError("No RGB frames loaded.")

    total_frames = imgs_cpu.shape[0]
    H, W = imgs_cpu.shape[2], imgs_cpu.shape[3]
    print(f"Loaded {total_frames} RGB frames. Original resolution: {orig_hw[0]}x{orig_hw[1]}, model resolution: {H}x{W}")

    gt_depths_cpu = None
    depth_match_mode = None
    used_depth_unit_scale = None
    use_depth_eval = not args.rgb_only and args.eval_source == "raw_folder"
    if use_depth_eval:
        args.depth_path = resolve_depth_path(args.data_path, args.depth_path)
        if args.depth_path is None:
            use_depth_eval = False
            print("No depth_path provided or auto-detected; running RGB-only inference/evaluation.")
        else:
            print(f"Loading depth maps from {args.depth_path}...")
            gt_depths_cpu, depth_match_mode, used_depth_unit_scale = load_depth_sequence(
                args.depth_path,
                frame_items=frame_items,
                target_hw=target_hw,
                resize_mode=args.depth_resize_mode,
                depth_unit_scale=args.depth_unit_scale,
                max_eval_depth=args.max_eval_depth,
            )
            if gt_depths_cpu.shape[0] != total_frames:
                raise RuntimeError(f"Depth frame count mismatch: {gt_depths_cpu.shape[0]} vs {total_frames}")
            print(f"Depth maps matched by: {depth_match_mode}")
            print(f"Depth unit scale in use: {used_depth_unit_scale}")
    else:
        args.depth_path = None
        if args.eval_source == "three_sixty_v2_dataset":
            print("Dataset-backed 360_v2 mode has no depth target here; skipping depth metrics.")
        else:
            print("RGB-only mode enabled; skipping depth loading and depth metrics.")

    if args.save_aligned_depth and use_depth_eval:
        os.makedirs(aligned_dir, exist_ok=True)
    elif args.save_aligned_depth:
        print("--save_aligned_depth ignored because depth evaluation is disabled.")

    print("Initializing RGB metrics (PSNR, SSIM, LPIPS)...")
    metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    metric_lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.metric_lpips_net, normalize=True).to(device)
    all_rgb_metrics = {"psnr": [], "ssim": [], "lpips": []}

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    pred_depths_cpu = []
    frame_reports = []
    scene_gaussian_reports = []
    global_candidate_results = None
    global_alignment = None
    scene_mode = "per_chunk_scene" if args.per_chunk_scene else "single_scene"
    full_scene_gs_indices = build_gaussian_input_indices(total_frames, args.gs_view_stride)

    def render_scene_predictions(res, imgs_batch, frame_indices, source_indices, scene_label):
        if not frame_indices:
            return

        if args.gaussian_mode == "hunyuan_like":
            gaussians = build_hunyuan_like_dense_gaussians(res, imgs_batch, source_indices, args)
        else:
            gaussians = res["gaussians"]
        pred_c2w = res["camera_poses"]
        pred_K = res["intrinsics"]
        if pred_c2w.shape[1] != len(frame_indices):
            raise RuntimeError(
                f"Model returned {pred_c2w.shape[1]} camera poses for {len(frame_indices)} render frames."
            )

        pred_w2c = se3_inverse(pred_c2w)
        raw_gaussians = select_render_gaussians(gaussians, batch_index=0)
        if args.color_source == "input" and args.gaussian_mode == "model":
            if "competition_color" not in raw_gaussians:
                raise RuntimeError(
                    "--color_source input requires gaussians['competition_color'], "
                    "which is produced by pi3.models.pi3_3dgs_9."
                )
            raw_gaussians = dict(raw_gaussians)
            raw_gaussians["color"] = raw_gaussians["competition_color"].to(
                device=raw_gaussians["color"].device,
                dtype=raw_gaussians["color"].dtype,
            )
        current_gaussians = filter_gaussians_by_opacity(raw_gaussians, args.render_opacity_threshold)
        # Render all valid Gaussians. Some model variants append pushed/far
        # support Gaussians after the near slice; truncating by num_near leaves
        # visible RGB holes. Optional opacity filtering above is the only render
        # pruning applied here.
        current_num_near = None
        raw_opacity_flat = raw_gaussians["opacity"].detach().float().reshape(-1)
        raw_total_count = int(raw_opacity_flat.numel())
        raw_active_count = int((raw_opacity_flat > 0.05).sum().item())
        opacity_flat = current_gaussians["opacity"].detach().float().reshape(-1)
        rendered_count = int(opacity_flat.numel())
        rendered_active_count = int((opacity_flat > 0.05).sum().item())
        redundancy_suppression_mask = None
        redundancy_suppression_count = 0
        if args.save_redundancy_overlay and not args.skip_save_frames:
            redundancy_suppression_mask = _redundancy_suppression_mask_tensor(
                raw_gaussians,
                raw_total_count,
                raw_gaussians["opacity"].device,
                redundancy_coef_threshold=args.redundancy_overlay_coef_threshold,
            )
            redundancy_appearance_mask = _redundancy_appearance_multiview_mask(
                raw_gaussians,
                imgs_batch[0],
                pred_w2c,
                pred_K,
                H,
                W,
                color_threshold=args.redundancy_overlay_multiview_color_threshold,
                min_views=args.redundancy_overlay_min_views,
                match_radius=args.redundancy_overlay_match_radius,
                chunk_size=args.redundancy_overlay_chunk_size,
            )
            redundancy_suppression_mask &= redundancy_appearance_mask
            redundancy_suppression_count = int(redundancy_suppression_mask.sum().item())
        stat_values = {}
        gaussian_stats = res.get("gaussian_stats", {})
        if isinstance(gaussian_stats, dict):
            for stat_key, stat_tensor in gaussian_stats.items():
                if isinstance(stat_tensor, torch.Tensor) and stat_tensor.numel() > 0:
                    stat_flat = stat_tensor.detach().float().reshape(stat_tensor.shape[0], -1)
                    if stat_flat.shape[0] > 0 and stat_flat.shape[1] == 1:
                        stat_values[f"stat_{stat_key}"] = float(stat_flat[0, 0].cpu().item())

        diagnostic_values = {}
        if args.save_gaussian_diagnostics:
            diagnostic_values = write_gaussian_diagnostics(
                raw_gaussians,
                scene_label,
                args.output_dir,
                max_rows=args.gaussian_diagnostic_max_rows,
                opacity_threshold=0.05,
            )

        ply_filename = os.path.join(args.output_dir, f"gaussians_{scene_label}.ply")
        if args.skip_save_ply:
            print(f"Skipped PLY point cloud for scene {scene_label}")
            kept_count = rendered_count
            saved_ply_path = ""
        else:
            kept_count, ply_input_count = save_ply_binary(
                current_gaussians,
                ply_filename,
                opacity_threshold=args.ply_opacity_threshold,
            )
            removed_count = ply_input_count - kept_count
            saved_ply_path = ply_filename
            print(
                f"Saved PLY point cloud: {ply_filename} "
                f"({kept_count}/{ply_input_count} kept, "
                f"{removed_count} removed with opacity <= {args.ply_opacity_threshold:g})"
            )
        scene_report = {
            "scene": scene_label,
            "ply_path": saved_ply_path,
            "total_count": raw_total_count,
            "active_count_opacity_gt_005": raw_active_count,
            "raw_total_count": raw_total_count,
            "raw_active_count_opacity_gt_005": raw_active_count,
            "rendered_count": rendered_count,
            "rendered_active_count_opacity_gt_005": rendered_active_count,
            "ply_kept_count": kept_count,
            "redundancy_suppression_count": redundancy_suppression_count,
            "opacity_threshold": args.ply_opacity_threshold,
            "render_opacity_threshold": args.render_opacity_threshold,
            "ply_opacity_threshold": args.ply_opacity_threshold,
            **stat_values,
            **diagnostic_values,
        }
        scene_gaussian_reports.append(scene_report)

        for i, global_frame_idx in enumerate(frame_indices):
            frame_name = frame_items[global_frame_idx]["stem"]

            view_w2c = pred_w2c[0:1, i:i + 1]
            view_K = pred_K[0:1, i:i + 1]
            rgb_tensor, depth_tensor, alpha_tensor = render_frame(
                current_gaussians,
                view_w2c,
                view_K,
                H,
                W,
                num_gaussians=current_num_near,
            )

            rgb_out = rgb_tensor[0, 0].permute(2, 0, 1)
            depth_out = depth_tensor[0, 0].permute(2, 0, 1)
            alpha_out = alpha_tensor[0, 0].permute(2, 0, 1)

            gt_img = imgs_batch[0, i]
            with torch.no_grad():
                pred_for_metric = torch.clamp(rgb_out.unsqueeze(0).float(), 0.0, 1.0)
                gt_for_metric = gt_img.unsqueeze(0).float()
                all_rgb_metrics["psnr"].append(metric_psnr(pred_for_metric, gt_for_metric).item())
                all_rgb_metrics["ssim"].append(metric_ssim(pred_for_metric, gt_for_metric).item())
                all_rgb_metrics["lpips"].append(metric_lpips(pred_for_metric, gt_for_metric).item())

            if not args.skip_save_frames:
                rgb_path = os.path.join(args.output_dir, f"rgb_{global_frame_idx:04d}.png")
                depth_path = os.path.join(args.output_dir, f"depth_{global_frame_idx:04d}.png")
                opacity_path = os.path.join(args.output_dir, f"opacity_heatmap_{global_frame_idx:04d}.png")
                save_image(rgb_out, rgb_path)
                save_depth(depth_out, depth_path)
                save_heatmap(alpha_out, opacity_path)
                if args.save_redundancy_overlay:
                    overlay_tensor, _, _ = build_redundancy_suppression_overlay(
                        raw_gaussians,
                        gt_img,
                        view_w2c,
                        view_K,
                        H,
                        W,
                        overlay_color=args.redundancy_overlay_color_rgb,
                        overlay_alpha=args.redundancy_overlay_alpha,
                        min_gaussian_opacity=args.redundancy_overlay_min_gaussian_opacity,
                        redundancy_coef_threshold=args.redundancy_overlay_coef_threshold,
                        overlay_mask=redundancy_suppression_mask,
                        mask_threshold=args.redundancy_overlay_mask_threshold,
                        connect_radius=args.redundancy_overlay_connect_radius,
                        boundary_width=args.redundancy_overlay_boundary_width,
                        boundary_alpha=args.redundancy_overlay_boundary_alpha,
                        boundary_color=args.redundancy_overlay_boundary_color_rgb,
                    )
                    overlay_path = os.path.join(
                        args.output_dir,
                        f"redundancy_suppression_overlay_{global_frame_idx:04d}.png",
                    )
                    save_image(overlay_tensor, overlay_path)
                    del overlay_tensor
                if not args.skip_save_gaussian_scene:
                    gaussian_scene_tensor, gaussian_scene_alpha = render_gaussian_scene_frame(
                        current_gaussians,
                        view_w2c,
                        view_K,
                        H,
                        W,
                        num_gaussians=current_num_near,
                        scale_modifier=args.gaussian_scene_scale_modifier,
                        opacity_multiplier=args.gaussian_scene_opacity_multiplier,
                        opacity_power=args.gaussian_scene_opacity_power,
                        opacity_threshold=args.gaussian_scene_opacity_threshold,
                        ellipsoid_alpha_threshold=args.gaussian_scene_ellipsoid_alpha_threshold,
                        max_gaussians=args.gaussian_scene_max_gaussians,
                        selection_mode=args.gaussian_scene_selection_mode,
                        selection_tile_size=args.gaussian_scene_selection_tile_size,
                        exclude_pushed=args.gaussian_scene_exclude_pushed,
                        depth_test=args.gaussian_scene_depth_test,
                        max_screen_radius=args.gaussian_scene_max_screen_radius,
                    )
                    gaussian_scene_out = gaussian_scene_tensor[0, 0].permute(2, 0, 1)
                    gaussian_scene_path = os.path.join(args.output_dir, f"gaussian_scene_{global_frame_idx:04d}.png")
                    save_image(gaussian_scene_out, gaussian_scene_path)
                    del gaussian_scene_tensor, gaussian_scene_alpha, gaussian_scene_out

            if use_depth_eval:
                pred_depth_cpu = depth_out.squeeze(0).float().cpu()
                pred_depths_cpu.append(pred_depth_cpu)
                gt_depth_cpu = gt_depths_cpu[global_frame_idx]
                gt_valid_pixels = int((gt_depth_cpu > args.min_eval_depth).sum().item())
                frame_reports.append({
                    "frame_idx": global_frame_idx,
                    "frame_name": frame_name,
                    "gt_valid_pixels": gt_valid_pixels,
                })

            del rgb_tensor, depth_tensor, alpha_tensor, rgb_out, depth_out, alpha_out

        print(f"Saved rendered frames {frame_indices[0]} - {frame_indices[-1]}")

        del gaussians, raw_gaussians, pred_c2w, pred_K, pred_w2c, current_gaussians, current_num_near

    if args.per_chunk_scene:
        active_chunk_size = max(1, args.chunk_size)
        start_idx = 0
        while start_idx < total_frames:
            end_idx = min(start_idx + active_chunk_size, total_frames)
            current_batch_size = end_idx - start_idx
            frame_indices = list(range(start_idx, end_idx))
            chunk_gs_local = build_gaussian_input_indices(current_batch_size, args.gs_view_stride)
            chunk_gs_global = [start_idx + idx for idx in chunk_gs_local]
            print(f"\nProcessing chunk: frames {start_idx} to {end_idx - 1} ({current_batch_size} frames)...")
            print(
                "Gaussian branch input frames for this chunk: "
                f"{len(chunk_gs_global)}/{current_batch_size} -> [{format_index_preview(chunk_gs_global)}]"
            )

            imgs_batch = imgs_cpu[start_idx:end_idx].unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    with amp_context():
                        res = model(imgs_batch)
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

            render_scene_predictions(
                res,
                imgs_batch,
                frame_indices,
                chunk_gs_local,
                f"chunk_{start_idx:04d}_to_{end_idx - 1:04d}",
            )

            del res, imgs_batch
            maybe_cuda_empty_cache(device)
            gc.collect()
            start_idx = end_idx
    else:
        frame_indices = list(range(total_frames))
        if args.chunk_size > 0 and args.chunk_size < total_frames:
            print(
                f"--chunk_size={args.chunk_size} is ignored in single-scene mode. "
                "Use --per_chunk_scene to restore chunked inference."
            )
        print(
            "\nProcessing one shared Gaussian scene for all frames. "
            f"Gaussian branch input frames: {len(full_scene_gs_indices)}/{total_frames} "
            f"-> [{format_index_preview(full_scene_gs_indices)}]"
        )

        imgs_batch = imgs_cpu.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                with amp_context():
                    res = model(imgs_batch)
        except RuntimeError as e:
            if is_retryable_inference_error(e):
                maybe_cuda_empty_cache(device)
                gc.collect()
                del imgs_batch
                raise RuntimeError(
                    "Single-scene inference failed. Reduce --pixel_limit, increase --gs_view_stride, "
                    "or pass --per_chunk_scene to fall back to separate Gaussian spaces per chunk."
                ) from e
            raise

        render_scene_predictions(
            res,
            imgs_batch,
            frame_indices,
            full_scene_gs_indices,
            f"scene_{0:04d}_to_{total_frames - 1:04d}",
        )

        del res, imgs_batch
        maybe_cuda_empty_cache(device)
        gc.collect()

    depth_metric_values = {
        "abs_rel": [],
        "d_rmse": [],
        "sparse_order_acc": [],
        "local_residual_consistency": [],
        "sparse_lidar_dq": [],
    }
    depth_candidate_rows = []

    if use_depth_eval:
        full_src, full_tgt, valid_masks = gather_valid_pairs(
            pred_depths_cpu, gt_depths_cpu,
            min_eval_depth=args.min_eval_depth,
            max_eval_depth=args.max_eval_depth,
        )
        if full_src is None or full_src.numel() == 0:
            raise RuntimeError("No overlapping valid pixels between predicted depth and provided depth maps.")

        final_eval_masks = valid_masks
        sigma_frame_reports = [empty_sigma_filter_report(status="disabled") for _ in valid_masks]
        global_sigma_stats = empty_sigma_filter_report(status="disabled")

        print(f"\nTotal valid depth pairs for alignment/eval: {full_src.numel()}")
        if args.alignment_scope == "global":
            initial_global_alignment, initial_global_candidates = estimate_best_alignment(full_src, full_tgt, args)
            global_alignment = initial_global_alignment
            global_candidate_results = initial_global_candidates

            if args.use_depth_sigma_filter:
                global_sigma_stats = estimate_depth_sigma_stats(
                    full_src,
                    full_tgt,
                    initial_global_alignment["scale"],
                    initial_global_alignment["shift"],
                    args,
                )
                sigma_masks = []
                for frame_idx, (pred_depth, gt_depth, valid_mask) in enumerate(
                    zip(pred_depths_cpu, gt_depths_cpu, valid_masks)
                ):
                    sigma_mask, sigma_report = apply_depth_sigma_filter(
                        pred_depth,
                        gt_depth,
                        valid_mask,
                        initial_global_alignment["scale"],
                        initial_global_alignment["shift"],
                        global_sigma_stats,
                        args,
                    )
                    sigma_masks.append(sigma_mask)
                    sigma_frame_reports[frame_idx] = sigma_report

                sigma_src, sigma_tgt = gather_pairs_from_masks(pred_depths_cpu, gt_depths_cpu, sigma_masks)
                if sigma_src is not None and sigma_src.numel() >= args.min_sigma_points:
                    global_alignment, global_candidate_results = estimate_best_alignment(sigma_src, sigma_tgt, args)
                    final_eval_masks = sigma_masks
                    print(
                        "3-sigma depth filter: "
                        f"kept {sigma_src.numel()} / {full_src.numel()} points "
                        f"({sigma_src.numel() / full_src.numel():.4f})"
                    )
                else:
                    print(
                        "3-sigma depth filter kept too few points; "
                        "falling back to unfiltered global alignment/metrics."
                    )

            print(
                "Selected global alignment: "
                f"{global_alignment['mode']} | scale={global_alignment['scale']:.6f} | shift={global_alignment['shift']:.6f}"
            )

        for idx, (pred_depth, gt_depth, valid_mask) in enumerate(zip(pred_depths_cpu, gt_depths_cpu, valid_masks)):
            frame_report = frame_reports[idx]
            sigma_report = sigma_frame_reports[idx]
            eval_base_mask = final_eval_masks[idx]

            if args.alignment_scope == "global":
                best_alignment = global_alignment
                candidate_results = global_candidate_results
            else:
                src = pred_depth[valid_mask].double()
                tgt = gt_depth[valid_mask].double()
                if src.numel() == 0:
                    best_alignment = None
                    candidate_results = []
                else:
                    initial_alignment, initial_candidates = estimate_best_alignment(src, tgt, args)
                    best_alignment = initial_alignment
                    candidate_results = initial_candidates
                    if args.use_depth_sigma_filter:
                        sigma_stats = estimate_depth_sigma_stats(
                            src,
                            tgt,
                            initial_alignment["scale"],
                            initial_alignment["shift"],
                            args,
                        )
                        sigma_mask, sigma_report = apply_depth_sigma_filter(
                            pred_depth,
                            gt_depth,
                            valid_mask,
                            initial_alignment["scale"],
                            initial_alignment["shift"],
                            sigma_stats,
                            args,
                        )
                        sigma_src = pred_depth[sigma_mask].double()
                        sigma_tgt = gt_depth[sigma_mask].double()
                        if sigma_src.numel() >= args.min_sigma_points:
                            best_alignment, candidate_results = estimate_best_alignment(sigma_src, sigma_tgt, args)
                            eval_base_mask = sigma_mask

            if best_alignment is None:
                frame_report.update({
                    "alignment_mode": "unavailable",
                    "scale": None,
                    "shift": None,
                    "valid_pixels": 0,
                    "abs_rel": None,
                    "d_rmse": None,
                })
                frame_report.update(sigma_report)
                frame_report.update(empty_sparse_structure_metrics())
                continue

            aligned_depth = pred_depth * best_alignment["scale"] + best_alignment["shift"]
            eval_mask = eval_base_mask.clone()
            eval_mask &= torch.isfinite(aligned_depth) & torch.isfinite(gt_depth)
            eval_mask &= aligned_depth > args.min_eval_depth
            eval_mask &= gt_depth > args.min_eval_depth
            if args.max_eval_depth is not None:
                eval_mask &= gt_depth <= args.max_eval_depth

            metrics = compute_depth_metrics(aligned_depth, gt_depth, eval_mask)
            valid_pixels = 0 if metrics is None else metrics["valid_pixels"]
            if metrics is None or valid_pixels < args.min_valid_pixels:
                frame_report.update({
                    "alignment_mode": best_alignment["mode"],
                    "scale": best_alignment["scale"],
                    "shift": best_alignment["shift"],
                    "valid_pixels": valid_pixels,
                    "abs_rel": None,
                    "d_rmse": None,
                })
                frame_report.update(sigma_report)
                frame_report.update(empty_sparse_structure_metrics())
            else:
                sparse_metrics = compute_sparse_structure_metrics(
                    aligned_depth,
                    gt_depth,
                    eval_mask,
                    args,
                    seed=args.alignment_seed + int(frame_report["frame_idx"]),
                )
                frame_report.update({
                    "alignment_mode": best_alignment["mode"],
                    "scale": best_alignment["scale"],
                    "shift": best_alignment["shift"],
                    "valid_pixels": valid_pixels,
                    "abs_rel": metrics["abs_rel"],
                    "d_rmse": metrics["d_rmse"],
                    **sigma_report,
                    **sparse_metrics,
                })
                depth_metric_values["abs_rel"].append(metrics["abs_rel"])
                depth_metric_values["d_rmse"].append(metrics["d_rmse"])
                for metric_name in ("sparse_order_acc", "local_residual_consistency", "sparse_lidar_dq"):
                    metric_value = sparse_metrics.get(metric_name)
                    if metric_value is not None and np.isfinite(metric_value):
                        depth_metric_values[metric_name].append(metric_value)

            if args.alignment_scope == "per_frame":
                for candidate in candidate_results:
                    depth_candidate_rows.append({
                        "frame_idx": frame_report["frame_idx"],
                        "frame_name": frame_report["frame_name"],
                        **candidate,
                    })

            if args.save_aligned_depth:
                aligned_path = os.path.join(aligned_dir, f"aligned_depth_{frame_report['frame_idx']:04d}.png")
                save_depth(aligned_depth.unsqueeze(0), aligned_path)

    avg_psnr = float(np.mean(all_rgb_metrics["psnr"])) if all_rgb_metrics["psnr"] else float("nan")
    avg_ssim = float(np.mean(all_rgb_metrics["ssim"])) if all_rgb_metrics["ssim"] else float("nan")
    avg_lpips = float(np.mean(all_rgb_metrics["lpips"])) if all_rgb_metrics["lpips"] else float("nan")
    avg_abs_rel = float(np.mean(depth_metric_values["abs_rel"])) if depth_metric_values["abs_rel"] else float("nan")
    avg_d_rmse = float(np.mean(depth_metric_values["d_rmse"])) if depth_metric_values["d_rmse"] else float("nan")
    avg_sro = float(np.mean(depth_metric_values["sparse_order_acc"])) if depth_metric_values["sparse_order_acc"] else float("nan")
    avg_lrc = float(np.mean(depth_metric_values["local_residual_consistency"])) if depth_metric_values["local_residual_consistency"] else float("nan")
    avg_sldq = float(np.mean(depth_metric_values["sparse_lidar_dq"])) if depth_metric_values["sparse_lidar_dq"] else float("nan")

    print("\n" + "=" * 48)
    print("Final Evaluation Metrics")
    print("=" * 48)
    print(f"Average PSNR:   {avg_psnr:.4f} dB")
    print(f"Average SSIM:   {avg_ssim:.4f}")
    print(f"Average LPIPS:  {avg_lpips:.4f}")
    if use_depth_eval:
        print(f"Average AbsRel: {avg_abs_rel:.6f}")
        print(f"Average dRMSE:  {avg_d_rmse:.6f}")
        print(f"Average SRO:    {avg_sro:.6f} (higher is better)")
        print(f"Average LRC:    {avg_lrc:.6f} (lower is better)")
        print(f"Average S-LDQ:  {avg_sldq:.6f} (higher is better)")
    else:
        print("Depth metrics:  skipped (RGB-only)")
    print("=" * 48)

    metrics_file = os.path.join(args.output_dir, "metrics_report.txt")
    with open(metrics_file, "w") as f:
        report_title = "Pi3_3DGS RGB + Depth Evaluation Report" if use_depth_eval else "Pi3_3DGS RGB Evaluation Report"
        f.write(report_title + "\n")
        f.write("=" * 48 + "\n")
        f.write(f"eval_source: {args.eval_source}\n")
        f.write(f"data_path: {args.data_path or 'N/A'}\n")
        if args.eval_source == "three_sixty_v2_dataset":
            f.write(f"dataset: 360_v2\n")
            f.write(f"split: {args.dataset_split}\n")
            f.write(f"scene: {args.dataset_scene}\n")
            f.write(f"dataset_root: {args.dataset_root}\n")
            f.write(f"dataset_image_dir_name: {args.dataset_image_dir_name}\n")
            f.write(f"hold_every: {args.dataset_hold_every}\n")
            f.write(f"dataset_frame_num: {args.dataset_frame_num}\n")
            f.write(f"dataset_seed: {args.dataset_seed}\n")
            f.write(f"dataset_shuffle_views: {args.dataset_shuffle_views}\n")
        f.write(f"model_impl: {model_impl}\n")
        f.write(f"ablation_name: {args.ablation_name}\n")
        f.write(f"quadtree_enabled: {not args.disable_quadtree}\n")
        f.write(f"local_competition_enabled: {not args.disable_local_competition}\n")
        f.write(f"hard_prune_redundant_gaussians_eval: {args.hard_prune_redundant_gaussians_eval}\n")
        f.write("local_competition_hard_cap_enabled: False\n")
        f.write(f"learnable_sampling_enabled: {not args.disable_learnable_sampling}\n")
        f.write(f"density_opacity_gate_enabled: {not args.disable_density_opacity_gate}\n")
        f.write(f"proposal_sampling_mode: {args.proposal_sampling_mode}\n")
        f.write(f"random_sampling_seed: {args.random_sampling_seed}\n")
        f.write(f"render_opacity_threshold: {args.render_opacity_threshold}\n")
        f.write(f"ply_opacity_threshold: {args.ply_opacity_threshold}\n")
        f.write(f"gaussian_mode: {args.gaussian_mode}\n")
        f.write(f"color_source: {args.color_source}\n")
        f.write(f"hunyuan_like_pixel_scale: {args.hunyuan_like_pixel_scale}\n")
        f.write(f"hunyuan_like_opacity: {args.hunyuan_like_opacity}\n")
        f.write(f"hunyuan_like_scale_min: {args.hunyuan_like_scale_min}\n")
        f.write(f"hunyuan_like_scale_max: {args.hunyuan_like_scale_max}\n")
        f.write(f"hunyuan_like_prune: {args.hunyuan_like_prune}\n")
        f.write(f"hunyuan_like_voxel_size: {args.hunyuan_like_voxel_size}\n")
        for arg_name in (
            "density_gate_min_prob",
            "density_gate_opacity_power",
            "opacity_filter_threshold",
            "scale_bias_strength",
            "scale_activation_multiplier",
            "low_conf_scale_boost",
            "geometry_head_view_chunk_size",
            "gs_decoder_view_chunk_size",
            "gs_head_view_chunk_size",
        ):
            f.write(f"{arg_name}: {getattr(args, arg_name)}\n")
        if loaded_config_path:
            f.write(f"model_config: {loaded_config_path}\n")
        f.write(f"render_frames_saved: {not args.skip_save_frames}\n")
        f.write(f"gaussian_scene_saved: {not args.skip_save_frames and not args.skip_save_gaussian_scene}\n")
        f.write(f"gaussian_scene_scale_modifier: {args.gaussian_scene_scale_modifier}\n")
        f.write(f"gaussian_scene_opacity_multiplier: {args.gaussian_scene_opacity_multiplier}\n")
        f.write(f"gaussian_scene_opacity_power: {args.gaussian_scene_opacity_power}\n")
        f.write(f"gaussian_scene_opacity_threshold: {args.gaussian_scene_opacity_threshold}\n")
        f.write(f"gaussian_scene_ellipsoid_alpha_threshold: {args.gaussian_scene_ellipsoid_alpha_threshold}\n")
        f.write(f"gaussian_scene_max_gaussians: {args.gaussian_scene_max_gaussians}\n")
        f.write(f"gaussian_scene_selection_mode: {args.gaussian_scene_selection_mode}\n")
        f.write(f"gaussian_scene_selection_tile_size: {args.gaussian_scene_selection_tile_size}\n")
        f.write(f"gaussian_scene_exclude_pushed: {args.gaussian_scene_exclude_pushed}\n")
        f.write(f"gaussian_scene_depth_test: {args.gaussian_scene_depth_test}\n")
        f.write(f"gaussian_scene_max_screen_radius: {args.gaussian_scene_max_screen_radius}\n")
        f.write(f"ply_saved: {not args.skip_save_ply}\n")
        f.write(f"redundancy_overlay_saved: {not args.skip_save_frames and args.save_redundancy_overlay}\n")
        f.write(f"redundancy_overlay_color: {args.redundancy_overlay_color_rgb}\n")
        f.write(f"redundancy_overlay_alpha: {args.redundancy_overlay_alpha}\n")
        f.write(f"redundancy_overlay_min_gaussian_opacity: {args.redundancy_overlay_min_gaussian_opacity}\n")
        f.write(f"redundancy_overlay_mask_threshold: {args.redundancy_overlay_mask_threshold}\n")
        f.write(f"redundancy_overlay_connect_radius: {args.redundancy_overlay_connect_radius}\n")
        f.write(f"redundancy_overlay_boundary_width: {args.redundancy_overlay_boundary_width}\n")
        f.write(f"redundancy_overlay_boundary_alpha: {args.redundancy_overlay_boundary_alpha}\n")
        f.write(f"redundancy_overlay_boundary_color: {args.redundancy_overlay_boundary_color_rgb}\n")
        f.write(f"redundancy_overlay_coef_threshold: {args.redundancy_overlay_coef_threshold}\n")
        f.write(f"redundancy_overlay_multiview_color_threshold: {args.redundancy_overlay_multiview_color_threshold}\n")
        f.write(f"redundancy_overlay_match_radius: {args.redundancy_overlay_match_radius}\n")
        f.write(f"redundancy_overlay_min_views: {args.redundancy_overlay_min_views}\n")
        f.write(f"gaussian_diagnostics_saved: {args.save_gaussian_diagnostics}\n")
        f.write(f"gaussian_diagnostic_max_rows: {args.gaussian_diagnostic_max_rows}\n")
        f.write(f"scene_mode: {scene_mode}\n")
        f.write(f"target_frame_count: {args.target_frame_count}\n")
        f.write(f"gs_view_stride: {args.gs_view_stride}\n")
        if args.per_chunk_scene:
            f.write(f"chunk_size: {args.chunk_size}\n")
        else:
            f.write(f"gaussian_input_frames: {len(full_scene_gs_indices)} / {total_frames}\n")
            f.write(f"gaussian_input_indices: {format_index_preview(full_scene_gs_indices, max_items=120)}\n")
        f.write(f"depth_eval: {use_depth_eval}\n")
        if use_depth_eval:
            f.write(f"depth_path: {args.depth_path}\n")
            f.write(f"depth_match_mode: {depth_match_mode}\n")
        f.write(f"frames: {total_frames}\n")
        f.write(f"model_resolution: {H}x{W}\n")
        f.write(f"metric_lpips_net: {args.metric_lpips_net}\n")
        if use_depth_eval:
            f.write(f"depth_unit_scale: {used_depth_unit_scale}\n")
            f.write(f"alignment_scope: {args.alignment_scope}\n")
            f.write(f"alignment_mode: {args.alignment_mode}\n")
            f.write(f"allow_affine_in_auto: {args.allow_affine}\n")
            f.write(f"alignment_select_metric: {args.alignment_select_metric}\n")
            f.write(f"depth_sigma_filter: {args.use_depth_sigma_filter}\n")
            if args.use_depth_sigma_filter:
                f.write(f"depth_sigma: {args.depth_sigma}\n")
                f.write(f"min_sigma_points: {args.min_sigma_points}\n")
                f.write(f"sigma_status: {global_sigma_stats.get('sigma_status', 'per_frame')}\n")
                f.write(f"sigma_input_pixels: {format_optional(global_sigma_stats.get('sigma_input_pixels'))}\n")
                f.write(f"sigma_kept_pixels: {format_optional(global_sigma_stats.get('sigma_kept_pixels'))}\n")
                f.write(f"sigma_keep_ratio: {format_optional(global_sigma_stats.get('sigma_keep_ratio'))}\n")
                f.write(f"sigma_residual_mean: {format_optional(global_sigma_stats.get('sigma_residual_mean'))}\n")
                f.write(f"sigma_residual_std: {format_optional(global_sigma_stats.get('sigma_residual_std'))}\n")
            if global_alignment is not None:
                f.write(f"selected_alignment: {global_alignment['mode']}\n")
                f.write(f"selected_scale: {global_alignment['scale']:.8f}\n")
                f.write(f"selected_shift: {global_alignment['shift']:.8f}\n")
        f.write(f"frame_names: {format_frame_name_preview(frame_items, max_items=240)}\n")
        f.write("\n")
        f.write(f"Average PSNR:   {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM:   {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS:  {avg_lpips:.4f}\n")
        if use_depth_eval:
            f.write(f"Average AbsRel: {avg_abs_rel:.6f}\n")
            f.write(f"Average dRMSE:  {avg_d_rmse:.6f}\n")
            f.write(f"Average SRO:    {avg_sro:.6f} (higher is better)\n")
            f.write(f"Average LRC:    {avg_lrc:.6f} (lower is better)\n")
            f.write(f"Average S-LDQ:  {avg_sldq:.6f} (higher is better)\n")
            f.write(f"Depth frames used: {len(depth_metric_values['abs_rel'])} / {total_frames}\n")
            f.write(
                "Sparse structure metric config: "
                f"radius={args.sparse_metric_max_radius}, "
                f"max_pairs={args.sparse_metric_max_pairs}, "
                f"sro_min_depth_delta={args.sro_min_depth_delta}, "
                f"sro_min_rel_delta={args.sro_min_rel_delta}, "
                f"lrc_max_depth_delta={args.lrc_max_depth_delta}, "
                f"lrc_max_rel_delta={args.lrc_max_rel_delta}\n"
            )

        if global_candidate_results is not None:
            f.write("\nGlobal alignment candidates:\n")
            for candidate in global_candidate_results:
                f.write(
                    f"  {candidate['mode']}: status={candidate.get('status')} "
                    f"scale={format_optional(candidate.get('scale'))} "
                    f"shift={format_optional(candidate.get('shift'))} "
                    f"abs_rel={format_optional(candidate.get('abs_rel'))} "
                    f"d_rmse={format_optional(candidate.get('d_rmse'))}\n"
                )

        if scene_gaussian_reports:
            f.write("\nGaussian scene counts:\n")
            for report in scene_gaussian_reports:
                f.write(
                    f"  {report['scene']}: "
                    f"raw_active>0.05={report['raw_active_count_opacity_gt_005']} / {report['raw_total_count']}, "
                    f"rendered={report['rendered_count']}, "
                    f"ply_kept={report['ply_kept_count']} "
                    f"redundancy_suppression={report['redundancy_suppression_count']} "
                    f"ply={report['ply_path'] or 'skipped'}\n"
                )
    print(f"Metrics saved to: {metrics_file}")

    if scene_gaussian_reports:
        gaussian_csv = os.path.join(args.output_dir, "gaussian_counts.csv")
        stat_fieldnames = sorted({
            key
            for report in scene_gaussian_reports
            for key in report.keys()
            if key.startswith("stat_")
        })
        with open(gaussian_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scene", "ply_path", "total_count",
                    "active_count_opacity_gt_005", "raw_total_count",
                    "raw_active_count_opacity_gt_005", "rendered_count",
                    "rendered_active_count_opacity_gt_005", "ply_kept_count",
                    "redundancy_suppression_count",
                    "opacity_threshold", "render_opacity_threshold", "ply_opacity_threshold",
                    "diagnostic_csv", "diagnostic_rows", "diagnostic_total_gaussians", "opacity_cdf_path",
                    *stat_fieldnames,
                ],
            )
            writer.writeheader()
            writer.writerows(scene_gaussian_reports)
        print(f"Gaussian counts saved to: {gaussian_csv}")

    if use_depth_eval:
        frame_csv = os.path.join(args.output_dir, "depth_metrics_per_frame.csv")
        with open(frame_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx", "frame_name", "gt_valid_pixels", "alignment_mode",
                    "scale", "shift", "valid_pixels", "abs_rel", "d_rmse",
                    "sigma_status", "sigma_input_pixels", "sigma_kept_pixels",
                    "sigma_keep_ratio", "sigma_residual_mean", "sigma_residual_std",
                    "local_pair_count", "sparse_order_acc", "sparse_order_pairs",
                    "local_residual_consistency", "local_residual_pairs", "sparse_lidar_dq",
                ],
            )
            writer.writeheader()
            writer.writerows(frame_reports)
        print(f"Per-frame depth metrics saved to: {frame_csv}")

        if global_candidate_results is not None:
            candidate_csv = os.path.join(args.output_dir, "alignment_candidates_global.csv")
            with open(candidate_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["mode", "status", "scale", "shift", "abs_rel", "d_rmse", "valid_pixels", "positive_ratio"],
                )
                writer.writeheader()
                writer.writerows(global_candidate_results)
            print(f"Global alignment candidates saved to: {candidate_csv}")
        elif depth_candidate_rows:
            candidate_csv = os.path.join(args.output_dir, "alignment_candidates_per_frame.csv")
            with open(candidate_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["frame_idx", "frame_name", "mode", "status", "scale", "shift",
                                "abs_rel", "d_rmse", "valid_pixels", "positive_ratio"],
                )
                writer.writeheader()
                writer.writerows(depth_candidate_rows)
            print(f"Per-frame alignment candidates saved to: {candidate_csv}")


if __name__ == "__main__":
    main()
