import argparse
import csv
import gc
import math
import os
import sys
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    LearnedPerceptualImagePatchSimilarity

from pi3.models.pi3_3dgs_woquad import Pi3_3DGS
from pi3.utils.alignment import align_depth_affine, align_depth_scale


RGB_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEPTH_EXTS = RGB_EXTS + (".npy", ".npz")


def save_heatmap(tensor, path):
    alpha_np = tensor.squeeze().detach().cpu().numpy()
    alpha_np = np.clip(alpha_np, 0, 1)
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap_color)


def save_ply_binary(gaussians, path, opacity_threshold=0.05):
    xyz = gaussians["xyz"].detach().cpu().float().numpy().reshape(-1, 3)
    rot = gaussians["rotation"].detach().cpu().float().numpy().reshape(-1, 4)
    scale = gaussians["scale"].detach().cpu().float().numpy().reshape(-1, 3)
    opacity = gaussians["opacity"].detach().cpu().float().numpy().reshape(-1)
    color = gaussians["color"].detach().cpu().float().numpy().reshape(-1, 3)

    total_count = xyz.shape[0]
    keep_mask = opacity >= opacity_threshold
    xyz = xyz[keep_mask]
    rot = rot[keep_mask]
    scale = scale[keep_mask]
    opacity = opacity[keep_mask]
    color = color[keep_mask]

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

    return xyz.shape[0], total_count


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


def list_sorted_files(directory, extensions):
    filenames = [
        x for x in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, x)) and x.lower().endswith(extensions)
    ]
    return sorted(filenames)


def build_selected_indices(total_count, interval=1, subset_start=None, subset_end=None, subset_step=1):
    base_indices = list(range(0, total_count, interval))
    return base_indices[slice(subset_start, subset_end, subset_step)]


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
    to_tensor = torch.from_numpy

    tensor_list = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.asarray(resized, dtype=np.float32) / 255.0
        img_tensor = to_tensor(img_np).permute(2, 0, 1)
        tensor_list.append(img_tensor)

    imgs = torch.stack(tensor_list, dim=0)
    return imgs, frame_items, (orig_h, orig_w), (target_h, target_w)


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


def main():
    parser = argparse.ArgumentParser(description="Pi3_3DGS inference with RGB/depth evaluation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, default=None,
                        help="Directory or stack file of depth maps. If omitted and not auto-detected, depth eval is skipped.")
    parser.add_argument("--output_dir", type=str, default="output_render_campus_0425")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Number of frames per inference chunk when --per_chunk_scene is enabled.")
    parser.add_argument("--per_chunk_scene", action="store_true",
                        help="Use the old behavior: build one Gaussian space per chunk instead of one for all frames.")
    parser.add_argument("--gs_view_stride", type=int, default=2,
                        help="View stride used by the model Gaussian branch. 1 uses all input views.")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--subset_start", type=int, default=None, help="Start index after interval sampling")
    parser.add_argument("--subset_end", type=int, default=None, help="End index after interval sampling")
    parser.add_argument("--subset_step", type=int, default=1, help="Step after interval sampling")
    parser.add_argument("--depth_unit_scale", type=float, default=None,
                        help="Scale factor applied to raw depth values. Default: auto infer (uint16 mm -> 0.001).")
    parser.add_argument("--depth_resize_mode", type=str, choices=["sparse", "nearest"], default="sparse")
    parser.add_argument("--rgb_only", action="store_true",
                        help="Skip depth loading, alignment, and depth metrics even if depth_path can be inferred.")
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

    os.makedirs(args.output_dir, exist_ok=True)
    aligned_dir = os.path.join(args.output_dir, "aligned_depth")

    device = torch.device(args.device)
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith(".mp4") else 1

    args.ckpt = resolve_checkpoint_path(args.ckpt)
    print(f"Loading Pi3_3DGS model from {args.ckpt}...")
    print(f"Using Pi3_3DGS implementation: pi3.models.pi3_3dgs_woquad")
    print(f"Gaussian branch view stride: {args.gs_view_stride}")
    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
        gs_view_stride=args.gs_view_stride,
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

    gt_depths_cpu = None
    depth_match_mode = None
    used_depth_unit_scale = None
    use_depth_eval = not args.rgb_only
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
        print("RGB-only mode enabled; skipping depth loading and depth metrics.")

    if args.save_aligned_depth and use_depth_eval:
        os.makedirs(aligned_dir, exist_ok=True)
    elif args.save_aligned_depth:
        print("--save_aligned_depth ignored because depth evaluation is disabled.")

    print("Initializing RGB metrics (PSNR, SSIM, LPIPS)...")
    metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    metric_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)
    all_rgb_metrics = {"psnr": [], "ssim": [], "lpips": []}

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    pred_depths_cpu = []
    frame_reports = []
    global_candidate_results = None
    global_alignment = None
    scene_mode = "per_chunk_scene" if args.per_chunk_scene else "single_scene"
    full_scene_gs_indices = build_gaussian_input_indices(total_frames, args.gs_view_stride)

    def render_scene_predictions(res, imgs_batch, frame_indices, scene_label):
        if not frame_indices:
            return

        gaussians = res["gaussians"]
        pred_c2w = res["camera_poses"]
        pred_K = res["intrinsics"]
        if pred_c2w.shape[1] != len(frame_indices):
            raise RuntimeError(
                f"Model returned {pred_c2w.shape[1]} camera poses for {len(frame_indices)} render frames."
            )

        num_near = gaussians.get("num_near", None)

        pred_w2c = se3_inverse(pred_c2w)
        current_gaussians = select_render_gaussians(gaussians, batch_index=0)
        current_num_near = num_near[0:1] if num_near is not None else None

        ply_filename = os.path.join(args.output_dir, f"gaussians_{scene_label}.ply")
        if args.skip_save_ply:
            print(f"Skipped PLY point cloud for scene {scene_label}")
        else:
            kept_count, total_count = save_ply_binary(current_gaussians, ply_filename)
            removed_count = total_count - kept_count
            print(
                f"Saved PLY point cloud: {ply_filename} "
                f"({kept_count}/{total_count} kept, {removed_count} removed with opacity < 0.05)"
            )

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

        del gaussians, pred_c2w, pred_K, pred_w2c, current_gaussians, current_num_near

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
        f.write(f"data_path: {args.data_path}\n")
        f.write("model_impl: pi3.models.pi3_3dgs_woquad\n")
        f.write(f"scene_mode: {scene_mode}\n")
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
    print(f"Metrics saved to: {metrics_file}")

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
