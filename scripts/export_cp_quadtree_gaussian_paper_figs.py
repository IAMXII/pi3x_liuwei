import argparse
import csv
import math
import os
import re
import sys
from contextlib import nullcontext

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from gsplat import rasterization
from pi3.models.pi3_3dgs_7 import Pi3_3DGS


def resolve_checkpoint_path(path):
    if os.path.isdir(path):
        for name in ("model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"No supported checkpoint file found in {path}")
    return path


def compute_target_size(width, height, pixel_limit):
    if pixel_limit <= 0:
        return (width // 14) * 14, (height // 14) * 14
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    w_target = width * scale
    h_target = height * scale
    k = max(1, round(w_target / 14))
    m = max(1, round(h_target / 14))
    while (k * 14) * (m * 14) > pixel_limit and (k > 1 or m > 1):
        if k / max(m, 1) > w_target / max(h_target, 1e-8):
            k = max(1, k - 1)
        else:
            m = max(1, m - 1)
    return k * 14, m * 14


def load_frame_range(data_root, frame_start, frame_end, frame_step, pixel_limit):
    rgb_dir = os.path.join(data_root, "rgb")
    frame_ids = list(range(frame_start, frame_end + 1, frame_step))
    items = []
    sources = []
    for frame_id in frame_ids:
        path = os.path.join(rgb_dir, f"{frame_id:06d}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        img = Image.open(path).convert("RGB")
        sources.append(img)
        items.append({"frame_id": frame_id, "stem": f"{frame_id:06d}", "path": path})

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


def save_rgb(path, image_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def robust_limits(values, low=1.0, high=99.0, mask=None):
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return 0.0, 1.0
    vals = arr[valid]
    lo = float(np.percentile(vals, low))
    hi = float(np.percentile(vals, high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def colorize_scalar(values, min_val=None, max_val=None, cmap=cv2.COLORMAP_TURBO):
    arr = np.asarray(values, dtype=np.float32)
    if min_val is None or max_val is None:
        min_val, max_val = robust_limits(arr)
    norm = (arr - min_val) / (max_val - min_val + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    color_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def blend_heatmap(image_rgb, heat_rgb, strength_map, max_alpha=0.72, base_dim=0.68):
    image = np.asarray(image_rgb, dtype=np.float32)
    heat = np.asarray(heat_rgb, dtype=np.float32)
    alpha = np.clip(strength_map, 0.0, 1.0)[..., None] * max_alpha
    base = image * base_dim
    out = base * (1.0 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def dilate_mask(mask, radius=1):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)


def support_overlay(image_rgb, quad_mask, low_conf_mask, support_mask):
    overlay = np.asarray(image_rgb, dtype=np.float32) * 0.52
    support = dilate_mask(support_mask, radius=1)
    quad = dilate_mask(quad_mask, radius=2)
    low = np.asarray(low_conf_mask, dtype=bool)
    overlay[support] = 0.35 * overlay[support] + 0.65 * np.array([238, 238, 238], dtype=np.float32)
    overlay[quad] = 0.16 * overlay[quad] + 0.84 * np.array([44, 214, 130], dtype=np.float32)
    overlay[low] = 0.20 * overlay[low] + 0.80 * np.array([226, 74, 74], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def gaussian_density_from_support(support_mask, sigma=3.0):
    density = np.asarray(support_mask, dtype=np.float32)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if float(density.max()) > 0:
        density = density / float(density.max())
    return density


def save_panel(images, titles, path, figsize_per_image=3.0, dpi=260):
    fig, axes = plt.subplots(1, len(images), figsize=(figsize_per_image * len(images), figsize_per_image))
    if len(images) == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=9, pad=5)
        ax.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_contact_sheet(rows, column_titles, path, dpi=240):
    if not rows:
        return
    n_rows = len(rows)
    n_cols = len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.35 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            ax = axes[r, c]
            ax.imshow(image)
            ax.axis("off")
            if r == 0:
                ax.set_title(column_titles[c], fontsize=9, pad=5)
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


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


def select_render_gaussians(gaussians, min_opacity=0.01, max_gaussians=180000):
    opacity = gaussians["opacity"][0, :, 0].detach().float()
    keep = opacity > min_opacity
    if not bool(keep.any()):
        keep = opacity >= opacity.max()
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    if max_gaussians > 0 and idx.numel() > max_gaussians:
        scores = opacity[idx]
        top = torch.topk(scores, k=max_gaussians, largest=True).indices
        idx = idx[top]

    out = {}
    for key in ("xyz", "rotation", "scale", "opacity", "color", "conf"):
        value = gaussians.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = value[:, idx].contiguous()
    return out, int(idx.numel())


def render_alpha_maps(gaussians, camera_poses, intrinsics, height, width):
    pred_w2c = se3_inverse(camera_poses)
    _, alpha, _ = rasterization(
        means=gaussians["xyz"].contiguous().float(),
        quats=gaussians["rotation"].contiguous().float(),
        scales=gaussians["scale"].contiguous().float(),
        opacities=gaussians["opacity"].squeeze(-1).contiguous().float(),
        colors=gaussians["color"].contiguous().float(),
        viewmats=pred_w2c.float(),
        Ks=intrinsics.float(),
        width=width,
        height=height,
        render_mode="RGB",
        packed=False,
    )
    return alpha[0, :, :, :, 0].detach().float().cpu().numpy()


def axis_limits(points, idx):
    lo, hi = robust_limits(points[:, idx], low=0.5, high=99.5)
    pad = 0.06 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def save_gaussian_spatial_distribution(gaussians, path, max_points=80000, seed=0):
    xyz = gaussians["xyz"][0].detach().float().cpu().numpy()
    scale = gaussians["scale"][0].detach().float().cpu().numpy()
    opacity = gaussians["opacity"][0, :, 0].detach().float().cpu().numpy()
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1) & (opacity > 0)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return False
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)

    points = xyz[idx]
    op = opacity[idx]
    scale_radius = np.cbrt(np.prod(np.clip(scale[idx], 1e-8, None), axis=1))
    s_lo, s_hi = robust_limits(scale_radius, low=5, high=95)
    sizes = 0.8 + 5.5 * np.clip((scale_radius - s_lo) / (s_hi - s_lo + 1e-8), 0.0, 1.0)
    op_norm = np.clip((op - robust_limits(op, 5, 95)[0]) / (robust_limits(op, 5, 95)[1] - robust_limits(op, 5, 95)[0] + 1e-8), 0.0, 1.0)

    fig = plt.figure(figsize=(11.2, 8.2))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    projections = [(0, 1, "X", "Y", "XY projection"), (0, 2, "X", "Z", "XZ projection"), (1, 2, "Y", "Z", "YZ projection")]
    for ax, (a, b, la, lb, title) in zip(axes[:3], projections):
        ax.scatter(points[:, a], points[:, b], s=sizes, c=op_norm, cmap="viridis", alpha=0.32, linewidths=0, rasterized=True)
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(axis_limits(points, a))
        ax.set_ylim(axis_limits(points, b))
        ax.grid(True, linewidth=0.3, alpha=0.35)

    ax3d = axes[3]
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=sizes, c=op_norm, cmap="viridis", alpha=0.24, linewidths=0)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_xlim(axis_limits(points, 0))
    ax3d.set_ylim(axis_limits(points, 1))
    ax3d.set_zlim(axis_limits(points, 2))
    ax3d.view_init(elev=24, azim=-55)
    ax3d.set_title("3D Gaussian distribution", fontsize=10)

    fig.suptitle(f"Gaussian Spatial Distribution | active={points.shape[0]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return True


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def tensor_to_numpy_map(viz, key, index):
    return viz[key][0, index].detach().float().cpu().numpy()


def write_manifest(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export paper figures for Pi3_3DGS_7 quadtree score maps and Gaussian distributions.")
    parser.add_argument("--data_root", type=str, default="/data/liuwei/dataset/ntu_seq/cp")
    parser.add_argument("--output_dir", type=str, default="outputs/cp_800_900_pi3_3dgs7_paper_figs")
    parser.add_argument("--ckpt", type=str, default="outputs/pi3_highres_0506_v6_local_comp/ckpts/best_model/model.safetensors")
    parser.add_argument("--frame_start", type=int, default=800)
    parser.add_argument("--frame_end", type=int, default=900)
    parser.add_argument("--frame_step", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--render_min_opacity", type=float, default=0.01)
    parser.add_argument("--max_render_gaussians", type=int, default=180000)
    parser.add_argument("--spatial_max_points", type=int, default=80000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    per_frame_dir = os.path.join(args.output_dir, "per_frame")
    overview_dir = os.path.join(args.output_dir, "overview")
    os.makedirs(per_frame_dir, exist_ok=True)
    os.makedirs(overview_dir, exist_ok=True)

    ckpt = resolve_checkpoint_path(args.ckpt)
    device = torch.device(args.device)
    imgs_cpu, input_rgbs, frame_items, orig_hw, target_hw = load_frame_range(
        args.data_root,
        args.frame_start,
        args.frame_end,
        args.frame_step,
        args.pixel_limit,
    )
    height, width = target_hw

    print(f"Frames: {[item['stem'] for item in frame_items]}")
    print(f"Original resolution: {orig_hw[0]}x{orig_hw[1]} | model resolution: {height}x{width}")
    print(f"Checkpoint: {ckpt}")

    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
        gs_view_stride=args.gs_view_stride,
    ).to(device).eval()

    if ckpt.endswith(".safetensors"):
        weight = load_file(ckpt, device="cpu")
    else:
        weight = torch.load(ckpt, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(weight, strict=False)
    print(f"load_state_dict: {load_result}")
    del weight

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    imgs_batch = imgs_cpu.unsqueeze(0).to(device)
    with torch.no_grad():
        with amp_context():
            result = model(imgs_batch, return_viz=True)

    viz = result["routing_viz"]
    sub_idx = viz["sub_idx"].detach().cpu().numpy().astype(int).tolist()
    if sub_idx != list(range(len(frame_items))):
        print(f"Warning: gs sub_idx is {sub_idx}; only those frames have quadtree score maps.")

    render_gaussians, render_count = select_render_gaussians(
        result["gaussians"],
        min_opacity=args.render_min_opacity,
        max_gaussians=args.max_render_gaussians,
    )
    alpha_maps = render_alpha_maps(render_gaussians, result["camera_poses"], result["intrinsics"], height, width)

    save_gaussian_spatial_distribution(
        render_gaussians,
        os.path.join(overview_dir, "gaussian_spatial_distribution.png"),
        max_points=args.spatial_max_points,
    )

    rows = []
    contact_rows = []
    for local_viz_index, loaded_index in enumerate(sub_idx):
        item = frame_items[loaded_index]
        frame_id = item["frame_id"]
        stem = safe_name(item["stem"])
        frame_rgb = input_rgbs[loaded_index]

        score_map = tensor_to_numpy_map(viz, "score_map", local_viz_index)
        color_score = tensor_to_numpy_map(viz, "color_score", local_viz_index)
        depth_score = tensor_to_numpy_map(viz, "depth_score", local_viz_index)
        scale_map = tensor_to_numpy_map(viz, "scale_map", local_viz_index)
        quad_mask = viz["quad_keep_mask"][0, local_viz_index].detach().bool().cpu().numpy()
        low_conf_mask = viz["low_conf_mask"][0, local_viz_index].detach().bool().cpu().numpy()
        support_mask = viz["support_mask"][0, local_viz_index].detach().bool().cpu().numpy()

        score_hi = max(robust_limits(score_map, low=1, high=99)[1], 1e-4)
        score_img = colorize_scalar(score_map, min_val=0.0, max_val=score_hi, cmap=cv2.COLORMAP_INFERNO)
        score_overlay = blend_heatmap(frame_rgb, score_img, np.clip(score_map / score_hi, 0.0, 1.0), max_alpha=0.76)

        color_hi = max(robust_limits(color_score, low=1, high=99)[1], 1e-4)
        depth_hi = max(robust_limits(depth_score, low=1, high=99)[1], 1e-4)
        color_img = colorize_scalar(color_score, min_val=0.0, max_val=color_hi, cmap=cv2.COLORMAP_VIRIDIS)
        depth_img = colorize_scalar(depth_score, min_val=0.0, max_val=depth_hi, cmap=cv2.COLORMAP_MAGMA)

        support_img = support_overlay(frame_rgb, quad_mask, low_conf_mask, support_mask)
        support_density = gaussian_density_from_support(support_mask, sigma=3.2)
        density_img = colorize_scalar(support_density, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_TURBO)
        density_overlay = blend_heatmap(frame_rgb, density_img, support_density, max_alpha=0.78, base_dim=0.56)

        alpha = np.clip(alpha_maps[loaded_index], 0.0, 1.0)
        alpha_lo, alpha_hi = robust_limits(alpha, low=2, high=99.5)
        alpha_hi = max(alpha_hi, alpha_lo + 1e-4)
        alpha_norm = np.clip((alpha - alpha_lo) / (alpha_hi - alpha_lo + 1e-8), 0.0, 1.0)
        alpha_img = colorize_scalar(alpha_norm, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_MAGMA)
        alpha_overlay = blend_heatmap(frame_rgb, alpha_img, alpha_norm, max_alpha=0.74, base_dim=0.56)

        save_rgb(os.path.join(per_frame_dir, f"input_{stem}.png"), frame_rgb)
        save_rgb(os.path.join(per_frame_dir, f"score_map_before_quadtree_{stem}.png"), score_img)
        save_rgb(os.path.join(per_frame_dir, f"score_overlay_before_quadtree_{stem}.png"), score_overlay)
        save_rgb(os.path.join(per_frame_dir, f"color_score_{stem}.png"), color_img)
        save_rgb(os.path.join(per_frame_dir, f"depth_score_{stem}.png"), depth_img)
        save_rgb(os.path.join(per_frame_dir, f"gaussian_support_points_{stem}.png"), support_img)
        save_rgb(os.path.join(per_frame_dir, f"gaussian_distribution_map_{stem}.png"), density_overlay)
        save_rgb(os.path.join(per_frame_dir, f"rendered_gaussian_opacity_{stem}.png"), alpha_img)
        save_rgb(os.path.join(per_frame_dir, f"rendered_gaussian_opacity_overlay_{stem}.png"), alpha_overlay)

        save_panel(
            [frame_rgb, score_img, density_overlay, alpha_overlay],
            ["Input", "Pre-quadtree score", "Gaussian support density", "Rendered opacity"],
            os.path.join(per_frame_dir, f"paper_panel_{stem}.png"),
            figsize_per_image=2.8,
            dpi=300,
        )
        save_panel(
            [frame_rgb, color_img, depth_img, score_img, support_img, alpha_overlay],
            ["Input", "Color score", "Depth score", "Score map", "Support points", "Opacity"],
            os.path.join(per_frame_dir, f"pipeline_panel_{stem}.png"),
            figsize_per_image=2.35,
            dpi=260,
        )

        contact_rows.append([frame_rgb, score_img, density_overlay, alpha_overlay])
        rows.append({
            "frame_id": frame_id,
            "source_path": item["path"],
            "score_mean": f"{float(np.mean(score_map)):.8f}",
            "score_p99": f"{float(np.percentile(score_map, 99)):.8f}",
            "support_points": int(np.count_nonzero(support_mask)),
            "quad_points": int(np.count_nonzero(quad_mask)),
            "low_conf_points": int(np.count_nonzero(low_conf_mask)),
            "render_alpha_mean": f"{float(np.mean(alpha)):.8f}",
            "render_alpha_p99": f"{float(np.percentile(alpha, 99)):.8f}",
        })

    save_contact_sheet(
        contact_rows,
        ["Input", "Pre-quadtree score", "Gaussian support density", "Rendered opacity"],
        os.path.join(overview_dir, "cp_800_900_score_gaussian_contact_sheet.png"),
        dpi=220,
    )
    write_manifest(rows, os.path.join(args.output_dir, "manifest.csv"))

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("Pi3_3DGS_7 CP 800-900 quadtree/Gaussian paper figures\n")
        f.write("=" * 64 + "\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"frames: {args.frame_start}-{args.frame_end} step {args.frame_step}\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"model_resolution: {height}x{width}\n")
        f.write(f"gs_view_stride: {args.gs_view_stride}\n")
        f.write(f"render_gaussians_kept: {render_count}\n")
        f.write(f"per_frame_dir: {per_frame_dir}\n")
        f.write(f"overview_dir: {overview_dir}\n")

    print(f"Saved per-frame figures: {per_frame_dir}")
    print(f"Saved overview figures: {overview_dir}")
    print(f"Manifest: {os.path.join(args.output_dir, 'manifest.csv')}")


if __name__ == "__main__":
    main()
