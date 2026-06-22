import argparse
import csv
import gc
import inspect
import os
import sys
from contextlib import nullcontext

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from export_cp_quadtree_gaussian_paper_figs import (
    axis_limits,
    blend_heatmap,
    colorize_scalar,
    load_frame_range,
    render_alpha_maps,
    robust_limits,
    safe_name,
    save_contact_sheet,
    save_panel,
    save_rgb,
)
from example_3dgs_5 import (
    MODEL_IMPL_ALIASES,
    import_model_class,
    infer_hydra_config_path,
    load_hydra_config,
    load_model_kwargs_from_config,
    resolve_checkpoint_path,
)

LOCAL_MODEL_IMPL_ALIASES = {
    **MODEL_IMPL_ALIASES,
    "_7": "pi3.models.pi3_3dgs_7.Pi3_3DGS",
    "7": "pi3.models.pi3_3dgs_7.Pi3_3DGS",
}


def load_model(args, device):
    ckpt = resolve_checkpoint_path(args.ckpt)
    model_impl = LOCAL_MODEL_IMPL_ALIASES.get(args.model_impl, args.model_impl)
    model_cls = import_model_class(model_impl)
    config_path = None if args.ignore_model_config else (args.model_config or infer_hydra_config_path(ckpt))
    cfg, loaded_config_path = load_hydra_config(config_path)
    model_kwargs = {
        "pos_type": "rope100",
        "decoder_size": "large",
        "ckpt": None,
        "debug_mem": False,
    }
    model_kwargs.update(load_model_kwargs_from_config(cfg, model_cls))
    model_kwargs.update({
        "ckpt": None,
        "debug_mem": False,
        "gs_view_stride": args.gs_view_stride,
        "enable_local_competition": True,
    })
    valid_keys = set(inspect.signature(model_cls.__init__).parameters)
    valid_keys.discard("self")
    model_kwargs = {key: value for key, value in model_kwargs.items() if key in valid_keys}
    model = model_cls(**model_kwargs).to(device).eval()
    if ckpt.endswith(".safetensors"):
        weight = load_file(ckpt, device="cpu")
    else:
        weight = torch.load(ckpt, map_location="cpu", weights_only=False)
    if hasattr(model, "_load_state_dict_flexible"):
        load_result = model._load_state_dict_flexible(weight)
    else:
        load_result = model.load_state_dict(weight, strict=False)
    del weight
    print(f"Checkpoint: {ckpt}")
    print(f"Model implementation: {model_impl}")
    if loaded_config_path:
        print(f"Model config: {loaded_config_path}")
    print(f"load_state_dict: {load_result}")
    return model, ckpt, model_impl, loaded_config_path


def run_forward(model, imgs_batch, device, amp_context, enable_local_competition, competition_strength=None):
    old_enable = getattr(model, "enable_local_competition", None)
    old_strength = getattr(model, "competition_strength", None)
    model.enable_local_competition = bool(enable_local_competition)
    if competition_strength is not None and hasattr(model, "competition_strength"):
        model.competition_strength = float(competition_strength)
    tag = "with" if enable_local_competition else "without"
    if competition_strength is not None:
        tag += f" strength={float(competition_strength):g}"
    print(f"Running inference {tag} local competition...")
    try:
        with torch.no_grad():
            with amp_context():
                return model(imgs_batch, return_viz=False)
    finally:
        if old_enable is not None:
            model.enable_local_competition = old_enable
        if old_strength is not None and hasattr(model, "competition_strength"):
            model.competition_strength = old_strength


def select_indices_by_opacity(gaussians, min_opacity=0.01, max_gaussians=180000):
    opacity = gaussians["opacity"][0, :, 0].detach().float()
    keep = opacity > float(min_opacity)
    if not bool(keep.any()):
        keep = opacity >= opacity.max()
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    if max_gaussians > 0 and idx.numel() > max_gaussians:
        scores = opacity[idx]
        top = torch.topk(scores, k=max_gaussians, largest=True).indices
        idx = idx[top]
    return idx


def gather_gaussians(gaussians, idx):
    out = {}
    for key in (
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
    ):
        value = gaussians.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = value[:, idx].contiguous()
    return out


def gaussians_to_cpu(gaussians):
    return {
        key: value.detach().float().cpu()
        for key, value in gaussians.items()
        if isinstance(value, torch.Tensor)
    }


def stats_to_cpu(stats):
    out = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().float().mean().cpu().item())
    return out


def run_variant_and_render(
        model,
        imgs_batch,
        device,
        amp_context,
        enable_comp,
        idx=None,
        args=None,
        height=None,
        width=None,
        competition_strength=None,
):
    result = run_forward(
        model,
        imgs_batch,
        device,
        amp_context,
        enable_comp,
        competition_strength=competition_strength,
    )
    if idx is None:
        idx = select_indices_by_opacity(
            result["gaussians"],
            min_opacity=args.render_min_opacity,
            max_gaussians=args.max_render_gaussians,
        )
    idx_device = idx.to(result["gaussians"]["xyz"].device)
    render_gaussians = gather_gaussians(result["gaussians"], idx_device)
    alpha_maps = render_alpha_maps(render_gaussians, result["camera_poses"], result["intrinsics"], height, width)
    cpu_gaussians = gaussians_to_cpu(render_gaussians)
    cpu_stats = stats_to_cpu(result.get("gaussian_stats", {}))
    del result, render_gaussians
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return idx.detach().cpu(), cpu_gaussians, cpu_stats, alpha_maps


def alpha_vis(alpha, shared_limits=None, cmap=cv2.COLORMAP_MAGMA):
    alpha = np.clip(alpha, 0.0, 1.0)
    if shared_limits is None:
        lo, hi = robust_limits(alpha, low=2, high=99.5)
    else:
        lo, hi = shared_limits
    hi = max(hi, lo + 1e-4)
    norm = np.clip((alpha - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return colorize_scalar(norm, min_val=0.0, max_val=1.0, cmap=cmap), norm


def save_histograms(opacity_before, opacity_after, gate_proxy, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    before = np.clip(opacity_before, 1e-6, 1.0)
    after = np.clip(opacity_after, 1e-6, 1.0)
    gate = np.clip(gate_proxy, 0.0, 1.0)
    suppression = 1.0 - gate

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    bins = np.linspace(0.0, 1.0, 80)
    axes[0].hist(before, bins=bins, color="#B44CC2", alpha=0.56, density=True, label="before")
    axes[0].hist(after, bins=bins, color="#2BAE8A", alpha=0.56, density=True, label="after")
    axes[0].set_title("Opacity distribution")
    axes[0].set_xlabel("opacity")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    axes[0].grid(True, linewidth=0.3, alpha=0.35)

    axes[1].hist(suppression, bins=bins, color="#D95F3D", alpha=0.72, density=True)
    axes[1].set_title("Local competition suppression")
    axes[1].set_xlabel("1 - opacity_after / opacity_before")
    axes[1].set_ylabel("density")
    axes[1].grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout(pad=0.8)
    path = os.path.join(out_dir, "local_competition_opacity_gate_histograms.png")
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def save_opacity_cdf(opacity_before, opacity_after, path):
    before = np.clip(opacity_before[np.isfinite(opacity_before)], 0.0, 1.0)
    after = np.clip(opacity_after[np.isfinite(opacity_after)], 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 401)

    def cdf(values):
        if values.size == 0:
            return np.zeros((bins.size - 1,), dtype=np.float32)
        hist, _ = np.histogram(values, bins=bins)
        return np.cumsum(hist).astype(np.float64) / max(1, int(hist.sum()))

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    x = bins[1:]
    ax.plot(x, cdf(before), color="#3B6FB6", linewidth=2.0, label="pre-gate")
    ax.plot(x, cdf(after), color="#D4513C", linewidth=2.0, label="post-gate")
    ax.axvline(0.05, color="#202020", linestyle="--", linewidth=1.1, label="0.05 threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("opacity")
    ax.set_ylabel("CDF")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def vector_from_gaussians(gaussians, key, length, fill=0.0):
    value = gaussians.get(key)
    if not isinstance(value, torch.Tensor):
        return np.full((length,), fill, dtype=np.float32)
    arr = value.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0, :, 0]
    elif arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1).astype(np.float32, copy=False)
    if arr.shape[0] == length:
        return arr
    out = np.full((length,), fill, dtype=np.float32)
    n = min(length, arr.shape[0])
    out[:n] = arr[:n]
    return out


def matrix_from_gaussians(gaussians, key, length, channels, fill=0.0):
    value = gaussians.get(key)
    if not isinstance(value, torch.Tensor):
        return np.full((length, channels), fill, dtype=np.float32)
    arr = value.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
    if arr.shape[0] == length and arr.shape[1] >= channels:
        return arr[:, :channels]
    out = np.full((length, channels), fill, dtype=np.float32)
    n = min(length, arr.shape[0])
    c = min(channels, arr.shape[1])
    out[:n, :c] = arr[:n, :c]
    return out


def write_per_gaussian_diagnostics(gauss_before, gauss_after, path, max_rows=250000):
    opacity_pre = gauss_before["opacity"][0, :, 0].numpy()
    opacity_post = gauss_after["opacity"][0, :, 0].numpy()
    n = int(opacity_post.shape[0])
    redundancy = vector_from_gaussians(gauss_after, "redundancy_score", n, fill=0.0)
    coef = vector_from_gaussians(gauss_after, "redundancy_coef", n, fill=0.0)
    gate = vector_from_gaussians(gauss_after, "competition_gate", n, fill=1.0)
    active = vector_from_gaussians(gauss_after, "competition_active", n, fill=0.0)
    source_view = vector_from_gaussians(gauss_before, "source_view", n, fill=-1.0)
    xyz = matrix_from_gaussians(gauss_before, "xyz", n, 3)
    color = matrix_from_gaussians(gauss_before, "color", n, 3)

    indices = np.arange(n, dtype=np.int64)
    if max_rows and max_rows > 0 and n > max_rows:
        priority = np.maximum(opacity_pre, opacity_post) + coef + np.clip(1.0 - gate, 0.0, 1.0)
        indices = np.argpartition(priority, -max_rows)[-max_rows:]

    rows = []
    for idx in indices:
        rows.append({
            "gaussian_index": int(idx),
            "redundancy_score": f"{float(redundancy[idx]):.8f}",
            "redundancy_coef": f"{float(coef[idx]):.8f}",
            "competition_gate": f"{float(gate[idx]):.8f}",
            "competition_active": int(active[idx] > 0.5),
            "opacity_pre": f"{float(opacity_pre[idx]):.8f}",
            "opacity_post": f"{float(opacity_post[idx]):.8f}",
            "active_005": int(opacity_post[idx] > 0.05),
            "source_view": int(round(float(source_view[idx]))),
            "xyz_x": f"{float(xyz[idx, 0]):.8f}",
            "xyz_y": f"{float(xyz[idx, 1]):.8f}",
            "xyz_z": f"{float(xyz[idx, 2]):.8f}",
            "color_r": f"{float(color[idx, 0]):.8f}",
            "color_g": f"{float(color[idx, 1]):.8f}",
            "color_b": f"{float(color[idx, 2]):.8f}",
        })
    write_csv(rows, path)
    return path, int(indices.shape[0]), n


def save_spatial_suppression_distribution(gaussians_before, gate_proxy, path, max_points=80000, seed=0):
    xyz = gaussians_before["xyz"][0].numpy()
    opacity = gaussians_before["opacity"][0, :, 0].numpy()
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity) & (opacity > 0)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return False
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    points = xyz[idx]
    suppression = np.clip(1.0 - gate_proxy[idx], 0.0, 1.0)
    sizes = 1.0 + 6.0 * suppression

    fig = plt.figure(figsize=(11.4, 8.2))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    projections = [(0, 1, "X", "Y", "XY projection"), (0, 2, "X", "Z", "XZ projection"), (1, 2, "Y", "Z", "YZ projection")]
    for ax, (a, b, la, lb, title) in zip(axes[:3], projections):
        sc = ax.scatter(points[:, a], points[:, b], s=sizes, c=suppression, cmap="inferno", vmin=0, vmax=1,
                        alpha=0.36, linewidths=0, rasterized=True)
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(axis_limits(points, a))
        ax.set_ylim(axis_limits(points, b))
        ax.grid(True, linewidth=0.3, alpha=0.35)

    ax3d = axes[3]
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=sizes, c=suppression, cmap="inferno",
                 vmin=0, vmax=1, alpha=0.28, linewidths=0)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_xlim(axis_limits(points, 0))
    ax3d.set_ylim(axis_limits(points, 1))
    ax3d.set_zlim(axis_limits(points, 2))
    ax3d.view_init(elev=24, azim=-55)
    ax3d.set_title("3D suppression view", fontsize=10)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.018)
    cbar.set_label("suppression strength")
    fig.suptitle(f"Local Competition Suppression in Gaussian Space | sampled={points.shape[0]}", fontsize=12)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return True


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export redundancy-aware local opacity competition paper figures for Pi3_3DGS_10."
    )
    parser.add_argument("--data_root", type=str, default="/data/liuwei/dataset/ntu_seq/cp")
    parser.add_argument("--output_dir", type=str, default="outputs/cp_800_900_pi3_3dgs10_local_competition_paper_figs")
    parser.add_argument("--ckpt", type=str, default="outputs/pi3_3dgs10_hunyuan_decoder/ckpts/best_model/model.safetensors")
    parser.add_argument("--model_impl", type=str, default="_10", help="Full import path or shorthand _8/_9/_10.")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Hydra config.yaml used to restore model construction args. Default: auto-detect near ckpt.")
    parser.add_argument("--ignore_model_config", action="store_true",
                        help="Use constructor defaults instead of checkpoint-side config.")
    parser.add_argument("--frame_start", type=int, default=800)
    parser.add_argument("--frame_end", type=int, default=900)
    parser.add_argument("--frame_step", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--render_min_opacity", type=float, default=0.01)
    parser.add_argument("--max_render_gaussians", type=int, default=180000)
    parser.add_argument("--spatial_max_points", type=int, default=80000)
    parser.add_argument("--post_gate_competition_strength", type=float, default=None,
                        help="Post-gate competition strength. Default keeps the model constructor/config value.")
    parser.add_argument("--diagnostic_max_rows", type=int, default=250000,
                        help="Max rows in the per-Gaussian diagnostic CSV. Use 0 for every compared Gaussian.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    per_frame_dir = os.path.join(args.output_dir, "per_frame")
    overview_dir = os.path.join(args.output_dir, "overview")
    os.makedirs(per_frame_dir, exist_ok=True)
    os.makedirs(overview_dir, exist_ok=True)

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

    model, ckpt, model_impl, loaded_config_path = load_model(args, device)
    post_strength = args.post_gate_competition_strength
    if post_strength is None and hasattr(model, "competition_strength"):
        post_strength = float(model.competition_strength)
    if post_strength is None:
        post_strength = 1.0
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    imgs_batch = imgs_cpu.unsqueeze(0).to(device)
    idx, gauss_before, stats_before, alpha_before = run_variant_and_render(
        model,
        imgs_batch,
        device,
        amp_context,
        True,
        idx=None,
        args=args,
        height=height,
        width=width,
        competition_strength=0.0,
    )
    _, gauss_after, stats_after, alpha_after = run_variant_and_render(
        model,
        imgs_batch,
        device,
        amp_context,
        True,
        idx=idx,
        args=args,
        height=height,
        width=width,
        competition_strength=post_strength,
    )

    opacity_before = gauss_before["opacity"][0, :, 0].numpy()
    opacity_after = gauss_after["opacity"][0, :, 0].numpy()
    gate_proxy = np.clip(opacity_after / np.clip(opacity_before, 1e-6, None), 0.0, 1.0)
    suppression_strength = 1.0 - gate_proxy

    hist_path = save_histograms(opacity_before, opacity_after, gate_proxy, overview_dir)
    cdf_path = save_opacity_cdf(
        opacity_before,
        opacity_after,
        os.path.join(overview_dir, "local_competition_opacity_cdf_pre_post.png"),
    )
    diag_csv_path, diag_rows, diag_total = write_per_gaussian_diagnostics(
        gauss_before,
        gauss_after,
        os.path.join(args.output_dir, "per_gaussian_local_competition_diagnostics.csv"),
        max_rows=args.diagnostic_max_rows,
    )
    spatial_path = os.path.join(overview_dir, "local_competition_spatial_suppression.png")
    save_spatial_suppression_distribution(
        gauss_before,
        gate_proxy,
        spatial_path,
        max_points=args.spatial_max_points,
    )

    rows = []
    contact_rows = []
    for i, item in enumerate(frame_items):
        frame_rgb = input_rgbs[i]
        stem = safe_name(item["stem"])
        before = np.clip(alpha_before[i], 0.0, 1.0)
        after = np.clip(alpha_after[i], 0.0, 1.0)
        suppressed = np.clip(before - after, 0.0, 1.0)
        relative = suppressed / np.clip(before, 1e-4, None)

        shared_lo, shared_hi = robust_limits(np.concatenate([before.reshape(-1), after.reshape(-1)]), low=2, high=99.5)
        before_img, before_norm = alpha_vis(before, shared_limits=(shared_lo, shared_hi), cmap=cv2.COLORMAP_MAGMA)
        after_img, after_norm = alpha_vis(after, shared_limits=(shared_lo, shared_hi), cmap=cv2.COLORMAP_MAGMA)

        supp_hi = max(robust_limits(suppressed, low=1, high=99.5)[1], 1e-4)
        supp_norm = np.clip(suppressed / supp_hi, 0.0, 1.0)
        supp_img = colorize_scalar(supp_norm, min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_INFERNO)
        supp_overlay = blend_heatmap(frame_rgb, supp_img, supp_norm, max_alpha=0.80, base_dim=0.56)

        rel_img = colorize_scalar(np.clip(relative, 0.0, 1.0), min_val=0.0, max_val=1.0, cmap=cv2.COLORMAP_TURBO)
        rel_overlay = blend_heatmap(frame_rgb, rel_img, np.clip(relative, 0.0, 1.0), max_alpha=0.78, base_dim=0.58)

        save_rgb(os.path.join(per_frame_dir, f"input_{stem}.png"), frame_rgb)
        save_rgb(os.path.join(per_frame_dir, f"opacity_pre_gate_strength0_{stem}.png"), before_img)
        save_rgb(os.path.join(per_frame_dir, f"opacity_post_gate_{stem}.png"), after_img)
        save_rgb(os.path.join(per_frame_dir, f"local_competition_suppression_map_{stem}.png"), supp_img)
        save_rgb(os.path.join(per_frame_dir, f"local_competition_suppression_overlay_{stem}.png"), supp_overlay)
        save_rgb(os.path.join(per_frame_dir, f"local_competition_relative_overlay_{stem}.png"), rel_overlay)

        save_panel(
            [frame_rgb, before_img, after_img, supp_overlay],
            ["Input", "Pre-gate strength=0", "Post-gate", "Suppressed opacity"],
            os.path.join(per_frame_dir, f"local_competition_panel_{stem}.png"),
            figsize_per_image=2.8,
            dpi=300,
        )
        save_panel(
            [frame_rgb, before_img, after_img, supp_img, supp_overlay, rel_overlay],
            ["Input", "Pre-gate", "Post-gate", "Suppression", "Suppression overlay", "Relative effect"],
            os.path.join(per_frame_dir, f"local_competition_pipeline_{stem}.png"),
            figsize_per_image=2.35,
            dpi=260,
        )

        contact_rows.append([frame_rgb, before_img, after_img, supp_overlay])
        rows.append({
            "frame_id": item["frame_id"],
            "source_path": item["path"],
            "alpha_before_mean": f"{float(before.mean()):.8f}",
            "alpha_after_mean": f"{float(after.mean()):.8f}",
            "alpha_suppressed_mean": f"{float(suppressed.mean()):.8f}",
            "alpha_suppressed_p99": f"{float(np.percentile(suppressed, 99)):.8f}",
            "relative_suppressed_mean": f"{float(relative.mean()):.8f}",
        })

    save_contact_sheet(
        contact_rows,
        ["Input", "Pre-gate", "Post-gate", "Suppressed opacity"],
        os.path.join(overview_dir, "cp_800_900_local_competition_contact_sheet.png"),
        dpi=220,
    )
    write_csv(rows, os.path.join(args.output_dir, "manifest.csv"))

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("Pi3_3DGS_10 redundancy-aware local opacity competition paper figures\n")
        f.write("=" * 64 + "\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"frames: {args.frame_start}-{args.frame_end} step {args.frame_step}\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"model_impl: {model_impl}\n")
        f.write(f"model_config: {loaded_config_path or 'N/A'}\n")
        f.write(f"model_resolution: {height}x{width}\n")
        f.write(f"gs_view_stride: {args.gs_view_stride}\n")
        f.write("pre_gate_competition_strength: 0.0\n")
        f.write(f"post_gate_competition_strength: {float(post_strength):.8f}\n")
        f.write(f"render_gaussians_compared: {int(idx.numel())}\n")
        f.write(f"opacity_before_mean: {float(opacity_before.mean()):.8f}\n")
        f.write(f"opacity_after_mean: {float(opacity_after.mean()):.8f}\n")
        f.write(f"gate_proxy_mean: {float(gate_proxy.mean()):.8f}\n")
        f.write(f"suppression_strength_mean: {float(suppression_strength.mean()):.8f}\n")
        f.write(f"diagnostic_csv: {diag_csv_path}\n")
        f.write(f"diagnostic_rows: {diag_rows} / {diag_total}\n")
        f.write("\npre_gate_strength0_stats:\n")
        for key in sorted(stats_before):
            f.write(f"  {key}: {stats_before[key]:.8f}\n")
        f.write("\npost_gate_local_competition_stats:\n")
        for key in sorted(stats_after):
            f.write(f"  {key}: {stats_after[key]:.8f}\n")
        f.write(f"\nhistogram: {hist_path}\n")
        f.write(f"opacity_cdf: {cdf_path}\n")
        f.write(f"spatial_suppression: {spatial_path}\n")
        f.write(f"per_frame_dir: {per_frame_dir}\n")
        f.write(f"overview_dir: {overview_dir}\n")

    print(f"Saved per-frame figures: {per_frame_dir}")
    print(f"Saved overview figures: {overview_dir}")
    print(f"Manifest: {os.path.join(args.output_dir, 'manifest.csv')}")


if __name__ == "__main__":
    main()
