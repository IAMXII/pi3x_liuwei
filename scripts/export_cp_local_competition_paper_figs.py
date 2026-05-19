import argparse
import csv
import gc
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
    resolve_checkpoint_path,
    robust_limits,
    safe_name,
    save_contact_sheet,
    save_panel,
    save_rgb,
)
from pi3.models.pi3_3dgs_7 import Pi3_3DGS


def load_model(args, device):
    ckpt = resolve_checkpoint_path(args.ckpt)
    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
        gs_view_stride=args.gs_view_stride,
        enable_local_competition=False,
    ).to(device).eval()
    if ckpt.endswith(".safetensors"):
        weight = load_file(ckpt, device="cpu")
    else:
        weight = torch.load(ckpt, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(weight, strict=False)
    del weight
    print(f"Checkpoint: {ckpt}")
    print(f"load_state_dict: {load_result}")
    return model, ckpt


def run_forward(model, imgs_batch, device, amp_context, enable_local_competition):
    model.enable_local_competition = bool(enable_local_competition)
    tag = "with" if enable_local_competition else "without"
    print(f"Running inference {tag} local competition...")
    with torch.no_grad():
        with amp_context():
            return model(imgs_batch, return_viz=False)


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
    for key in ("xyz", "rotation", "scale", "opacity", "color", "conf"):
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


def run_variant_and_render(model, imgs_batch, device, amp_context, enable_comp, idx=None, args=None, height=None, width=None):
    result = run_forward(model, imgs_batch, device, amp_context, enable_comp)
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
    parser = argparse.ArgumentParser(description="Export local competition paper figures for Pi3_3DGS_7.")
    parser.add_argument("--data_root", type=str, default="/data/liuwei/dataset/ntu_seq/cp")
    parser.add_argument("--output_dir", type=str, default="outputs/cp_800_900_local_competition_paper_figs")
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

    model, ckpt = load_model(args, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    imgs_batch = imgs_cpu.unsqueeze(0).to(device)
    idx, gauss_before, stats_before, alpha_before = run_variant_and_render(
        model, imgs_batch, device, amp_context, False, idx=None, args=args, height=height, width=width
    )
    _, gauss_after, stats_after, alpha_after = run_variant_and_render(
        model, imgs_batch, device, amp_context, True, idx=idx, args=args, height=height, width=width
    )

    opacity_before = gauss_before["opacity"][0, :, 0].numpy()
    opacity_after = gauss_after["opacity"][0, :, 0].numpy()
    gate_proxy = np.clip(opacity_after / np.clip(opacity_before, 1e-6, None), 0.0, 1.0)
    suppression_strength = 1.0 - gate_proxy

    hist_path = save_histograms(opacity_before, opacity_after, gate_proxy, overview_dir)
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
        save_rgb(os.path.join(per_frame_dir, f"opacity_without_local_competition_{stem}.png"), before_img)
        save_rgb(os.path.join(per_frame_dir, f"opacity_with_local_competition_{stem}.png"), after_img)
        save_rgb(os.path.join(per_frame_dir, f"local_competition_suppression_map_{stem}.png"), supp_img)
        save_rgb(os.path.join(per_frame_dir, f"local_competition_suppression_overlay_{stem}.png"), supp_overlay)
        save_rgb(os.path.join(per_frame_dir, f"local_competition_relative_overlay_{stem}.png"), rel_overlay)

        save_panel(
            [frame_rgb, before_img, after_img, supp_overlay],
            ["Input", "Without local competition", "With local competition", "Suppressed opacity"],
            os.path.join(per_frame_dir, f"local_competition_panel_{stem}.png"),
            figsize_per_image=2.8,
            dpi=300,
        )
        save_panel(
            [frame_rgb, before_img, after_img, supp_img, supp_overlay, rel_overlay],
            ["Input", "Before", "After", "Suppression", "Suppression overlay", "Relative effect"],
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
        ["Input", "Without LC", "With LC", "Suppressed opacity"],
        os.path.join(overview_dir, "cp_800_900_local_competition_contact_sheet.png"),
        dpi=220,
    )
    write_csv(rows, os.path.join(args.output_dir, "manifest.csv"))

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("Pi3_3DGS_7 CP 800-900 local competition paper figures\n")
        f.write("=" * 64 + "\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"frames: {args.frame_start}-{args.frame_end} step {args.frame_step}\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"model_resolution: {height}x{width}\n")
        f.write(f"gs_view_stride: {args.gs_view_stride}\n")
        f.write(f"render_gaussians_compared: {int(idx.numel())}\n")
        f.write(f"opacity_before_mean: {float(opacity_before.mean()):.8f}\n")
        f.write(f"opacity_after_mean: {float(opacity_after.mean()):.8f}\n")
        f.write(f"gate_proxy_mean: {float(gate_proxy.mean()):.8f}\n")
        f.write(f"suppression_strength_mean: {float(suppression_strength.mean()):.8f}\n")
        f.write("\nwithout_local_competition_stats:\n")
        for key in sorted(stats_before):
            f.write(f"  {key}: {stats_before[key]:.8f}\n")
        f.write("\nwith_local_competition_stats:\n")
        for key in sorted(stats_after):
            f.write(f"  {key}: {stats_after[key]:.8f}\n")
        f.write(f"\nhistogram: {hist_path}\n")
        f.write(f"spatial_suppression: {spatial_path}\n")
        f.write(f"per_frame_dir: {per_frame_dir}\n")
        f.write(f"overview_dir: {overview_dir}\n")

    print(f"Saved per-frame figures: {per_frame_dir}")
    print(f"Saved overview figures: {overview_dir}")
    print(f"Manifest: {os.path.join(args.output_dir, 'manifest.csv')}")


if __name__ == "__main__":
    main()
