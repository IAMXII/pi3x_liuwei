import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path


METRIC_PATTERNS = {
    "psnr": r"Average PSNR:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "ssim": r"Average SSIM:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "lpips": r"Average LPIPS:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
}


VARIANTS = {
    # Baseline from the checkpoint-side Hydra config.
    "full": [],
    # Isolate local opacity suppression.
    "no_local_competition": ["--disable_local_competition"],
    # Isolate quadtree candidate thinning.
    "no_quadtree": ["--disable_quadtree"],
    # Dense candidate field without the local competition suppressor.
    "dense_no_competition": ["--disable_quadtree", "--disable_local_competition"],
    # Test whether the learned RGB head is the main bottleneck.
    "input_color": ["--color_source", "input"],
    # Test whether the learned density gate is over-attenuating opacity.
    "no_density_opacity_gate": ["--disable_density_opacity_gate"],
    # Closest feed-forward comparison to Hunyuan's dense per-pixel splats:
    # dense candidates, no local competition, no opacity gate, no opacity pre-filter,
    # and RGB taken from the source pixel like Hunyuan's RGB2SH DC initialization.
    "dense_input_color_no_gate_no_filter": [
        "--disable_quadtree",
        "--disable_local_competition",
        "--disable_density_opacity_gate",
        "--opacity_filter_threshold",
        "0.0",
        "--color_source",
        "input",
    ],
    # Explicit HunyuanWorld-Mirror-style output strategy comparison:
    # dense per-pixel source-view splats, source RGB as DC color, pixel-footprint
    # scale, high opacity, and the same Pi3 geometry/cameras for control.
    "hunyuan_like": [
        "--gaussian_mode",
        "hunyuan_like",
        "--hunyuan_like_pixel_scale",
        "1.0",
        "--hunyuan_like_opacity",
        "0.95",
    ],
    "hunyuan_like_scale0.5": [
        "--gaussian_mode",
        "hunyuan_like",
        "--hunyuan_like_pixel_scale",
        "0.5",
        "--hunyuan_like_opacity",
        "0.95",
    ],
    "hunyuan_like_pruned": [
        "--gaussian_mode",
        "hunyuan_like",
        "--hunyuan_like_pixel_scale",
        "1.0",
        "--hunyuan_like_opacity",
        "0.95",
        "--hunyuan_like_prune",
    ],
    # If this helps, current Gaussians are too large and causing blur.
    "scale_x0.2": ["--scale_activation_multiplier", "0.02"],
}


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_report(report_path):
    metrics = {key: math.nan for key in METRIC_PATTERNS}
    metrics.update({
        "frames": "",
        "gaussian_input_frames": "",
        "color_source": "",
        "quadtree_enabled": "",
        "local_competition_enabled": "",
        "learnable_sampling_enabled": "",
        "density_opacity_gate_enabled": "",
        "gaussian_mode": "",
        "frame_names": "",
    })
    if not report_path.is_file():
        return metrics

    text = report_path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = parse_float(match.group(1))

    for key in (
        "frames",
        "gaussian_input_frames",
        "color_source",
        "quadtree_enabled",
        "local_competition_enabled",
        "learnable_sampling_enabled",
        "density_opacity_gate_enabled",
        "gaussian_mode",
        "frame_names",
    ):
        match = re.search(rf"^{key}:\s*(.*)$", text, flags=re.MULTILINE)
        if match:
            metrics[key] = match.group(1).strip()
    return metrics


def parse_counts(counts_path):
    total = 0
    active = 0
    stats = {}
    if not counts_path.is_file():
        return total, active, stats

    with counts_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += int(float(row.get("total_count") or 0))
            active += int(float(row.get("active_count_opacity_gt_005") or 0))
            for key, value in row.items():
                if not key.startswith("stat_") or value in (None, ""):
                    continue
                try:
                    stats[key] = float(value)
                except ValueError:
                    pass
    return total, active, stats


def build_command(args, variant):
    out_dir = Path(args.output_root) / variant
    command = [
        args.python,
        "example_3dgs_5.py",
        "--ckpt",
        args.ckpt,
        "--output_dir",
        str(out_dir),
        "--device",
        args.device,
        "--eval_source",
        "three_sixty_v2_dataset",
        "--dataset_root",
        args.dataset_root,
        "--dataset_scene",
        args.dataset_scene,
        "--dataset_split",
        args.dataset_split,
        "--dataset_image_dir_name",
        args.dataset_image_dir_name,
        "--dataset_hold_every",
        str(args.dataset_hold_every),
        "--dataset_frame_num",
        str(args.dataset_frame_num),
        "--dataset_resolution",
        args.dataset_resolution,
        "--dataset_seed",
        str(args.dataset_seed),
        "--dataset_index",
        str(args.dataset_index),
        "--gs_view_stride",
        str(args.gs_view_stride),
        "--chunk_size",
        str(args.chunk_size),
        "--metric_lpips_net",
        args.metric_lpips_net,
        "--ablation_name",
        variant,
    ]
    if not args.dataset_shuffle_views:
        command.append("--no-dataset_shuffle_views")
    if args.model_impl:
        command.extend(["--model_impl", args.model_impl])
    if args.model_config:
        command.extend(["--model_config", args.model_config])
    if args.per_chunk_scene:
        command.append("--per_chunk_scene")
    if args.skip_save_ply:
        command.append("--skip_save_ply")
    if args.skip_save_frames:
        command.append("--skip_save_frames")
    command.extend(VARIANTS[variant])
    return command, out_dir


def summarize(output_root, variants):
    rows = []
    for variant in variants:
        out_dir = Path(output_root) / variant
        metrics = parse_report(out_dir / "metrics_report.txt")
        total, active, stats = parse_counts(out_dir / "gaussian_counts.csv")
        rows.append({
            "variant": variant,
            "output_dir": str(out_dir),
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "gaussian_total": total,
            "gaussian_active_opacity_gt_005": active,
            "frames": metrics["frames"],
            "gaussian_input_frames": metrics["gaussian_input_frames"],
            "color_source": metrics["color_source"],
            "quadtree_enabled": metrics["quadtree_enabled"],
            "local_competition_enabled": metrics["local_competition_enabled"],
            "learnable_sampling_enabled": metrics["learnable_sampling_enabled"],
            "density_opacity_gate_enabled": metrics["density_opacity_gate_enabled"],
            "gaussian_mode": metrics["gaussian_mode"],
            "stat_selected_count": stats.get("stat_selected_count", math.nan),
            "stat_proposal_count": stats.get("stat_proposal_count", math.nan),
            "stat_density_gate_mean": stats.get("stat_density_gate_mean", math.nan),
            "stat_competition_gate_mean": stats.get("stat_competition_gate_mean", math.nan),
            "frame_names": metrics["frame_names"],
        })

    summary_path = Path(output_root) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to: {summary_path}")
    for row in rows:
        print(
            f"{row['variant']:36s} "
            f"SSIM={row['ssim']:.4f} PSNR={row['psnr']:.3f} LPIPS={row['lpips']:.4f} "
            f"active={row['gaussian_active_opacity_gt_005']}/{row['gaussian_total']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Controlled Pi3 Gaussian-head verification on 360_v2 bicycle."
    )
    parser.add_argument(
        "--ckpt",
        default="outputs/pi3_highres_0528_v9_360_only_bicycle_val/ckpts/best_model/model.safetensors",
    )
    parser.add_argument("--output_root", default="outputs/pi3_gs_head_verification_0530_bicycle")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_impl", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--dataset_root", default="/data/liuwei/dataset/360_v2")
    parser.add_argument("--dataset_scene", default="bicycle")
    parser.add_argument("--dataset_split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--dataset_image_dir_name", default="images_4")
    parser.add_argument("--dataset_hold_every", type=int, default=8)
    parser.add_argument("--dataset_frame_num", type=int, default=8)
    parser.add_argument("--dataset_resolution", default="518x336")
    parser.add_argument("--dataset_seed", type=int, default=2024)
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--dataset_shuffle_views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--metric_lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--per_chunk_scene", action="store_true")
    parser.add_argument("--skip_save_ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_save_frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS.keys(), default=list(VARIANTS.keys()))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summarize(output_root, args.variants)
        return

    for variant in args.variants:
        command, out_dir = build_command(args, variant)
        out_dir.mkdir(parents=True, exist_ok=True)
        command_log = out_dir / "command.txt"
        command_log.write_text(" ".join(command) + "\n", encoding="utf-8")

        if args.skip_existing and (out_dir / "metrics_report.txt").is_file():
            print(f"Skipping existing result: {variant}")
            continue

        print("\n" + "=" * 80)
        print(f"Running {variant}")
        print(" ".join(command))
        if args.dry_run:
            continue

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED {variant}: returncode={exc.returncode}")
            if not args.continue_on_error:
                raise

    if args.dry_run:
        print("Dry run complete; no metrics were generated.")
    else:
        summarize(output_root, args.variants)


if __name__ == "__main__":
    main()
