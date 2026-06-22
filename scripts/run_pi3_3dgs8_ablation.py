import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path


DATASETS = {
    # "cp": {
    #     "rgb": "/data/liuwei/dataset/ntu_seq/cp/rgb",
    #     "depth": "/data/liuwei/dataset/ntu_seq/cp/depth",
    #     "subset_start": 1146,
    #     "subset_end": 1346,
    #     "subset_step": 5,
    # },
    # "campus": {
    #     "rgb": "/data/liuwei/dataset/ntu_seq/campus/rgb",
    #     "depth": "/data/liuwei/dataset/ntu_seq/campus/depth",
    #     "subset_start": 148,
    #     "subset_end": 248,
    #     "subset_step": 5,
    # },
    # "hku": {
    #     "rgb": "/data/liuwei/dataset/ntu_seq/hku1/rgb",
    #     "depth": "/data/liuwei/dataset/ntu_seq/hku1/depth",
    #     "subset_start": 412,
    #     "subset_end": 512,
    #     "subset_step": 5,
    # },
    "bicycle": {
        "rgb": "/data/liuwei/dataset/360_v2/bicycle/images_4",
        "depth": None,
        "subset_start": 0,
        "subset_end": 90,
        "subset_step": 8,
    },
    "stump": {
        "rgb": "/data/liuwei/dataset/360_v2/stump/images_4",
        "depth": None,
        "subset_start": 0,
        "subset_end": 60,
        "subset_step": 3,
    },
}


VARIANTS = {
    # Full Pi3_3DGS_10 path: quadtree proposals + local competition.
    # RGB rendering must keep pushed/far support Gaussians; only PLY export uses
    # the opacity threshold.
    "full": [
        "--proposal_sampling_mode", "quadtree",
        "--render_opacity_threshold", "0.0",
        "--ply_opacity_threshold", "0.05",
    ],
    # Quadtree sampling ablation: keep quadtree's per-view proposal count, but choose random pixels.
    "random_equal_sample": [
        "--proposal_sampling_mode", "random_equal",
        "--render_opacity_threshold", "0.0",
        "--ply_opacity_threshold", "0.05",
    ],
    # Local competition ablation requested here: no local competition and no opacity>0.05 render filter.
    "no_local_no_opacity_filter": [
        "--proposal_sampling_mode", "quadtree",
        "--disable_local_competition",
        "--render_opacity_threshold", "0.0",
        "--ply_opacity_threshold", "0.0",
    ],
    # Dense all-Gaussian output: no quadtree, no local competition, no model/render/export opacity filter.
    "pure_all_gaussians": [
        "--disable_quadtree",
        "--disable_local_competition",
        "--opacity_filter_threshold", "0.0",
        "--render_opacity_threshold", "0.0",
        "--ply_opacity_threshold", "0.0",
    ],
}


METRIC_PATTERNS = {
    "psnr": r"Average PSNR:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "ssim": r"Average SSIM:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "lpips": r"Average LPIPS:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "abs_rel": r"Average AbsRel:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "d_rmse": r"Average dRMSE:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "sro": r"Average SRO:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "lrc": r"Average LRC:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
    "sldq": r"Average S-LDQ:\s+([+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+))",
}


COUNT_LIKE_STAT_KEYS = {
    "stat_redundancy_candidates",
    "stat_competition_candidates",
    "stat_competition_expected_suppressed",
    "stat_redundancy_expected_suppressed",
    "stat_opacity_mass",
    "stat_active_count_opacity_005",
    "stat_active_count_opacity_002",
    "stat_count_before",
    "stat_count_after",
    "stat_physical_count",
    "stat_proposal_count",
    "stat_learned_extra_count",
    "stat_selected_count",
}
MEAN_LIKE_STAT_KEYS = {
    "stat_redundancy_coef_mean",
    "stat_competition_gate_mean",
    "stat_redundancy_gate_mean",
    "stat_redundancy_threshold",
    "stat_local_radius_mean",
    "stat_local_radius_median",
    "stat_cube_size_mean",
    "stat_cube_size_median",
}
MAX_LIKE_STAT_KEYS = {
    "stat_redundancy_coef_max",
}
SUMMARY_STAT_KEYS = [
    "stat_redundancy_candidates",
    "stat_redundancy_coef_mean",
    "stat_redundancy_coef_max",
    "stat_competition_gate_mean",
    "stat_redundancy_gate_mean",
    "stat_opacity_mass",
    "stat_active_count_opacity_005",
    "stat_competition_expected_suppressed",
    "stat_redundancy_expected_suppressed",
    "stat_redundancy_threshold",
    "stat_local_radius_mean",
    "stat_local_radius_median",
    "stat_cube_size_mean",
    "stat_cube_size_median",
    "stat_count_before",
    "stat_count_after",
    "stat_physical_count",
    "stat_proposal_count",
    "stat_learned_extra_count",
    "stat_selected_count",
]


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_metrics(report_path):
    metrics = {key: math.nan for key in METRIC_PATTERNS}
    metrics["depth_frames_used"] = ""
    metrics["depth_frames_total"] = ""
    metrics["eval_source"] = ""
    metrics["split"] = ""
    metrics["scene"] = ""
    metrics["model_impl"] = ""
    metrics["metric_lpips_net"] = ""
    metrics["frame_names"] = ""
    metrics["proposal_sampling_mode"] = ""
    metrics["render_opacity_threshold"] = ""
    metrics["ply_opacity_threshold"] = ""
    if not report_path.is_file():
        return metrics

    text = report_path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = parse_float(match.group(1))

    frame_match = re.search(r"Depth frames used:\s+(\d+)\s*/\s*(\d+)", text)
    if frame_match:
        metrics["depth_frames_used"] = frame_match.group(1)
        metrics["depth_frames_total"] = frame_match.group(2)
    for key in (
        "eval_source",
        "split",
        "scene",
        "model_impl",
        "metric_lpips_net",
        "frame_names",
        "proposal_sampling_mode",
        "render_opacity_threshold",
        "ply_opacity_threshold",
    ):
        meta_match = re.search(rf"^{key}:\s*(.*)$", text, flags=re.MULTILINE)
        if meta_match:
            metrics[key] = meta_match.group(1).strip()
    return metrics


def is_stale_report(report_path, command_path):
    if not report_path.is_file() or not command_path.is_file():
        return False
    return report_path.stat().st_mtime < command_path.stat().st_mtime


def parse_gaussian_counts(csv_path):
    total = 0
    active = 0
    rendered = 0
    saved = 0
    ply_paths = []
    stat_sums = {key: 0.0 for key in COUNT_LIKE_STAT_KEYS}
    stat_means = {key: [] for key in MEAN_LIKE_STAT_KEYS}
    stat_max = {key: math.nan for key in MAX_LIKE_STAT_KEYS}
    if not csv_path.is_file():
        return total, active, rendered, saved, "", {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_total = row.get("raw_total_count") or row.get("total_count") or 0
            raw_active = row.get("raw_active_count_opacity_gt_005") or row.get("active_count_opacity_gt_005") or 0
            rendered_count = row.get("rendered_count") or row.get("active_count_opacity_gt_005") or 0
            saved_count = row.get("ply_kept_count") or row.get("active_count_opacity_gt_005") or 0
            total += int(float(raw_total))
            active += int(float(raw_active))
            rendered += int(float(rendered_count))
            saved += int(float(saved_count))
            ply_path = row.get("ply_path") or ""
            if ply_path:
                ply_paths.append(ply_path)
            for key in COUNT_LIKE_STAT_KEYS:
                value = row.get(key)
                if value not in (None, ""):
                    try:
                        stat_sums[key] += float(value)
                    except ValueError:
                        pass
            for key in MEAN_LIKE_STAT_KEYS:
                value = row.get(key)
                if value not in (None, ""):
                    try:
                        stat_means[key].append(float(value))
                    except ValueError:
                        pass
            for key in MAX_LIKE_STAT_KEYS:
                value = row.get(key)
                if value not in (None, ""):
                    try:
                        parsed = float(value)
                    except ValueError:
                        continue
                    stat_max[key] = parsed if math.isnan(stat_max[key]) else max(stat_max[key], parsed)

    stats = {}
    stats.update(stat_sums)
    for key, values in stat_means.items():
        stats[key] = (sum(values) / len(values)) if values else math.nan
    stats.update(stat_max)
    return total, active, rendered, saved, ";".join(ply_paths), stats


def summarize_results(output_root, datasets, variants):
    rows = []
    for dataset in datasets:
        for variant in variants:
            out_dir = output_root / dataset / variant
            report_path = out_dir / "metrics_report.txt"
            command_path = out_dir / "command.txt"
            stale_report = is_stale_report(report_path, command_path)
            if stale_report:
                print(f"WARNING: stale metrics ignored for {dataset}/{variant}: {report_path}")
                metrics = {key: math.nan for key in METRIC_PATTERNS}
                metrics["depth_frames_used"] = ""
                metrics["depth_frames_total"] = ""
                metrics["eval_source"] = ""
                metrics["split"] = ""
                metrics["scene"] = ""
                metrics["model_impl"] = ""
                metrics["metric_lpips_net"] = ""
                metrics["frame_names"] = ""
                metrics["proposal_sampling_mode"] = ""
                metrics["render_opacity_threshold"] = ""
                metrics["ply_opacity_threshold"] = ""
                total, active, rendered, saved, ply_paths, stat_summary = 0, 0, 0, 0, "", {}
            else:
                metrics = parse_metrics(report_path)
                total, active, rendered, saved, ply_paths, stat_summary = parse_gaussian_counts(out_dir / "gaussian_counts.csv")
            row = {
                "dataset": dataset,
                "variant": variant,
                "output_dir": str(out_dir),
                "metrics_status": "stale_command_newer_than_report" if stale_report else "ok",
                "eval_source": metrics["eval_source"],
                "split": metrics["split"],
                "scene": metrics["scene"],
                "model_impl": metrics["model_impl"],
                "metric_lpips_net": metrics["metric_lpips_net"],
                "proposal_sampling_mode": metrics["proposal_sampling_mode"],
                "render_opacity_threshold": metrics["render_opacity_threshold"],
                "ply_opacity_threshold": metrics["ply_opacity_threshold"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "abs_rel": metrics["abs_rel"],
                "d_rmse": metrics["d_rmse"],
                "sro": metrics["sro"],
                "lrc": metrics["lrc"],
                "sldq": metrics["sldq"],
                "depth_frames_used": metrics["depth_frames_used"],
                "depth_frames_total": metrics["depth_frames_total"],
                "gaussian_total": total,
                "gaussian_active_opacity_gt_005": active,
                "gaussian_rendered": rendered,
                "gaussian_saved_ply": saved,
                "ply_paths": ply_paths,
                "frame_names": metrics["frame_names"],
            }
            for stat_key in SUMMARY_STAT_KEYS:
                row[stat_key] = stat_summary.get(stat_key, math.nan)
            rows.append(row)

    summary_path = output_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to: {summary_path}")


def build_command(args, dataset_name, variant_name):
    dataset = DATASETS[dataset_name]
    out_dir = Path(args.output_root) / dataset_name / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        "example_3dgs_6.py",
        "--ckpt", args.ckpt,
        "--output_dir", str(out_dir),
        "--device", args.device,
        "--eval_source", args.eval_source,
        "--pixel_limit", str(args.pixel_limit),
        "--gs_view_stride", str(args.gs_view_stride),
        "--chunk_size", str(args.chunk_size),
        "--ablation_name", variant_name,
        "--metric_lpips_net", args.metric_lpips_net,
    ]
    if args.model_impl:
        command.extend(["--model_impl", args.model_impl])
    if args.eval_source == "raw_folder":
        command.extend([
            "--data_path", dataset["rgb"],
            "--interval", "1",
            "--subset_start", str(dataset["subset_start"]),
            "--subset_end", str(dataset["subset_end"]),
            "--subset_step", str(dataset["subset_step"]),
        ])
    else:
        command.extend([
            "--dataset_root", args.dataset_root,
            "--dataset_scene", dataset.get("scene", dataset_name),
            "--dataset_split", args.dataset_split,
            "--dataset_image_dir_name", args.dataset_image_dir_name,
            "--dataset_hold_every", str(args.dataset_hold_every),
            "--dataset_frame_num", str(args.dataset_frame_num),
            "--dataset_resolution", args.dataset_resolution,
            "--dataset_seed", str(args.dataset_seed),
            "--dataset_index", str(args.dataset_index),
        ])
        if not args.dataset_shuffle_views:
            command.append("--no-dataset_shuffle_views")
    if args.eval_source == "raw_folder" and dataset.get("depth"):
        command.extend(["--depth_path", dataset["depth"]])
    if args.per_chunk_scene:
        command.append("--per_chunk_scene")
    if args.save_aligned_depth:
        command.append("--save_aligned_depth")
    if args.save_gaussian_diagnostics:
        command.append("--save_gaussian_diagnostics")
        command.extend(["--gaussian_diagnostic_max_rows", str(args.gaussian_diagnostic_max_rows)])
    command.extend(VARIANTS[variant_name])
    if variant_name == "random_equal_sample":
        command.extend(["--random_sampling_seed", str(args.random_sampling_seed)])
    return command, out_dir


def main():
    parser = argparse.ArgumentParser(description="Run Pi3_3DGS_10 ablations on the configured datasets.")
    parser.add_argument("--ckpt", default="outputs/pi3_3dgs10_hunyuan_decoder/ckpts/best_model/model.safetensors")
    parser.add_argument("--output_root", default="outputs/pi3_3dgs10_ablation_0530_rendered")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_impl", default="_10", help="Full model import path or shorthand _8/_9/_10.")
    parser.add_argument("--eval_source", choices=["raw_folder", "three_sixty_v2_dataset"], default="raw_folder")
    parser.add_argument("--dataset_root", default="/data/liuwei/dataset/360_v2")
    parser.add_argument("--dataset_split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--dataset_image_dir_name", default="images_4")
    parser.add_argument("--dataset_hold_every", type=int, default=8)
    parser.add_argument("--dataset_frame_num", type=int, default=8)
    parser.add_argument("--dataset_resolution", default="518x336")
    parser.add_argument("--dataset_seed", type=int, default=2024)
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--dataset_shuffle_views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--random_sampling_seed", type=int, default=2024)
    parser.add_argument("--per_chunk_scene", action="store_true")
    parser.add_argument("--save_aligned_depth", action="store_true")
    parser.add_argument("--save_gaussian_diagnostics", action="store_true")
    parser.add_argument("--gaussian_diagnostic_max_rows", type=int, default=250000)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=list(DATASETS.keys()))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS.keys(), default=list(VARIANTS.keys()))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        for variant_name in args.variants:
            command, out_dir = build_command(args, dataset_name, variant_name)
            command_log = out_dir / "command.txt"

            if args.skip_existing and (out_dir / "metrics_report.txt").is_file():
                print(f"Skipping existing result: {dataset_name}/{variant_name}")
                continue

            command_log.write_text(" ".join(command) + "\n", encoding="utf-8")

            print("\n" + "=" * 80)
            print(f"Running {dataset_name}/{variant_name}")
            print(" ".join(command))
            if args.dry_run:
                continue

            try:
                env = os.environ.copy()
                env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
                Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                print(f"FAILED {dataset_name}/{variant_name}: returncode={exc.returncode}")
                if not args.continue_on_error:
                    raise

    if args.dry_run:
        print("Dry run complete; no metrics were generated.")
    else:
        summarize_results(output_root, args.datasets, args.variants)


if __name__ == "__main__":
    main()
