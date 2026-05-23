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
    "campus": {
        "rgb": "/data/liuwei/dataset/ntu_seq/campus/rgb",
        "depth": "/data/liuwei/dataset/ntu_seq/campus/depth",
        "subset_start": 148,
        "subset_end": 248,
        "subset_step": 5,
    },
    "hku": {
        "rgb": "/data/liuwei/dataset/ntu_seq/hku1/rgb",
        "depth": "/data/liuwei/dataset/ntu_seq/hku1/depth",
        "subset_start": 412,
        "subset_end": 512,
        "subset_step": 5,
    },
    "bicycle": {
        "rgb": "/data/liuwei/dataset/360_v2/bicycle/images",
        "depth": None,
        "subset_start": 0,
        "subset_end": 194,
        "subset_step": 5,
    },
    "stump": {
        "rgb": "/data/liuwei/dataset/360_v2/stump/images",
        "depth": None,
        "subset_start": 0,
        "subset_end": 125,
        "subset_step": 3,
    },
}


VARIANTS = {
    "full": [],
    "no_quadtree": ["--disable_quadtree"],
    "no_local_competition": ["--disable_local_competition"],
    "pure_gaussian": ["--disable_quadtree", "--disable_local_competition"],
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


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_metrics(report_path):
    metrics = {key: math.nan for key in METRIC_PATTERNS}
    metrics["depth_frames_used"] = ""
    metrics["depth_frames_total"] = ""
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
    return metrics


def parse_gaussian_counts(csv_path):
    total = 0
    active = 0
    ply_paths = []
    if not csv_path.is_file():
        return total, active, ""

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += int(float(row.get("total_count") or 0))
            active += int(float(row.get("active_count_opacity_gt_005") or 0))
            ply_path = row.get("ply_path") or ""
            if ply_path:
                ply_paths.append(ply_path)
    return total, active, ";".join(ply_paths)


def summarize_results(output_root, datasets, variants):
    rows = []
    for dataset in datasets:
        for variant in variants:
            out_dir = output_root / dataset / variant
            metrics = parse_metrics(out_dir / "metrics_report.txt")
            total, active, ply_paths = parse_gaussian_counts(out_dir / "gaussian_counts.csv")
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "output_dir": str(out_dir),
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
                "ply_paths": ply_paths,
            })

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
        "example_3dgs_5.py",
        "--ckpt", args.ckpt,
        "--data_path", dataset["rgb"],
        "--output_dir", str(out_dir),
        "--device", args.device,
        "--interval", "1",
        "--subset_start", str(dataset["subset_start"]),
        "--subset_end", str(dataset["subset_end"]),
        "--subset_step", str(dataset["subset_step"]),
        "--pixel_limit", str(args.pixel_limit),
        "--gs_view_stride", str(args.gs_view_stride),
        "--chunk_size", str(args.chunk_size),
        "--ablation_name", variant_name,
    ]
    if dataset.get("depth"):
        command.extend(["--depth_path", dataset["depth"]])
    if args.per_chunk_scene:
        command.append("--per_chunk_scene")
    if args.save_aligned_depth:
        command.append("--save_aligned_depth")
    command.extend(VARIANTS[variant_name])
    return command, out_dir


def main():
    parser = argparse.ArgumentParser(description="Run Pi3_3DGS_8 ablations on the readme_data.md datasets.")
    parser.add_argument("--ckpt", default="outputs/pi3_highres_0517_v8_local_comp/ckpts/best_model/model.safetensors")
    parser.add_argument("--output_root", default="outputs/pi3_3dgs8_ablation_0517_rendered")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--per_chunk_scene", action="store_true")
    parser.add_argument("--save_aligned_depth", action="store_true")
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
            command_log.write_text(" ".join(command) + "\n", encoding="utf-8")

            if args.skip_existing and (out_dir / "metrics_report.txt").is_file():
                print(f"Skipping existing result: {dataset_name}/{variant_name}")
                continue

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
