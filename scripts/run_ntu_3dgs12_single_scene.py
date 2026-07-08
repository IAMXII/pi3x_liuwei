#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DATASETS = {
    "campus": {
        "rgb": "/data/liuwei/dataset/ntu_seq/campus/rgb",
        "depth": "/data/liuwei/dataset/ntu_seq/campus/depth",
    },
    "cp": {
        "rgb": "/data/liuwei/dataset/ntu_seq/cp/rgb",
        "depth": "/data/liuwei/dataset/ntu_seq/cp/depth",
    },
}


PROFILES = {
    # Best first: one shared scene, all selected views feed the Gaussian branch.
    "quality": {
        "target_frames": None,
        "pixel_limit": None,
        "gs_view_stride": 1,
        "geometry_head_view_chunk_size": 12,
        "gs_decoder_view_chunk_size": 12,
        "gs_head_view_chunk_size": 1,
    },
    # Still one shared scene; reduces only GS attribute generation pressure.
    "balanced": {
        "target_frames": None,
        "pixel_limit": None,
        "gs_view_stride": 2,
        "geometry_head_view_chunk_size": 8,
        "gs_decoder_view_chunk_size": 8,
        "gs_head_view_chunk_size": 1,
    },
    # Same frame coverage, lower resolution if the long sequence OOMs.
    "safe": {
        "target_frames": None,
        "pixel_limit": 196000,
        "gs_view_stride": 2,
        "geometry_head_view_chunk_size": 6,
        "gs_decoder_view_chunk_size": 6,
        "gs_head_view_chunk_size": 1,
    },
    # Last resort before using per-chunk scenes.
    "compact": {
        "target_frames": 160,
        "pixel_limit": 160000,
        "gs_view_stride": 4,
        "geometry_head_view_chunk_size": 4,
        "gs_decoder_view_chunk_size": 4,
        "gs_head_view_chunk_size": 1,
    },
}


def default_python():
    env_python = Path("/home/liuwei/anaconda3/envs/pi3-liuwei/bin/python3.10")
    return str(env_python) if env_python.is_file() else sys.executable


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_profile(profile_name, args):
    profile = dict(PROFILES[profile_name])
    target_frames = profile["target_frames"]
    if target_frames is None:
        target_frames = args.target_frames
    else:
        target_frames = min(int(target_frames), int(args.target_frames))

    pixel_limit = profile["pixel_limit"] if profile["pixel_limit"] is not None else args.pixel_limit
    profile["target_frames"] = int(target_frames)
    profile["pixel_limit"] = int(pixel_limit)
    return profile


def build_command(args, dataset_name, profile_name, out_dir):
    dataset = DATASETS[dataset_name]
    profile = resolve_profile(profile_name, args)

    command = [
        args.python,
        "example_3dgs_6.py",
        "--ckpt", args.ckpt,
        "--model_impl", "_12",
        "--output_dir", str(out_dir),
        "--device", "cuda",
        "--eval_source", "raw_folder",
        "--data_path", dataset["rgb"],
        "--depth_path", dataset["depth"],
        "--target_frame_count", str(profile["target_frames"]),
        "--pixel_limit", str(profile["pixel_limit"]),
        "--gs_view_stride", str(profile["gs_view_stride"]),
        "--geometry_head_view_chunk_size", str(profile["geometry_head_view_chunk_size"]),
        "--gs_decoder_view_chunk_size", str(profile["gs_decoder_view_chunk_size"]),
        "--gs_head_view_chunk_size", str(profile["gs_head_view_chunk_size"]),
        "--chunk_size", "0",
        "--ablation_name", profile_name,
        "--metric_lpips_net", args.metric_lpips_net,
        "--render_opacity_threshold", "0.0",
        "--ply_opacity_threshold", str(args.ply_opacity_threshold),
    ]

    if args.color_source != "predicted":
        command.extend(["--color_source", args.color_source])
    if args.rgb_only:
        command.append("--rgb_only")
    if args.skip_save_ply:
        command.append("--skip_save_ply")
    if args.save_gaussian_diagnostics:
        command.append("--save_gaussian_diagnostics")
        command.extend(["--gaussian_diagnostic_max_rows", str(args.gaussian_diagnostic_max_rows)])
    if args.save_redundancy_overlay:
        command.append("--save_redundancy_overlay")
    else:
        command.append("--no-save_redundancy_overlay")
    if not args.save_gaussian_scene:
        command.append("--skip_save_gaussian_scene")

    return command


def run_command(command, out_dir, gpu, dry_run=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(
        f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(command) + "\n",
        encoding="utf-8",
    )

    if dry_run:
        print(f"[dry-run][gpu {gpu}] " + " ".join(command))
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return proc.returncode


def run_dataset(args, dataset_name, gpu):
    profiles = args.profiles
    dataset_root = Path(args.output_root) / dataset_name

    for profile_name in profiles:
        out_dir = dataset_root / profile_name
        if args.skip_existing and (out_dir / "metrics_report.txt").is_file():
            print(f"[{dataset_name}][gpu {gpu}] skipping existing {profile_name}")
            return 0, profile_name

        command = build_command(args, dataset_name, profile_name, out_dir)
        print(f"[{dataset_name}][gpu {gpu}] running profile={profile_name} -> {out_dir}")
        rc = run_command(command, out_dir, gpu, dry_run=args.dry_run)
        if rc == 0:
            (dataset_root / "best_success.txt").write_text(
                f"profile: {profile_name}\noutput_dir: {out_dir}\ngpu: {gpu}\n",
                encoding="utf-8",
            )
            print(f"[{dataset_name}][gpu {gpu}] success profile={profile_name}")
            return 0, profile_name

        print(f"[{dataset_name}][gpu {gpu}] failed profile={profile_name} returncode={rc}")
        if not args.fallback:
            return rc, profile_name

    return 1, profiles[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Run Pi3_3DGS_12 on NTU campus/cp as one shared Gaussian scene per dataset."
    )
    parser.add_argument("--ckpt", default="outputs/pi3_3dgs12_hunyuan_decoder_0620_fix/ckpts/best_model/model.safetensors")
    parser.add_argument("--output_root", default="outputs/pi3_3dgs12_ntu_single_scene_200")
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=["campus", "cp"])
    parser.add_argument("--gpus", default="8,9", help="Comma-separated physical GPU ids. Default: 8,9.")
    parser.add_argument("--target_frames", type=int, default=200)
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES.keys(), default=["quality", "balanced", "safe", "compact"])
    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric_lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--color_source", choices=["predicted", "input"], default="predicted")
    parser.add_argument("--ply_opacity_threshold", type=float, default=0.03)
    parser.add_argument("--rgb_only", action="store_true", help="Skip depth loading/eval for faster visual-only rendering.")
    parser.add_argument("--skip_save_ply", action="store_true")
    parser.add_argument("--save_gaussian_scene", action="store_true",
                        help="Also save slow MonoGS-style ellipsoid visualizations.")
    parser.add_argument("--save_redundancy_overlay", action="store_true",
                        help="Also save slow redundancy suppression overlays.")
    parser.add_argument("--save_gaussian_diagnostics", action="store_true")
    parser.add_argument("--gaussian_diagnostic_max_rows", type=int, default=250000)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.target_frames <= 0:
        raise ValueError("--target_frames must be positive.")
    if args.pixel_limit <= 0:
        raise ValueError("--pixel_limit must be positive.")

    gpus = parse_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    assignments = {
        dataset_name: gpus[i % len(gpus)]
        for i, dataset_name in enumerate(args.datasets)
    }

    max_workers = min(len(args.datasets), len(gpus))
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_dataset, args, dataset_name, gpu): dataset_name
            for dataset_name, gpu in assignments.items()
        }
        for future in as_completed(future_map):
            dataset_name = future_map[future]
            try:
                results[dataset_name] = future.result()
            except Exception as exc:
                results[dataset_name] = (1, "exception")
                print(f"[{dataset_name}] exception: {exc}")

    failed = [name for name, (rc, _) in results.items() if rc != 0]
    if failed:
        print("Failed datasets: " + ", ".join(failed))
        raise SystemExit(1)

    print(f"All requested datasets completed. Outputs under: {output_root}")


if __name__ == "__main__":
    main()
