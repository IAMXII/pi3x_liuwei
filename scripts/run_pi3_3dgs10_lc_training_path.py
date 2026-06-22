import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/liuwei/anaconda3/envs/pi3-liuwei/bin/python3.10"
DEFAULT_INIT_CKPT = "outputs/pi3_3dgs10_hunyuan_decoder/ckpts/checkpoint_1/model.safetensors"

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

STAT_KEYS = [
    "stat_redundancy_candidates",
    "stat_redundancy_coef_mean",
    "stat_redundancy_coef_max",
    "stat_competition_gate_mean",
    "stat_redundancy_gate_mean",
    "stat_opacity_mass",
    "stat_active_count_opacity_005",
    "stat_competition_expected_suppressed",
    "stat_redundancy_expected_suppressed",
]

TRAIN_VARIANTS = {
    "lc_on": True,
    "lc_off": False,
}

EVAL_VARIANTS = {
    "train_lc_on_eval_lc_on": ("lc_on", True),
    "train_lc_on_eval_lc_off": ("lc_on", False),
    "train_lc_off_eval_lc_off": ("lc_off", False),
    "train_lc_off_eval_lc_on": ("lc_off", True),
}


def shell_quote(arg):
    if not arg or any(ch in arg for ch in " \t\n'\"[]~+"):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return arg


def write_run_sh(path, command, gpus=None):
    lines = [
        "#!/usr/bin/env zsh",
        "set -euo pipefail",
        f"cd {shell_quote(str(REPO_ROOT))}",
    ]
    if gpus:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shell_quote(gpus)}")
    lines.extend([
        "export WANDB_MODE=offline",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export MPLCONFIGDIR=/tmp/matplotlib-cache",
        "export HYDRA_FULL_ERROR=1",
        "",
        "exec " + " ".join(shell_quote(arg) for arg in command),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def train_output_name(args, variant):
    return f"{Path(args.output_root).name}_train_{variant}"


def train_output_dir(args, variant):
    return REPO_ROOT / "outputs" / train_output_name(args, variant)


def build_train_command(args, variant, port):
    enabled = TRAIN_VARIANTS[variant]
    gpus = [item for item in args.gpus.split(",") if item.strip()]
    command = [
        args.python,
        "-m",
        "accelerate.commands.launch",
    ]
    if len(gpus) > 1:
        command.append("--multi_gpu")
    command.extend([
        "--num_processes",
        str(max(1, len(gpus))),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        str(port),
        "scripts/train_pi3.py",
        "train=train_pi3_highres",
        f"name={train_output_name(args, variant)}",
        "model._target_=pi3.models.pi3_3dgs_10.Pi3_3DGS",
        f"model.ckpt={args.init_ckpt}",
        f"+model.enable_local_competition={str(enabled).lower()}",
        "train.num_epoch=2",
        "train.iters_per_epoch=5000",
        "test.iters_per_test=300",
        "train.auto_resume=false",
        "train.resume=null",
        "train.global_step=0",
        "train.start_epoch=0",
        "log.ckpt_interval=1",
        "log.max_checkpoints=3",
    ])
    command.extend(args.train_override)
    return command


def find_model_ckpt(output_dir):
    candidates = [
        output_dir / "ckpts" / "best_model" / "model.safetensors",
        output_dir / "ckpts" / "checkpoint_2" / "model.safetensors",
        output_dir / "ckpts" / "checkpoint_1" / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checkpoint_dirs = sorted((output_dir / "ckpts").glob("checkpoint_*"))
    for checkpoint_dir in reversed(checkpoint_dirs):
        candidate = checkpoint_dir / "model.safetensors"
        if candidate.is_file():
            return candidate
    return candidates[0]


def build_eval_command(args, eval_name, ckpt, eval_lc_enabled):
    out_dir = Path(args.output_root) / "eval" / eval_name
    command = [
        args.python,
        "example_3dgs_6.py",
        "--ckpt",
        str(ckpt),
        "--output_dir",
        str(out_dir),
        "--device",
        args.device,
        "--model_impl",
        "_10",
        "--eval_source",
        args.eval_source,
        "--pixel_limit",
        str(args.pixel_limit),
        "--gs_view_stride",
        str(args.gs_view_stride),
        "--chunk_size",
        str(args.chunk_size),
        "--metric_lpips_net",
        args.metric_lpips_net,
        "--ablation_name",
        eval_name,
    ]
    if args.eval_source == "raw_folder":
        command.extend([
            "--data_path",
            args.data_path,
            "--depth_path",
            args.depth_path,
            "--interval",
            "1",
            "--subset_start",
            str(args.subset_start),
            "--subset_end",
            str(args.subset_end),
            "--subset_step",
            str(args.subset_step),
        ])
    else:
        command.extend([
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
        ])
    if not eval_lc_enabled:
        command.append("--disable_local_competition")
    if args.save_gaussian_diagnostics:
        command.append("--save_gaussian_diagnostics")
        command.extend(["--gaussian_diagnostic_max_rows", str(args.gaussian_diagnostic_max_rows)])
    if args.skip_save_frames:
        command.append("--skip_save_frames")
    command.extend(args.eval_override)
    return command, out_dir


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_metrics(path):
    metrics = {key: math.nan for key in METRIC_PATTERNS}
    if not path.is_file():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = parse_float(match.group(1))
    return metrics


def parse_counts(path):
    total = active = 0
    stats = {key: math.nan for key in STAT_KEYS}
    stat_accum = {key: [] for key in STAT_KEYS}
    if not path.is_file():
        return total, active, stats
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += int(float(row.get("raw_total_count") or row.get("total_count") or 0))
            active += int(float(row.get("raw_active_count_opacity_gt_005") or row.get("active_count_opacity_gt_005") or 0))
            for key in STAT_KEYS:
                value = row.get(key)
                if value in (None, ""):
                    continue
                try:
                    stat_accum[key].append(float(value))
                except ValueError:
                    pass
    for key, values in stat_accum.items():
        if not values:
            continue
        if key.endswith("_max"):
            stats[key] = max(values)
        elif "candidates" in key or "suppressed" in key or key.endswith("_mass") or key.endswith("_005"):
            stats[key] = sum(values)
        else:
            stats[key] = sum(values) / len(values)
    return total, active, stats


def summarize(args):
    rows = []
    for eval_name, (train_variant, eval_lc_enabled) in EVAL_VARIANTS.items():
        out_dir = Path(args.output_root) / "eval" / eval_name
        metrics = parse_metrics(out_dir / "metrics_report.txt")
        total, active, stats = parse_counts(out_dir / "gaussian_counts.csv")
        row = {
            "eval_name": eval_name,
            "train_local_competition": TRAIN_VARIANTS[train_variant],
            "eval_local_competition": eval_lc_enabled,
            "output_dir": str(out_dir),
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "abs_rel": metrics["abs_rel"],
            "d_rmse": metrics["d_rmse"],
            "sro": metrics["sro"],
            "lrc": metrics["lrc"],
            "sldq": metrics["sldq"],
            "gaussian_total": total,
            "gaussian_active_opacity_gt_005": active,
        }
        row.update(stats)
        rows.append(row)

    summary_path = Path(args.output_root) / "training_path_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to: {summary_path}")


def run_command(command, cwd, env, log_path, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    if dry_run:
        print(f"[dry-run] {log_path}")
        return 0
    with log_path.with_suffix(".log").open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run LC-on/LC-off 10k training-path comparison for Pi3_3DGS_10.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--port", type=int, default=29741)
    parser.add_argument("--init_ckpt", default=DEFAULT_INIT_CKPT)
    parser.add_argument("--output_root", default="outputs/pi3_3dgs10_lc_training_path_10k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=["train", "eval", "all", "summarize"], default="all")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--train_override", nargs="*", default=[])
    parser.add_argument("--eval_override", nargs="*", default=[])

    parser.add_argument("--eval_source", choices=["raw_folder", "three_sixty_v2_dataset"], default="raw_folder")
    parser.add_argument("--data_path", default="/data/liuwei/dataset/ntu_seq/cp/rgb")
    parser.add_argument("--depth_path", default="/data/liuwei/dataset/ntu_seq/cp/depth")
    parser.add_argument("--subset_start", type=int, default=1146)
    parser.add_argument("--subset_end", type=int, default=1346)
    parser.add_argument("--subset_step", type=int, default=5)
    parser.add_argument("--dataset_root", default="/data/liuwei/dataset/360_v2")
    parser.add_argument("--dataset_scene", default="bicycle")
    parser.add_argument("--dataset_split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--dataset_image_dir_name", default="images_4")
    parser.add_argument("--dataset_hold_every", type=int, default=8)
    parser.add_argument("--dataset_frame_num", type=int, default=8)
    parser.add_argument("--dataset_resolution", default="518x336")
    parser.add_argument("--dataset_seed", type=int, default=2024)
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--metric_lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--save_gaussian_diagnostics", action="store_true")
    parser.add_argument("--gaussian_diagnostic_max_rows", type=int, default=250000)
    parser.add_argument("--skip_save_frames", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    if args.stage in ("train", "all"):
        for idx, variant in enumerate(TRAIN_VARIANTS):
            out_dir = train_output_dir(args, variant)
            run_dir = Path(args.output_root) / "train_commands" / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            command = build_train_command(args, variant, args.port + idx)
            write_run_sh(run_dir / "run.sh", command, gpus=args.gpus)
            if args.skip_existing and find_model_ckpt(out_dir).is_file():
                print(f"[skip] training {variant}: {find_model_ckpt(out_dir)}")
                continue
            code = run_command(command, REPO_ROOT, env, run_dir / "command.txt", dry_run=args.dry_run)
            if code != 0:
                print(f"[failed] training {variant}: returncode={code}")
                if not args.continue_on_error:
                    raise SystemExit(code)

    if args.stage in ("eval", "all"):
        for eval_name, (train_variant, eval_lc_enabled) in EVAL_VARIANTS.items():
            ckpt = find_model_ckpt(train_output_dir(args, train_variant))
            command, out_dir = build_eval_command(args, eval_name, ckpt, eval_lc_enabled)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_run_sh(out_dir / "run.sh", command, gpus=args.gpus)
            if args.skip_existing and (out_dir / "metrics_report.txt").is_file():
                print(f"[skip] eval {eval_name}: {out_dir / 'metrics_report.txt'}")
                continue
            code = run_command(command, REPO_ROOT, env, out_dir / "command.txt", dry_run=args.dry_run)
            if code != 0:
                print(f"[failed] eval {eval_name}: returncode={code}")
                if not args.continue_on_error:
                    raise SystemExit(code)

    if not args.dry_run and args.stage in ("eval", "all", "summarize"):
        summarize(args)
    elif args.dry_run:
        print(f"Dry run complete. Commands are under {Path(args.output_root)}")


if __name__ == "__main__":
    main()
