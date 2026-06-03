import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/liuwei/anaconda3/envs/pi3-liuwei/bin/python3.10"
DEFAULT_CKPT = "outputs/pi3_highres_0523_v8_local_comp/ckpts/best_model/model.safetensors"
AVERAGED_METRIC_RE = re.compile(
    r"(?<!/)([A-Za-z][A-Za-z0-9_]*):\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?\s+\(([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
)


DATASET_OFF_OVERRIDES = [
    "~train_dataset.weights.co3dv2",
    "~train_dataset.weights.wildrgbd",
    "~train_dataset.weights.matrixcity",
    "~train_dataset.weights.midair",
    "~train_dataset.weights.ntu",
    "~train_dataset.weights.real",
    "~train_dataset.weights.kitti",
    "~train_dataset.weights.waymo",
    "~train_dataset.weights.hypersim",
    "~test_dataset.weights.co3dv2",
    "~test_dataset.weights.wildrgbd",
    "~test_dataset.weights.matrixcity",
    "~test_dataset.weights.midair",
    "~test_dataset.weights.hypersim",
    "~test_dataset.weights.ntu",
    "~test_dataset.weights.real",
    "~test_dataset.weights.kitti",
    "~test_dataset.weights.waymo",
]


VARIANTS = {
    "B0_fixed_baseline": {
        "description": "Fixed-resolution 360_v2 baseline with the current v9 360-only settings.",
        "overrides": [
            "+model.enable_local_competition=true",
        ],
    },
    "E1_norm_median": {
        "description": "Use median pseudo-mask scene normalization instead of mean normalization.",
        "overrides": [
            "+model.enable_local_competition=true",
            "+loss.train_loss.norm_strategy=median",
            "+loss.test_loss.norm_strategy=median",
        ],
    },
    "E2_stride2_easy": {
        "description": "Restrict 360_v2 frame sampling to max_stride=2.",
        "overrides": [
            "+model.enable_local_competition=true",
            "train_dataset.360_v2.max_stride=2",
            "test_dataset.360_v2.max_stride=2",
        ],
    },
    "E3_no_opacity_reg": {
        "description": "Disable sparsity and alpha regularization.",
        "overrides": [
            "+model.enable_local_competition=true",
            "loss.train_loss.enable_sparsity_loss=false",
            "loss.test_loss.enable_sparsity_loss=false",
            "loss.train_loss.lambda_sparsity=0",
            "loss.test_loss.lambda_sparsity=0",
            "loss.train_loss.enable_alpha_regularization=false",
            "loss.test_loss.enable_alpha_regularization=false",
            "loss.train_loss.lambda_alpha_regul=0",
            "loss.test_loss.lambda_alpha_regul=0",
        ],
    },
    "E4_no_local_competition": {
        "description": "Disable local competition opacity suppression.",
        "overrides": [
            "+model.enable_local_competition=false",
        ],
    },
    "E5_structural_loss": {
        "description": "Disable LPIPS and add explicit edge loss with SSIM weight 0.5.",
        "overrides": [
            "+model.enable_local_competition=true",
            "loss.train_loss.lambda_lpips=0",
            "loss.test_loss.lambda_lpips=0",
            "+loss.train_loss.lambda_edge=0.05",
            "+loss.test_loss.lambda_edge=0.05",
            "+loss.train_loss.lambda_ssim=0.5",
            "+loss.test_loss.lambda_ssim=0.5",
        ],
    },
}


def base_overrides(name, ckpt):
    return [
        "train=train_pi3_highres",
        f"name={name}",
        "model._target_=pi3.models.pi3_3dgs_9.Pi3_3DGS",
        f"model.ckpt={ckpt}",
        "model.enable_redundancy_pruning=false",
        "+model.use_input_intrinsics=false",
        "+model.competition_radius_scale=1.5",
        "+model.competition_color_bins=16",
        "+model.competition_temperature=0.8",
        "+model.competition_target_power=0.5",
        "+model.competition_strength=0.6",
        "+model.competition_min_gate=0.05",
        "model.redundancy_lambda_mercy=0.1",
        "+model.redundancy_use_color=true",
        "+model.redundancy_color_threshold=0.08",
        "+model.redundancy_color_bins=16",
        "+model.redundancy_soft_suppression=true",
        "+model.redundancy_suppression_gamma=1.5",
        "+model.redundancy_suppression_max=0.85",
        "+model.gs_decoder_view_chunk_size=12",
        "+model.quadtree_base_threshold=0.04",
        "+model.quadtree_relax_factor=0.1",
        "loss.train_loss._target_=pi3.models.loss_3dgs_1.Pi3LossGS",
        "loss.test_loss._target_=pi3.models.loss_3dgs_1.Pi3LossGS",
        "train.num_epoch=1",
        "train.iters_per_epoch=800",
        "test.iters_per_test=300",
        "train.random_reslution=false",
        "train.num_resolution=1",
        "+train.resolution=[[518,336]]",
        "train.image_num_range=[8,8]",
        "train.max_img_per_gpu=8",
        "train.auto_resume=false",
        "train.resume=null",
        "+log.save_checkpoints=false",
        "+log.save_best_model=false",
        "test.eval_only_dataset=360_v2",
        "test.eval_only_scene=null",
        *DATASET_OFF_OVERRIDES,
    ]


def build_command(python, gpus, port, name, ckpt, variant_name):
    variant = VARIANTS[variant_name]
    return [
        python,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        str(len(gpus.split(","))),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        str(port),
        "scripts/train_pi3.py",
        *base_overrides(name, ckpt),
        *variant["overrides"],
    ]


def shell_quote(arg):
    if not arg or any(ch in arg for ch in " \t\n'\"[]~+"):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return arg


def write_run_sh(path, command, gpus):
    lines = [
        "#!/usr/bin/env zsh",
        "set -euo pipefail",
        f"cd {shell_quote(str(REPO_ROOT))}",
        f"export CUDA_VISIBLE_DEVICES={shell_quote(gpus)}",
        "export WANDB_MODE=offline",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export MPLCONFIGDIR=/tmp/matplotlib-cache",
        "export HYDRA_FULL_ERROR=1",
        "",
        "exec " + " ".join(shell_quote(arg) for arg in command),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _parse_averaged_metrics(line, prefix):
    metrics = {}
    for key, avg_value in AVERAGED_METRIC_RE.findall(line):
        try:
            metrics[f"{prefix}_{key}"] = float(avg_value)
        except ValueError:
            continue
    return metrics


def read_log_metrics(log_path):
    if not log_path.exists():
        return None

    metrics = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Validation results:" in line:
                metrics.update(_parse_averaged_metrics(line, "val"))
            elif "Epoch: [0]" in line and "Validation Epoch" not in line and "loss_ssim" in line:
                metrics.update(_parse_averaged_metrics(line, "train"))

    return metrics or None


def read_last_metrics(output_dir, run_log=None):
    log_path = output_dir / "ckpts" / "log.txt"
    last = None
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    if last is None and run_log is not None:
        last = read_log_metrics(run_log)
    return last


def summarize(result_path, rows):
    baseline = next((row for row in rows if row["variant"] == "B0_fixed_baseline"), None)
    baseline_ssim = None
    if baseline and baseline.get("metrics"):
        baseline_ssim = baseline["metrics"].get("val_loss_ssim")

    for row in rows:
        metrics = row.get("metrics") or {}
        val_ssim = metrics.get("val_loss_ssim")
        row["delta_vs_B0_val_loss_ssim"] = (
            val_ssim - baseline_ssim
            if baseline_ssim is not None and val_ssim is not None
            else None
        )

    result_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run controlled 360_v2 SSIM-loss ablations.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="6,7,8,9")
    parser.add_argument("--port", type=int, default=29691)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--output-root", default="outputs/pi3_highres_0529_v9_360_ssim_ablation")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    os.chdir(REPO_ROOT)
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, variant_name in enumerate(args.variants):
        run_name = f"{Path(args.output_root).name}_{variant_name}"
        output_dir = REPO_ROOT / "outputs" / run_name
        variant_dir = output_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        command = build_command(
            args.python,
            args.gpus,
            args.port + index,
            run_name,
            args.ckpt,
            variant_name,
        )
        run_sh = variant_dir / "run.sh"
        write_run_sh(run_sh, command, args.gpus)

        row = {
            "variant": variant_name,
            "description": VARIANTS[variant_name]["description"],
            "output_dir": str(output_dir),
            "run_sh": str(run_sh),
            "status": "dry_run" if args.dry_run else "pending",
        }

        if args.skip_existing:
            existing_metrics = read_last_metrics(output_dir, variant_dir / "run.log")
            if existing_metrics is not None:
                row["status"] = "skipped_existing"
                row["metrics"] = existing_metrics
                rows.append(row)
                summarize(output_root / "summary.json", rows)
                continue

        if args.dry_run:
            print(f"[dry-run] {variant_name}: {run_sh}")
            rows.append(row)
            summarize(output_root / "summary.json", rows)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        env["WANDB_MODE"] = "offline"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
        env["HYDRA_FULL_ERROR"] = "1"
        log_path = variant_dir / "run.log"
        print(f"[run] {variant_name}: log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        metrics = read_last_metrics(output_dir, log_path)
        row["metrics"] = metrics
        if proc.returncode == 0:
            row["status"] = "completed"
        elif metrics is not None:
            row["status"] = f"metrics_available_after_failure_{proc.returncode}"
            print(
                f"[warn] {variant_name} exited with {proc.returncode}, "
                "but metrics were written; continuing.",
                flush=True,
            )
        else:
            row["status"] = f"failed_{proc.returncode}"
        rows.append(row)
        summarize(output_root / "summary.json", rows)

        if proc.returncode != 0 and metrics is None:
            raise SystemExit(f"{variant_name} failed with return code {proc.returncode}. See {log_path}")

    summarize(output_root / "summary.json", rows)
    print(f"Wrote {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
