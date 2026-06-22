import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/liuwei/anaconda3/envs/pi3-liuwei/bin/python3.10"
DEFAULT_CKPT = "outputs/pi3_3dgs10_hunyuan_decoder/ckpts/best_model/model.safetensors"

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


def parse_number(value, default=math.nan):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value, default=0):
    number = parse_number(value, math.nan)
    if not math.isfinite(number):
        return default
    return int(round(number))


def shell_quote(arg):
    arg = str(arg)
    if not arg or any(ch in arg for ch in " \t\n'\"[]~+"):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return arg


def write_run_sh(path, command, gpus=None):
    lines = [
        "#!/usr/bin/env zsh",
        "set -euo pipefail",
        f"cd {shell_quote(REPO_ROOT)}",
    ]
    if gpus:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shell_quote(gpus)}")
    lines.extend([
        "exec " + " ".join(shell_quote(arg) for arg in command),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def parse_metrics(path):
    values = {key: math.nan for key in METRIC_PATTERNS}
    values.update({
        "frames": 0,
        "input_views": 0,
        "total_frames_for_views": 0,
        "reported_stride": 0,
        "scene_mode": "",
        "model_impl": "",
    })
    if not path.is_file():
        return values

    text = path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            values[key] = parse_number(match.group(1))

    match = re.search(r"frames:\s+(\d+)", text)
    if match:
        values["frames"] = int(match.group(1))
    match = re.search(r"gaussian_input_frames:\s+(\d+)\s*/\s*(\d+)", text)
    if match:
        values["input_views"] = int(match.group(1))
        values["total_frames_for_views"] = int(match.group(2))
    match = re.search(r"gs_view_stride:\s+(\d+)", text)
    if match:
        values["reported_stride"] = int(match.group(1))
    for key in ("scene_mode", "model_impl"):
        match = re.search(rf"{key}:\s+(.+)", text)
        if match:
            values[key] = match.group(1).strip()
    return values


def aggregate_stat(field, values):
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    count_like = (
        "count" in field
        or "pruned" in field
        or "candidates" in field
        or "suppressed" in field
        or "selected" in field
        or "proposal" in field
        or "extra" in field
        or field.endswith("_mass")
    )
    if field.endswith("_max"):
        return max(clean)
    if count_like:
        return sum(clean)
    return sum(clean) / len(clean)


def parse_counts(path):
    out = {
        "gaussian_total": 0,
        "gaussian_active_005": 0,
        "rendered_count": 0,
        "rendered_active_005": 0,
        "ply_kept_count": 0,
        "redundancy_suppression_count": 0,
    }
    stat_values = {}
    if not path.is_file():
        return out

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["gaussian_total"] += parse_int(row.get("raw_total_count") or row.get("total_count"))
            out["gaussian_active_005"] += parse_int(
                row.get("raw_active_count_opacity_gt_005") or row.get("active_count_opacity_gt_005")
            )
            out["rendered_count"] += parse_int(row.get("rendered_count"))
            out["rendered_active_005"] += parse_int(row.get("rendered_active_count_opacity_gt_005"))
            out["ply_kept_count"] += parse_int(row.get("ply_kept_count"))
            out["redundancy_suppression_count"] += parse_int(row.get("redundancy_suppression_count"))
            for key, value in row.items():
                if not key.startswith("stat_"):
                    continue
                stat_values.setdefault(key, []).append(parse_number(value))

    for key, values in stat_values.items():
        out[key] = aggregate_stat(key, values)

    hard_pruned = out.get("stat_hard_redundancy_pruned", math.nan)
    if not math.isfinite(hard_pruned):
        hard_pruned = out.get("stat_physical_pruned", math.nan)
    out["hard_redundancy_pruned"] = hard_pruned
    out["redundancy_candidates"] = out.get("stat_redundancy_candidates", math.nan)
    out["soft_expected_suppressed"] = out.get("stat_redundancy_expected_suppressed", math.nan)

    before_count = out.get("stat_count_before", math.nan)
    if not math.isfinite(before_count):
        before_count = out["gaussian_total"] + (hard_pruned if math.isfinite(hard_pruned) else 0)
    out["gaussian_before_prune"] = before_count
    out["hard_prune_rate"] = (
        hard_pruned / before_count
        if math.isfinite(hard_pruned) and math.isfinite(before_count) and before_count > 0
        else math.nan
    )
    out["soft_suppression_rate"] = (
        out["soft_expected_suppressed"] / before_count
        if math.isfinite(out["soft_expected_suppressed"]) and math.isfinite(before_count) and before_count > 0
        else math.nan
    )
    return out


def parse_csv_numbers(value):
    return [int(item) for item in str(value).replace(";", ",").split(",") if item.strip()]


def build_eval_command(args, stride, out_dir):
    command = [
        args.python,
        "example_3dgs_6.py",
        "--ckpt",
        args.ckpt,
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
        str(stride),
        "--chunk_size",
        str(args.chunk_size),
        "--metric_lpips_net",
        args.metric_lpips_net,
        "--ablation_name",
        f"views_stride_{stride}",
        "--skip_save_frames",
        "--skip_save_ply",
    ]
    if args.rgb_only:
        command.append("--rgb_only")
    if args.competition_mode == "hard":
        command.append("--hard_prune_redundant_gaussians_eval")

    if args.eval_source == "raw_folder":
        command.extend([
            "--data_path",
            args.data_path,
            "--interval",
            str(args.interval),
            "--subset_start",
            str(args.subset_start),
            "--subset_end",
            str(args.subset_end),
            "--subset_step",
            str(args.subset_step),
        ])
        if args.depth_path:
            command.extend(["--depth_path", args.depth_path])
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

    command.extend(args.extra_eval_args)
    return command


def run_command(command, env, log_path, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(" ".join(shell_quote(arg) for arg in command) + "\n", encoding="utf-8")
    if dry_run:
        print("[dry-run] " + " ".join(shell_quote(arg) for arg in command))
        return 0
    with log_path.with_suffix(".log").open("w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def summarize(args):
    rows = []
    for stride in parse_csv_numbers(args.strides):
        out_dir = Path(args.output_root) / f"stride_{stride:02d}"
        metrics = parse_metrics(out_dir / "metrics_report.txt")
        counts = parse_counts(out_dir / "gaussian_counts.csv")
        row = {
            "stride": stride,
            "input_views": metrics["input_views"],
            "frames": metrics["frames"],
            "output_dir": str(out_dir),
            **{key: metrics[key] for key in METRIC_PATTERNS},
            **counts,
        }
        rows.append(row)

    rows.sort(key=lambda row: (row.get("input_views", 0), -row.get("stride", 0)))
    summary_path = Path(args.output_root) / "view_sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to: {summary_path}")
    return summary_path


def load_summary(path):
    rows = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in ("output_dir",):
                    parsed[key] = value
                else:
                    parsed[key] = parse_number(value)
            if parsed.get("input_views", 0) > 0:
                rows.append(parsed)
    rows.sort(key=lambda row: row["input_views"])
    return rows


def plot_summary(args, summary_path=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    summary_path = Path(summary_path or Path(args.output_root) / "view_sweep_summary.csv")
    rows = load_summary(summary_path)
    if not rows:
        raise RuntimeError(f"No valid rows found in {summary_path}")

    x = [row["input_views"] for row in rows]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
    colors = {
        "psnr": "#176D75",
        "ssim": "#B65C38",
        "gaussian": "#4B4E9A",
        "active": "#7D8C2E",
        "pruned": "#C88719",
        "rate": "#8B3A62",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 320,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf7")
    for ax in axes.ravel():
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#d9dedc", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Gaussian input views")

    def line_with_point_labels(ax, y, color, label, ylabel, value_fmt=None):
        ax.plot(x, y, color=color, linewidth=2.4, alpha=0.9, label=label, zorder=2)
        for idx, (x_val, y_val) in enumerate(zip(x, y)):
            if not math.isfinite(y_val):
                continue
            ax.scatter(
                [x_val],
                [y_val],
                s=76,
                marker=markers[idx % len(markers)],
                color=color,
                edgecolor="#ffffff",
                linewidth=1.3,
                zorder=3,
            )
            if value_fmt is None:
                note = f"{int(x_val)}v"
            else:
                note = f"{int(x_val)}v\n{value_fmt(y_val)}"
            ax.annotate(
                note,
                (x_val, y_val),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#263238",
            )
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="best")

    psnr = [row.get("psnr", math.nan) for row in rows]
    ssim = [row.get("ssim", math.nan) for row in rows]
    lpips = [row.get("lpips", math.nan) for row in rows]
    total_m = [row.get("gaussian_total", math.nan) / 1e6 for row in rows]
    active_m = [row.get("gaussian_active_005", math.nan) / 1e6 for row in rows]
    if args.competition_mode == "hard":
        competition_m = [row.get("hard_redundancy_pruned", math.nan) / 1e6 for row in rows]
        competition_rate = [row.get("hard_prune_rate", math.nan) for row in rows]
        competition_label = "Hard-pruned redundant"
        competition_ylabel = "Pruned Gaussians (M)"
        competition_rate_label = "Prune rate"
        competition_title = "Redundancy Removed Locally"
    else:
        competition_m = [row.get("redundancy_candidates", math.nan) / 1e6 for row in rows]
        competition_rate = [row.get("soft_suppression_rate", math.nan) for row in rows]
        competition_label = "Redundancy candidates"
        competition_ylabel = "Candidate Gaussians (M)"
        competition_rate_label = "Soft suppression rate"
        competition_title = "Local Redundancy Soft Gate"

    ax = axes[0, 0]
    line_with_point_labels(ax, psnr, colors["psnr"], "PSNR", "PSNR (dB) ↑", lambda v: f"{v:.2f}")
    ax.set_title("Rendering Quality")

    ax = axes[0, 1]
    line_with_point_labels(ax, ssim, colors["ssim"], "SSIM", "SSIM ↑", lambda v: f"{v:.3f}")
    if any(math.isfinite(v) for v in lpips):
        ax2 = ax.twinx()
        ax2.plot(x, lpips, color="#68717A", linewidth=2.0, linestyle="--", label="LPIPS")
        ax2.set_ylabel("LPIPS ↓")
        ax2.spines["top"].set_visible(False)
        ax2.legend(frameon=False, loc="lower right")
    ax.set_title("Perceptual Fidelity")

    ax = axes[1, 0]
    ax.plot(x, total_m, color=colors["gaussian"], linewidth=2.4, label="Physical Gaussians")
    ax.plot(x, active_m, color=colors["active"], linewidth=2.0, linestyle="--", label="Opacity > 0.05")
    for idx, (x_val, y_val) in enumerate(zip(x, total_m)):
        if not math.isfinite(y_val):
            continue
        ax.scatter(
            [x_val],
            [y_val],
            s=76,
            marker=markers[idx % len(markers)],
            color=colors["gaussian"],
            edgecolor="#ffffff",
            linewidth=1.3,
            zorder=3,
        )
        ax.annotate(
            f"{int(x_val)}v\n{y_val:.2f}M",
            (x_val, y_val),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#263238",
        )
    if args.competition_mode == "hard":
        ax.set_title("Gaussian Budget After Hard Prune")
    else:
        ax.set_title("Gaussian Budget After Soft Gate")
    ax.set_ylabel("# Gaussians (M)")
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 1]
    line_with_point_labels(
        ax,
        competition_m,
        colors["pruned"],
        competition_label,
        competition_ylabel,
        lambda v: f"{v:.2f}M",
    )
    ax2 = ax.twinx()
    ax2.plot(x, competition_rate, color=colors["rate"], linewidth=2.0, linestyle="--", label=competition_rate_label)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax2.set_ylabel(competition_rate_label)
    ax2.spines["top"].set_visible(False)
    ax2.legend(frameon=False, loc="lower right")
    ax.set_title(competition_title)

    scene = args.figure_title
    if not scene:
        if args.competition_mode == "hard":
            scene = "Pi3-10 View-Count Scaling with Inference Hard Redundancy Pruning"
        else:
            scene = "Pi3-10 View-Count Scaling with Local Redundancy Soft Gate"
    fig.suptitle(scene, fontsize=15, fontweight="bold", color="#1f2a2e")
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "pi3_3dgs10_view_sweep_scaling"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {stem.with_suffix('.png')}")
    print(f"Vector outputs: {stem.with_suffix('.pdf')}, {stem.with_suffix('.svg')}")
    return stem.with_suffix(".png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and plot a Pi3_3DGS_10 view-count sweep with inference-time hard redundancy pruning."
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--output_root", default="outputs/pi3_3dgs10_view_sweep_hard_prune_cp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpus",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value. Leave empty when the local CUDA runtime needs all devices visible.",
    )
    parser.add_argument("--stage", choices=["run", "summarize", "plot", "all"], default="all")
    parser.add_argument("--strides", default="19,10,7,5,4,3,2,1")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--figure_title", default="")
    parser.add_argument("--competition_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--extra_eval_args", nargs="*", default=[])

    parser.add_argument("--eval_source", choices=["raw_folder", "three_sixty_v2_dataset"], default="raw_folder")
    parser.add_argument("--data_path", default="/data/liuwei/dataset/ntu_seq/cp/rgb")
    parser.add_argument("--depth_path", default="/data/liuwei/dataset/ntu_seq/cp/depth")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--subset_start", type=int, default=1146)
    parser.add_argument("--subset_end", type=int, default=1346)
    parser.add_argument("--subset_step", type=int, default=10)
    parser.add_argument("--rgb_only", action="store_true")

    parser.add_argument("--dataset_root", default="/data/liuwei/dataset/360_v2")
    parser.add_argument("--dataset_scene", default="bicycle")
    parser.add_argument("--dataset_split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--dataset_image_dir_name", default="images_4")
    parser.add_argument("--dataset_hold_every", type=int, default=8)
    parser.add_argument("--dataset_frame_num", type=int, default=20)
    parser.add_argument("--dataset_resolution", default="518x336")
    parser.add_argument("--dataset_seed", type=int, default=2024)
    parser.add_argument("--dataset_index", type=int, default=0)

    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--metric_lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    env = None
    if args.gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    if args.stage in ("run", "all"):
        for stride in parse_csv_numbers(args.strides):
            out_dir = Path(args.output_root) / f"stride_{stride:02d}"
            command = build_eval_command(args, stride, out_dir)
            write_run_sh(out_dir / "run.sh", command, gpus=args.gpus)
            if args.skip_existing and (out_dir / "metrics_report.txt").is_file():
                print(f"[skip] stride={stride}: {out_dir / 'metrics_report.txt'}")
                continue
            code = run_command(command, env, out_dir / "command.txt", dry_run=args.dry_run)
            if code != 0:
                print(f"[failed] stride={stride}: returncode={code}")
                if not args.continue_on_error:
                    raise SystemExit(code)

    summary_path = Path(args.output_root) / "view_sweep_summary.csv"
    if not args.dry_run and args.stage in ("summarize", "all"):
        summary_path = summarize(args)
    elif args.dry_run:
        print(f"Dry run complete. Commands are under {Path(args.output_root)}")

    if not args.dry_run and args.stage in ("plot", "all"):
        plot_summary(args, summary_path)


if __name__ == "__main__":
    main()
