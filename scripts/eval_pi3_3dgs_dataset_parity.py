import argparse
import csv
import importlib
import inspect
import math
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.base.base_dataset import unified_collate_fn


MODEL_IMPL_ALIASES = {
    "_8": "pi3.models.pi3_3dgs_8.Pi3_3DGS",
    "8": "pi3.models.pi3_3dgs_8.Pi3_3DGS",
    "_9": "pi3.models.pi3_3dgs_9.Pi3_3DGS",
    "9": "pi3.models.pi3_3dgs_9.Pi3_3DGS",
    "_10": "pi3.models.pi3_3dgs_10.Pi3_3DGS",
    "10": "pi3.models.pi3_3dgs_10.Pi3_3DGS",
}


def resolve_model_impl(model_impl, cfg):
    if model_impl:
        return MODEL_IMPL_ALIASES.get(model_impl, model_impl)
    return str(cfg.model._target_)


def import_class(import_path):
    module_name, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def scalar_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def tensors_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.float() if torch.is_floating_point(value) else value
    if isinstance(value, dict):
        return {key: tensors_to_float(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensors_to_float(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensors_to_float(item) for item in value)
    return value


def parse_resolution(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        if "x" in value:
            width, height = value.lower().split("x", 1)
        elif "," in value:
            width, height = value.split(",", 1)
        else:
            raise ValueError(f"Resolution must look like 518x336 or 518,336, got {value!r}")
        return [[int(width), int(height)]]
    return value


def cfg_to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def instantiate_model(cfg, ckpt_path, model_impl, device):
    model_cls = import_class(model_impl)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("_target_", None)

    valid_keys = set(inspect.signature(model_cls.__init__).parameters)
    valid_keys.discard("self")
    model_kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key in valid_keys and key != "ckpt"
    }
    if "ckpt" in valid_keys:
        model_kwargs["ckpt"] = None

    if getattr(instantiate_model, "disable_learnable_sampling", False):
        model_kwargs["enable_learnable_sampling"] = False
    learned_ratio = getattr(instantiate_model, "learned_sampling_extra_ratio", None)
    if learned_ratio is not None:
        model_kwargs["learned_sampling_extra_ratio"] = float(learned_ratio)
    density_power = getattr(instantiate_model, "density_gate_opacity_power", None)
    if density_power is not None:
        model_kwargs["density_gate_opacity_power"] = float(density_power)

    model = model_cls(**model_kwargs).to(device).eval()
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path), device="cpu")
    else:
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if hasattr(model, "_load_state_dict_flexible"):
        model._load_state_dict_flexible(state_dict)
    else:
        model.load_state_dict(state_dict, strict=False)
    return model


def instantiate_loss(cfg, device):
    loss = hydra.utils.instantiate(cfg.loss.test_loss)
    loss = loss.to(device)
    loss.eval()
    return loss


def build_dataset(cfg, args, split):
    if args.dataset != "360_v2":
        raise ValueError("This diagnostic currently supports --dataset 360_v2.")

    cfg_group = cfg.train_dataset if split == "train" else cfg.test_dataset
    dataset_cfg = cfg_group[args.dataset]

    resolution = parse_resolution(args.resolution)
    if resolution is None:
        resolution = cfg_to_plain(cfg.test_dataset[args.dataset].resolution)

    overrides = {
        "mode": split,
        "resolution": resolution,
        "scene_names": args.scene,
    }
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.image_dir_name:
        overrides["image_dir_name"] = args.image_dir_name
    if args.hold_every is not None:
        overrides["hold_every"] = args.hold_every
    if args.frame_num is not None:
        overrides["frame_num"] = args.frame_num

    dataset = hydra.utils.instantiate(dataset_cfg, **overrides)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples for dataset={args.dataset}, scene={args.scene}, split={split}. "
            "Check data_root/image_dir_name/scene name."
        )
    return dataset


def get_frame_names(batch):
    names = []
    for view in batch:
        instance = view.get("instance")
        if isinstance(instance, list):
            names.append(str(instance[0]))
        elif isinstance(instance, tuple):
            names.append(str(instance[0]))
        else:
            names.append(str(instance))
    return names


def forward_batch(model, batch, global_step=0):
    imgs = torch.stack([view["img"] for view in batch], dim=1)
    intrinsics = torch.stack([view["camera_intrinsics"] for view in batch], dim=1)

    forward_params = inspect.signature(model.forward).parameters
    model_kwargs = {}
    if "intrinsics" in forward_params:
        model_kwargs["intrinsics"] = intrinsics
    if "global_step" in forward_params:
        model_kwargs["global_step"] = global_step
    return model(imgs, **model_kwargs)


def split_list(split_arg):
    if split_arg == "both":
        return ["train", "test"]
    return [split_arg]


def mean_numeric(rows, key):
    values = [row[key] for row in rows if key in row and isinstance(row[key], float) and math.isfinite(row[key])]
    return float(np.mean(values)) if values else math.nan


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_idx",
        "dataset_index",
        "dataset",
        "scene",
        "model_impl",
        "total_loss",
        "loss_rgb",
        "loss_ssim",
        "loss_depth",
        "loss_lpips",
        "frame_names",
    ]
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra_keys)

    csv_path = output_dir / "per_sample_losses.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.txt"
    metric_keys = [
        "total_loss",
        "loss_rgb",
        "loss_ssim",
        "loss_depth",
        "loss_lpips",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Pi3_3DGS dataset parity evaluation\n")
        f.write("=" * 48 + "\n")
        f.write(f"samples: {len(rows)}\n")
        for split in sorted({row["split"] for row in rows}):
            split_rows = [row for row in rows if row["split"] == split]
            f.write(f"\n[{split}]\n")
            f.write(f"samples: {len(split_rows)}\n")
            for key in metric_keys:
                f.write(f"{key}: {mean_numeric(split_rows, key):.6f}\n")
        f.write("\nframe_names_by_sample:\n")
        for row in rows:
            f.write(f"{row['split']} sample {row['sample_idx']}: {row['frame_names']}\n")

    print(f"Per-sample losses saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pi3_3DGS on the real dataset pipeline used by training/validation."
    )
    parser.add_argument("--config", default="outputs/pi3_highres_0523_v8_local_comp/.hydra/config.yaml")
    parser.add_argument("--ckpt", default="outputs/pi3_highres_0523_v8_local_comp/ckpts/best_model/model.safetensors")
    parser.add_argument("--dataset", default="360_v2")
    parser.add_argument("--scene", default="bicycle")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_impl", default=None, help="Use full import path or shorthand _8/_9/_10.")
    parser.add_argument("--output_dir", default="outputs/pi3_3dgs_dataset_parity")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--image_dir_name", default=None)
    parser.add_argument("--hold_every", type=int, default=None)
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--resolution", default=None, help="Override validation resolution, e.g. 518x336.")
    parser.add_argument("--batch_idx", type=int, default=20000)
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA autocast during model forward.")
    parser.add_argument("--disable_learnable_sampling", action="store_true")
    parser.add_argument("--learned_sampling_extra_ratio", type=float, default=None)
    parser.add_argument("--density_gate_opacity_power", type=float, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    model_impl = resolve_model_impl(args.model_impl, cfg)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model implementation: {model_impl}")
    print(f"disable_learnable_sampling: {args.disable_learnable_sampling}")
    print(f"learned_sampling_extra_ratio: {args.learned_sampling_extra_ratio}")
    print(f"density_gate_opacity_power: {args.density_gate_opacity_power}")

    instantiate_model.disable_learnable_sampling = args.disable_learnable_sampling
    instantiate_model.learned_sampling_extra_ratio = args.learned_sampling_extra_ratio
    instantiate_model.density_gate_opacity_power = args.density_gate_opacity_power
    model = instantiate_model(cfg, ckpt_path, model_impl, device)
    loss_fn = instantiate_loss(cfg, device)

    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = torch.no_grad

    rows = []
    sample_count = args.num_samples if args.num_samples > 0 else 1
    for split in split_list(args.split):
        dataset = build_dataset(cfg, args, split)
        print(f"[{split}] dataset scenes: {len(dataset)}")
        for sample_idx in range(sample_count):
            dataset_index = args.dataset_index % len(dataset)
            dataset._rng = np.random.default_rng(args.seed + sample_idx)
            views = dataset[dataset_index]
            batch = unified_collate_fn([views])
            batch = move_to_device(batch, device)

            with torch.no_grad():
                with amp_context():
                    pred = forward_batch(model, batch, global_step=args.global_step)
                pred = tensors_to_float(pred)
                loss, details = loss_fn(
                    pred,
                    batch,
                    current_epoch=0,
                    total_epochs=1,
                    batch_idx=args.batch_idx,
                )

            row = {
                "split": split,
                "sample_idx": sample_idx,
                "dataset_index": dataset_index,
                "dataset": args.dataset,
                "scene": args.scene,
                "model_impl": model_impl,
                "total_loss": scalar_float(loss),
                "frame_names": ";".join(get_frame_names(batch)),
            }
            for key, value in details.items():
                scalar = scalar_float(value)
                if scalar is not None:
                    row[key] = scalar
            rows.append(row)

            metric_text = ", ".join(
                f"{key}={row[key]:.6f}"
                for key in ("total_loss", "loss_rgb", "loss_ssim", "loss_depth", "loss_lpips")
                if key in row and row[key] is not None
            )
            print(f"[{split}] sample {sample_idx}: {metric_text}")
            print(f"[{split}] frames: {row['frame_names']}")

    output_dir = Path(args.output_dir) / f"{args.dataset}_{args.scene}_{model_impl.split('.')[-2]}"
    write_outputs(rows, output_dir)


if __name__ == "__main__":
    main()
