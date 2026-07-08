#!/usr/bin/env python3
import argparse
import csv
import gc
import inspect
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure

from gsplat import rasterization

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_3dgs_5 import (  # noqa: E402
    DEFAULT_MODEL_IMPL,
    MODEL_IMPL_ALIASES,
    import_model_class,
    infer_hydra_config_path,
    load_hydra_config,
    load_model_kwargs_from_config,
    load_rgb_sequence,
    resolve_checkpoint_path,
    se3_inverse,
)


SH_C0 = 0.28209479177387814
PLY_FIELDS = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def read_pi3_ply(path):
    path = Path(path)
    properties = []
    vertex_count = None
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} ended before PLY header finished")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property "):
                parts = text.split()
                if len(parts) == 3:
                    properties.append((parts[1], parts[2]))
            elif text == "end_header":
                data_offset = handle.tell()
                break

    if vertex_count is None:
        raise ValueError(f"{path} has no vertex count")
    prop_names = [name for _, name in properties]
    missing = [name for name in PLY_FIELDS if name not in prop_names]
    if missing:
        raise ValueError(f"{path} is missing required PLY fields: {missing}")
    if any(dtype != "float" for dtype, _ in properties):
        raise ValueError(f"{path} must use float properties only; got {properties}")

    dtype = np.dtype([(name, "<f4") for _, name in properties])
    with path.open("rb") as handle:
        handle.seek(data_offset)
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)
    if data.shape[0] != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices from {path}, got {data.shape[0]}")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32, copy=True)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32, copy=True)
    opacity = data["opacity"].astype(np.float32, copy=True)
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32, copy=True)
    quats = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32, copy=True)
    return xyz, f_dc, opacity, scales, quats


def write_pi3_ply(path, params, chunk_size=1_000_000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(params["means"].shape[0])
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for start in range(0, n, int(chunk_size)):
            end = min(start + int(chunk_size), n)
            means = params["means"][start:end].detach().float().cpu().numpy()
            colors = params["colors"][start:end, 0].detach().float().cpu().numpy()
            opacities = params["opacities"][start:end].detach().float().cpu().numpy()[:, None]
            scales = params["scales"][start:end].detach().float().cpu().numpy()
            quats = F.normalize(params["quats"][start:end].detach().float(), dim=-1).cpu().numpy()
            zeros = np.zeros_like(means, dtype=np.float32)
            packed = np.concatenate([means, zeros, colors, opacities, scales, quats], axis=1).astype(
                np.float32,
                copy=False,
            )
            handle.write(packed.tobytes(order="C"))


def maybe_sample_gaussians(xyz, f_dc, opacity, scales, quats, max_gaussians, seed):
    if max_gaussians <= 0 or xyz.shape[0] <= max_gaussians:
        return xyz, f_dc, opacity, scales, quats, np.arange(xyz.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    probs = 1.0 / (1.0 + np.exp(-np.clip(opacity, -80.0, 80.0)))
    top_pool = min(xyz.shape[0], max(max_gaussians * 3, max_gaussians))
    pool = np.argpartition(-probs, top_pool - 1)[:top_pool]
    selected = np.sort(rng.choice(pool, size=max_gaussians, replace=False))
    return xyz[selected], f_dc[selected], opacity[selected], scales[selected], quats[selected], selected


def create_params(
    xyz,
    f_dc,
    opacity,
    scales,
    quats,
    device,
    train_means=True,
    train_scales=True,
    train_quats=True,
    train_opacities=True,
    train_colors=True,
    input_color_mode="rgb",
):
    if input_color_mode == "rgb":
        color_init = (np.clip(f_dc, 0.0, 1.0) - 0.5) / SH_C0
    elif input_color_mode == "sh0":
        color_init = f_dc
    else:
        raise ValueError(f"Unsupported input_color_mode: {input_color_mode}")
    params = nn.ParameterDict(
        {
            "means": nn.Parameter(torch.from_numpy(xyz).to(device=device, dtype=torch.float32)),
            "scales": nn.Parameter(torch.from_numpy(scales).to(device=device, dtype=torch.float32)),
            "quats": nn.Parameter(torch.from_numpy(quats).to(device=device, dtype=torch.float32)),
            "opacities": nn.Parameter(torch.from_numpy(opacity).to(device=device, dtype=torch.float32)),
            "colors": nn.Parameter(torch.from_numpy(color_init[:, None, :]).to(device=device, dtype=torch.float32)),
        }
    )
    trainable = {
        "means": bool(train_means),
        "scales": bool(train_scales),
        "quats": bool(train_quats),
        "opacities": bool(train_opacities),
        "colors": bool(train_colors),
    }
    for name, should_train in trainable.items():
        params[name].requires_grad_(should_train)
    return params


def build_optimizers(params, args):
    lrs = {
        "means": args.lr_means,
        "scales": args.lr_scales,
        "quats": args.lr_quats,
        "opacities": args.lr_opacities,
        "colors": args.lr_colors,
    }
    return {
        name: torch.optim.Adam([param], lr=lrs[name], eps=1e-15)
        for name, param in params.items()
        if param.requires_grad and lrs[name] > 0.0
    }


def render(params, w2c, K, height, width, background):
    colors, alphas, meta = rasterization(
        means=params["means"],
        quats=F.normalize(params["quats"], dim=-1),
        scales=torch.exp(params["scales"]).clamp_min(1e-8),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=w2c.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=int(width),
        height=int(height),
        sh_degree=0,
        packed=True,
        backgrounds=background,
        render_mode="RGB",
    )
    return colors[0].permute(2, 0, 1).contiguous(), alphas[0], meta


def psnr_from_tensors(pred, gt):
    mse = F.mse_loss(pred.clamp(0.0, 1.0), gt.clamp(0.0, 1.0)).item()
    return -10.0 * math.log10(max(mse, 1e-12))


def load_or_predict_cameras(args, imgs_cpu, frame_items, device):
    cache_path = Path(args.camera_cache)
    if cache_path.is_file() and not args.force_recompute_cameras:
        with np.load(cache_path, allow_pickle=True) as data:
            cached_names = [str(x) for x in data["frame_names"].tolist()]
            expected_names = [str(item["stem"]) for item in frame_items]
            if cached_names != expected_names:
                raise RuntimeError(
                    f"Camera cache frame names do not match current data: {cache_path}. "
                    "Pass --force_recompute_cameras to rebuild it."
                )
            w2c = torch.from_numpy(data["w2c"].astype(np.float32))
            K = torch.from_numpy(data["K"].astype(np.float32))
        print(f"Loaded cached cameras: {cache_path}", flush=True)
        return w2c, K

    args.ckpt = resolve_checkpoint_path(args.ckpt)
    config_path = args.model_config or infer_hydra_config_path(args.ckpt)
    cfg, loaded_config_path = load_hydra_config(config_path)
    model_cfg = cfg.get("model") or {}
    model_impl = args.model_impl or model_cfg.get("_target_") or DEFAULT_MODEL_IMPL
    model_impl = MODEL_IMPL_ALIASES.get(model_impl, model_impl)
    model_cls = import_model_class(model_impl)
    print(f"Loading camera source model: {model_impl}", flush=True)
    if loaded_config_path:
        print(f"Loaded model config: {loaded_config_path}", flush=True)

    model_kwargs = {
        "pos_type": "rope100",
        "decoder_size": "large",
        "ckpt": None,
        "debug_mem": False,
        "gs_view_stride": args.gs_view_stride,
        "enable_quadtree": not args.disable_quadtree,
        "enable_local_competition": not args.disable_local_competition,
    }
    model_kwargs.update(load_model_kwargs_from_config(cfg, model_cls))
    model_kwargs.update({
        "ckpt": None,
        "debug_mem": False,
        "gs_view_stride": args.gs_view_stride,
        "enable_quadtree": not args.disable_quadtree,
        "enable_local_competition": not args.disable_local_competition,
    })
    for arg_name in (
        "geometry_head_view_chunk_size",
        "gs_decoder_view_chunk_size",
        "gs_head_view_chunk_size",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            model_kwargs[arg_name] = value
    valid_model_keys = set(inspect.signature(model_cls.__init__).parameters)
    model_kwargs = {key: value for key, value in model_kwargs.items() if key in valid_model_keys}

    model = model_cls(**model_kwargs).to(device).eval()
    if str(args.ckpt).endswith(".safetensors"):
        from safetensors.torch import load_file

        weight = load_file(args.ckpt)
        if hasattr(model, "_load_state_dict_flexible"):
            model._load_state_dict_flexible(weight)
        else:
            model.load_state_dict(weight, strict=False)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        if hasattr(model, "_load_state_dict_flexible"):
            model._load_state_dict_flexible(weight)
        else:
            model.load_state_dict(weight, strict=False)

    imgs_batch = imgs_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    print(f"Predicting Pi3 cameras for {imgs_batch.shape[1]} frames...", flush=True)
    with torch.no_grad():
        with amp_context():
            res = model(imgs_batch)
    pred_w2c = se3_inverse(res["camera_poses"]).detach().float().cpu()[0]
    pred_K = res["intrinsics"].detach().float().cpu()[0]
    frame_names = np.array([str(item["stem"]) for item in frame_items], dtype=object)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        w2c=pred_w2c.numpy(),
        K=pred_K.numpy(),
        frame_names=frame_names,
        height=np.array([imgs_cpu.shape[2]], dtype=np.int32),
        width=np.array([imgs_cpu.shape[3]], dtype=np.int32),
    )
    print(f"Saved camera cache: {cache_path}", flush=True)

    del res, imgs_batch, model, weight
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_w2c, pred_K


@torch.no_grad()
def evaluate_subset(params, imgs_cpu, w2c_cpu, K_cpu, device, background, frame_indices):
    rows = []
    height = int(imgs_cpu.shape[2])
    width = int(imgs_cpu.shape[3])
    for idx in frame_indices:
        pred, _, _ = render(
            params,
            w2c_cpu[idx].to(device=device, dtype=torch.float32),
            K_cpu[idx].to(device=device, dtype=torch.float32),
            height,
            width,
            background,
        )
        gt = imgs_cpu[idx].to(device=device, dtype=torch.float32)
        pred_b = pred.clamp(0.0, 1.0).unsqueeze(0)
        gt_b = gt.clamp(0.0, 1.0).unsqueeze(0)
        ssim = structural_similarity_index_measure(pred_b, gt_b, data_range=1.0).item()
        rows.append({"frame_idx": int(idx), "psnr": psnr_from_tensors(pred, gt), "ssim": ssim})
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a Pi3-exported PLY with gsplat SH degree 0.")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-impl", default="_12")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-cache", default=None)
    parser.add_argument("--force-recompute-cameras", action="store_true")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--subset-start", type=int, default=None)
    parser.add_argument("--subset-end", type=int, default=None)
    parser.add_argument("--subset-step", type=int, default=1)
    parser.add_argument("--target-frame-count", type=int, default=None)
    parser.add_argument("--pixel-limit", type=int, default=255000)
    parser.add_argument("--gs-view-stride", type=int, default=1)
    parser.add_argument("--geometry-head-view-chunk-size", type=int, default=100)
    parser.add_argument("--gs-decoder-view-chunk-size", type=int, default=48)
    parser.add_argument("--gs-head-view-chunk-size", type=int, default=1)
    parser.add_argument("--disable-quadtree", action="store_true")
    parser.add_argument("--disable-local-competition", action="store_true")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--train-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-means", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-scales", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-quats", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-opacities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-colors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-color-mode", choices=["rgb", "sh0"], default="rgb")
    parser.add_argument("--lr-means", type=float, default=1.6e-4)
    parser.add_argument("--lr-scales", type=float, default=5e-3)
    parser.add_argument("--lr-quats", type=float, default=1e-3)
    parser.add_argument("--lr-opacities", type=float, default=5e-2)
    parser.add_argument("--lr-colors", type=float, default=2.5e-3)
    parser.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-frame-count", type=int, default=24)
    parser.add_argument("--export-every", type=int, default=0)
    parser.add_argument("--write-chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.camera_cache is None:
        args.camera_cache = str(output_dir / "pi3_predicted_cameras.npz")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    print(f"Loading target frames from {args.data_path}", flush=True)
    imgs_cpu, frame_items, orig_hw, target_hw = load_rgb_sequence(
        args.data_path,
        interval=args.interval,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        subset_step=args.subset_step,
        pixel_limit=args.pixel_limit,
        target_frame_count=args.target_frame_count,
    )
    if imgs_cpu.numel() == 0:
        raise RuntimeError("No target frames loaded.")
    height = int(imgs_cpu.shape[2])
    width = int(imgs_cpu.shape[3])
    print(f"Loaded {imgs_cpu.shape[0]} frames at {height}x{width}; original={orig_hw}", flush=True)

    w2c_cpu, K_cpu = load_or_predict_cameras(args, imgs_cpu, frame_items, device)
    if w2c_cpu.shape[0] != imgs_cpu.shape[0] or K_cpu.shape[0] != imgs_cpu.shape[0]:
        raise RuntimeError(f"Camera count mismatch: w2c={w2c_cpu.shape}, K={K_cpu.shape}, imgs={imgs_cpu.shape}")

    print(f"Loading PLY: {args.ply}", flush=True)
    xyz, f_dc, opacity, scales, quats = read_pi3_ply(args.ply)
    original_count = int(xyz.shape[0])
    xyz, f_dc, opacity, scales, quats, selected_indices = maybe_sample_gaussians(
        xyz, f_dc, opacity, scales, quats, args.max_gaussians, args.seed
    )
    if selected_indices.shape[0] != original_count:
        np.save(output_dir / "selected_gaussian_indices.npy", selected_indices)
        print(f"Sampled {selected_indices.shape[0]} / {original_count} gaussians", flush=True)
    else:
        print(f"Using all {original_count} gaussians", flush=True)

    train_means = args.train_geometry if args.train_means is None else args.train_means
    train_scales = args.train_geometry if args.train_scales is None else args.train_scales
    train_quats = args.train_geometry if args.train_quats is None else args.train_quats
    params = create_params(
        xyz,
        f_dc,
        opacity,
        scales,
        quats,
        device,
        train_means=train_means,
        train_scales=train_scales,
        train_quats=train_quats,
        train_opacities=args.train_opacities,
        train_colors=args.train_colors,
        input_color_mode=args.input_color_mode,
    )
    del xyz, f_dc, opacity, scales, quats
    gc.collect()
    optimizers = build_optimizers(params, args)
    background = torch.tensor(args.background, device=device, dtype=torch.float32)
    log_path = output_dir / "train_log.csv"
    summary_path = output_dir / "summary.json"
    eval_indices = np.linspace(0, imgs_cpu.shape[0] - 1, num=min(args.eval_frame_count, imgs_cpu.shape[0]))
    eval_indices = sorted({int(round(x)) for x in eval_indices})
    config = {
        **vars(args),
        "sh_degree": 0,
        "original_gaussian_count": original_count,
        "optimized_gaussian_count": int(params["means"].shape[0]),
        "frame_count": int(imgs_cpu.shape[0]),
        "height": height,
        "width": width,
        "target_hw": target_hw,
        "orig_hw": orig_hw,
        "trainable_params": {
            "means": bool(params["means"].requires_grad),
            "scales": bool(params["scales"].requires_grad),
            "quats": bool(params["quats"].requires_grad),
            "opacities": bool(params["opacities"].requires_grad),
            "colors": bool(params["colors"].requires_grad),
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Starting gsplat SH=0 optimization for {args.iters} iterations", flush=True)
    rows = []
    start_time = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "frame_idx",
                "loss",
                "loss_l1",
                "loss_ssim",
                "train_psnr",
                "train_ssim",
                "num_gaussians",
                "elapsed_sec",
            ],
        )
        writer.writeheader()

        for step in range(args.iters):
            frame_idx = step % int(imgs_cpu.shape[0])
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            pred, _, _ = render(
                params,
                w2c_cpu[frame_idx].to(device=device, dtype=torch.float32),
                K_cpu[frame_idx].to(device=device, dtype=torch.float32),
                height,
                width,
                background,
            )
            gt = imgs_cpu[frame_idx].to(device=device, dtype=torch.float32)
            pred_b = pred.clamp(0.0, 1.0).unsqueeze(0)
            gt_b = gt.clamp(0.0, 1.0).unsqueeze(0)
            loss_l1 = F.l1_loss(pred_b, gt_b)
            loss_ssim = 1.0 - structural_similarity_index_measure(pred_b, gt_b, data_range=1.0)
            loss = (1.0 - args.lambda_ssim) * loss_l1 + args.lambda_ssim * loss_ssim
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()

            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.iters:
                with torch.no_grad():
                    train_psnr = psnr_from_tensors(pred, gt)
                    train_ssim = 1.0 - float(loss_ssim.item())
                row = {
                    "step": step + 1,
                    "frame_idx": frame_idx,
                    "loss": float(loss.item()),
                    "loss_l1": float(loss_l1.item()),
                    "loss_ssim": float(loss_ssim.item()),
                    "train_psnr": float(train_psnr),
                    "train_ssim": float(train_ssim),
                    "num_gaussians": int(params["means"].shape[0]),
                    "elapsed_sec": float(time.time() - start_time),
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(
                    f"[gsplat-sh0] step {step + 1}/{args.iters} "
                    f"frame={frame_idx} loss={row['loss']:.5f} "
                    f"psnr={row['train_psnr']:.3f} ssim={row['train_ssim']:.4f}",
                    flush=True,
                )

            if args.export_every > 0 and (step + 1) % args.export_every == 0:
                export_path = output_dir / f"optimized_sh0_step_{step + 1:06d}.ply"
                print(f"Exporting checkpoint PLY: {export_path}", flush=True)
                write_pi3_ply(export_path, params, chunk_size=args.write_chunk_size)

            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                eval_rows = evaluate_subset(params, imgs_cpu, w2c_cpu, K_cpu, device, background, eval_indices)
                mean_psnr = float(np.mean([row["psnr"] for row in eval_rows]))
                mean_ssim = float(np.mean([row["ssim"] for row in eval_rows]))
                with (output_dir / f"eval_step_{step + 1:06d}.json").open("w", encoding="utf-8") as eval_handle:
                    json.dump({"step": step + 1, "mean_psnr": mean_psnr, "mean_ssim": mean_ssim, "rows": eval_rows}, eval_handle, indent=2)
                print(f"[eval] step {step + 1}: psnr={mean_psnr:.3f} ssim={mean_ssim:.4f}", flush=True)

    final_ply = output_dir / "optimized_sh0_10000.ply"
    if args.iters != 10000:
        final_ply = output_dir / f"optimized_sh0_{args.iters}.ply"
    print(f"Exporting final PLY: {final_ply}", flush=True)
    write_pi3_ply(final_ply, params, chunk_size=args.write_chunk_size)

    final_eval = evaluate_subset(params, imgs_cpu, w2c_cpu, K_cpu, device, background, eval_indices)
    summary = {
        "final_ply": str(final_ply),
        "iters": int(args.iters),
        "sh_degree": 0,
        "original_gaussian_count": original_count,
        "optimized_gaussian_count": int(params["means"].shape[0]),
        "elapsed_sec": float(time.time() - start_time),
        "last_log": rows[-1] if rows else None,
        "eval_frame_indices": eval_indices,
        "eval_psnr": float(np.mean([row["psnr"] for row in final_eval])),
        "eval_ssim": float(np.mean([row["ssim"] for row in final_eval])),
        "eval_rows": final_eval,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
