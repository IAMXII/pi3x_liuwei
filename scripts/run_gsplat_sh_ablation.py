import argparse
import csv
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.utils import save_image

from gsplat import DefaultStrategy, rasterization


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.basic import colmap_to_opencv_intrinsics


RGB_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
SH_C0 = 0.28209479177387814


def write_gsplat_sh0_ply(path, params, opacity_threshold=0.0, chunk_size=1_000_000):
    if int(params["colors"].shape[1]) != 1:
        raise ValueError("write_gsplat_sh0_ply only supports SH degree 0 params.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opacities = params["opacities"].detach()
    if opacity_threshold > 0.0:
        keep_indices = torch.nonzero(torch.sigmoid(opacities) > float(opacity_threshold), as_tuple=False).flatten()
        n = int(keep_indices.numel())
    else:
        keep_indices = None
        n = int(opacities.shape[0])

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
            if keep_indices is None:
                index = slice(start, end)
            else:
                index = keep_indices[start:end]
            means = params["means"][index].detach().float().cpu().numpy()
            zeros = np.zeros_like(means, dtype=np.float32)
            colors = params["colors"][index, 0].detach().float().cpu().numpy()
            opa = params["opacities"][index].detach().float().cpu().numpy()[:, None]
            scales = params["scales"][index].detach().float().cpu().numpy()
            quats = F.normalize(params["quats"][index].detach().float(), dim=-1).cpu().numpy()
            packed = np.concatenate([means, zeros, colors, opa, scales, quats], axis=1).astype(
                np.float32,
                copy=False,
            )
            handle.write(packed.tobytes(order="C"))


def read_next_bytes(handle, num_bytes, fmt):
    return struct.unpack("<" + fmt, handle.read(num_bytes))


def qvec_to_rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as handle:
        num_cameras = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(handle, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.array(read_next_bytes(handle, 8 * num_params, "d" * num_params), dtype=np.float32)
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as handle:
        num_images = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(handle, 4, "i")[0]
            qvec = np.array(read_next_bytes(handle, 32, "dddd"), dtype=np.float32)
            tvec = np.array(read_next_bytes(handle, 24, "ddd"), dtype=np.float32)
            camera_id = read_next_bytes(handle, 4, "i")[0]

            name = b""
            while True:
                char = handle.read(1)
                if char == b"\x00":
                    break
                name += char
            image_name = name.decode("utf-8")

            num_points2d = read_next_bytes(handle, 8, "Q")[0]
            handle.seek(num_points2d * 24, os.SEEK_CUR)

            rot_w2c = qvec_to_rotmat(qvec)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = rot_w2c
            w2c[:3, 3] = tvec
            images[image_name] = {
                "image_id": int(image_id),
                "camera_id": int(camera_id),
                "w2c": w2c,
            }
    return images


def read_points3d_binary(path):
    xyzs = []
    rgbs = []
    errors = []
    with open(path, "rb") as handle:
        num_points = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_points):
            handle.read(8)
            xyz = read_next_bytes(handle, 24, "ddd")
            rgb = read_next_bytes(handle, 3, "BBB")
            error = read_next_bytes(handle, 8, "d")[0]
            track_len = read_next_bytes(handle, 8, "Q")[0]
            handle.seek(track_len * 8, os.SEEK_CUR)
            xyzs.append(xyz)
            rgbs.append(rgb)
            errors.append(error)
    return (
        np.asarray(xyzs, dtype=np.float32),
        np.asarray(rgbs, dtype=np.float32) / 255.0,
        np.asarray(errors, dtype=np.float32),
    )


def camera_to_intrinsics(camera, image_size):
    model = camera["model"]
    params = camera["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    else:
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])

    intrinsics = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = colmap_to_opencv_intrinsics(intrinsics)

    image_width, image_height = image_size
    intrinsics[0, :] *= image_width / float(camera["width"])
    intrinsics[1, :] *= image_height / float(camera["height"])
    return intrinsics.astype(np.float32)


def resize_image_and_intrinsics(image, intrinsics, max_width):
    if max_width <= 0 or image.width <= max_width:
        tensor = image_to_tensor(image)
        return tensor, intrinsics.astype(np.float32)

    orig_width, orig_height = image.size
    scale = max_width / float(image.width)
    new_width = int(round(image.width * scale))
    new_height = int(round(image.height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    intrinsics = intrinsics.copy()
    intrinsics[0, :] *= new_width / float(orig_width)
    intrinsics[1, :] *= new_height / float(orig_height)
    return image_to_tensor(image), intrinsics.astype(np.float32)


def image_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_scene(scene_root, image_dir_name, hold_every, max_width):
    scene_root = Path(scene_root)
    sparse_dir = scene_root / "sparse" / "0"
    image_dir = scene_root / image_dir_name
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    image_metas = read_images_binary(sparse_dir / "images.bin")

    frames = []
    for image_name, meta in image_metas.items():
        image_path = image_dir / image_name
        if not image_path.is_file() or image_path.suffix.lower() not in RGB_EXTS:
            continue
        if meta["camera_id"] not in cameras:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            intrinsics = camera_to_intrinsics(cameras[meta["camera_id"]], image.size)
            image_tensor, intrinsics = resize_image_and_intrinsics(image, intrinsics, max_width)
        frames.append(
            {
                "name": image_name,
                "image_id": meta["image_id"],
                "image_path": str(image_path),
                "image": image_tensor,
                "w2c": torch.from_numpy(meta["w2c"]),
                "K": torch.from_numpy(intrinsics),
            }
        )

    frames = sorted(frames, key=lambda item: item["image_id"])
    train_frames = [frame for idx, frame in enumerate(frames) if idx % hold_every != 0]
    test_frames = [frame for idx, frame in enumerate(frames) if idx % hold_every == 0]
    return frames, train_frames, test_frames


def estimate_initial_scales(xyz, max_points_for_tree=200000):
    if xyz.shape[0] == 1:
        return np.full((1,), 0.01, dtype=np.float32)
    tree = cKDTree(xyz[:max_points_for_tree])
    dists, _ = tree.query(xyz, k=4, workers=-1)
    nn_dist = np.maximum(dists[:, 1:].mean(axis=1), 1e-4)
    return nn_dist.astype(np.float32)


def create_params(xyz, rgb, sh_degree, init_opacity, device):
    scales = estimate_initial_scales(xyz)
    n_bases = (int(sh_degree) + 1) ** 2
    sh = np.zeros((xyz.shape[0], n_bases, 3), dtype=np.float32)
    sh[:, 0, :] = (rgb - 0.5) / SH_C0

    params = nn.ParameterDict(
        {
            "means": nn.Parameter(torch.from_numpy(xyz).to(device)),
            "scales": nn.Parameter(torch.log(torch.from_numpy(scales[:, None]).repeat(1, 3)).to(device)),
            "quats": nn.Parameter(
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(xyz.shape[0], 1)
            ),
            "opacities": nn.Parameter(
                torch.full((xyz.shape[0],), torch.logit(torch.tensor(init_opacity)).item(), device=device)
            ),
            "colors": nn.Parameter(torch.from_numpy(sh).to(device)),
        }
    )
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
        name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15)
        for name, lr in lrs.items()
        if params[name].requires_grad
    }


def render(params, frame, device, sh_degree, background, absgrad=False):
    image = frame["image"]
    _, height, width = image.shape
    viewmat = frame["w2c"].to(device=device, dtype=torch.float32).unsqueeze(0)
    intrinsics = frame["K"].to(device=device, dtype=torch.float32).unsqueeze(0)
    backgrounds = torch.tensor(background, device=device, dtype=torch.float32)
    colors, alphas, meta = rasterization(
        means=params["means"],
        quats=F.normalize(params["quats"], dim=-1),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=viewmat,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=True,
        backgrounds=backgrounds,
        render_mode="RGB",
        absgrad=absgrad,
    )
    return colors[0].permute(2, 0, 1).contiguous(), alphas[0], meta


def rgb_metrics(pred, gt):
    pred = pred.clamp(0.0, 1.0).unsqueeze(0)
    gt = gt.clamp(0.0, 1.0).unsqueeze(0)
    mse = F.mse_loss(pred, gt).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    ssim = structural_similarity_index_measure(pred, gt, data_range=1.0).item()
    return psnr, ssim


@torch.no_grad()
def evaluate(params, frames, device, sh_degree, background, out_dir=None, max_save=8):
    rows = []
    for idx, frame in enumerate(frames):
        pred, _, _ = render(params, frame, device, sh_degree, background)
        gt = frame["image"].to(device=device, dtype=torch.float32)
        psnr, ssim = rgb_metrics(pred, gt)
        row = {
            "frame_idx": idx,
            "frame_name": frame["name"],
            "psnr": psnr,
            "ssim": ssim,
        }
        rows.append(row)
        if out_dir is not None and idx < max_save:
            render_path = out_dir / f"render_{idx:03d}_{Path(frame['name']).stem}.png"
            target_path = out_dir / f"target_{idx:03d}_{Path(frame['name']).stem}.png"
            save_image(pred.clamp(0.0, 1.0).cpu(), render_path)
            save_image(frame["image"].clamp(0.0, 1.0), target_path)

    psnrs = [row["psnr"] for row in rows]
    ssims = [row["ssim"] for row in rows]
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "rows": rows,
    }


def write_per_frame_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_idx", "frame_name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_scene_scale(frames):
    centers = []
    for frame in frames:
        c2w = torch.linalg.inv(frame["w2c"]).cpu().numpy()
        centers.append(c2w[:3, 3])
    centers = np.asarray(centers, dtype=np.float32)
    center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - center, axis=1)
    return float(max(np.percentile(distances, 90), 1e-6))


def build_density_strategy(args):
    if not args.enable_density_control:
        return None
    return DefaultStrategy(
        prune_opa=args.strategy_prune_opa,
        grow_grad2d=args.strategy_grow_grad2d,
        grow_scale3d=args.strategy_grow_scale3d,
        grow_scale2d=args.strategy_grow_scale2d,
        prune_scale3d=args.strategy_prune_scale3d,
        prune_scale2d=args.strategy_prune_scale2d,
        refine_scale2d_stop_iter=args.strategy_refine_scale2d_stop_iter,
        refine_start_iter=args.strategy_refine_start_iter,
        refine_stop_iter=args.strategy_refine_stop_iter,
        reset_every=args.strategy_reset_every,
        refine_every=args.strategy_refine_every,
        pause_refine_after_reset=args.strategy_pause_refine_after_reset,
        absgrad=args.strategy_absgrad,
        revised_opacity=args.strategy_revised_opacity,
        verbose=args.strategy_verbose,
    )


def maybe_sample_points(xyz, rgb, errors, max_points, seed):
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    finite_error = np.isfinite(errors)
    if finite_error.any():
        ranks = np.argsort(errors[finite_error])
        finite_indices = np.where(finite_error)[0][ranks]
        candidate = finite_indices[: max(max_points * 3, max_points)]
        if candidate.shape[0] >= max_points:
            selected = rng.choice(candidate, size=max_points, replace=False)
        else:
            selected = rng.choice(xyz.shape[0], size=max_points, replace=False)
    else:
        selected = rng.choice(xyz.shape[0], size=max_points, replace=False)
    selected = np.sort(selected)
    return xyz[selected], rgb[selected]


def train_one_variant(args, scene, xyz, rgb, sh_degree):
    seed_everything(args.seed)
    device = torch.device(args.device)
    variant_dir = Path(args.output_root) / f"sh{sh_degree}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "renders").mkdir(parents=True, exist_ok=True)

    all_frames, train_frames, test_frames = scene
    params = create_params(xyz, rgb, sh_degree, args.init_opacity, device)
    optimizers = build_optimizers(params, args)
    initial_num_points = int(params["means"].shape[0])
    scene_scale = estimate_scene_scale(all_frames)
    strategy = build_density_strategy(args)
    strategy_state = None
    if strategy is not None:
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    background = [float(x) for x in args.background]
    log_rows = []
    start_time = time.time()

    config = {
        **vars(args),
        "sh_degree": int(sh_degree),
        "initial_num_points": initial_num_points,
        "scene_scale": scene_scale,
        "density_control": strategy is not None,
        "num_frames_all": len(all_frames),
        "num_frames_train": len(train_frames),
        "num_frames_test": len(test_frames),
        "height": int(all_frames[0]["image"].shape[1]),
        "width": int(all_frames[0]["image"].shape[2]),
    }
    with open(variant_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for step in range(args.iters):
        frame = train_frames[step % len(train_frames)]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        pred, _, info = render(
            params,
            frame,
            device,
            sh_degree,
            background,
            absgrad=bool(strategy.absgrad) if strategy is not None else False,
        )
        gt = frame["image"].to(device=device, dtype=torch.float32)
        pred_b = pred.clamp(0.0, 1.0).unsqueeze(0)
        gt_b = gt.unsqueeze(0)
        loss_l1 = F.l1_loss(pred_b, gt_b)
        loss_ssim = 1.0 - structural_similarity_index_measure(pred_b, gt_b, data_range=1.0)
        loss = (1.0 - args.lambda_ssim) * loss_l1 + args.lambda_ssim * loss_ssim
        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

        if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == args.iters:
            with torch.no_grad():
                psnr, ssim = rgb_metrics(pred, gt)
            row = {
                "step": step + 1,
                "loss": float(loss.item()),
                "loss_l1": float(loss_l1.item()),
                "loss_ssim": float(loss_ssim.item()),
                "train_psnr": float(psnr),
                "train_ssim": float(ssim),
                "num_points": int(params["means"].shape[0]),
                "elapsed_sec": float(time.time() - start_time),
            }
            log_rows.append(row)
            print(
                f"[sh={sh_degree}] step {step + 1}/{args.iters} "
                f"loss={row['loss']:.5f} train_ssim={row['train_ssim']:.4f} "
                f"points={row['num_points']}",
                flush=True,
            )

    train_eval = evaluate(
        params,
        train_frames[: min(len(train_frames), args.eval_train_frames)],
        device,
        sh_degree,
        background,
        out_dir=variant_dir / "renders" if args.save_renders else None,
        max_save=args.max_save_renders,
    )
    test_eval = evaluate(
        params,
        test_frames,
        device,
        sh_degree,
        background,
        out_dir=variant_dir / "renders" if args.save_renders else None,
        max_save=args.max_save_renders,
    )

    write_per_frame_csv(variant_dir / "test_metrics_per_frame.csv", test_eval["rows"])
    final_ply = None
    if args.save_ply:
        if int(sh_degree) != 0:
            raise ValueError("--save-ply currently supports SH degree 0 only.")
        final_ply = variant_dir / f"optimized_sh0_{args.iters}.ply"
        write_gsplat_sh0_ply(
            final_ply,
            params,
            opacity_threshold=args.ply_opacity_threshold,
            chunk_size=args.ply_chunk_size,
        )
        print(f"[sh={sh_degree}] saved PLY: {final_ply}", flush=True)

    summary = {
        "sh_degree": int(sh_degree),
        "initial_num_points": initial_num_points,
        "final_num_points": int(params["means"].shape[0]),
        "final_ply": str(final_ply) if final_ply is not None else "",
        "ply_opacity_threshold": float(args.ply_opacity_threshold),
        "scene_scale": scene_scale,
        "density_control": strategy is not None,
        "train_eval_frames": int(len(train_eval["rows"])),
        "test_eval_frames": int(len(test_eval["rows"])),
        "train_psnr": train_eval["psnr"],
        "train_ssim": train_eval["ssim"],
        "train_loss_ssim": 1.0 - train_eval["ssim"],
        "test_psnr": test_eval["psnr"],
        "test_ssim": test_eval["ssim"],
        "test_loss_ssim": 1.0 - test_eval["ssim"],
        "elapsed_sec": float(time.time() - start_time),
        "log": log_rows,
    }
    torch.save({name: value.detach().cpu() for name, value in params.items()}, variant_dir / "params.pt")
    with open(variant_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def write_pair_summary(output_root, summaries):
    rows = []
    by_degree = {item["sh_degree"]: item for item in summaries}
    default = by_degree.get(3) or summaries[0]
    for item in summaries:
        rows.append(
            {
                "sh_degree": item["sh_degree"],
                "initial_num_points": item["initial_num_points"],
                "final_num_points": item["final_num_points"],
                "final_ply": item.get("final_ply", ""),
                "density_control": item["density_control"],
                "train_ssim": item["train_ssim"],
                "test_ssim": item["test_ssim"],
                "test_loss_ssim": item["test_loss_ssim"],
                "delta_test_ssim_vs_sh3": item["test_ssim"] - default["test_ssim"],
                "train_psnr": item["train_psnr"],
                "test_psnr": item["test_psnr"],
                "elapsed_sec": item["elapsed_sec"],
            }
        )

    output_root = Path(output_root)
    with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled gsplat SH-degree ablation on 360_v2 bicycle.")
    parser.add_argument("--scene-root", default="/data/liuwei/dataset/360_v2/bicycle")
    parser.add_argument("--image-dir-name", default="images_4")
    parser.add_argument("--output-root", default="outputs/gsplat_sh_ablation_0530_bicycle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--hold-every", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=518)
    parser.add_argument("--max-points", type=int, default=54275)
    parser.add_argument("--sh-degrees", type=int, nargs="+", default=[3, 0])
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--init-opacity", type=float, default=0.1)
    parser.add_argument("--lr-means", type=float, default=1.6e-4)
    parser.add_argument("--lr-scales", type=float, default=5e-3)
    parser.add_argument("--lr-quats", type=float, default=1e-3)
    parser.add_argument("--lr-opacities", type=float, default=5e-2)
    parser.add_argument("--lr-colors", type=float, default=2.5e-3)
    parser.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-train-frames", type=int, default=25)
    parser.add_argument("--save-renders", action="store_true")
    parser.add_argument("--max-save-renders", type=int, default=8)
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--ply-opacity-threshold", type=float, default=0.0)
    parser.add_argument("--ply-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--enable-density-control", action="store_true")
    parser.add_argument("--strategy-prune-opa", type=float, default=0.005)
    parser.add_argument("--strategy-grow-grad2d", type=float, default=0.0002)
    parser.add_argument("--strategy-grow-scale3d", type=float, default=0.01)
    parser.add_argument("--strategy-grow-scale2d", type=float, default=0.05)
    parser.add_argument("--strategy-prune-scale3d", type=float, default=0.1)
    parser.add_argument("--strategy-prune-scale2d", type=float, default=0.15)
    parser.add_argument("--strategy-refine-scale2d-stop-iter", type=int, default=0)
    parser.add_argument("--strategy-refine-start-iter", type=int, default=500)
    parser.add_argument("--strategy-refine-stop-iter", type=int, default=15000)
    parser.add_argument("--strategy-reset-every", type=int, default=3000)
    parser.add_argument("--strategy-refine-every", type=int, default=100)
    parser.add_argument("--strategy-pause-refine-after-reset", type=int, default=0)
    parser.add_argument("--strategy-absgrad", action="store_true")
    parser.add_argument("--strategy-revised-opacity", action="store_true")
    parser.add_argument("--strategy-verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.scene_root, args.image_dir_name, args.hold_every, args.max_width)
    xyz, rgb, errors = read_points3d_binary(Path(args.scene_root) / "sparse" / "0" / "points3D.bin")
    xyz, rgb = maybe_sample_points(xyz, rgb, errors, args.max_points, args.seed)

    summaries = []
    for sh_degree in args.sh_degrees:
        summaries.append(train_one_variant(args, scene, xyz, rgb, sh_degree))
        torch.cuda.empty_cache()

    write_pair_summary(output_root, summaries)
    print(f"Wrote {output_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
