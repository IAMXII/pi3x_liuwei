import argparse
import csv
import gc
import math
import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

from pi3.models.pi3_3dgs_topk import Pi3_3DGS


RGB_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_CKPT = "outputs/pi3_highres_0402/ckpts/best_model/model.safetensors"


def save_heatmap(tensor, path):
    alpha_np = tensor.squeeze().detach().cpu().numpy()
    alpha_np = np.clip(alpha_np, 0, 1)
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap_color)


def save_ply_binary(gaussians, path):
    xyz = gaussians["xyz"].detach().cpu().float().numpy().reshape(-1, 3)
    rot = gaussians["rotation"].detach().cpu().float().numpy().reshape(-1, 4)
    scale = gaussians["scale"].detach().cpu().float().numpy().reshape(-1, 3)
    opacity = gaussians["opacity"].detach().cpu().float().numpy().reshape(-1)
    color = gaussians["color"].detach().cpu().float().numpy().reshape(-1, 3)

    scale_ply = np.log(np.clip(scale, 1e-10, None))
    opacity_clipped = np.clip(opacity, 1e-6, 1 - 1e-6)
    opacity_ply = np.log(opacity_clipped / (1 - opacity_clipped))

    sh_c0 = 0.28209479177387814
    f_dc = (color - 0.5) / sh_c0
    normals = np.zeros_like(xyz)

    attributes = np.concatenate((
        xyz, normals, f_dc, opacity_ply[..., np.newaxis], scale_ply, rot
    ), axis=-1).astype(np.float32)

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n".encode("utf-8"))
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write(b"end_header\n")
        f.write(attributes.tobytes())


def se3_inverse(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:4] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def save_image(tensor, path):
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_np)


def save_depth(tensor, path):
    depth_np = tensor.squeeze().detach().cpu().numpy()
    valid_mask = depth_np > 1e-5
    if valid_mask.sum() > 0:
        d_min = np.percentile(depth_np[valid_mask], 2)
        d_max = np.percentile(depth_np[valid_mask], 98)
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
        depth_norm = np.clip(depth_norm, 0, 1)
    else:
        depth_norm = np.zeros_like(depth_np, dtype=np.float32)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(path, depth_gray)


def render_frame(gaussians, w2c, K, H, W, num_gaussians=None):
    means = gaussians["xyz"]
    quats = gaussians["rotation"]
    scales = gaussians["scale"]
    opacities = gaussians["opacity"]
    colors = gaussians["color"]
    conf = gaussians.get("conf", None)

    B = means.shape[0]
    if num_gaussians is not None:
        if isinstance(num_gaussians, torch.Tensor):
            num_gaussians = num_gaussians.to(device=means.device, dtype=torch.long).reshape(B)
            max_N = max(1, int(num_gaussians.max().item()))
            means = means[:, :max_N]
            quats = quats[:, :max_N]
            scales = scales[:, :max_N]
            opacities = opacities[:, :max_N].clone()
            colors = colors[:, :max_N]
            if isinstance(conf, torch.Tensor):
                conf = conf[:, :max_N]

            range_seq = torch.arange(max_N, device=means.device).expand(B, max_N)
            valid_mask = range_seq < num_gaussians.unsqueeze(1)
            opacities[~valid_mask] = 0.0
        else:
            limit = max(1, int(num_gaussians))
            means = means[:, :limit]
            quats = quats[:, :limit]
            scales = scales[:, :limit]
            opacities = opacities[:, :limit]
            colors = colors[:, :limit]
            if isinstance(conf, torch.Tensor):
                conf = conf[:, :limit]

    rgb, alpha, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W,
        height=H,
        render_mode="RGB",
        packed=False,
    )

    if isinstance(conf, torch.Tensor):
        conf_prob = torch.sigmoid(conf)
        if conf_prob.ndim == 2:
            conf_prob = conf_prob.unsqueeze(-1)
        opacities_depth = torch.where(conf_prob < 0.1, torch.zeros_like(opacities), opacities)
    else:
        opacities_depth = opacities

    depth, _, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities_depth.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W,
        height=H,
        render_mode="ED",
        packed=False,
    )

    return rgb, depth, alpha


def list_sorted_files(directory, extensions):
    filenames = [
        x for x in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, x)) and x.lower().endswith(extensions)
    ]
    return sorted(filenames)


def build_selected_indices(total_count, interval=1, subset_start=None, subset_end=None, subset_step=1):
    base_indices = list(range(0, total_count, interval))
    return base_indices[slice(subset_start, subset_end, subset_step)]


def compute_target_size(width, height, pixel_limit):
    if pixel_limit is None or pixel_limit <= 0:
        return max(1, round(width / 14)) * 14, max(1, round(height / 14)) * 14

    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    scale = min(scale, 1.0)
    w_target = width * scale
    h_target = height * scale
    k = max(1, round(w_target / 14))
    m = max(1, round(h_target / 14))
    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(m, 1) > w_target / max(h_target, 1e-8):
            k -= 1
        else:
            m -= 1
        k = max(k, 1)
        m = max(m, 1)
    return k * 14, m * 14


def load_rgb_sequence(path, interval=1, subset_start=None, subset_end=None, subset_step=1, pixel_limit=255000):
    frame_items = []
    sources = []

    if os.path.isdir(path):
        filenames = list_sorted_files(path, RGB_EXTS)
        selected_indices = build_selected_indices(
            len(filenames),
            interval=interval,
            subset_start=subset_start,
            subset_end=subset_end,
            subset_step=subset_step,
        )
        for source_index in selected_indices:
            filename = filenames[source_index]
            full_path = os.path.join(path, filename)
            img = Image.open(full_path).convert("RGB")
            sources.append(img)
            frame_items.append({
                "stem": os.path.splitext(filename)[0],
                "source_index": source_index,
                "path": full_path,
            })
    elif path.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        raw_sources = []
        raw_items = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_sources.append(Image.fromarray(rgb_frame))
                raw_items.append({
                    "stem": f"{frame_idx:06d}",
                    "source_index": frame_idx,
                    "path": None,
                })
            frame_idx += 1
        cap.release()

        subset = slice(subset_start, subset_end, subset_step)
        sources = raw_sources[subset]
        frame_items = raw_items[subset]
    else:
        raise ValueError(f"Unsupported data_path. Must be a directory or .mp4: {path}")

    if not sources:
        return torch.empty(0), [], None, None

    orig_w, orig_h = sources[0].size
    target_w, target_h = compute_target_size(orig_w, orig_h, pixel_limit)
    resample_lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

    tensor_list = []
    for img in sources:
        resized = img.resize((target_w, target_h), resample_lanczos)
        img_np = np.asarray(resized, dtype=np.float32) / 255.0
        tensor_list.append(torch.from_numpy(img_np).permute(2, 0, 1))

    imgs = torch.stack(tensor_list, dim=0)
    return imgs, frame_items, (orig_h, orig_w), (target_h, target_w)


def maybe_cuda_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_retryable_inference_error(exc):
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "32-bit index math" in message
        or "input tensor must fit into 32-bit index math" in message
    )


def select_render_gaussians(gaussians, batch_index=0):
    render_keys = {"xyz", "rotation", "scale", "opacity", "color"}
    current = {
        k: v[batch_index:batch_index + 1]
        for k, v in gaussians.items()
        if isinstance(v, torch.Tensor) and k in render_keys
    }

    conf_tensor = gaussians.get("conf")
    if isinstance(conf_tensor, torch.Tensor):
        xyz_tensor = current["xyz"]
        if (
            conf_tensor.ndim in (2, 3)
            and conf_tensor.shape[0] > batch_index
            and conf_tensor.shape[1] == xyz_tensor.shape[1]
        ):
            current["conf"] = conf_tensor[batch_index:batch_index + 1]

    return current


def init_metrics(device, skip_lpips=False):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    metrics = {
        "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        "lpips": None,
    }

    if not skip_lpips:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True,
        ).to(device)

    return metrics


def compute_rgb_metrics(metrics, pred, gt):
    pred_for_metric = torch.clamp(pred.unsqueeze(0).float(), 0.0, 1.0)
    gt_for_metric = gt.unsqueeze(0).float()

    out = {
        "psnr": metrics["psnr"](pred_for_metric, gt_for_metric).item(),
        "ssim": metrics["ssim"](pred_for_metric, gt_for_metric).item(),
        "lpips": None,
    }
    if metrics["lpips"] is not None:
        out["lpips"] = metrics["lpips"](pred_for_metric, gt_for_metric).item()
    return out


def load_checkpoint(model, ckpt_path, device):
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(ckpt_path, device="cpu")
    else:
        weight = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    load_result = model.load_state_dict(weight, strict=False)
    del weight
    maybe_cuda_empty_cache(device)
    return load_result


def write_metrics_report(path, args, total_frames, H, W, metrics_rows, load_result):
    psnr_values = [x["psnr"] for x in metrics_rows if x.get("psnr") is not None]
    ssim_values = [x["ssim"] for x in metrics_rows if x.get("ssim") is not None]
    lpips_values = [x["lpips"] for x in metrics_rows if x.get("lpips") is not None]

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else float("nan")
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")
    avg_lpips = float(np.mean(lpips_values)) if lpips_values else float("nan")

    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])

    with open(path, "w") as f:
        f.write("Pi3_3DGS TopK Inference Report\n")
        f.write("=" * 48 + "\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"model_class: pi3.models.pi3_3dgs_topk.Pi3_3DGS\n")
        f.write(f"frames: {total_frames}\n")
        f.write(f"model_resolution: {H}x{W}\n")
        f.write(f"interval: {args.interval}\n")
        f.write(f"subset_start: {args.subset_start}\n")
        f.write(f"subset_end: {args.subset_end}\n")
        f.write(f"subset_step: {args.subset_step}\n")
        f.write(f"pixel_limit: {args.pixel_limit}\n")
        f.write(f"chunk_size: {args.chunk_size}\n")
        f.write(f"missing_keys: {len(missing_keys)}\n")
        f.write(f"unexpected_keys: {len(unexpected_keys)}\n")
        f.write("\n")
        f.write(f"Average PSNR:  {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")

    return avg_psnr, avg_ssim, avg_lpips


def main():
    parser = argparse.ArgumentParser(description="Pi3_3DGS TopK inference and rendering")
    parser.add_argument("--data_path", type=str, default="examples/skating.mp4")
    parser.add_argument("--output_dir", type=str, default="output_render_cp_topk")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of frames per inference chunk")
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--subset_start", type=int, default=None, help="Start index after interval sampling")
    parser.add_argument("--subset_end", type=int, default=None, help="End index after interval sampling")
    parser.add_argument("--subset_step", type=int, default=1, help="Step after interval sampling")
    parser.add_argument("--skip_metrics", action="store_true", help="Only save renders, depth, opacity and PLY")
    parser.add_argument("--skip_lpips", action="store_true", help="Compute PSNR/SSIM but skip LPIPS")
    parser.add_argument("--save_inputs", action="store_true", help="Also save resized input frames")
    parser.add_argument("--save_raw_depth", action="store_true", help="Also save raw rendered depth as .npy")
    args = parser.parse_args()

    if args.interval < 0:
        args.interval = 10 if args.data_path.lower().endswith(".mp4") else 1

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_inputs:
        os.makedirs(os.path.join(args.output_dir, "inputs"), exist_ok=True)
    if args.save_raw_depth:
        os.makedirs(os.path.join(args.output_dir, "depth_raw"), exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    print(f"Loading Pi3_3DGS TopK model from {args.ckpt}...")
    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
    ).to(device).eval()
    load_result = load_checkpoint(model, args.ckpt, device)
    print(f"Checkpoint load result: {load_result}")

    print(f"Loading RGB frames from {args.data_path}...")
    imgs_cpu, frame_items, orig_hw, target_hw = load_rgb_sequence(
        args.data_path,
        interval=args.interval,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        subset_step=args.subset_step,
        pixel_limit=args.pixel_limit,
    )
    if imgs_cpu.numel() == 0:
        raise RuntimeError("No RGB frames loaded.")

    total_frames = imgs_cpu.shape[0]
    H, W = imgs_cpu.shape[2], imgs_cpu.shape[3]
    print(f"Loaded {total_frames} frames. Original resolution: {orig_hw[0]}x{orig_hw[1]}, model resolution: {H}x{W}")

    if args.skip_metrics:
        metrics = None
    else:
        print("Initializing RGB metrics (PSNR, SSIM, LPIPS)...")
        metrics = init_metrics(device, skip_lpips=args.skip_lpips)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    metrics_rows = []
    camera_pose_rows = []
    intrinsic_rows = []
    manifest_rows = []

    active_chunk_size = max(1, args.chunk_size)
    start_idx = 0
    while start_idx < total_frames:
        end_idx = min(start_idx + active_chunk_size, total_frames)
        current_batch_size = end_idx - start_idx
        print(f"\nProcessing chunk: frames {start_idx} to {end_idx - 1} ({current_batch_size} frames)...")

        imgs_batch = imgs_cpu[start_idx:end_idx].unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                with amp_context():
                    res = model(imgs_batch)
        except RuntimeError as e:
            if is_retryable_inference_error(e):
                maybe_cuda_empty_cache(device)
                gc.collect()
                if active_chunk_size <= 1:
                    print(f"ERROR: inference failed even with chunk_size=1 on chunk {start_idx}-{end_idx - 1}.")
                    raise

                new_chunk_size = max(1, active_chunk_size // 2)
                print(
                    f"Chunk {start_idx}-{end_idx - 1} failed with chunk_size={active_chunk_size}: {e}\n"
                    f"Retrying from frame {start_idx} with chunk_size={new_chunk_size}."
                )
                active_chunk_size = new_chunk_size
                del imgs_batch
                continue
            raise

        gaussians = res["gaussians"]
        pred_c2w = res["camera_poses"]
        pred_K = res["intrinsics"]
        pred_w2c = se3_inverse(pred_c2w)
        current_gaussians = select_render_gaussians(gaussians, batch_index=0)
        # Render all valid Gaussians. Some model variants append pushed/far
        # support Gaussians after the near slice; truncating by num_near leaves
        # visible RGB holes.
        current_num_near = None

        ply_filename = os.path.join(args.output_dir, f"gaussians_chunk_{start_idx:04d}_to_{end_idx - 1:04d}.ply")
        save_ply_binary(current_gaussians, ply_filename)
        print(f"Saved PLY point cloud: {ply_filename}")

        for i in range(current_batch_size):
            global_frame_idx = start_idx + i
            frame_item = frame_items[global_frame_idx]
            frame_name = frame_item["stem"]

            view_w2c = pred_w2c[0:1, i:i + 1]
            view_K = pred_K[0:1, i:i + 1]
            rgb_tensor, depth_tensor, alpha_tensor = render_frame(
                current_gaussians,
                view_w2c,
                view_K,
                H,
                W,
                num_gaussians=current_num_near,
            )

            rgb_out = rgb_tensor[0, 0].permute(2, 0, 1)
            depth_out = depth_tensor[0, 0].permute(2, 0, 1)
            alpha_out = alpha_tensor[0, 0].permute(2, 0, 1)

            metric_values = {"psnr": None, "ssim": None, "lpips": None}
            if metrics is not None:
                with torch.no_grad():
                    metric_values = compute_rgb_metrics(metrics, rgb_out, imgs_batch[0, i])
                metrics_rows.append({
                    "frame_idx": global_frame_idx,
                    "frame_name": frame_name,
                    **metric_values,
                })

            rgb_path = os.path.join(args.output_dir, f"rgb_{global_frame_idx:04d}.png")
            depth_path = os.path.join(args.output_dir, f"depth_{global_frame_idx:04d}.png")
            opacity_path = os.path.join(args.output_dir, f"opacity_heatmap_{global_frame_idx:04d}.png")
            save_image(rgb_out, rgb_path)
            save_depth(depth_out, depth_path)
            save_heatmap(alpha_out, opacity_path)

            raw_depth_path = None
            if args.save_raw_depth:
                raw_depth_path = os.path.join(args.output_dir, "depth_raw", f"depth_{global_frame_idx:04d}.npy")
                np.save(raw_depth_path, depth_out.squeeze(0).detach().cpu().float().numpy())

            input_path = None
            if args.save_inputs:
                input_path = os.path.join(args.output_dir, "inputs", f"input_{global_frame_idx:04d}.png")
                save_image(imgs_batch[0, i], input_path)

            camera_pose_rows.append(pred_c2w[0, i].detach().cpu().float().numpy())
            intrinsic_rows.append(pred_K[0, i].detach().cpu().float().numpy())
            manifest_rows.append({
                "frame_idx": global_frame_idx,
                "frame_name": frame_name,
                "source_index": frame_item["source_index"],
                "source_path": frame_item["path"],
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "opacity_path": opacity_path,
                "raw_depth_path": raw_depth_path,
                "input_path": input_path,
                **metric_values,
            })

            del rgb_tensor, depth_tensor, alpha_tensor, rgb_out, depth_out, alpha_out

        print(f"Saved frames {start_idx} - {end_idx - 1}")

        del gaussians, pred_c2w, pred_w2c, pred_K, current_gaussians, current_num_near, res, imgs_batch
        maybe_cuda_empty_cache(device)
        gc.collect()
        start_idx = end_idx

    camera_npz = os.path.join(args.output_dir, "camera_params.npz")
    np.savez(
        camera_npz,
        camera_poses=np.stack(camera_pose_rows, axis=0),
        intrinsics=np.stack(intrinsic_rows, axis=0),
        frame_indices=np.arange(total_frames, dtype=np.int64),
        frame_names=np.array([x["stem"] for x in frame_items]),
        source_indices=np.array([x["source_index"] for x in frame_items], dtype=np.int64),
    )
    print(f"Camera poses and intrinsics saved to: {camera_npz}")

    manifest_csv = os.path.join(args.output_dir, "render_manifest.csv")
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx", "frame_name", "source_index", "source_path",
                "rgb_path", "depth_path", "opacity_path", "raw_depth_path", "input_path",
                "psnr", "ssim", "lpips",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Render manifest saved to: {manifest_csv}")

    if metrics_rows:
        metrics_csv = os.path.join(args.output_dir, "rgb_metrics_per_frame.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "frame_name", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"Per-frame RGB metrics saved to: {metrics_csv}")

    metrics_file = os.path.join(args.output_dir, "metrics_report.txt")
    avg_psnr, avg_ssim, avg_lpips = write_metrics_report(
        metrics_file,
        args,
        total_frames,
        H,
        W,
        metrics_rows,
        load_result,
    )

    print("\n" + "=" * 48)
    print("Final Render Evaluation Metrics")
    print("=" * 48)
    print(f"Average PSNR:  {avg_psnr:.4f} dB")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print("=" * 48)
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
