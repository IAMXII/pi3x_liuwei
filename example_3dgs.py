import torch
import torch.nn.functional as F
import argparse
import os
import cv2
import numpy as np
from gsplat import rasterization
import sys
import gc

# --- 【新增】：引入 torchmetrics 计算评测指标 ---
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    LearnedPerceptualImagePatchSimilarity

# 假设你的项目结构如下
from pi3.utils.basic import load_images_as_tensor, load_images_and_intrinsics
from pi3.models.pi3_3dgs_1 import Pi3_3DGS


# ==========================================
# Helper Functions
# ==========================================
def save_heatmap(tensor, path):
    alpha_np = tensor.squeeze().detach().cpu().numpy()
    alpha_np = np.clip(alpha_np, 0, 1)
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap_color)


def save_ply_binary(gaussians, path):
    xyz = gaussians["xyz"].detach().cpu().float().numpy().squeeze()
    rot = gaussians["rotation"].detach().cpu().float().numpy().squeeze()
    scale = gaussians["scale"].detach().cpu().float().numpy().squeeze()
    opacity = gaussians["opacity"].detach().cpu().float().numpy().squeeze()
    color = gaussians["color"].detach().cpu().float().numpy().squeeze()

    scale_ply = np.log(np.clip(scale, 1e-10, None))
    opacity_clipped = np.clip(opacity, 1e-6, 1 - 1e-6)
    opacity_ply = np.log(opacity_clipped / (1 - opacity_clipped))

    SH_C0 = 0.28209479177387814
    f_dc = (color - 0.5) / SH_C0
    normals = np.zeros_like(xyz)

    attributes = np.concatenate((
        xyz, normals, f_dc, opacity_ply[..., np.newaxis], scale_ply, rot
    ), axis=-1).astype(np.float32)

    with open(path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n".encode('utf-8'))
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
        depth_norm = depth_np
    depth_gray = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(path, depth_gray)


# ==========================================
# Rendering Logic
# ==========================================
def render_frame(gaussians, w2c, K, H, W, num_gaussians=None):
    means = gaussians["xyz"]
    quats = gaussians["rotation"]
    scales = gaussians["scale"]
    opacities = gaussians["opacity"]
    colors = gaussians["color"]

    B = means.shape[0]

    if num_gaussians is not None:
        if isinstance(num_gaussians, torch.Tensor):
            max_N = int(num_gaussians.max().item())
            means = means[:, :max_N]
            quats = quats[:, :max_N]
            scales = scales[:, :max_N]
            opacities = opacities[:, :max_N].clone()
            colors = colors[:, :max_N]

            if "conf" in gaussians:
                gaussians["conf"] = gaussians["conf"][:, :max_N]

            range_seq = torch.arange(max_N, device=means.device).expand(B, max_N)
            valid_mask = range_seq < num_gaussians.unsqueeze(1)
            opacities[~valid_mask] = 0.0
        else:
            limit = int(num_gaussians)
            means = means[:, :limit]
            quats = quats[:, :limit]
            scales = scales[:, :limit]
            opacities = opacities[:, :limit]
            colors = colors[:, :limit]
            if "conf" in gaussians:
                gaussians["conf"] = gaussians["conf"][:, :limit]

    rgb, alpha, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W, height=H,
        render_mode='RGB',
        packed=False
    )

    opacities_depth = opacities.clone()
    if "conf" in gaussians:
        conf_prob = torch.sigmoid(gaussians["conf"])
        opacities_depth = torch.where(
            conf_prob < 0.1,
            torch.zeros_like(opacities_depth),
            opacities_depth
        )

    depth, _, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities_depth.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W, height=H,
        render_mode='ED',
        packed=False
    )

    return rgb, depth, alpha


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pi3_3DGS Inference with Predicted Intrinsics")

    parser.add_argument("--data_path", type=str, default='examples/skating.mp4')
    parser.add_argument("--output_dir", type=str, default='output_render_cp')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of frames to process at once to avoid OOM")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1

    # 1. Load Model
    print(f"Loading Pi3_3DGS model from {args.ckpt}...")
    model = Pi3_3DGS(
        pos_type='rope100',
        decoder_size='large',
        ckpt=None,
        debug_mem=False
    ).to(device).eval()

    if args.ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file

        weight = load_file(args.ckpt)
        model.load_state_dict(weight, strict=False)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)

    print(f"Loading data from {args.data_path}...")
    imgs_cpu = load_images_as_tensor(args.data_path, interval=args.interval)
    imgs_cpu = imgs_cpu[0:150:3, ...]
    total_frames = imgs_cpu.shape[0]
    H, W = imgs_cpu.shape[2], imgs_cpu.shape[3]

    print(f"Total frames: {total_frames}, Resolution: {H}x{W}")
    print(f"Processing in chunks of {args.chunk_size}...")

    # --- 【新增】：初始化计算指标 (使用 3DGS 标准的 VGG 网络测 LPIPS) ---
    print("Initializing metrics (PSNR, SSIM, LPIPS)...")
    metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # normalize=True 表示输入图像的数据范围在 [0, 1] 之间，库会在内部将其映射到 [-1, 1]
    metric_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    # 记录每个图像的评测结果
    all_metrics = {"psnr": [], "ssim": [], "lpips": []}
    # -------------------------------------------------------------

    # 4. Inference & Render Loop
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    for start_idx in range(0, total_frames, args.chunk_size):
        end_idx = min(start_idx + args.chunk_size, total_frames)
        current_batch_size = end_idx - start_idx

        print(f"\nProcessing chunk: frames {start_idx} to {end_idx} ({current_batch_size} frames)...")

        # Prepare Batch on GPU
        imgs_batch = imgs_cpu[start_idx:end_idx].unsqueeze(0).to(device)
        N_chunk = imgs_batch.shape[1]

        # Inference
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=dtype):
                    res = model(imgs_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"ERROR: Out of Memory during inference on chunk {start_idx}-{end_idx}.")
                print("Try reducing --chunk_size (e.g., --chunk_size 10)")
                sys.exit(1)
            else:
                raise e

        # Prepare Rendering Data
        gaussians = res['gaussians']
        pred_c2w = res['camera_poses']
        pred_K = res['intrinsics']

        num_near = gaussians.get('num_near', None)
        torch.cuda.empty_cache()

        pred_w2c = se3_inverse(pred_c2w)
        Ks = pred_K

        b_idx = 0
        current_gaussians = {k: v[b_idx:b_idx + 1] for k, v in gaussians.items() if isinstance(v, torch.Tensor)}
        current_num_near = num_near[b_idx:b_idx + 1] if num_near is not None else None

        ply_filename = os.path.join(args.output_dir, f"gaussians_chunk_{start_idx:04d}_to_{end_idx - 1:04d}.ply")
        save_ply_binary(current_gaussians, ply_filename)
        print(f"Saved PLY point cloud: {ply_filename}")

        for i in range(current_batch_size):
            global_frame_idx = start_idx + i
            view_w2c = pred_w2c[b_idx:b_idx + 1, i:i + 1]
            view_K = Ks[b_idx:b_idx + 1, i:i + 1]

            rgb_tensor, depth_tensor, alpha_tensor = render_frame(
                current_gaussians,
                view_w2c,
                view_K,
                H, W,
                num_gaussians=None
            )

            rgb_out = rgb_tensor[0, 0].permute(2, 0, 1)
            depth_out = depth_tensor[0, 0].permute(2, 0, 1)
            alpha_out = alpha_tensor[0, 0].permute(2, 0, 1)

            # --- 【新增】：计算当前帧的 PSNR, SSIM, LPIPS ---
            # 提取 Ground Truth 并转为 [1, C, H, W] 的浮点数以匹配预测值的格式
            gt_img = imgs_batch[0, i].to(device)

            with torch.no_grad():
                pred_for_metric = rgb_out.unsqueeze(0).float()
                gt_for_metric = gt_img.unsqueeze(0).float()

                # 对齐边界情况，保证范围卡在 [0, 1] 内计算更为准确
                pred_for_metric = torch.clamp(pred_for_metric, 0.0, 1.0)

                psnr_val = metric_psnr(pred_for_metric, gt_for_metric).item()
                ssim_val = metric_ssim(pred_for_metric, gt_for_metric).item()
                lpips_val = metric_lpips(pred_for_metric, gt_for_metric).item()

                all_metrics["psnr"].append(psnr_val)
                all_metrics["ssim"].append(ssim_val)
                all_metrics["lpips"].append(lpips_val)
            # --------------------------------------------------

            rgb_path = os.path.join(args.output_dir, f"rgb_{global_frame_idx:04d}.png")
            depth_path = os.path.join(args.output_dir, f"depth_{global_frame_idx:04d}.png")
            opacity_path = os.path.join(args.output_dir, f"opacity_heatmap_{global_frame_idx:04d}.png")

            save_image(rgb_out, rgb_path)
            save_depth(depth_out, depth_path)
            save_heatmap(alpha_out, opacity_path)

            del rgb_tensor, depth_tensor, alpha_tensor, rgb_out, depth_out, alpha_out

        print(f"Saved frames {start_idx} - {end_idx - 1}")

        del gaussians, pred_c2w, pred_w2c, current_gaussians
        torch.cuda.empty_cache()
        gc.collect()

    # --- 【新增】：运行结束后汇总并保存评测结果 ---
    print("\n" + "=" * 40)
    print("Final Render Evaluation Metrics")
    print("=" * 40)

    avg_psnr = np.mean(all_metrics["psnr"])
    avg_ssim = np.mean(all_metrics["ssim"])
    avg_lpips = np.mean(all_metrics["lpips"])

    print(f"Average PSNR:  {avg_psnr:.4f} dB")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print("=" * 40)

    # 将结果写入 output_dir 方便后续分析对比
    metrics_file = os.path.join(args.output_dir, "metrics_report.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Average PSNR:  {avg_psnr:.4f}\n")
        f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
    print(f"Metrics saved to: {metrics_file}")
    # --------------------------------------------------