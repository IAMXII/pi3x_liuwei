
import torch
import torch.nn.functional as F
import argparse
import os
import cv2
import numpy as np
from gsplat import rasterization
import sys
import gc

# 假设你的项目结构如下
from pi3.utils.basic import load_images_as_tensor,load_images_and_intrinsics
from pi3.models.pi3_3dgs import Pi3_3DGS


# ==========================================
# Helper Functions
# ==========================================

def se3_inverse(T):
    """SE(3) Matrix Inverse: C2W -> W2C"""
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
    """保存 RGB [3, H, W]"""
    # 纯净的维度转换，不需要任何反归一化，因为模型已经通过 sigmoid 输出了 [0, 1] 的值
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()

    # 限制在 [0, 1] 并转为 uint8
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)

    # 转为 OpenCV 的 BGR 格式保存
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_np)


def save_depth(tensor, path):
    """保存深度图 [1, H, W] (灰度图)"""
    depth_np = tensor.squeeze().detach().cpu().numpy()

    # 鲁棒的归一化用于可视化
    valid_mask = depth_np > 1e-5
    if valid_mask.sum() > 0:
        d_min = np.percentile(depth_np[valid_mask], 2)
        d_max = np.percentile(depth_np[valid_mask], 98)
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
        depth_norm = np.clip(depth_norm, 0, 1)
    else:
        depth_norm = depth_np

    # 直接保存为灰度图，移除伪彩色 (ColorMap)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(path, depth_gray)


# ==========================================
# Rendering Logic
# ==========================================
def render_frame(gaussians, w2c, K, H, W, num_gaussians=None):
    means = gaussians["xyz"]
    # 规范化四元数是个好习惯，防止渲染出现黑斑
    quats = gaussians["rotation"]
    scales = gaussians["scale"]
    opacities = gaussians["opacity"]

    # === 核心修复：应用 Sigmoid 激活颜色 ===
    # 将模型输出的无界 logits 映射到完美的 [0, 1] 色彩空间
    colors = gaussians["color"]

    B = means.shape[0]

    # Handle Ragged Batching (num_near)
    if num_gaussians is not None:
        if isinstance(num_gaussians, torch.Tensor):
            max_N = int(num_gaussians.max().item())
            means = means[:, :max_N]
            quats = quats[:, :max_N]
            scales = scales[:, :max_N]
            opacities = opacities[:, :max_N].clone()
            colors = colors[:, :max_N]

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

    # Render RGB
    rgb, _, _ = rasterization(
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

    # Render Depth
    depth, _, _ = rasterization(
        means=means.contiguous().float(),
        quats=quats.contiguous().float(),
        scales=scales.contiguous().float(),
        opacities=opacities.squeeze(-1).contiguous().float(),
        colors=colors.contiguous().float(),
        viewmats=w2c.float(),
        Ks=K.float(),
        width=W, height=H,
        render_mode='ED',  # Expected Depth
        packed=False
    )

    return rgb, depth


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pi3_3DGS Inference with Custom Intrinsics")

    parser.add_argument("--data_path", type=str, default='examples/skating.mp4')
    parser.add_argument("--output_dir", type=str, default='output_render')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of frames to process at once to avoid OOM")

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
    K_base = torch.tensor([
        [332.232689, 0.000000, 333.058485],
        [0.000000, 332.644823, 240.998586],
        [0.000000, 0.000000, 1.000000]
    ], device=device)
    # 2. Load Data (Keep on CPU first!)
    print(f"Loading data from {args.data_path}...")
    imgs_cpu,K_base = load_images_and_intrinsics(args.data_path, K_base, interval=args.interval)

    total_frames = imgs_cpu.shape[0]
    H, W = imgs_cpu.shape[2], imgs_cpu.shape[3]

    print(f"Total frames: {total_frames}, Resolution: {H}x{W}")
    print(f"Processing in chunks of {args.chunk_size}...")

    # 3. Setup Intrinsics


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
        K_batch = K_base.unsqueeze(0).unsqueeze(0).expand(1, N_chunk, 3, 3).to(device)

        # Inference
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=dtype):
                    res = model(imgs_batch, K_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"ERROR: Out of Memory during inference on chunk {start_idx}-{end_idx}.")
                print("Try reducing --chunk_size (e.g., --chunk_size 10)")
                sys.exit(1)
            else:
                raise e

        # Prepare Rendering Data
        # 我们只保留渲染真正需要的变量
        gaussians = res['gaussians']
        pred_c2w = res['camera_poses']

        # 【优化核心：立即丢弃推理阶段无用的中间张量和输入，释放大块显存】
        num_near = gaussians.get('num_near', None)
        del res['points'], res['conf'], res['local_points'], res
        del imgs_batch  # 渲染阶段不再需要输入图像
        torch.cuda.empty_cache()  # 强制执行显存碎片整理

        pred_w2c = se3_inverse(pred_c2w)

        # Expand Intrinsics for this batch
        Ks = K_base.unsqueeze(0).unsqueeze(0).expand(1, current_batch_size, -1, -1)

        # Render Loop for current chunk
        b_idx = 0
        current_gaussians = {k: v[b_idx:b_idx + 1] for k, v in gaussians.items() if isinstance(v, torch.Tensor)}
        current_num_near = num_near[b_idx:b_idx + 1] if num_near is not None else None

        for i in range(current_batch_size):
            global_frame_idx = start_idx + i

            view_w2c = pred_w2c[b_idx:b_idx + 1, i:i + 1]
            view_K = Ks[b_idx:b_idx + 1, i:i + 1]

            rgb_tensor, depth_tensor = render_frame(
                current_gaussians,
                view_w2c,
                view_K,
                H, W,
                num_gaussians=current_num_near
            )

            rgb_out = rgb_tensor[0, 0].permute(2, 0, 1)
            depth_out = depth_tensor[0, 0].permute(2, 0, 1)

            rgb_path = os.path.join(args.output_dir, f"rgb_{global_frame_idx:04d}.png")
            depth_path = os.path.join(args.output_dir, f"depth_{global_frame_idx:04d}.png")

            save_image(rgb_out, rgb_path)
            save_depth(depth_out, depth_path)

            # 单帧渲染结束后及时清理
            del rgb_tensor, depth_tensor, rgb_out, depth_out

        print(f"Saved frames {start_idx} - {end_idx - 1}")

        # 清理当前 chunk 显存
        del gaussians, pred_c2w, pred_w2c, current_gaussians, K_batch
        torch.cuda.empty_cache()
        gc.collect()