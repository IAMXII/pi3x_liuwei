# import torch
# import torch.nn.functional as F
# import argparse
# import os
# import cv2
# import numpy as np
# from gsplat import rasterization
# import sys
#
# # 假设你的项目结构如下
# from pi3.utils.basic import load_images_as_tensor
# # from pi3_3dgs import Pi3_3DGS
# from pi3.models.pi3_3dgs import Pi3_3DGS
#
#
# # ==========================================
# # Helper Functions
# # ==========================================
#
# def se3_inverse(T):
#     """SE(3) Matrix Inverse: C2W -> W2C"""
#     R = T[..., :3, :3]
#     t = T[..., :3, 3:4]
#     R_inv = R.transpose(-1, -2)
#     t_inv = -torch.matmul(R_inv, t)
#     T_inv = torch.zeros_like(T)
#     T_inv[..., :3, :3] = R_inv
#     T_inv[..., :3, 3:4] = t_inv
#     T_inv[..., 3, 3] = 1.0
#     return T_inv
#
#
# def save_image(tensor, path):
#     """保存 RGB [3, H, W]"""
#     img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
#     img_np = np.clip(img_np, 0, 1) * 255
#     img_np = img_np.astype(np.uint8)
#     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(path, img_np)
#
#
# def save_depth(tensor, path):
#     """保存深度图 [1, H, W] (伪彩色)"""
#     depth_np = tensor.squeeze().detach().cpu().numpy()
#     # 鲁棒的归一化用于可视化
#     valid_mask = depth_np > 1e-5
#     if valid_mask.sum() > 0:
#         d_min = np.percentile(depth_np[valid_mask], 2)
#         d_max = np.percentile(depth_np[valid_mask], 98)
#         depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
#         depth_norm = np.clip(depth_norm, 0, 1)
#     else:
#         depth_norm = depth_np
#
#     depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
#     cv2.imwrite(path, depth_colormap)
#
#
# # ==========================================
# # Rendering Logic
# # ==========================================
# def render_frame(gaussians, w2c, K, H, W, num_gaussians=None):
#     means = gaussians["xyz"]
#     quats = gaussians["rotation"]
#     scales = gaussians["scale"]
#     opacities = gaussians["opacity"]
#     colors = gaussians["color"]
#
#     B = means.shape[0]
#
#     # Handle Ragged Batching (num_near)
#     if num_gaussians is not None:
#         if isinstance(num_gaussians, torch.Tensor):
#             max_N = int(num_gaussians.max().item())
#             means = means[:, :max_N]
#             quats = quats[:, :max_N]
#             scales = scales[:, :max_N]
#             opacities = opacities[:, :max_N].clone()
#             colors = colors[:, :max_N]
#
#             range_seq = torch.arange(max_N, device=means.device).expand(B, max_N)
#             valid_mask = range_seq < num_gaussians.unsqueeze(1)
#             opacities[~valid_mask] = 0.0
#         else:
#             limit = int(num_gaussians)
#             means = means[:, :limit]
#             quats = quats[:, :limit]
#             scales = scales[:, :limit]
#             opacities = opacities[:, :limit]
#             colors = colors[:, :limit]
#
#     # Render RGB
#     rgb, _, _ = rasterization(
#         means=means.contiguous(),
#         quats=quats.contiguous(),
#         scales=scales.contiguous(),
#         opacities=opacities.squeeze(-1).contiguous(),
#         colors=colors.contiguous(),
#         viewmats=w2c,
#         Ks=K,
#         width=W, height=H,
#         render_mode='RGB',
#         packed=False
#     )
#
#     # Render Depth
#     depth, _, _ = rasterization(
#         means=means.contiguous(),
#         quats=quats.contiguous(),
#         scales=scales.contiguous(),
#         opacities=opacities.squeeze(-1).contiguous(),
#         colors=colors.contiguous(),
#         viewmats=w2c,
#         Ks=K,
#         width=W, height=H,
#         render_mode='ED',  # Expected Depth
#         packed=False
#     )
#
#     return rgb, depth
#
#
# # ==========================================
# # Main
# # ==========================================
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Pi3_3DGS Inference with Custom Intrinsics")
#
#     parser.add_argument("--data_path", type=str, default='examples/skating.mp4')
#     parser.add_argument("--output_dir", type=str, default='output_render')
#     parser.add_argument("--ckpt", type=str, required=True)
#     parser.add_argument("--interval", type=int, default=-1)
#     parser.add_argument("--device", type=str, default='cuda')
#
#     args = parser.parse_args()
#
#     os.makedirs(args.output_dir, exist_ok=True)
#     device = torch.device(args.device)
#
#     if args.interval < 0:
#         args.interval = 10 if args.data_path.endswith('.mp4') else 1
#
#     # 1. Load Model
#     print(f"Loading Pi3_3DGS model from {args.ckpt}...")
#     model = Pi3_3DGS(
#         pos_type='rope100',
#         decoder_size='large',
#         num_anchors=16384,
#         ckpt=None
#     ).to(device).eval()
#
#     if args.ckpt.endswith('.safetensors'):
#         from safetensors.torch import load_file
#
#         weight = load_file(args.ckpt)
#         model.load_state_dict(weight, strict=False)
#     else:
#         weight = torch.load(args.ckpt, map_location=device, weights_only=False)
#         model.load_state_dict(weight, strict=False)
#
#     # 2. Load Data
#     print(f"Loading data from {args.data_path}...")
#     imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device)
#     imgs_input = imgs.unsqueeze(0)  # [1, N, 3, H, W]
#     B, N, C, H, W = imgs_input.shape
#     print(f"Input shape: {imgs_input.shape} (H={H}, W={W})")
#
#     # 3. Load Custom Intrinsics
#     # 这一步调用上面定义的函数，请确保你已经修改了该函数
#     # K_custom = load_custom_intrinsics(device=device, H=H, W=W)
#     # print(f"Loaded custom intrinsics: \n{K_custom}")
#
#     # 4. Inference
#     print("Running inference...")
#     torch.backends.cuda.matmul.allow_tf32 = True
#     dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#
#     with torch.no_grad():
#         with torch.amp.autocast('cuda', dtype=dtype):
#             res = model(imgs_input)
#
#     # 5. Prepare Rendering
#     print("Preparing rendering data...")
#     gaussians = res['gaussians']
#     pred_c2w = res['camera_poses']
#     pred_w2c = se3_inverse(pred_c2w)
#     num_near = gaussians.get('num_near', None)
#
#     # 扩展内参以匹配 Batch 和 View 数量
#     # K_custom: [1, 3, 3] -> [B, N, 3, 3]
#     Ks = torch.tensor([332.232689, 0.000000, 333.058485],
#                       [0.000000, 332.644823, 240.998586],
#                       [0.000000, 0.000000, 1.000000])
#     # Ks = K_custom.unsqueeze(0).expand(B, N, -1, -1)
#
#     # 6. Render Loop (Single Batch Demo)
#     b_idx = 0
#     current_gaussians = {k: v[b_idx:b_idx + 1] for k, v in gaussians.items() if isinstance(v, torch.Tensor)}
#     current_num_near = num_near[b_idx:b_idx + 1] if num_near is not None else None
#
#     print(f"Start rendering {N} frames...")
#     for i in range(N):
#         view_w2c = pred_w2c[b_idx:b_idx + 1, i]
#         view_K = Ks[b_idx:b_idx + 1, i]
#
#         rgb_tensor, depth_tensor = render_frame(
#             current_gaussians,
#             view_w2c,
#             view_K,
#             H, W,
#             num_gaussians=current_num_near
#         )
#
#         rgb_out = rgb_tensor[0].permute(2, 0, 1)
#         depth_out = depth_tensor[0].permute(2, 0, 1)
#
#         rgb_path = os.path.join(args.output_dir, f"rgb_{i:04d}.png")
#         depth_path = os.path.join(args.output_dir, f"depth_{i:04d}.png")
#
#         save_image(rgb_out, rgb_path)
#         save_depth(depth_out, depth_path)
#
#         print(f"Saved Frame {i}")
#
#     print("All done!")

import torch
import torch.nn.functional as F
import argparse
import os
import cv2
import numpy as np
from gsplat import rasterization
import sys
import gc  # 引入垃圾回收

# 假设你的项目结构如下
from pi3.utils.basic import load_images_as_tensor
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
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_np)


def save_depth(tensor, path):
    """保存深度图 [1, H, W] (伪彩色)"""
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

    depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, depth_colormap)


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
    # 新增 chunk_size 参数，默认 30，可根据显存大小调整
    parser.add_argument("--chunk_size", type=int, default=30, help="Number of frames to process at once to avoid OOM")

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
        num_anchors=16384,
        ckpt=None
    ).to(device).eval()

    if args.ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file

        weight = load_file(args.ckpt)
        model.load_state_dict(weight, strict=False)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)

    # 2. Load Data (Keep on CPU first!)
    print(f"Loading data from {args.data_path}...")
    # 修改：不在这里调用 .to(device)，节省显存
    imgs_cpu = load_images_as_tensor(args.data_path, interval=args.interval)

    total_frames = imgs_cpu.shape[0]
    H, W = imgs_cpu.shape[2], imgs_cpu.shape[3]  # [N, C, H, W] or [N, H, W, C] check util?
    # load_images_as_tensor 通常返回 [N, 3, H, W]

    print(f"Total frames: {total_frames}, Resolution: {H}x{W}")
    print(f"Processing in chunks of {args.chunk_size}...")

    # 3. Setup Intrinsics (Fixed Syntax Error)
    # 修改：增加了最外层的方括号 []，使其成为合法的 3x3 矩阵结构
    K_base = torch.tensor([
        [332.232689, 0.000000, 333.058485],
        [0.000000, 332.644823, 240.998586],
        [0.000000, 0.000000, 1.000000]
    ], device=device)

    # 4. Inference & Render Loop
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    for start_idx in range(0, total_frames, args.chunk_size):
        end_idx = min(start_idx + args.chunk_size, total_frames)
        current_batch_size = end_idx - start_idx

        print(f"\nProcessing chunk: frames {start_idx} to {end_idx} ({current_batch_size} frames)...")

        # Prepare Batch on GPU
        # [N_chunk, 3, H, W] -> [1, N_chunk, 3, H, W]
        imgs_batch = imgs_cpu[start_idx:end_idx].unsqueeze(0).to(device)

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
        pred_w2c = se3_inverse(pred_c2w)
        num_near = gaussians.get('num_near', None)

        # Expand Intrinsics for this batch
        # [1, 3, 3] -> [1, N_chunk, 3, 3]
        Ks = K_base.unsqueeze(0).unsqueeze(0).expand(1, current_batch_size, -1, -1)

        # Render Loop for current chunk
        b_idx = 0  # Batch size is always 1 in this loop
        current_gaussians = {k: v[b_idx:b_idx + 1] for k, v in gaussians.items() if isinstance(v, torch.Tensor)}
        current_num_near = num_near[b_idx:b_idx + 1] if num_near is not None else None

        for i in range(current_batch_size):
            global_frame_idx = start_idx + i

            # === 修改 1: 使用 i:i+1 切片，保持 [1, 1, 4, 4] 的 4 维结构 ===
            # view_w2c shape: [1, 1, 4, 4]
            # view_K shape: [1, 1, 3, 3]
            view_w2c = pred_w2c[b_idx:b_idx + 1, i:i + 1]
            view_K = Ks[b_idx:b_idx + 1, i:i + 1]

            rgb_tensor, depth_tensor = render_frame(
                current_gaussians,
                view_w2c,
                view_K,
                H, W,
                num_gaussians=current_num_near
            )

            # === 修改 2: 输出维度变为 [Batch, Cam, H, W, C] -> [1, 1, H, W, 3] ===
            # 我们需要取 [0, 0] 来获得单张图像 [H, W, 3]
            rgb_out = rgb_tensor[0, 0].permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            depth_out = depth_tensor[0, 0].permute(2, 0, 1)

            rgb_path = os.path.join(args.output_dir, f"rgb_{global_frame_idx:04d}.png")
            depth_path = os.path.join(args.output_dir, f"depth_{global_frame_idx:04d}.png")

            save_image(rgb_out, rgb_path)
            save_depth(depth_out, depth_path)


        print(f"Saved frames {start_idx} - {end_idx - 1}")

        # === 关键步骤：清理显存 ===
        del res, gaussians, pred_c2w, pred_w2c, imgs_batch, current_gaussians
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll done! Rendered all frames.")