import torch
import torch.nn.functional as F
import argparse
import os
import cv2
import numpy as np
from gsplat import rasterization
import math

# 假设你的项目结构如下，请根据实际情况调整引用
from pi3.utils.basic import load_images_as_tensor
from pi3_3dgs import Pi3_3DGS  # 引用你上传的 pi3_3dgs.py 中定义的模型


# ==========================================
# Helper Functions (Ref: loss_3dgs.py)
# ==========================================

def se3_inverse(T):
    """SE(3) Matrix Inverse: 用于将 C2W 转换为 W2C"""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:4] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def get_default_intrinsics(H, W, fov_degrees=60.0, device='cuda'):
    """
    由于 Pi3_3DGS 预测的是相对位姿，通常我们需要一个假设的内参来进行渲染。
    这里根据图像尺寸和默认 FOV 生成内参矩阵 K。
    """
    focal = 0.5 * W / math.tan(0.5 * math.radians(fov_degrees))
    fx = focal
    fy = focal
    cx = W / 2.0
    cy = H / 2.0

    K = torch.eye(3, device=device)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K.unsqueeze(0)  # [1, 3, 3]


def save_image(tensor, path):
    """保存 RGB Tensor [3, H, W] 到文件"""
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_np)


def save_depth(tensor, path):
    """保存深度图 Tensor [1, H, W] 到文件 (伪彩色)"""
    depth_np = tensor.squeeze().detach().cpu().numpy()

    # 简单的归一化以便可视化
    # 注意：为了更好看清细节，这里采用了百分比截断
    d_min = np.percentile(depth_np, 2)
    d_max = np.percentile(depth_np, 98)
    depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, depth_colormap)


# ==========================================
# Rendering Logic (Ref: loss_3dgs.py -> _render_gs)
# ==========================================
def render_frame(gaussians, w2c, K, H, W, num_gaussians=None):
    """
    执行单帧或多帧渲染
    """
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
            # 截取需要的最大数量
            means = means[:, :max_N]
            quats = quats[:, :max_N]
            scales = scales[:, :max_N]
            opacities = opacities[:, :max_N].clone()
            colors = colors[:, :max_N]

            # 创建 Mask 将多余的 Gaussian 透明度设为 0
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

    # 渲染 RGB
    rgb, _, _ = rasterization(
        means=means.contiguous(),
        quats=quats.contiguous(),
        scales=scales.contiguous(),
        opacities=opacities.squeeze(-1).contiguous(),
        colors=colors.contiguous(),
        viewmats=w2c,  # gsplat 需要 W2C
        Ks=K,
        width=W, height=H,
        render_mode='RGB',
        packed=False
    )

    # 渲染 Depth (使用 Expected Depth 模式 'ED' 或类似，取决于 gsplat 版本)
    # 注意：某些旧版 gsplat 可能不支持直接 'ED'，如果报错请改为仅仅渲染 Alpha 也就是 accum
    depth, _, _ = rasterization(
        means=means.contiguous(),
        quats=quats.contiguous(),
        scales=scales.contiguous(),
        opacities=opacities.squeeze(-1).contiguous(),
        colors=colors.contiguous(),  # depth 模式下 color 其实不重要
        viewmats=w2c,
        Ks=K,
        width=W, height=H,
        render_mode='ED',  # Expected Depth
        packed=False
    )

    return rgb, depth


# ==========================================
# Main Inference Script
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with Pi3_3DGS and render results.")

    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to input video or images directory")
    parser.add_argument("--output_dir", type=str, default='output_render',
                        help="Directory to save rendered images")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the Pi3_3DGS model checkpoint (.safetensors or .pth)")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Sampling interval")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--fov", type=float, default=60.0,
                        help="Assumed Field of View for rendering intrinsics")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Interval logic
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1

    # 1. Load Model
    print(f"Loading Pi3_3DGS model from {args.ckpt}...")
    # 注意：这里初始化参数需要根据你训练时的配置进行调整
    # 比如 pos_type, decoder_size 等。这里使用了你提供的 light 版本的参数。
    model = Pi3_3DGS(
        pos_type='rope100',
        decoder_size='large',
        num_anchors=16384,
        ckpt=None  # 手动加载
    ).to(device).eval()

    # 加载权重
    if args.ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file

        weight = load_file(args.ckpt)
        model.load_state_dict(weight, strict=False)
    else:
        weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)

    # 2. Load Data
    print(f"Loading data from {args.data_path}...")
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device)
    # imgs shape: [N_views, 3, H, W]

    # 构造 Batch (Batch Size = 1)
    # Pi3_3DGS 输入期望是 [B, N, 3, H, W]
    imgs_input = imgs.unsqueeze(0)

    B, N, C, H, W = imgs_input.shape
    print(f"Input shape: {imgs_input.shape}")

    # 3. Inference
    print("Running inference...")
    # 开启 TF32 (参考你的 pi3_3dgs.py)
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs_input)

    # 4. Prepare for Rendering
    print("Preparing rendering data...")

    gaussians = res['gaussians']
    pred_c2w = res['camera_poses']  # [B, N, 4, 4]

    # 将预测的 Camera-to-World (C2W) 转换为 World-to-Camera (W2C)
    # gsplat 需要 W2C 矩阵
    pred_w2c = se3_inverse(pred_c2w)

    # 获取有效的高斯数量 (Ragged Batching)
    num_near = gaussians.get('num_near', None)

    # 准备内参 (Intrinsics)
    # 这里的 K 需要扩展到 [B, N, 3, 3]
    K_base = get_default_intrinsics(H, W, fov_degrees=args.fov, device=device)
    Ks = K_base.unsqueeze(0).expand(B, N, -1, -1)  # [B, N, 3, 3]

    # 5. Render Loop
    # 既然 gsplat 支持 batch 渲染，我们可以尝试一次性渲染，或者逐帧渲染以节省显存
    # 为了保险起见和方便保存，这里演示逐帧(View)渲染

    # 如果 batch size > 1，这里只取第一个 batch
    b_idx = 0

    # 提取当前 Batch 的高斯属性
    current_gaussians = {
        k: v[b_idx:b_idx + 1] for k, v in gaussians.items() if isinstance(v, torch.Tensor)
    }
    # num_near 也是 tensor [B]
    current_num_near = num_near[b_idx:b_idx + 1] if num_near is not None else None

    print(f"Start rendering {N} frames...")

    for i in range(N):
        # 取第 i 个视角的位姿 [1, 4, 4]
        view_w2c = pred_w2c[b_idx:b_idx + 1, i]
        view_K = Ks[b_idx:b_idx + 1, i]

        # 渲染
        # 注意：这里我们传入整个高斯云，gsplat 会根据 view_w2c 进行投影
        rgb_tensor, depth_tensor = render_frame(
            current_gaussians,
            view_w2c,
            view_K,
            H, W,
            num_gaussians=current_num_near
        )

        # rgb_tensor: [1, H, W, 3] -> [3, H, W] for saving
        rgb_out = rgb_tensor[0].permute(2, 0, 1)

        # depth_tensor: [1, H, W, 1] -> [1, H, W] for saving
        depth_out = depth_tensor[0].permute(2, 0, 1)

        # 保存路径
        rgb_path = os.path.join(args.output_dir, f"rgb_{i:04d}.png")
        depth_path = os.path.join(args.output_dir, f"depth_{i:04d}.png")

        save_image(rgb_out, rgb_path)
        save_depth(depth_out, depth_path)

        print(f"Saved Frame {i}: {rgb_path}")

    print("All done!")