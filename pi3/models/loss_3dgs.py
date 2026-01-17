import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rasterization import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim


def se3_inverse(T):
    """SE(3) 矩阵求逆: T = [R|t] -> T^-1 = [R^T | -R^T @ t]"""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:4] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def homogenize_points(pts):
    return F.pad(pts, (0, 1), value=1.0)


class Pi3LossGS(nn.Module):
    def __init__(
            self,
            lambda_rgb=1.0,
            lambda_ssim=0.2,
            lambda_depth=0.1
    ):
        super().__init__()
        # 只保留 GS 相关的权重
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth

    def prepare_gt(self, gt):
        """处理输入的 C2W 并转换为渲染所需的 W2C 和归一化尺度"""
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses_c2w = torch.stack([view['camera_pose'] for view in gt], dim=1)  # [B, N, 4, 4]
        gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
        gt_imgs = torch.stack([view['img'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # 1. 坐标系对齐：以第一帧相机中心为原点
        w2c_target = se3_inverse(poses_c2w[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses_rel_c2w = torch.einsum('bij, bnjk -> bnik', w2c_target, poses_c2w)

        # 2. 尺度归一化因子计算
        valid_batch = masks.sum([-1, -2, -3]) > 0
        norm_factor = torch.ones(B, device=gt_pts.device)
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            batch_factors = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)
            norm_factor[valid_batch] = batch_factors

            # 缩放平移和深度
            poses_rel_c2w[valid_batch, ..., :3, 3] /= batch_factors[..., None, None]
            gt_depths[valid_batch] /= batch_factors[..., None, None, None]

        # 3. 关键：将相对 C2W 转为渲染用的 W2C (viewmats)
        render_w2c = se3_inverse(poses_rel_c2w)

        return dict(
            imgs=gt_imgs,
            render_w2c=render_w2c,  # 用于 gsplat
            render_ks=gt_ks,
            gt_depths=gt_depths,
            norm_factor=norm_factor
        )

    def _render_gs(self, gaussians, w2c, ks, H, W, num_gaussians=None):
        """调用 gsplat 进行渲染"""
        # 注意：此处 w2c 需要是 [M, 4, 4]，由 render_w2c reshape 得到
        return rasterization(
            means3d=gaussians["xyz"][:, :num_gaussians].contiguous(),
            quats=gaussians["rotation"][:, :num_gaussians].contiguous(),
            scales=gaussians["scale"][:, :num_gaussians].contiguous(),
            opacities=gaussians["opacity"][:, :num_gaussians].squeeze(-1).contiguous(),
            colors=gaussians["color"][:, :num_gaussians].contiguous(),
            viewmats=w2c,
            Ks=ks,
            width=W, height=H,
            packed=False
        )

    def forward(self, pred, gt_raw):
        # 1. 准备 GT（处理 C2W -> W2C 和 Scale）
        gt = self.prepare_gt(gt_raw)

        B, N, C, H, W = gt['imgs'].shape
        render_w2c = gt['render_w2c'].reshape(B * N, 4, 4)
        render_ks = gt['render_ks'].reshape(B * N, 3, 3)

        gauss = pred['gaussians']
        num_near = gauss.get('num_near', None)

        # 2. 渲染 (Path A: RGB, Path B: Depth)
        rgb_full, _, _ = self._render_gs(gauss, render_w2c, render_ks, H, W, num_gaussians=None)

        # 只在有近景定义时计算深度
        depth_near = None
        if num_near is not None:
            _, _, info_near = self._render_gs(gauss, render_w2c, render_ks, H, W, num_gaussians=num_near)
            depth_near = info_near["depths"]  # [B*N, H, W, 1]

        # 3. 计算损失
        # RGB & SSIM
        rgb_full = rgb_full.reshape(B * N, H, W, 3).permute(0, 3, 1, 2)
        gt_imgs = gt['imgs'].reshape(B * N, 3, H, W)
        loss_rgb = F.l1_loss(rgb_full, gt_imgs)
        loss_ssim = 1.0 - ssim(rgb_full, gt_imgs, data_range=1.0)

        # Depth Loss
        loss_depth = torch.tensor(0.0, device=rgb_full.device)
        if depth_near is not None:
            gt_depths = gt['gt_depths'].reshape(B * N, H, W, 1)
            depth_mask = (gt_depths > 0).float()
            loss_depth = F.l1_loss(depth_near * depth_mask, gt_depths * depth_mask)

        # 4. 汇总
        final_loss = (
                self.lambda_rgb * loss_rgb +
                self.lambda_ssim * loss_ssim +
                self.lambda_depth * loss_depth
        )

        details = {
            "loss_gs_rgb": loss_rgb.item(),
            "loss_gs_ssim": loss_ssim.item(),
            "loss_gs_depth": loss_depth.item(),
            "total_loss": final_loss.item()
        }

        return final_loss, details