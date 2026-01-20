# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from gsplat import rasterization
# from torchmetrics.functional import structural_similarity_index_measure as ssim


# def se3_inverse(T):
#     """SE(3) 矩阵求逆"""
#     R = T[..., :3, :3]
#     t = T[..., :3, 3:4]
#     R_inv = R.transpose(-1, -2)
#     t_inv = -torch.matmul(R_inv, t)
#     T_inv = torch.zeros_like(T)
#     T_inv[..., :3, :3] = R_inv
#     T_inv[..., :3, 3:4] = t_inv
#     T_inv[..., 3, 3] = 1.0
#     return T_inv


# def homogenize_points(pts):
#     return F.pad(pts, (0, 1), value=1.0)


# class Pi3LossGS(nn.Module):
#     def __init__(
#             self,
#             lambda_rgb=1.0,
#             lambda_ssim=0.2,
#             lambda_depth=0.1,
#             lambda_repulsion=0.01,
#             lambda_sparsity=0.005,
#             lambda_consist=0.05
#     ):
#         super().__init__()
#         self.lambda_rgb = lambda_rgb
#         self.lambda_ssim = lambda_ssim
#         self.lambda_depth = lambda_depth
#         self.lambda_repulsion = lambda_repulsion
#         self.lambda_sparsity = lambda_sparsity
#         self.lambda_consist = lambda_consist

#     def prepare_gt(self, gt):
#         """处理输入的 C2W 并转换为渲染所需的 W2C 和归一化尺度"""
#         gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
#         masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
#         poses_c2w = torch.stack([view['camera_pose'] for view in gt], dim=1)  # [B, N, 4, 4]
#         gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
#         gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
#         gt_imgs = torch.stack([view['img'] for view in gt], dim=1)

#         B, N, H, W, _ = gt_pts.shape

#         # 1. 坐标系对齐
#         w2c_target = se3_inverse(poses_c2w[:, 0])
#         gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
#         poses_rel_c2w = torch.einsum('bij, bnjk -> bnik', w2c_target, poses_c2w)

#         # 2. 尺度归一化
#         valid_batch = masks.sum([-1, -2, -3]) > 0
#         norm_factor = torch.ones(B, device=gt_pts.device)
#         if valid_batch.sum() > 0:
#             B_ = valid_batch.sum()
#             all_pts = gt_pts[valid_batch].clone()
#             all_pts[~masks[valid_batch]] = 0
#             all_pts = all_pts.reshape(B_, N, -1, 3)
#             all_dis = all_pts.norm(dim=-1)
#             batch_factors = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)
#             norm_factor[valid_batch] = batch_factors

#             poses_rel_c2w[valid_batch, ..., :3, 3] /= batch_factors[..., None, None]
#             gt_depths[valid_batch] /= batch_factors[..., None, None, None]

#         render_w2c = se3_inverse(poses_rel_c2w)

#         return dict(
#             imgs=gt_imgs,
#             render_w2c=render_w2c, # [B, N, 4, 4]
#             render_ks=gt_ks,       # [B, N, 3, 3]
#             gt_depths=gt_depths,
#             norm_factor=norm_factor
#         )

#     def _render_gs(self, gaussians, w2c, ks, H, W, num_gaussians=None, render_mode='RGB'):
#         """
#         gsplat v1.0.0+ 渲染接口
#         支持变长 num_gaussians (Ragged Batch) 和多种渲染模式
#         """
#         # 1. 获取所有属性
#         means = gaussians["xyz"]
#         quats = gaussians["rotation"]
#         scales = gaussians["scale"]
#         opacities = gaussians["opacity"]
#         colors = gaussians["color"]
        
#         B = means.shape[0]

#         # 2. 处理 num_gaussians (Ragged Batch Masking)
#         if num_gaussians is not None:
#             if isinstance(num_gaussians, torch.Tensor):
#                 max_N = int(num_gaussians.max().item())
                
#                 # 切片到最大长度
#                 means = means[:, :max_N]
#                 quats = quats[:, :max_N]
#                 scales = scales[:, :max_N]
#                 # [Clone] 重要：这里必须 clone，因为下面要进行 in-place 修改
#                 opacities = opacities[:, :max_N].clone() 
#                 colors = colors[:, :max_N]
                
#                 # 构建掩码
#                 range_seq = torch.arange(max_N, device=means.device).expand(B, max_N)
#                 valid_mask = range_seq < num_gaussians.unsqueeze(1)
                
#                 # 将无效点的 Opacity 设为 0
#                 opacities[~valid_mask] = 0.0
                
#             else:
#                 limit = int(num_gaussians)
#                 means = means[:, :limit]
#                 quats = quats[:, :limit]
#                 scales = scales[:, :limit]
#                 opacities = opacities[:, :limit]
#                 colors = colors[:, :limit]

#         # 3. 渲染
#         return rasterization(
#             means=means.contiguous(),
#             quats=quats.contiguous(),
#             scales=scales.contiguous(),
#             opacities=opacities.squeeze(-1).contiguous(),
#             colors=colors.contiguous(),
#             viewmats=w2c,   
#             Ks=ks,          
#             width=W, height=H,
#             render_mode=render_mode, 
#             packed=False
#         )

#     def _calc_repulsion_loss(self, xyz, num_near, k=4, max_points=4096):
#         """
#         [Fix] 修复了 In-place 操作导致的 RuntimeError
#         """
#         B = xyz.shape[0]
#         loss = 0.0
        
#         for b in range(B):
#             N_valid = int(num_near[b].item())
#             if N_valid < k + 1: continue

#             # 只取有效点
#             points = xyz[b, :N_valid]
            
#             # Subsampling optimization
#             if N_valid > max_points:
#                 perm = torch.randperm(N_valid, device=points.device)[:max_points]
#                 points = points[perm]
            
#             dists = torch.cdist(points, points, p=2)
            
#             # [CRITICAL FIX] 
#             # 原始代码: dists.fill_diagonal_(1e10) -> In-place Error
#             # 修复代码: 使用加法生成新 Tensor，不修改 dists 原值
#             N_curr = dists.shape[0]
#             eye_mask = torch.eye(N_curr, device=dists.device)
#             dists = dists + eye_mask * 1e10 
            
#             min_dists, _ = dists.topk(k, dim=1, largest=False)
#             loss += torch.exp(-min_dists.pow(2) / 0.01).mean()
            
#         return loss / B

#     def forward(self, pred, gt_raw):
#         # 1. 准备 GT
#         gt = self.prepare_gt(gt_raw)

#         B, N, C, H, W = gt['imgs'].shape
#         render_w2c = gt['render_w2c']
#         render_ks = gt['render_ks']

#         gauss = pred['gaussians']
#         num_near = gauss.get('num_near', None) 
        
#         # 2. 渲染 RGB
#         rgb_full, _, _ = self._render_gs(
#             gauss, render_w2c, render_ks, H, W, 
#             num_gaussians=num_near, render_mode='RGB'
#         )

#         # 3. 渲染深度
#         depth_near = None
#         if num_near is not None:
#             depth_near_map, _, _ = self._render_gs(
#                 gauss, render_w2c, render_ks, H, W, 
#                 num_gaussians=num_near, render_mode='ED'
#             )
#             depth_near = depth_near_map

#         # 4. 计算损失
#         rgb_full = rgb_full.reshape(B * N, H, W, 3).permute(0, 3, 1, 2)
#         gt_imgs = gt['imgs'].reshape(B * N, 3, H, W)
        
#         loss_rgb = F.l1_loss(rgb_full, gt_imgs)
#         loss_ssim = 1.0 - ssim(rgb_full, gt_imgs, data_range=1.0)

#         loss_depth = torch.tensor(0.0, device=rgb_full.device)
#         if depth_near is not None:
#             gt_depths = gt['gt_depths'].reshape(B * N, H, W, 1)
#             depth_near = depth_near.reshape(B * N, H, W, 1)
#             depth_mask = (gt_depths > 0).float()
#             loss_depth = F.l1_loss(depth_near * depth_mask, gt_depths * depth_mask)

#         # 5. 其他正则项
#         loss_repulsion = torch.tensor(0.0, device=rgb_full.device)
#         loss_sparsity = torch.tensor(0.0, device=rgb_full.device)
#         loss_consist = torch.tensor(0.0, device=rgb_full.device) 

#         # Repulsion
#         if self.lambda_repulsion > 0 and num_near is not None:
#             loss_repulsion = self._calc_repulsion_loss(gauss['xyz'], num_near)

#         # Sparsity
#         if self.lambda_sparsity > 0:
#             opacity = gauss['opacity'] 
#             total_sparsity = 0.0
#             for b in range(B):
#                 n = int(num_near[b].item())
#                 if n > 0:
#                     total_sparsity += opacity[b, :n].mean()
#             loss_sparsity = total_sparsity / B

#         # Geometry Consistency
#         if self.lambda_consist > 0 and 'base_anchors' in gauss:
#             xyz = gauss['xyz']
#             base = gauss['base_anchors']
#             total_consist = 0.0
#             for b in range(B):
#                 n = int(num_near[b].item())
#                 if n > 0:
#                     diff = xyz[b, :n] - base[b, :n]
#                     total_consist += torch.norm(diff, dim=-1).mean()
#             loss_consist = total_consist / B

#         final_loss = (
#                 self.lambda_rgb * loss_rgb +
#                 self.lambda_ssim * loss_ssim +
#                 self.lambda_depth * loss_depth +
#                 self.lambda_repulsion * loss_repulsion +
#                 self.lambda_sparsity * loss_sparsity +
#                 self.lambda_consist * loss_consist
#         )

#         # [FIX] 去掉 .item()，保持 Tensor 格式，以便 accelerator.gather 能够处理
#         details = {
#             "loss_gs_rgb": loss_rgb,
#             "loss_gs_ssim": loss_ssim,
#             "loss_gs_depth": loss_depth,
#             "loss_repulsion": loss_repulsion,
#             "loss_sparsity": loss_sparsity,
#             "loss_consist": loss_consist,
#             "total_loss": final_loss
#         }

#         return final_loss, details

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim


def se3_inverse(T):
    """SE(3) 矩阵求逆"""
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
            lambda_depth=0.1,
            # [优化 1] Repulsion 默认为 0，如需开启建议设得很小
            lambda_repulsion=0.0, 
            lambda_sparsity=0.005,
            lambda_consist=0.05
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_repulsion = lambda_repulsion
        self.lambda_sparsity = lambda_sparsity
        self.lambda_consist = lambda_consist

    def prepare_gt(self, gt):
        """处理输入的 C2W 并转换为渲染所需的 W2C 和归一化尺度"""
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses_c2w = torch.stack([view['camera_pose'] for view in gt], dim=1)  # [B, N, 4, 4]
        gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
        gt_imgs = torch.stack([view['img'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # 1. 坐标系对齐
        w2c_target = se3_inverse(poses_c2w[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses_rel_c2w = torch.einsum('bij, bnjk -> bnik', w2c_target, poses_c2w)

        # 2. 尺度归一化
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

            poses_rel_c2w[valid_batch, ..., :3, 3] /= batch_factors[..., None, None]
            gt_depths[valid_batch] /= batch_factors[..., None, None, None]

        render_w2c = se3_inverse(poses_rel_c2w)

        return dict(
            imgs=gt_imgs,
            render_w2c=render_w2c, # [B, N, 4, 4]
            render_ks=gt_ks,       # [B, N, 3, 3]
            gt_depths=gt_depths,
            norm_factor=norm_factor
        )

    def _render_gs(self, gaussians, w2c, ks, H, W, num_gaussians=None, render_mode='RGB'):
        # 1. 获取所有属性
        means = gaussians["xyz"]
        quats = gaussians["rotation"]
        scales = gaussians["scale"]
        opacities = gaussians["opacity"]
        colors = gaussians["color"]
        
        B = means.shape[0]

        # 2. 处理 num_gaussians (Ragged Batch Masking)
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

        # 3. 渲染
        return rasterization(
            means=means.contiguous(),
            quats=quats.contiguous(),
            scales=scales.contiguous(),
            opacities=opacities.squeeze(-1).contiguous(),
            colors=colors.contiguous(),
            viewmats=w2c,   
            Ks=ks,          
            width=W, height=H,
            render_mode=render_mode, 
            packed=False
        )

    # [优化 2] 减少采样点数: max_points 4096 -> 1024
    def _calc_repulsion_loss(self, xyz, num_near, k=4, max_points=1024):
        B = xyz.shape[0]
        loss = 0.0
        
        for b in range(B):
            N_valid = int(num_near[b].item())
            if N_valid < k + 1: continue

            # 只取有效点
            points = xyz[b, :N_valid]
            
            # Subsampling optimization
            if N_valid > max_points:
                perm = torch.randperm(N_valid, device=points.device)[:max_points]
                points = points[perm]
            
            dists = torch.cdist(points, points, p=2)
            
            N_curr = dists.shape[0]
            eye_mask = torch.eye(N_curr, device=dists.device)
            dists = dists + eye_mask * 1e10 
            
            min_dists, _ = dists.topk(k, dim=1, largest=False)
            loss += torch.exp(-min_dists.pow(2) / 0.01).mean()
            
        return loss / B

    def forward(self, pred, gt_raw):
        # 1. 准备 GT
        gt = self.prepare_gt(gt_raw)

        B, N, C, H, W = gt['imgs'].shape
        render_w2c = gt['render_w2c']
        render_ks = gt['render_ks']

        gauss = pred['gaussians']
        num_near = gauss.get('num_near', None) 
        
        # 2. 渲染 RGB
        rgb_full, _, _ = self._render_gs(
            gauss, render_w2c, render_ks, H, W, 
            num_gaussians=num_near, render_mode='RGB'
        )

        # 3. 渲染深度
        depth_near = None
        if num_near is not None:
            depth_near_map, _, _ = self._render_gs(
                gauss, render_w2c, render_ks, H, W, 
                num_gaussians=num_near, render_mode='ED'
            )
            depth_near = depth_near_map

        # 4. 计算损失
        rgb_full = rgb_full.reshape(B * N, H, W, 3).permute(0, 3, 1, 2)
        gt_imgs = gt['imgs'].reshape(B * N, 3, H, W)
        
        loss_rgb = F.l1_loss(rgb_full, gt_imgs)
        loss_ssim = 1.0 - ssim(rgb_full, gt_imgs, data_range=1.0)

        loss_depth = torch.tensor(0.0, device=rgb_full.device)
        if depth_near is not None:
            gt_depths = gt['gt_depths'].reshape(B * N, H, W, 1)
            depth_near = depth_near.reshape(B * N, H, W, 1)
            depth_mask = (gt_depths > 0).float()
            loss_depth = F.l1_loss(depth_near * depth_mask, gt_depths * depth_mask)

        # 5. 其他正则项
        loss_repulsion = torch.tensor(0.0, device=rgb_full.device)
        loss_sparsity = torch.tensor(0.0, device=rgb_full.device)
        loss_consist = torch.tensor(0.0, device=rgb_full.device) 

        # [优化 3] Repulsion Loss 性能瓶颈解决
        # 策略：仅有 10% 的概率计算此 Loss，且计算时权重放大 10 倍
        # 这样能极大减少 O(N^2) 计算的频率，同时保持梯度期望一致
        if self.lambda_repulsion > 0 and num_near is not None:
            if torch.rand(1).item() < 0.1:
                loss_repulsion = self._calc_repulsion_loss(gauss['xyz'], num_near) * 10.0
            else:
                # 不计算时必须保留一个梯度挂载点，防止 DDP 报错
                # loss_repulsion = 0.0 * gauss['xyz'].sum() 
                pass

        # Sparsity
        if self.lambda_sparsity > 0:
            opacity = gauss['opacity'] 
            total_sparsity = 0.0
            for b in range(B):
                n = int(num_near[b].item())
                if n > 0:
                    total_sparsity += opacity[b, :n].mean()
            loss_sparsity = total_sparsity / B

        # Geometry Consistency
        if self.lambda_consist > 0 and 'base_anchors' in gauss:
            xyz = gauss['xyz']
            base = gauss['base_anchors']
            total_consist = 0.0
            for b in range(B):
                n = int(num_near[b].item())
                if n > 0:
                    diff = xyz[b, :n] - base[b, :n]
                    total_consist += torch.norm(diff, dim=-1).mean()
            loss_consist = total_consist / B

        final_loss = (
                self.lambda_rgb * loss_rgb +
                self.lambda_ssim * loss_ssim +
                self.lambda_depth * loss_depth +
                self.lambda_repulsion * loss_repulsion +
                self.lambda_sparsity * loss_sparsity +
                self.lambda_consist * loss_consist
        )

        details = {
            "loss_gs_rgb": loss_rgb,
            "loss_gs_ssim": loss_ssim,
            "loss_gs_depth": loss_depth,
            "loss_repulsion": loss_repulsion,
            "loss_sparsity": loss_sparsity,
            "loss_consist": loss_consist,
            "total_loss": final_loss
        }

        return final_loss, details