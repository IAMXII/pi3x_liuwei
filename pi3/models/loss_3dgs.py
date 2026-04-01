import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .pi3_3dgs import matrix_to_quaternion, quat_mult
from ..utils.alignment import align_points_scale
from ..utils.geometry import depth_edge, homogenize_points
import lpips
from math import exp

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

# ==========================================
# === 新增：Sobel 边缘损失提取函数 ===
# ==========================================
def sobel_edge_loss(pred, gt):
    """
    计算图像的 Sobel 边缘/梯度 L1 Loss。
    输入: pred, gt 形状均为 [B, C, H, W]
    """
    device = pred.device
    channels = pred.size(1)
    
    # 定义 3x3 Sobel 算子
    sobel_x = torch.tensor([[-1., 0., 1.], 
                            [-2., 0., 2.], 
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], 
                            [0.,  0.,  0.], 
                            [1.,  2.,  1.]], device=device).view(1, 1, 3, 3)
    
    # 扩展到所有通道，使用分组卷积 (groups=channels) 独立计算每个通道的梯度
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
    # 提取水平和垂直梯度
    pred_dx = F.conv2d(pred, sobel_x, padding=1, groups=channels)
    pred_dy = F.conv2d(pred, sobel_y, padding=1, groups=channels)
    
    gt_dx = F.conv2d(gt, sobel_x, padding=1, groups=channels)
    gt_dy = F.conv2d(gt, sobel_y, padding=1, groups=channels)
    
    # 计算梯度域的 L1 损失
    loss_dx = F.l1_loss(pred_dx, gt_dx)
    loss_dy = F.l1_loss(pred_dy, gt_dy)
    
    return loss_dx + loss_dy
# ==========================================

# class Pi3LossGS(nn.Module):
#     def __init__(
#             self, lambda_rgb=1, lambda_ssim=0.5, lambda_depth=0.3, 
#             lambda_pose=0.2, lambda_scale=0.1, lambda_pts=0.3, train_stage=1, local_align_res=4096,
#             train_conf=False, num_sky_anchors=8196 
#     ):
#         super().__init__()
#         self.lambda_rgb = lambda_rgb
#         self.lambda_ssim = lambda_ssim
#         self.lambda_depth = lambda_depth
#         self.lambda_pose = lambda_pose
#         self.lambda_scale = lambda_scale
#         self.lambda_pts = lambda_pts 
#         self.lambda_lpips = 0.1
#         self.lambda_edge = 0.15 # === 新增：Edge Loss 权重，可调 ===
        
#         # 强制转换为 int，防止 YAML 解析为字符串导致的幽灵 Bug
#         self.train_stage = int(train_stage) 
#         self.local_align_res = local_align_res
        
#         self.train_conf = train_conf 
#         self.num_sky_anchors = num_sky_anchors
#         self.camera_loss_fn = CameraPoseLoss()
#         self.lpips_loss_fn = lpips.LPIPS(net='alex').to('cuda')
        

#     def prepare_gt(self, gt):
#         """支持自动从 Depth+Pose+Intrinsics 反投影生成 pts3d 的对齐与 norm 逻辑"""
#         # 1. 安全提取 Dataloader 传来的基础数据
#         imgs = torch.stack([view['img'] for view in gt], dim=1)
#         gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
#         poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
#         gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)

#         B, N, H, W = gt_depths.shape
#         device = gt_depths.device

#         # ==========================================
#         # 核心补全：动态生成 valid_mask 和 pts3d
#         # ==========================================
#         # A. 生成掩模: 深度值有效的区域 (大于极小值)
#         masks = (gt_depths > 1e-4).unsqueeze(-1) # [B, N, H, W, 1]

#         # B. 像素坐标网格反投影
#         grid_y, grid_x = torch.meshgrid(
#             torch.arange(H, device=device), 
#             torch.arange(W, device=device), 
#             indexing='ij'
#         )
#         grid_x = grid_x.expand(B, N, -1, -1).float()
#         grid_y = grid_y.expand(B, N, -1, -1).float()

#         # 提取内参
#         fx = gt_ks[..., 0, 0].view(B, N, 1, 1)
#         fy = gt_ks[..., 1, 1].view(B, N, 1, 1)
#         cx = gt_ks[..., 0, 2].view(B, N, 1, 1)
#         cy = gt_ks[..., 1, 2].view(B, N, 1, 1)

#         # 计算相机局部坐标 (Local Points)
#         local_x = (grid_x - cx) * gt_depths / fx
#         local_y = (grid_y - cy) * gt_depths / fy
#         local_z = gt_depths
#         gt_local_pts_raw = torch.stack([local_x, local_y, local_z], dim=-1) # [B, N, H, W, 3]

#         # 转换到全局坐标 (Global Points)
#         gt_local_pts_h = homogenize_points(gt_local_pts_raw).view(B, N, -1, 4).transpose(2, 3) # [B, N, 4, H*W]
#         # 假设 poses 是 Camera-to-World (c2w)
#         gt_pts = torch.matmul(poses, gt_local_pts_h).transpose(2, 3).reshape(B, N, H, W, 4)[..., :3] 
#         # ==========================================

#         # --- 以下无缝衔接你原本的坐标系统一与对齐逻辑 ---
#         # 统一坐标系到第一个视角
#         w2c_target = se3_inverse(poses[:, 0])
#         gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
#         poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

#         # ==========================================
#         # 规范化全局尺度 (极其关键，防止 NaN)
#         # ==========================================
#         # [修复 1]: 确保 valid_batch 是一维张量 [B]
#         valid_batch = masks.view(B, -1).sum(dim=-1) > 0 
        
#         if valid_batch.sum() > 0:
#             B_ = valid_batch.sum()
#             all_pts = gt_pts[valid_batch].clone() # 形状: [B_, N, H, W, 3]
            
#             # 使用展开后的布尔掩模置零无效点
#             mask_bool = masks[valid_batch].squeeze(-1) # 形状: [B_, N, H, W]
#             all_pts[~mask_bool] = 0
            
#             # [修复 2]: 将 N 和 H*W 展平，避免复杂的维度计算
#             all_pts = all_pts.reshape(B_, -1, 3) # 形状: [B_, N*H*W, 3]
#             all_dis = all_pts.norm(dim=-1)       # 形状: [B_, N*H*W]
            
#             # 分母：计算每个 Batch 有多少个有效点
#             num_valid_pts = mask_bool.view(B_, -1).float().sum(dim=-1) # 形状: [B_]
            
#             # 计算缩放因子
#             norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) # 形状: [B_]
#             norm_factor = norm_factor.clamp_min(1e-4)

#             # 执行缩放 (利用 None 自动对齐广播维度)
#             gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
#             poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
#             gt_depths[valid_batch] /= norm_factor[..., None, None, None]
#         # ==========================================

#         # 重新转换出安全的 local_pts 供后续 loss 使用
#         extrinsics = se3_inverse(poses)
#         gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]

#         return dict(
#             imgs = imgs,
#             gt_ks = gt_ks,
#             gt_depths = gt_depths,
#             global_points = gt_pts,
#             gt_local_pts = gt_local_pts, 
#             masks = masks,
#             gt_c2w = poses
#         )

#     def normalize_pred(self, pred, gt):
#         """恢复原版预测结果的尺度对齐 (已彻底修复维度匹配问题)"""
#         local_points = pred['local_points']
#         camera_poses = pred['camera_poses']
#         B, N, H, W, _ = local_points.shape
#         masks = gt['masks'] # 原始形状: [B, N, H, W, 1]
        
#         # 1. 挤掉最后一维，变成纯纯的布尔掩模 [B, N, H, W]
#         mask_bool = masks.squeeze(-1) 

#         all_pts = local_points.clone()
#         # 此时形状完美匹配，不会报错了！
#         all_pts[~mask_bool] = 0 
        
#         # 2. 展平空间维度，避免复杂的多维 sum
#         all_pts = all_pts.reshape(B, -1, 3) # 形状: [B, N*H*W, 3]
#         all_dis = all_pts.norm(dim=-1)      # 形状: [B, N*H*W]
        
#         # 3. 精准计算每个 Batch 有多少个有效点
#         num_valid_pts = mask_bool.view(B, -1).float().sum(dim=-1) # 形状: [B]
        
#         # 计算缩放因子
#         norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) # 形状: [B]
#         norm_factor = norm_factor.clamp_min(1e-4) # 防护除零
        
#         # 4. 执行缩放对齐
#         local_points = local_points / norm_factor[..., None, None, None, None]
        
#         camera_poses_normalized = camera_poses.clone()
#         camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

#         pred['local_points'] = local_points
#         pred['camera_poses'] = camera_poses_normalized

#         # 同步缩放高斯
#         if 'gaussians' in pred:
#             pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
#             pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

#         return pred

#     def prepare_ROE(self, pts, mask, target_size=4096):
#         """
#         优化版：消除 for 循环中因 shape[0] 动态导致的 CPU-GPU 同步阻塞
#         直接使用 linspace 索引采样，速度提升几个数量级。
#         """
#         B, N, H, W, C = pts.shape
#         pts_flat = pts.reshape(B, -1, C)
#         mask_flat = mask.reshape(B, -1)
        
#         output = torch.ones((B, target_size, C), device=pts.device)
        
#         for i in range(B):
#             valid_pts = pts_flat[i][mask_flat[i]]
#             num_valid = valid_pts.shape[0]
#             if num_valid > 0:
#                 # 生成均匀索引并直接 gather 采样
#                 idx = torch.linspace(0, num_valid - 1, target_size, device=pts.device, dtype=torch.long)
#                 output[i] = valid_pts[idx]
                
#         return output

#     def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
#         opacities = gaussians["opacity"].clone()
        
#         if self.train_stage == 3 or not self.training:
#             means = gaussians["xyz"].detach().contiguous()
#             quats = gaussians["rotation"].detach().contiguous()
#             scales = gaussians["scale"].detach().contiguous()
#             colors = gaussians["color"].detach().contiguous()
#             conf_prob = torch.sigmoid(gaussians["conf"])
#             opacities = (opacities.detach() * conf_prob).squeeze(-1).contiguous()
#         else:
#             means = gaussians["xyz"].contiguous()
#             quats = gaussians["rotation"].contiguous()
#             scales = gaussians["scale"].contiguous()
#             colors = gaussians["color"].contiguous()
#             opacities = opacities.squeeze(-1).contiguous()

#         return rasterization(
#             means=means, quats=quats, scales=scales,
#             opacities=opacities, colors=colors,
#             viewmats=w2c, Ks=ks, width=W, height=H, render_mode=render_mode, packed=False
#         )

#     def forward(self, pred, gt_raw, batch_idx=0, current_epoch=None, total_epochs=None, **kwargs):
#         gt = self.prepare_gt(gt_raw)
        
#         B, N_total, C, H, W = gt['imgs'].shape
#         sub_idx = torch.arange(0, N_total, 1, device=gt['imgs'].device)
#         N_sub = len(sub_idx)

#         gt_sub_mask = {'masks': gt['masks'][:, sub_idx]}
#         pred = self.normalize_pred(pred, gt_sub_mask) 
        
#         gt_ks, gt_c2w, gt_imgs, gt_depths, valid_masks = gt['gt_ks'], gt['gt_c2w'], gt['imgs'], gt['gt_depths'], gt['masks']
#         gt_local_pts = gt['gt_local_pts']
        
#         valid_masks_sub = valid_masks[:, sub_idx].squeeze(-1)
#         gt_local_pts_sub = gt_local_pts[:, sub_idx]

#         gauss_raw = pred['gaussians']
#         pred_c2w = pred['camera_poses'] # 此时全量位姿预测为 [B, N_total, 4, 4]
        
#         pred_local_pts = torch.clamp(pred['local_points'], min=-1e4, max=1e4)

#         # === 新增：初始化 loss_edge 为 0 ===
#         loss_rgb = loss_ssim = loss_depth = loss_pose = loss_conf = loss_scale = loss_pts = loss_edge = torch.tensor(0.0, device=pred_c2w.device)
#         details = {}

#         scale_opt = torch.ones((B,), device=pred_c2w.device)
        
#         if self.train_stage in [1, 2]:
#             weights_ = gt_local_pts_sub[..., 2].clamp_min(1e-3)
#             weights_ = 1 / (weights_ + 1e-6)
            
#             xyz_pred_ROE = self.prepare_ROE(pred_local_pts.reshape(B, N_sub, H, W, 3), valid_masks_sub, target_size=self.local_align_res)
#             with torch.no_grad():
#                 xyz_gt_ROE = self.prepare_ROE(gt_local_pts_sub.reshape(B, N_sub, H, W, 3), valid_masks_sub, target_size=self.local_align_res)
#                 xyz_w_ROE = self.prepare_ROE((weights_[..., None]).reshape(B, N_sub, H, W, 1), valid_masks_sub, target_size=self.local_align_res)[..., 0]
            
#             scale_opt = align_points_scale(xyz_pred_ROE, xyz_gt_ROE, xyz_w_ROE)
            
#             if valid_masks_sub.sum() > 0:
#                 aligned_local_pts = pred_local_pts * scale_opt.view(B, 1, 1, 1, 1)
                
#                 loss_pts = F.l1_loss(aligned_local_pts[valid_masks_sub], gt_local_pts_sub[valid_masks_sub])
#             else:
#                 loss_pts = (pred_local_pts.sum() * 0.0)

#         gauss_render = {k: v for k, v in gauss_raw.items()} 
#         render_c2w = pred_c2w.clone()

#         if self.train_stage in [1, 2]:
#             detach_scale = scale_opt.detach().view(B, 1, 1)
#             render_c2w[..., :3, 3] *= detach_scale
#             gauss_render["xyz"] = gauss_raw["xyz"] * detach_scale
#             gauss_render["scale"] = gauss_raw["scale"] * detach_scale

#         render_w2c = se3_inverse(render_c2w)
        
#         render_out, _, _ = self._render_gs(gauss_render, render_w2c, gt_ks, H, W, render_mode='RGB+ED')
        
#         rgb_full = render_out[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
#         depth_map = render_out[..., 3:4].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
        
#         gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)
#         gt_depth_reshaped = gt_depths.reshape(B * N_total, 1, H, W)
        
#         if self.train_stage in [1, 2]:
#             aligned_depth_map = depth_map * scale_opt.detach().repeat_interleave(N_total).view(B * N_total, 1, 1, 1)
#         else:
#             aligned_depth_map = depth_map
            
#         gt_depth_reshaped = gt_depths.reshape(B * N_total, 1, H, W)
#         mask_depth = (gt_depth_reshaped > 1e-4)
            
#         with torch.no_grad():
#             batch_idx = 0
#             num_viz = min(4, B * N_total)
#             rgb_viz = torch.cat([gt_imgs_reshaped[:num_viz], rgb_full[:num_viz]], dim=2) 
#             d_pred_viz = aligned_depth_map[:num_viz] / (aligned_depth_map[:num_viz].max() + 1e-5)
#             d_gt_viz = gt_depth_reshaped[:num_viz] / (gt_depth_reshaped[:num_viz].max() + 1e-5)
#             d_pred_viz = d_pred_viz.repeat(1, 3, 1, 1)
#             d_gt_viz = d_gt_viz.repeat(1, 3, 1, 1)
#             depth_viz = torch.cat([d_gt_viz, d_pred_viz], dim=2)
            
#             final_viz = torch.cat([rgb_viz, depth_viz], dim=3)
#             torchvision.utils.save_image(final_viz, f"debug_output/step_{batch_idx}_stage_{self.train_stage}.png")

#         if self.train_stage in [1, 2]:
#             loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
#             loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped,data_range=1.0)
#             loss_lpips = self.lpips_loss_fn(rgb_full, gt_imgs_reshaped).mean()
            
#             # === 新增：计算 Sobel Edge Loss ===
#             loss_edge = sobel_edge_loss(rgb_full, gt_imgs_reshaped)
            
#             mask_depth = (gt_depth_reshaped > 1e-4) & (gt_depth_reshaped < 58982.4)
#             if self.lambda_depth > 0 and mask_depth.sum() > 10:
#                 loss_depth = F.l1_loss(aligned_depth_map[mask_depth], gt_depth_reshaped[mask_depth])

#         elif self.train_stage == 3:
#             rgb_sub = rgb_full.view(B, N_total, 3, H, W)[:, sub_idx].reshape(B * N_sub, 3, H, W)
#             gt_imgs_sub = gt_imgs[:, sub_idx].reshape(B * N_sub, 3, H, W)
#             pixel_error = torch.abs(rgb_sub - gt_imgs_sub).mean(dim=1, keepdim=True).detach()
#             valid_target = (pixel_error < 0.1).float() 
#             dense_conf_logits = pred['conf'].reshape(B * N_sub, 1, H, W)
#             loss_conf = F.binary_cross_entropy_with_logits(dense_conf_logits, valid_target)
            
#         final_loss = (
#             self.lambda_rgb * loss_rgb + 
#             self.lambda_ssim * loss_ssim +
#             self.lambda_depth * loss_depth +
#             self.lambda_lpips * loss_lpips +
#             self.lambda_edge * loss_edge +  # === 新增：将 Edge Loss 加入总 Loss ===
#             self.lambda_pts * loss_pts 
#         )

#         if final_loss == 0.0:
#              final_loss = (pred_local_pts.sum() * 0.0)

#         details.update({
#             "loss_rgb": loss_rgb,
#             "loss_ssim": loss_ssim,
#             "loss_depth": loss_depth, "loss_pts": loss_pts,
#             "loss_lpips": loss_lpips, 
#             "loss_edge": loss_edge,         # === 新增：记录到 details 以便监控 ===
#             "total_loss": final_loss
#         })

#         return final_loss, details

class Pi3LossGS(nn.Module):
    def __init__(
            self, lambda_rgb=1, lambda_ssim=0.5, lambda_depth=1.5, 
            lambda_pose=0.2, lambda_scale=0.1, train_stage=1, local_align_res=4096,
            train_conf=False, num_sky_anchors=8196 
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_pose = lambda_pose
        self.lambda_scale = lambda_scale
        self.lambda_lpips = 0.1
        self.lambda_edge = 0.15 
        
        self.train_stage = int(train_stage) 
        self.local_align_res = local_align_res
        
        self.train_conf = train_conf 
        self.num_sky_anchors = num_sky_anchors
        # self.lpips_loss_fn = lpips.LPIPS(net='alex')
        self.lpips_loss_fn = lpips.LPIPS(net='alex').eval()
        # 冻结 LPIPS 参数，避免无谓的梯度计算
        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False

        self.register_buffer('cached_grid_x', None)
        self.register_buffer('cached_grid_y', None)
        

    def prepare_gt(self, gt):
        # ... (保持不变) ...
        imgs = torch.stack([view['img'] for view in gt], dim=1)
        gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)

        B, N, H, W = gt_depths.shape
        device = gt_depths.device
        masks = (gt_depths > 1e-4).unsqueeze(-1) 
        H, W = gt_depths.shape[-2:]

        if self.cached_grid_x is None or self.cached_grid_x.shape != (H, W):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device), 
                torch.arange(W, device=device), 
                indexing='ij'
            )
            self.cached_grid_x = grid_x.float()
            self.cached_grid_y = grid_y.float()

        grid_x = self.cached_grid_x.expand(B, N, -1, -1)
        grid_y = self.cached_grid_y.expand(B, N, -1, -1)

        fx = gt_ks[..., 0, 0].view(B, N, 1, 1)
        fy = gt_ks[..., 1, 1].view(B, N, 1, 1)
        cx = gt_ks[..., 0, 2].view(B, N, 1, 1)
        cy = gt_ks[..., 1, 2].view(B, N, 1, 1)

        local_x = (grid_x - cx) * gt_depths / fx
        local_y = (grid_y - cy) * gt_depths / fy
        local_z = gt_depths
        gt_local_pts_raw = torch.stack([local_x, local_y, local_z], dim=-1) 

        # gt_local_pts_h = homogenize_points(gt_local_pts_raw).view(B, N, -1, 4).transpose(2, 3) 
        # gt_pts = torch.matmul(poses, gt_local_pts_h).transpose(2, 3).reshape(B, N, H, W, 4)[..., :3] 
        R = poses[..., :3, :3]       # [B, N, 3, 3]
        t = poses[..., :3, 3:4]      # [B, N, 3, 1]
        # [B, N, 3, H*W]
        pts_flat = gt_local_pts_raw.reshape(B, N, -1, 3).transpose(-1, -2) 
        # R @ pts + t，然后变回原形状
        gt_pts = (torch.matmul(R, pts_flat) + t).transpose(-1, -2).reshape(B, N, H, W, 3)

        # w2c_target = se3_inverse(poses[:, 0])
        # gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        # poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)
        w2c_target = se3_inverse(poses[:, 0])
        R_w2c = w2c_target[..., :3, :3]    # [B, 3, 3]
        t_w2c = w2c_target[..., :3, 3:4]   # [B, 3, 1]
        # 将 gt_pts 从 [B, N, H, W, 3] 展平为 [B, 3, N*H*W]
        # gt_pts_flat = gt_pts.view(B, -1, 3).transpose(-1, -2)
        gt_pts_flat = gt_pts.reshape(B, -1, 3).transpose(-1, -2)
        # 用 bmm (Batch Matrix Multiply) 加速运算
        gt_pts = (torch.bmm(R_w2c, gt_pts_flat) + t_w2c).transpose(-1, -2).reshape(B, N, H, W, 3)
        
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        valid_batch = masks.view(B, -1).sum(dim=-1) > 0 
        
        # if valid_batch.sum() > 0:
        #     B_ = valid_batch.sum()
        #     all_pts = gt_pts[valid_batch].clone() 
            
        #     mask_bool = masks[valid_batch].squeeze(-1) 
        #     all_pts[~mask_bool] = 0
            
        #     all_pts = all_pts.reshape(B_, -1, 3) 
        #     all_dis = all_pts.norm(dim=-1)       
            
        #     num_valid_pts = mask_bool.view(B_, -1).float().sum(dim=-1) 
            
        #     norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) 
        #     norm_factor = norm_factor.clamp_min(1e-4)

        #     gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
        #     poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
        #     gt_depths[valid_batch] /= norm_factor[..., None, None, None]

        # extrinsics = se3_inverse(poses)
        # gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        # extrinsics = se3_inverse(poses)
        # R_ext = extrinsics[..., :3, :3]    # [B, N, 3, 3]
        # t_ext = extrinsics[..., :3, 3:4]   # [B, N, 3, 1]
        # gt_pts_flat2 = gt_pts.view(B, N, -1, 3).transpose(-1, -2) # [B, N, 3, H*W]
        # gt_local_pts = (torch.matmul(R_ext, gt_pts_flat2) + t_ext).transpose(-1, -2).reshape(B, N, H, W, 3)
        return dict(
            imgs = imgs,
            gt_ks = gt_ks,
            gt_depths = gt_depths,
            # global_points = gt_pts,
            # gt_local_pts = gt_local_pts, 
            masks = masks,
            gt_c2w = poses
        )

    def normalize_pred(self, pred, gt):
        # ... (保持不变) ...
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['masks'] 
        
        mask_bool = masks.squeeze(-1) 

        all_pts = local_points.clone()
        all_pts[~mask_bool] = 0 
        
        all_pts = all_pts.reshape(B, -1, 3) 
        all_dis = all_pts.norm(dim=-1)      
        
        num_valid_pts = mask_bool.view(B, -1).float().sum(dim=-1) 
        
        norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) 
        norm_factor = norm_factor.clamp_min(1e-4) 
        
        local_points = local_points / norm_factor[..., None, None, None, None]
        
        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        if 'gaussians' in pred:
            pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
            pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

        return pred

    # === 注意：彻底删除了 prepare_ROE ===

    def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
        # ... (保持不变) ...
        opacities = gaussians["opacity"].clone()
        
        # if self.train_stage == 3 or not self.training:
        #     means = gaussians["xyz"].detach().contiguous()
        #     quats = gaussians["rotation"].detach().contiguous()
        #     scales = gaussians["scale"].detach().contiguous()
        #     colors = gaussians["color"].detach().contiguous()
        #     conf_prob = torch.sigmoid(gaussians["conf"])
        #     opacities = (opacities.detach() * conf_prob).squeeze(-1).contiguous()
        # else:
        means = gaussians["xyz"].contiguous()
        quats = gaussians["rotation"].contiguous()
        scales = gaussians["scale"].contiguous()
        colors = gaussians["color"].contiguous()
        opacities = opacities.squeeze(-1).contiguous()

        return rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=w2c, Ks=ks, width=W, height=H, render_mode=render_mode, packed=False
        )

    def forward(self, pred, gt_raw, batch_idx=0, current_epoch=None, total_epochs=None, **kwargs):
        gt = self.prepare_gt(gt_raw)
        
        B, N_total, C, H, W = gt['imgs'].shape
        sub_idx = torch.arange(0, N_total, 1, device=gt['imgs'].device)
        self.lpips_loss_fn = self.lpips_loss_fn.to(gt['imgs'].device)  # 确保 LPIPS 损失函数在正确的设备上
        N_sub = len(sub_idx)

        gt_sub_mask = {'masks': gt['masks']}
        pred = self.normalize_pred(pred, gt_sub_mask) 
        
        gt_ks, gt_c2w, gt_imgs, gt_depths, valid_masks = gt['gt_ks'], gt['gt_c2w'], gt['imgs'], gt['gt_depths'], gt['masks']
        # gt_local_pts = gt['gt_local_pts']
        
        valid_masks_sub = valid_masks[:, sub_idx].squeeze(-1)
        # gt_local_pts_sub = gt_local_pts[:, sub_idx]

        gauss_raw = pred['gaussians']
        pred_c2w = pred['camera_poses']
        intrinsics_pred = pred['intrinsics'] 
        # print("Ground Truth Intrinsics:", gt_ks[0, 0])  # 打印第一个视角的 GT 内参以供调试
        # print("Predicted Intrinsics:", intrinsics_pred[0, 0])  # 打印第一个视角的预测内参以供调试
        warmup_steps = 7500.0 # 前 60% 步数用于预热光度损失
        progress = min(batch_idx / warmup_steps, 1.0)

        # 光度损失 (RGB/SSIM) 从 10% 平滑增加到 100%
        start_ratio = 0.1
        photo_ratio = start_ratio + (1.0 - start_ratio) * progress
        cur_lambda_rgb = self.lambda_rgb * photo_ratio
        cur_lambda_ssim = self.lambda_ssim * photo_ratio
        lpips_start_step = 2500.0
        lpips_total_steps = 22500.0  # 100000 - 50000
        if batch_idx > lpips_start_step:
            lpips_progress = min((batch_idx - lpips_start_step) / lpips_total_steps, 1.0)
        else:
            lpips_progress = 0.0
            
        cur_lambda_lpips = self.lambda_lpips * lpips_progress
        # cur_lambda_lpips = self.lambda_lpips * max(photo_ratio, 1.0)
        # 几何损失 (Depth) 从 1.0 平滑衰减到 0.5，前期强行主导几何框架
        depth_ratio = 1.0 - 0.5 * progress 
        cur_lambda_depth = self.lambda_depth * depth_ratio
        pred_local_pts = torch.clamp(pred['local_points'], min=-1e4, max=1e4)

        loss_rgb = loss_ssim = loss_depth = loss_pose = loss_conf = loss_scale = loss_edge = torch.tensor(0.0, device=pred_c2w.device)
        details = {}

        # === 删除了所有计算和应用 scale_opt 的逻辑 ===

        gauss_render = {k: v for k, v in gauss_raw.items()} 
        render_c2w = pred_c2w.clone()

        # === 删除了渲染前的尺度对齐干预 ===

        render_w2c = se3_inverse(render_c2w)
        
        render_out, _, _ = self._render_gs(gauss_render, render_w2c, intrinsics_pred, H, W, render_mode='RGB+ED')
        
        rgb_full = render_out[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
        depth_map = render_out[..., 3:4].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
        
        gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)
        # gt_depth_reshaped = gt_depths.reshape(B * N_total, 1, H, W)
            
        # with torch.no_grad():
        #     batch_idx = 0
        #     num_viz = min(4, B * N_total)
        #     rgb_viz = torch.cat([gt_imgs_reshaped[:num_viz], rgb_full[:num_viz]], dim=2) 
            
        #     # 直接使用 depth_map 进行可视化
        #     d_pred_viz = depth_map[:num_viz] / (depth_map[:num_viz].max() + 1e-5)
        #     d_gt_viz = gt_depth_reshaped[:num_viz] / (gt_depth_reshaped[:num_viz].max() + 1e-5)
        #     d_pred_viz = d_pred_viz.repeat(1, 3, 1, 1)
        #     d_gt_viz = d_gt_viz.repeat(1, 3, 1, 1)
        #     depth_viz = torch.cat([d_gt_viz, d_pred_viz], dim=2)
            
        #     final_viz = torch.cat([rgb_viz, depth_viz], dim=3)
        #     torchvision.utils.save_image(final_viz, f"debug_output/step_{batch_idx}_stage_{self.train_stage}.png")

        if self.train_stage in [1, 2, 3]:
            loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
            loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)
            loss_lpips = self.lpips_loss_fn(rgb_full, gt_imgs_reshaped).mean()
            # 使用 pred_local_pts 提取深度作为伪标签约束
            # conf_mask = torch.sigmoid(pred['conf'][..., 0]) > 0.1 
            # non_edge_mask = ~depth_edge(pred['local_points'][..., 2], rtol=0.03) 
            
            # pseudo_mask = torch.logical_and(conf_mask, non_edge_mask)
            # pseudo_mask = pseudo_mask.reshape(B * N_total, 1, H, W)
            pseudo_gt_depth = pred_local_pts[..., 2:3].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
            
            if self.lambda_depth > 0:# and pseudo_mask.sum() > 10:
                loss_depth = F.l1_loss(depth_map, pseudo_gt_depth.detach())

        # ==========================================================
        # 【修改】：应用动态权重
        # ==========================================================
        final_loss = (
            cur_lambda_rgb * loss_rgb + 
            cur_lambda_ssim * loss_ssim +
            cur_lambda_depth * loss_depth +
            cur_lambda_lpips * loss_lpips
        )

        if final_loss == 0.0:
             final_loss = (pred_local_pts.sum() * 0.0)

        details.update({
            "loss_rgb": loss_rgb,
            "loss_ssim": loss_ssim,
            "loss_depth": loss_depth, 
            "loss_lpips": loss_lpips,

            "cur_weight_rgb": torch.tensor(cur_lambda_rgb, device=pred_c2w.device),   # 添加监控
            "cur_weight_depth": torch.tensor(cur_lambda_depth, device=pred_c2w.device), # 添加监控
            "total_loss": final_loss
        })

        return final_loss, details