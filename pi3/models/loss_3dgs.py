# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from gsplat import rasterization
# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from .pi3_3dgs import matrix_to_quaternion, quat_mult
# from ..utils.alignment import align_points_scale
# from ..utils.geometry import depth_edge, homogenize_points
# import lpips
# from math import exp

# def se3_inverse(T):
#     R = T[..., :3, :3]
#     t = T[..., :3, 3:4]
#     R_inv = R.transpose(-1, -2)
#     t_inv = -torch.matmul(R_inv, t)
#     T_inv = torch.zeros_like(T)
#     T_inv[..., :3, :3] = R_inv
#     T_inv[..., :3, 3:4] = t_inv
#     T_inv[..., 3, 3] = 1.0
#     return T_inv

# # ==========================================
# # === 新增：Sobel 边缘损失提取函数 ===
# # ==========================================
# def sobel_edge_loss(pred, gt):
#     """
#     计算图像的 Sobel 边缘/梯度 L1 Loss。
#     输入: pred, gt 形状均为 [B, C, H, W]
#     """
#     device = pred.device
#     channels = pred.size(1)
    
#     # 定义 3x3 Sobel 算子
#     sobel_x = torch.tensor([[-1., 0., 1.], 
#                             [-2., 0., 2.], 
#                             [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
#     sobel_y = torch.tensor([[-1., -2., -1.], 
#                             [0.,  0.,  0.], 
#                             [1.,  2.,  1.]], device=device).view(1, 1, 3, 3)
    
#     # 扩展到所有通道，使用分组卷积 (groups=channels) 独立计算每个通道的梯度
#     sobel_x = sobel_x.repeat(channels, 1, 1, 1)
#     sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
#     # 提取水平和垂直梯度
#     pred_dx = F.conv2d(pred, sobel_x, padding=1, groups=channels)
#     pred_dy = F.conv2d(pred, sobel_y, padding=1, groups=channels)
    
#     gt_dx = F.conv2d(gt, sobel_x, padding=1, groups=channels)
#     gt_dy = F.conv2d(gt, sobel_y, padding=1, groups=channels)
    
#     # 计算梯度域的 L1 损失
#     loss_dx = F.l1_loss(pred_dx, gt_dx)
#     loss_dy = F.l1_loss(pred_dy, gt_dy)
    
#     return loss_dx + loss_dy

# class Pi3LossGS(nn.Module):
#     def __init__(
#             self, lambda_rgb=1, lambda_ssim=0.5, lambda_depth=1.5, 
#             lambda_pose=0.2, lambda_scale=0.1, train_stage=1, local_align_res=4096,
#             train_conf=False, num_sky_anchors=8196 
#     ):
#         super().__init__()
#         self.lambda_rgb = lambda_rgb
#         self.lambda_ssim = lambda_ssim
#         self.lambda_depth = lambda_depth
#         self.lambda_pose = lambda_pose
#         self.lambda_scale = lambda_scale
#         self.lambda_lpips = 0.1
#         self.lambda_edge = 0.15 
        
#         self.train_stage = int(train_stage) 
#         self.local_align_res = local_align_res
        
#         self.train_conf = train_conf 
#         self.num_sky_anchors = num_sky_anchors
#         # self.lpips_loss_fn = lpips.LPIPS(net='alex')
#         self.lpips_loss_fn = lpips.LPIPS(net='alex').eval()
#         # 冻结 LPIPS 参数，避免无谓的梯度计算
#         for param in self.lpips_loss_fn.parameters():
#             param.requires_grad = False

#         self.register_buffer('cached_grid_x', None)
#         self.register_buffer('cached_grid_y', None)
        

#     def prepare_gt(self, gt):
#         # ... (保持不变) ...
#         imgs = torch.stack([view['img'] for view in gt], dim=1)
#         # gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
#         # poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
#         gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)

#         # B, N, C, H, W = imgs.shape
#         # device = imgs.device
#         # masks = (imgs > 1e-4).unsqueeze(-1) 
#         # H, W = imgs.shape[-2:]

#         # if self.cached_grid_x is None or self.cached_grid_x.shape != (H, W):
#         #     grid_y, grid_x = torch.meshgrid(
#         #         torch.arange(H, device=device), 
#         #         torch.arange(W, device=device), 
#         #         indexing='ij'
#         #     )
#         #     self.cached_grid_x = grid_x.float()
#         #     self.cached_grid_y = grid_y.float()

#         # grid_x = self.cached_grid_x.expand(B, N, -1, -1)
#         # grid_y = self.cached_grid_y.expand(B, N, -1, -1)

#         # fx = gt_ks[..., 0, 0].view(B, N, 1, 1)
#         # fy = gt_ks[..., 1, 1].view(B, N, 1, 1)
#         # cx = gt_ks[..., 0, 2].view(B, N, 1, 1)
#         # cy = gt_ks[..., 1, 2].view(B, N, 1, 1)

#         # local_x = (grid_x - cx) * gt_depths / fx
#         # local_y = (grid_y - cy) * gt_depths / fy
#         # local_z = gt_depths
#         # gt_local_pts_raw = torch.stack([local_x, local_y, local_z], dim=-1) 

#         # gt_local_pts_h = homogenize_points(gt_local_pts_raw).view(B, N, -1, 4).transpose(2, 3) 
#         # gt_pts = torch.matmul(poses, gt_local_pts_h).transpose(2, 3).reshape(B, N, H, W, 4)[..., :3] 
#         # R = poses[..., :3, :3]       # [B, N, 3, 3]
#         # t = poses[..., :3, 3:4]      # [B, N, 3, 1]
#         # # [B, N, 3, H*W]
#         # pts_flat = gt_local_pts_raw.reshape(B, N, -1, 3).transpose(-1, -2) 
#         # # R @ pts + t，然后变回原形状
#         # gt_pts = (torch.matmul(R, pts_flat) + t).transpose(-1, -2).reshape(B, N, H, W, 3)

#         # # w2c_target = se3_inverse(poses[:, 0])
#         # # gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
#         # # poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)
#         # w2c_target = se3_inverse(poses[:, 0])
#         # R_w2c = w2c_target[..., :3, :3]    # [B, 3, 3]
#         # t_w2c = w2c_target[..., :3, 3:4]   # [B, 3, 1]
#         # # 将 gt_pts 从 [B, N, H, W, 3] 展平为 [B, 3, N*H*W]
#         # # gt_pts_flat = gt_pts.view(B, -1, 3).transpose(-1, -2)
#         # gt_pts_flat = gt_pts.reshape(B, -1, 3).transpose(-1, -2)
#         # # 用 bmm (Batch Matrix Multiply) 加速运算
#         # gt_pts = (torch.bmm(R_w2c, gt_pts_flat) + t_w2c).transpose(-1, -2).reshape(B, N, H, W, 3)
        
#         # poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

#         # valid_batch = masks.view(B, -1).sum(dim=-1) > 0 
        
#         # if valid_batch.sum() > 0:
#         #     B_ = valid_batch.sum()
#         #     all_pts = gt_pts[valid_batch].clone() 
            
#         #     mask_bool = masks[valid_batch].squeeze(-1) 
#         #     all_pts[~mask_bool] = 0
            
#         #     all_pts = all_pts.reshape(B_, -1, 3) 
#         #     all_dis = all_pts.norm(dim=-1)       
            
#         #     num_valid_pts = mask_bool.view(B_, -1).float().sum(dim=-1) 
            
#         #     norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) 
#         #     norm_factor = norm_factor.clamp_min(1e-4)

#         #     gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
#         #     poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
#         #     gt_depths[valid_batch] /= norm_factor[..., None, None, None]

#         # extrinsics = se3_inverse(poses)
#         # gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
#         # extrinsics = se3_inverse(poses)
#         # R_ext = extrinsics[..., :3, :3]    # [B, N, 3, 3]
#         # t_ext = extrinsics[..., :3, 3:4]   # [B, N, 3, 1]
#         # gt_pts_flat2 = gt_pts.view(B, N, -1, 3).transpose(-1, -2) # [B, N, 3, H*W]
#         # gt_local_pts = (torch.matmul(R_ext, gt_pts_flat2) + t_ext).transpose(-1, -2).reshape(B, N, H, W, 3)
#         return dict(
#             imgs = imgs,
#             gt_ks = gt_ks,
#             # gt_depths = gt_depths,
#             # global_points = gt_pts,
#             # gt_local_pts = gt_local_pts, 
#             # masks = masks,
#             # gt_c2w = poses
#         )

#     def normalize_pred(self, pred, gt):
#         # ... (保持不变) ...
#         local_points = pred['local_points']
#         camera_poses = pred['camera_poses']
#         B, N, H, W, _ = local_points.shape
#         masks = gt['masks'] 
        
#         mask_bool = masks.squeeze(-1) 

#         all_pts = local_points.clone()
#         all_pts[~mask_bool] = 0 
        
#         all_pts = all_pts.reshape(B, -1, 3) 
#         all_dis = all_pts.norm(dim=-1)      
        
#         num_valid_pts = mask_bool.view(B, -1).float().sum(dim=-1) 
        
#         norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) 
#         norm_factor = norm_factor.clamp_min(1e-4) 
        
#         local_points = local_points / norm_factor[..., None, None, None, None]
        
#         camera_poses_normalized = camera_poses.clone()
#         camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

#         pred['local_points'] = local_points
#         pred['camera_poses'] = camera_poses_normalized

#         if 'gaussians' in pred:
#             pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
#             pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

#         return pred

#     # === 注意：彻底删除了 prepare_ROE ===

#     def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
#         # ... (保持不变) ...
#         opacities = gaussians["opacity"].clone()
        
#         # if self.train_stage == 3 or not self.training:
#         #     means = gaussians["xyz"].detach().contiguous()
#         #     quats = gaussians["rotation"].detach().contiguous()
#         #     scales = gaussians["scale"].detach().contiguous()
#         #     colors = gaussians["color"].detach().contiguous()
#         #     conf_prob = torch.sigmoid(gaussians["conf"])
#         #     opacities = (opacities.detach() * conf_prob).squeeze(-1).contiguous()
#         # else:
#         means = gaussians["xyz"].contiguous()
#         quats = gaussians["rotation"].contiguous()
#         scales = gaussians["scale"].contiguous()
#         colors = gaussians["color"].contiguous()
#         opacities = opacities.squeeze(-1).contiguous()

#         return rasterization(
#             means=means, quats=quats, scales=scales,
#             opacities=opacities, colors=colors,
#             viewmats=w2c, Ks=ks, width=W, height=H, render_mode=render_mode, packed=False
#         )

#     def forward(self, pred, gt_raw, batch_idx=0, current_epoch=None, total_epochs=None, **kwargs):
#         gt = self.prepare_gt(gt_raw)
        
#         B, N_total, C, H, W = gt['imgs'].shape
#         sub_idx = torch.arange(0, N_total, 1, device=gt['imgs'].device)
#         self.lpips_loss_fn = self.lpips_loss_fn.to(gt['imgs'].device)  # 确保 LPIPS 损失函数在正确的设备上
#         N_sub = len(sub_idx)

#         gt_sub_mask = {'masks': gt['masks']}
#         pred = self.normalize_pred(pred, gt_sub_mask) 
        
#         gt_ks, gt_imgs = gt['gt_ks'], gt['imgs']
#         # gt_local_pts = gt['gt_local_pts']
        
#         # valid_masks_sub = valid_masks[:, sub_idx].squeeze(-1)
#         # gt_local_pts_sub = gt_local_pts[:, sub_idx]

#         gauss_raw = pred['gaussians']
#         pred_c2w = pred['camera_poses']
#         intrinsics_pred = pred['intrinsics'] 
#         # print("Ground Truth Intrinsics:", gt_ks[0, 0])  # 打印第一个视角的 GT 内参以供调试
#         # print("Predicted Intrinsics:", intrinsics_pred[0, 0])  # 打印第一个视角的预测内参以供调试
#         warmup_steps = 7500.0 # 前 60% 步数用于预热光度损失
#         progress = min(batch_idx / warmup_steps, 1.0)

#         # 光度损失 (RGB/SSIM) 从 10% 平滑增加到 100%
#         start_ratio = 0.1
#         photo_ratio = start_ratio + (1.0 - start_ratio) * progress
#         cur_lambda_rgb = self.lambda_rgb * photo_ratio
#         cur_lambda_ssim = self.lambda_ssim * photo_ratio
#         lpips_start_step = 2500.0
#         lpips_total_steps = 12500.0  # 100000 - 50000
#         if batch_idx > lpips_start_step:
#             lpips_progress = min((batch_idx - lpips_start_step) / lpips_total_steps, 1.0)
#         else:
#             lpips_progress = 0.0
            
#         cur_lambda_lpips = self.lambda_lpips * lpips_progress
#         # cur_lambda_lpips = self.lambda_lpips * max(photo_ratio, 1.0)
#         # 几何损失 (Depth) 从 1.0 平滑衰减到 0.5，前期强行主导几何框架
#         depth_ratio = 1.0 - 0.5 * progress 
#         cur_lambda_depth = self.lambda_depth * depth_ratio
#         pred_local_pts = torch.clamp(pred['local_points'], min=-1e4, max=1e4)

#         loss_rgb = loss_ssim = loss_depth = loss_pose = loss_conf = loss_scale = loss_edge = torch.tensor(0.0, device=pred_c2w.device)
#         details = {}

#         # === 删除了所有计算和应用 scale_opt 的逻辑 ===

#         gauss_render = {k: v for k, v in gauss_raw.items()} 
#         render_c2w = pred_c2w.clone()

#         # === 删除了渲染前的尺度对齐干预 ===

#         render_w2c = se3_inverse(render_c2w)
        
#         # render_out, _, _ = self._render_gs(gauss_render, render_w2c, intrinsics_pred, H, W, render_mode='RGB+ED')
        
#         # rgb_full = render_out[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
#         # depth_map = render_out[..., 3:4].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
#         # === 修改后代码 ===
#         # 1. 渲染完整的 RGB (保留被推远的高斯作为背景和天空)
#         render_out_rgb, _, _ = self._render_gs(gauss_render, render_w2c, intrinsics_pred, H, W, render_mode='RGB')
#         rgb_full = render_out_rgb[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
        
#         # 2. 构造专用于渲染 Depth 的高斯字典，剔除低 conf 节点
#         gauss_render_depth = {k: v for k, v in gauss_render.items()}
#         if "conf" in gauss_render:
#             conf_prob = torch.sigmoid(gauss_render["conf"])
#             # 如果 conf < 0.1，将其透明度强制设为 0，防止其极大的 z 值污染近景 Expected Depth
#             gauss_render_depth["opacity"] = torch.where(
#                 conf_prob < 0.1, 
#                 torch.zeros_like(gauss_render["opacity"]), 
#                 gauss_render["opacity"]
#             )
            
#         # 使用 ED (Expected Depth) 模式重新渲染深度
#         render_out_depth, _, _ = self._render_gs(gauss_render_depth, render_w2c, intrinsics_pred, H, W, render_mode='ED')
        
#         # ED 模式返回的是单通道深度图，切片 [..., 0:1]
#         depth_map = render_out_depth[..., 0:1].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
        
#         gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)
#         # gt_depth_reshaped = gt_depths.reshape(B * N_total, 1, H, W)
            
#         # with torch.no_grad():
#         #     batch_idx = 0
#         #     num_viz = min(4, B * N_total)
#         #     rgb_viz = torch.cat([gt_imgs_reshaped[:num_viz], rgb_full[:num_viz]], dim=2) 
            
#         #     # 直接使用 depth_map 进行可视化
#         #     d_pred_viz = depth_map[:num_viz] / (depth_map[:num_viz].max() + 1e-5)
#         #     d_gt_viz = gt_depth_reshaped[:num_viz] / (gt_depth_reshaped[:num_viz].max() + 1e-5)
#         #     d_pred_viz = d_pred_viz.repeat(1, 3, 1, 1)
#         #     d_gt_viz = d_gt_viz.repeat(1, 3, 1, 1)
#         #     depth_viz = torch.cat([d_gt_viz, d_pred_viz], dim=2)
            
#         #     final_viz = torch.cat([rgb_viz, depth_viz], dim=3)
#         #     torchvision.utils.save_image(final_viz, f"debug_output/step_{batch_idx}_stage_{self.train_stage}.png")

#         if self.train_stage in [1, 2, 3]:
#             loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
#             loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)
#             loss_lpips = self.lpips_loss_fn(rgb_full, gt_imgs_reshaped).mean()
#             # 使用 pred_local_pts 提取深度作为伪标签约束
#             conf_mask = torch.sigmoid(pred['conf'][..., 0]) > 0.1 
#             non_edge_mask = ~depth_edge(pred['local_points'][..., 2], rtol=0.03) 
            
#             pseudo_mask = torch.logical_and(conf_mask, non_edge_mask)
#             pseudo_mask = pseudo_mask.reshape(B * N_total, 1, H, W)
#             pseudo_gt_depth = pred_local_pts[..., 2:3].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
            
#             if self.lambda_depth > 0 and pseudo_mask.sum() > 10:
#                 loss_depth = F.l1_loss(depth_map[pseudo_mask], pseudo_gt_depth[pseudo_mask].detach())

#         # ==========================================================
#         # 【修改】：应用动态权重
#         # ==========================================================
#         final_loss = (
#             cur_lambda_rgb * loss_rgb + 
#             cur_lambda_ssim * loss_ssim +
#             self.lambda_depth * loss_depth +
#             cur_lambda_lpips * loss_lpips
#         )

#         if final_loss == 0.0:
#              final_loss = (pred_local_pts.sum() * 0.0)

#         details.update({
#             "loss_rgb": loss_rgb,
#             "loss_ssim": loss_ssim,
#             "loss_depth": loss_depth, 
#             "loss_lpips": loss_lpips,

#             "cur_weight_rgb": torch.tensor(cur_lambda_rgb, device=pred_c2w.device),   # 添加监控
#             "cur_weight_depth": torch.tensor(cur_lambda_depth, device=pred_c2w.device), # 添加监控
#             "total_loss": final_loss
#         })

#         return final_loss, details

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

def sobel_edge_loss(pred, gt):
    """
    计算图像的 Sobel 边缘/梯度 L1 Loss。
    输入: pred, gt 形状均为 [B, C, H, W]
    """
    device = pred.device
    channels = pred.size(1)
    
    sobel_x = torch.tensor([[-1., 0., 1.], 
                            [-2., 0., 2.], 
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], 
                            [0.,  0.,  0.], 
                            [1.,  2.,  1.]], device=device).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
    pred_dx = F.conv2d(pred, sobel_x, padding=1, groups=channels)
    pred_dy = F.conv2d(pred, sobel_y, padding=1, groups=channels)
    
    gt_dx = F.conv2d(gt, sobel_x, padding=1, groups=channels)
    gt_dy = F.conv2d(gt, sobel_y, padding=1, groups=channels)
    
    loss_dx = F.l1_loss(pred_dx, gt_dx)
    loss_dy = F.l1_loss(pred_dy, gt_dy)
    
    return loss_dx + loss_dy

class Pi3LossGS(nn.Module):
    def __init__(
            self, lambda_rgb=1, lambda_ssim=0.5, lambda_depth=5, 
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
        self.lpips_loss_fn = lpips.LPIPS(net='alex').eval()
        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False

        self.register_buffer('cached_grid_x', None)
        self.register_buffer('cached_grid_y', None)
        

    def prepare_gt(self, gt):
        imgs = torch.stack([view['img'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
        return dict(
            imgs = imgs,
            gt_ks = gt_ks,
        )

    # def normalize_pred(self, pred, gt):
    #     local_points = pred['local_points']
    #     camera_poses = pred['camera_poses']
    #     B, N, H, W, _ = local_points.shape
    #     masks = gt['masks'] 
        
    #     mask_bool = masks.squeeze(-1) 

    #     all_pts = local_points.clone()
    #     all_pts[~mask_bool] = 0 
        
    #     all_pts = all_pts.reshape(B, -1, 3) 
    #     all_dis = all_pts.norm(dim=-1)      
        
    #     num_valid_pts = mask_bool.view(B, -1).float().sum(dim=-1) 
        
    #     norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) 
    #     norm_factor = norm_factor.clamp_min(1e-4) 
        
    #     local_points = local_points / norm_factor[..., None, None, None, None]
        
    #     camera_poses_normalized = camera_poses.clone()
    #     camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

    #     pred['local_points'] = local_points
    #     pred['camera_poses'] = camera_poses_normalized

    #     if 'gaussians' in pred:
    #         pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
    #         pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

    #     return pred

    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['masks'] 
        
        mask_bool = masks.squeeze(-1) 

        all_pts = local_points.clone()
        all_pts[~mask_bool] = 0 
        
        all_pts = all_pts.reshape(B, -1, 3) 
        all_dis = all_pts.norm(dim=-1)      
        
        norm_factor = torch.ones(B, device=all_dis.device, dtype=all_dis.dtype)
        for b in range(B):
            valid_dis = all_dis[b][mask_bool.view(B, -1)[b]]
            if valid_dis.numel() > 0:
                # 【核心修复 1】：使用中位数 (Median) 
                # 中位数对极端的“飞点”完全免疫，能永远把场景最核心的主体稳定锚定在距离 1.0 的安全区
                norm_factor[b] = torch.median(valid_dis)
                
        # 兜底保护
        norm_factor = norm_factor.clamp_min(1e-4) 
        
        # 将场景主体统一缩放归一化
        local_points = local_points / norm_factor[..., None, None, None, None]
        
        # 【核心修复 2】：硬性物理截断
        # 满足你的需求：归一化后，绝对不允许任何点飞出 1000 的范围，直接切平
        local_points = torch.clamp(local_points, min=-1000.0, max=1000.0)
        
        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        if 'gaussians' in pred:
            # 同步缩放高斯属性
            pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
            # 同样对高斯坐标施加 100 的硬截断
            pred['gaussians']['xyz'] = torch.clamp(pred['gaussians']['xyz'], min=-1000.0, max=1000.0)
            
            pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

        return pred


    def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
        opacities = gaussians["opacity"]
        
        means = gaussians["xyz"]
        quats = gaussians["rotation"]
        scales = gaussians["scale"]
        colors = gaussians["color"]
        opacities = opacities.squeeze(-1)

        return rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=w2c, Ks=ks, width=W, height=H, render_mode=render_mode, packed=True
        )

    def forward(self, pred, gt_raw, batch_idx=0, current_epoch=None, total_epochs=None, **kwargs):
        gt = self.prepare_gt(gt_raw)
        
        B, N_total, C, H, W = gt['imgs'].shape
        sub_idx = torch.arange(0, N_total, 1, device=gt['imgs'].device)
        self.lpips_loss_fn = self.lpips_loss_fn.to(gt['imgs'].device)  
        N_sub = len(sub_idx)

        # ==========================================
        # === 修改处 1: 提前计算 pseudo_mask 代替 gt_mask ===
        # ==========================================
        conf_mask = torch.sigmoid(pred['conf'][..., 0]) > 0.1 
        non_edge_mask = ~depth_edge(pred['local_points'][..., 2], rtol=0.03) 
        pseudo_mask_2d = torch.logical_and(conf_mask, non_edge_mask) # 形状: [B, N, H, W]
        
        # 增加最后通道维度以适配 normalize_pred 所需的 [B, N, H, W, 1] 形状
        pseudo_mask_dict = {'masks': pseudo_mask_2d.unsqueeze(-1)}
        
        # 传入 pseudo_mask 进行归一化
        pred = self.normalize_pred(pred, pseudo_mask_dict) 
        # ==========================================
        
        gt_ks, gt_imgs = gt['gt_ks'], gt['imgs']

        gauss_raw = pred['gaussians']
        pred_c2w = pred['camera_poses']
        intrinsics_pred = pred['intrinsics'] 

        warmup_steps = 7500.0 
        progress = min(batch_idx / warmup_steps, 1.0)

        start_ratio = 0.1
        photo_ratio = start_ratio + (1.0 - start_ratio) * progress
        cur_lambda_rgb = self.lambda_rgb * photo_ratio
        cur_lambda_ssim = self.lambda_ssim * photo_ratio
        
        lpips_start_step = 2500.0
        lpips_total_steps = 12500.0  
        if batch_idx > lpips_start_step:
            lpips_progress = min((batch_idx - lpips_start_step) / lpips_total_steps, 1.0)
        else:
            lpips_progress = 0.0
            
        cur_lambda_lpips = self.lambda_lpips * lpips_progress
        
        depth_ratio = 1.0 - 0.5 * progress 
        cur_lambda_depth = self.lambda_depth * depth_ratio
        pred_local_pts = torch.clamp(pred['local_points'], min=-1e4, max=1e4)

        loss_rgb = loss_ssim = loss_depth = loss_pose = loss_conf = loss_scale = loss_edge = torch.tensor(0.0, device=pred_c2w.device)
        details = {}

        gauss_render = {k: v for k, v in gauss_raw.items()} 
        render_c2w = pred_c2w.clone()

        render_w2c = se3_inverse(render_c2w)
        
        # 1. 渲染完整的 RGB
        render_out_rgb, _, _ = self._render_gs(gauss_render, render_w2c, intrinsics_pred, H, W, render_mode='RGB')
        rgb_full = render_out_rgb[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
        
        # 2. 构造专用于渲染 Depth 的高斯字典
        gauss_render_depth = {k: v for k, v in gauss_render.items()}
        if "conf" in gauss_render:
            conf_prob = torch.sigmoid(gauss_render["conf"])
            gauss_render_depth["opacity"] = torch.where(
                conf_prob < 0.1, 
                torch.zeros_like(gauss_render["opacity"]), 
                gauss_render["opacity"]
            )
            
        render_out_depth, _, _ = self._render_gs(gauss_render_depth, render_w2c, intrinsics_pred, H, W, render_mode='ED')
        
        depth_map = render_out_depth[..., 0:1].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
        
        gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)

        if self.train_stage in [1, 2, 3]:
            loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
            loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)
            loss_lpips = self.lpips_loss_fn(rgb_full, gt_imgs_reshaped).mean()
            
            # ==========================================
            # === 修改处 2: 复用之前算好的 pseudo_mask ===
            # ==========================================
            # 直接使用我们在 forward 开头计算出来的 pseudo_mask_2d，只需进行 reshape 即可
            pseudo_mask = pseudo_mask_2d.reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
            pseudo_gt_depth = pred_local_pts[..., 2:3].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
            
            if self.lambda_depth > 0 and pseudo_mask.sum() > 10:
                loss_depth = F.l1_loss(depth_map[pseudo_mask], pseudo_gt_depth[pseudo_mask].detach())
            # ==========================================

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
            "cur_weight_rgb": torch.tensor(cur_lambda_rgb, device=pred_c2w.device),   
            "cur_weight_depth": torch.tensor(cur_lambda_depth, device=pred_c2w.device), 
            "total_loss": final_loss
        })

        return final_loss, details