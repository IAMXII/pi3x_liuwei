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

class Pi3LossGS(nn.Module):
    def __init__(
            self, lambda_rgb=1, lambda_ssim=0.5, lambda_depth=5,
            lambda_pose=0.2, lambda_scale=0.1, train_stage=1, local_align_res=4096,
            lambda_anchor=0.2, lambda_budget=0.05, lambda_overlap=0.1,
            lambda_lowconf_flat=0.05, lambda_lowconf_coarse=0.05,
            lambda_proto_ownership=0.08, lambda_proto_usage=0.02, lambda_proto_compact=0.01,
            lpips_every=1
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_pose = lambda_pose
        self.lambda_scale = lambda_scale
        self.lambda_lpips = 0.1
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_budget = float(lambda_budget)
        self.lambda_overlap = float(lambda_overlap)
        self.lambda_lowconf_flat = float(lambda_lowconf_flat)
        self.lambda_lowconf_coarse = float(lambda_lowconf_coarse)
        self.lambda_proto_ownership = float(lambda_proto_ownership)
        self.lambda_proto_usage = float(lambda_proto_usage)
        self.lambda_proto_compact = float(lambda_proto_compact)
        self.lpips_every = max(1, int(lpips_every))
        
        self.train_stage = int(train_stage) 
        self.local_align_res = local_align_res
        
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

        loss_rgb = loss_ssim = loss_depth = torch.tensor(0.0, device=pred_c2w.device)
        details = {}

        gauss_render = {k: v for k, v in gauss_raw.items()} 
        render_c2w = pred_c2w.clone()

        render_w2c = se3_inverse(render_c2w)
        
        # 1. 渲染完整的 RGB
        render_out_rgb, _, _ = self._render_gs(gauss_render, render_w2c, intrinsics_pred, H, W, render_mode='RGB')
        rgb_full = render_out_rgb[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
        
        # 2. Depth 渲染与 RGB 使用同一组高斯；point conf 只用于监督区域选择，不直接裁剪高斯
        render_out_depth, _, _ = self._render_gs(gauss_render, render_w2c, intrinsics_pred, H, W, render_mode='ED')
        
        depth_map = render_out_depth[..., 0:1].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
        
        gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)

        if self.train_stage in [1, 2, 3]:
            loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
            loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)
            if batch_idx % self.lpips_every == 0:
                loss_lpips = self.lpips_loss_fn(rgb_full, gt_imgs_reshaped).mean()
            else:
                loss_lpips = rgb_full.new_tensor(0.0)

            pseudo_mask = pseudo_mask_2d.reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
            pseudo_gt_depth = pred_local_pts[..., 2:3].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)

            if self.lambda_depth > 0 and pseudo_mask.sum() > 10:
                loss_depth = F.l1_loss(depth_map[pseudo_mask], pseudo_gt_depth[pseudo_mask].detach())

        aux_regs = pred.get('regularizers', {})
        loss_anchor = aux_regs.get('loss_anchor', torch.tensor(0.0, device=pred_c2w.device))
        loss_budget = aux_regs.get('loss_budget', torch.tensor(0.0, device=pred_c2w.device))
        loss_overlap = aux_regs.get('loss_overlap', torch.tensor(0.0, device=pred_c2w.device))
        # loss_lowconf_flat = aux_regs.get('loss_lowconf_flat', torch.tensor(0.0, device=pred_c2w.device))
        # loss_lowconf_coarse = aux_regs.get('loss_lowconf_coarse', torch.tensor(0.0, device=pred_c2w.device))
        loss_proto_ownership = aux_regs.get('loss_proto_ownership', torch.tensor(0.0, device=pred_c2w.device))
        loss_proto_usage = aux_regs.get('loss_proto_usage', torch.tensor(0.0, device=pred_c2w.device))
        # loss_proto_compact = aux_regs.get('loss_proto_compact', torch.tensor(0.0, device=pred_c2w.device))
        loss_sparsity = torch.tensor(0.0, device=pred_c2w.device)
        # loss_volume = torch.tensor(0.0, device=pred_c2w.device)
        
        valid_render_mask = gauss_raw['opacity'].squeeze(-1) > 0.05
        if valid_render_mask.sum() > 0:
            active_opacities = gauss_raw['opacity'][valid_render_mask]
            active_scales = gauss_raw['scale'][valid_render_mask]
            
            # 【修复 3】：将 opacity 截断，防止极小的负浮点数导致 log 出现 nan
            eps = 1e-6
            active_opacities_safe = active_opacities.clamp(eps, 1.0 - eps)
            loss_sparsity = -(
                active_opacities_safe * torch.log(active_opacities_safe) + 
                (1.0 - active_opacities_safe) * torch.log(1.0 - active_opacities_safe)
            ).mean()
            
            # 【修复 1】：将 scale 强转为 fp32 计算体积，防止 fp16 溢出 (65504 上限)
            active_scales_fp32 = active_scales.float()
            vol = active_scales_fp32[..., 0] * active_scales_fp32[..., 1] * active_scales_fp32[..., 2]
            loss_volume = (vol * active_opacities.squeeze(-1).float()).mean()
        final_loss = (
            cur_lambda_rgb * loss_rgb +
            cur_lambda_ssim * loss_ssim +
            cur_lambda_depth * loss_depth +
            cur_lambda_lpips * loss_lpips +
            self.lambda_anchor * loss_anchor +
            self.lambda_budget * loss_budget +
            self.lambda_overlap * loss_overlap +
            self.lambda_proto_ownership * loss_proto_ownership +
            self.lambda_proto_usage * loss_proto_usage +
            # self.lambda_proto_compact * loss_proto_compact +
            # self.lambda_lowconf_flat * loss_lowconf_flat +
            # self.lambda_lowconf_coarse * loss_lowconf_coarse +
            0.05 * loss_sparsity
            # 0.001 * loss_volume
        )

        if final_loss == 0.0:
             final_loss = (pred_local_pts.sum() * 0.0)
             

        details.update({
            "loss_rgb": loss_rgb,
            "loss_ssim": loss_ssim,
            "loss_depth": loss_depth, 
            "loss_lpips": loss_lpips,
            "loss_anchor": loss_anchor,
            "loss_budget": loss_budget,
            "loss_overlap": loss_overlap,
            "loss_proto_ownership": loss_proto_ownership,
            "loss_proto_usage": loss_proto_usage,
            # "loss_proto_compact": loss_proto_compact,
            # "loss_lowconf_flat": loss_lowconf_flat,
            # "loss_lowconf_coarse": loss_lowconf_coarse,
            "loss_sparsity": loss_sparsity,
            # "loss_volume": loss_volume,
            "selected_before_fusion": aux_regs.get('selected_before_fusion', torch.tensor(0.0, device=pred_c2w.device)),
            "selected_after_fusion": aux_regs.get('selected_after_fusion', torch.tensor(0.0, device=pred_c2w.device)),
            "teacher_ratio": aux_regs.get('teacher_ratio', torch.tensor(0.0, device=pred_c2w.device)),
            "cur_weight_rgb": torch.tensor(cur_lambda_rgb, device=pred_c2w.device),
            "cur_weight_depth": torch.tensor(cur_lambda_depth, device=pred_c2w.device),
            "total_loss": final_loss
        })

        if "selected_gaussians" in pred:
            details["selected_gaussians"] = pred["selected_gaussians"].detach().float()

        return final_loss, details