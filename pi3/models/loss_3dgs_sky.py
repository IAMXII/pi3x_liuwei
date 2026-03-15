import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .pi3_3dgs import matrix_to_quaternion, quat_mult
from ..utils.alignment import align_points_scale
from ..utils.geometry import depth_edge, homogenize_points

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

class CameraPoseLoss(nn.Module):
    def __init__(self, alpha=10000):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))
        return R_err.mean()
    
    def forward(self, pred_pose, gt_pose, scale):
        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *= scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_rel_all = torch.matmul(pred_w2c.unsqueeze(2), pred_pose_align.unsqueeze(1))
        gt_rel_all = torch.matmul(gt_w2c.unsqueeze(2), gt_pose.unsqueeze(1))

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)
        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        rot_loss = self.rot_ang_loss(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3))
        
        return self.alpha * trans_loss + rot_loss, trans_loss, rot_loss

class Pi3LossGS(nn.Module):
    def __init__(
            self, lambda_rgb=1.2, lambda_ssim=0.7, lambda_depth=0.5, 
            lambda_pose=0.2, lambda_scale=0.1, lambda_pts=0.7, train_stage=1, local_align_res=4096,
            train_conf=False, num_sky_anchors=8196 
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_pose = lambda_pose
        self.lambda_scale = lambda_scale
        self.lambda_pts = lambda_pts 
        
        # 强制转换为 int，防止 YAML 解析为字符串导致的幽灵 Bug
        self.train_stage = int(train_stage) 
        self.local_align_res = local_align_res
        
        self.train_conf = train_conf 
        self.num_sky_anchors = num_sky_anchors
        self.camera_loss_fn = CameraPoseLoss()

    def prepare_gt(self, gt):
        """支持自动从 Depth+Pose+Intrinsics 反投影生成 pts3d 的对齐与 norm 逻辑"""
        # 1. 安全提取 Dataloader 传来的基础数据
        imgs = torch.stack([view['img'] for view in gt], dim=1)
        gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)

        B, N, H, W = gt_depths.shape
        device = gt_depths.device

        # ==========================================
        # 核心补全：动态生成 valid_mask 和 pts3d
        # ==========================================
        # A. 生成掩模: 深度值有效的区域 (大于极小值)
        masks = (gt_depths > 1e-4).unsqueeze(-1) # [B, N, H, W, 1]

        # B. 像素坐标网格反投影
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        grid_x = grid_x.expand(B, N, -1, -1).float()
        grid_y = grid_y.expand(B, N, -1, -1).float()

        # 提取内参
        fx = gt_ks[..., 0, 0].view(B, N, 1, 1)
        fy = gt_ks[..., 1, 1].view(B, N, 1, 1)
        cx = gt_ks[..., 0, 2].view(B, N, 1, 1)
        cy = gt_ks[..., 1, 2].view(B, N, 1, 1)

        # 计算相机局部坐标 (Local Points)
        local_x = (grid_x - cx) * gt_depths / fx
        local_y = (grid_y - cy) * gt_depths / fy
        local_z = gt_depths
        gt_local_pts_raw = torch.stack([local_x, local_y, local_z], dim=-1) # [B, N, H, W, 3]

        # 转换到全局坐标 (Global Points)
        gt_local_pts_h = homogenize_points(gt_local_pts_raw).view(B, N, -1, 4).transpose(2, 3) # [B, N, 4, H*W]
        # 假设 poses 是 Camera-to-World (c2w)
        gt_pts = torch.matmul(poses, gt_local_pts_h).transpose(2, 3).reshape(B, N, H, W, 4)[..., :3] 
        # ==========================================

        # --- 以下无缝衔接你原本的坐标系统一与对齐逻辑 ---
        # 统一坐标系到第一个视角
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # ==========================================
        # 规范化全局尺度 (极其关键，防止 NaN)
        # ==========================================
        # [修复 1]: 确保 valid_batch 是一维张量 [B]
        valid_batch = masks.view(B, -1).sum(dim=-1) > 0 
        
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone() # 形状: [B_, N, H, W, 3]
            
            # 使用展开后的布尔掩模置零无效点
            mask_bool = masks[valid_batch].squeeze(-1) # 形状: [B_, N, H, W]
            all_pts[~mask_bool] = 0
            
            # [修复 2]: 将 N 和 H*W 展平，避免复杂的维度计算
            all_pts = all_pts.reshape(B_, -1, 3) # 形状: [B_, N*H*W, 3]
            all_dis = all_pts.norm(dim=-1)       # 形状: [B_, N*H*W]
            
            # 分母：计算每个 Batch 有多少个有效点
            num_valid_pts = mask_bool.view(B_, -1).float().sum(dim=-1) # 形状: [B_]
            
            # 计算缩放因子
            norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) # 形状: [B_]
            norm_factor = norm_factor.clamp_min(1e-4)

            # 执行缩放 (利用 None 自动对齐广播维度)
            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
            gt_depths[valid_batch] /= norm_factor[..., None, None, None]
        # ==========================================

        # 重新转换出安全的 local_pts 供后续 loss 使用
        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]

        return dict(
            imgs = imgs,
            gt_ks = gt_ks,
            gt_depths = gt_depths,
            global_points = gt_pts,
            gt_local_pts = gt_local_pts, 
            masks = masks,
            gt_c2w = poses
        )

    def normalize_pred(self, pred, gt):
        """恢复原版预测结果的尺度对齐 (已彻底修复维度匹配问题)"""
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['masks'] # 原始形状: [B, N, H, W, 1]
        
        # 1. 挤掉最后一维，变成纯纯的布尔掩模 [B, N, H, W]
        mask_bool = masks.squeeze(-1) 

        all_pts = local_points.clone()
        # 此时形状完美匹配，不会报错了！
        all_pts[~mask_bool] = 0 
        
        # 2. 展平空间维度，避免复杂的多维 sum
        all_pts = all_pts.reshape(B, -1, 3) # 形状: [B, N*H*W, 3]
        all_dis = all_pts.norm(dim=-1)      # 形状: [B, N*H*W]
        
        # 3. 精准计算每个 Batch 有多少个有效点
        num_valid_pts = mask_bool.view(B, -1).float().sum(dim=-1) # 形状: [B]
        
        # 计算缩放因子
        norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8) # 形状: [B]
        norm_factor = norm_factor.clamp_min(1e-4) # 防护除零
        
        # 4. 执行缩放对齐
        local_points = local_points / norm_factor[..., None, None, None, None]
        
        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        # 同步缩放高斯
        if 'gaussians' in pred:
            pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
            pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

        return pred

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        for i in range(B):
            valid_pts = pts[i][mask[i]]
            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')
                valid_pts = valid_pts.squeeze(0).permute(1, 0)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)
            output.append(valid_pts)
        return torch.stack(output, dim=0)

    def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
        opacities = gaussians["opacity"].clone()
        
        if self.train_stage == 3 or not self.training:
            means = gaussians["xyz"].detach().contiguous()
            quats = gaussians["rotation"].detach().contiguous()
            scales = gaussians["scale"].detach().contiguous()
            colors = gaussians["color"].detach().contiguous()
            conf_prob = torch.sigmoid(gaussians["conf"])
            opacities = (opacities.detach() * conf_prob).squeeze(-1).contiguous()
        else:
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
        sub_idx = torch.arange(0, N_total, 2, device=gt['imgs'].device)
        N_sub = len(sub_idx)

        gt_sub_mask = {'masks': gt['masks'][:, sub_idx]}
        pred = self.normalize_pred(pred, gt_sub_mask) 
        
        gt_ks, gt_c2w, gt_imgs, gt_depths, valid_masks = gt['gt_ks'], gt['gt_c2w'], gt['imgs'], gt['gt_depths'], gt['masks']
        gt_local_pts = gt['gt_local_pts']
        
        valid_masks_sub = valid_masks[:, sub_idx].squeeze(-1)
        gt_local_pts_sub = gt_local_pts[:, sub_idx]

        gauss_raw = pred['gaussians']
        pred_c2w = pred['camera_poses'] # 此时全量位姿预测为 [B, N_total, 4, 4]
        
        pred_local_pts = torch.clamp(pred['local_points'], min=-1e4, max=1e4)

        loss_rgb = loss_ssim = loss_depth = loss_pose = loss_conf = loss_scale = loss_pts = torch.tensor(0.0, device=pred_c2w.device)
        details = {}

        scale_opt = torch.ones((B,), device=pred_c2w.device)
        
        if self.train_stage in [1, 2]:
            weights_ = gt_local_pts_sub[..., 2].clamp_min(1e-3)
            weights_ = 1 / (weights_ + 1e-6)
            
            xyz_pred_ROE = self.prepare_ROE(pred_local_pts.reshape(B, N_sub, H, W, 3), valid_masks_sub, target_size=self.local_align_res)
            with torch.no_grad():
                xyz_gt_ROE = self.prepare_ROE(gt_local_pts_sub.reshape(B, N_sub, H, W, 3), valid_masks_sub, target_size=self.local_align_res)
                xyz_w_ROE = self.prepare_ROE((weights_[..., None]).reshape(B, N_sub, H, W, 1), valid_masks_sub, target_size=self.local_align_res)[..., 0]
            
            scale_opt = align_points_scale(xyz_pred_ROE, xyz_gt_ROE, xyz_w_ROE)
            
            # scale_opt = torch.nan_to_num(scale_opt, nan=1.0, posinf=1.0, neginf=1.0)
            # scale_opt = torch.where(scale_opt <= 0, -scale_opt, scale_opt)
            # scale_opt = torch.clamp(scale_opt, min=1e-3, max=100.0) 

            # loss_scale = F.l1_loss(scale_opt, torch.ones_like(scale_opt))

            if valid_masks_sub.sum() > 0:
                aligned_local_pts = pred_local_pts * scale_opt.view(B, 1, 1, 1, 1)
                
                loss_pts = F.l1_loss(aligned_local_pts[valid_masks_sub], gt_local_pts_sub[valid_masks_sub])
            else:
                loss_pts = (pred_local_pts.sum() * 0.0)

        # ==========================================================
        # [修改点 5] 高斯渲染时全面使用 PRED Pose (覆盖所有 N_total 视角)
        # ==========================================================
        gauss_render = {k: v for k, v in gauss_raw.items()} 
        # 直接使用全网预测的 N_total 相机位姿（此处无需利用 GT 锚定，相机渲染是相对系）
        render_c2w = pred_c2w.clone()

        if self.train_stage in [1, 2]:
            detach_scale = scale_opt.detach().view(B, 1, 1)
            render_c2w[..., :3, 3] *= detach_scale
            gauss_render["xyz"] = gauss_raw["xyz"] * detach_scale
            gauss_render["scale"] = gauss_raw["scale"] * detach_scale

        render_w2c = se3_inverse(render_c2w)
        
        rgb_full, _, _ = self._render_gs(gauss_render, render_w2c, gt_ks, H, W, render_mode='RGB')
        rgb_full = rgb_full.reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
        gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)

        depth_map, _, _ = self._render_gs(gauss_render, render_w2c, gt_ks, H, W, render_mode='ED')
        depth_map = depth_map.reshape(B * N_total, 1, H, W)
        
        if self.train_stage in [1, 2]:
            aligned_depth_map = depth_map * scale_opt.detach().repeat_interleave(N_total).view(B * N_total, 1, 1, 1)
        else:
            aligned_depth_map = depth_map
            
        gt_depth_reshaped = gt_depths.reshape(B * N_total, 1, H, W)
        mask_depth = (gt_depth_reshaped > 1e-4)
        ###### matrixcity depth#############
        # 假设 max_sky_depth 是你的仿真数据集里天空深度的阈值 (比如 100.0)
        max_sky_depth = 200

        # gt_depth_reshaped 形状 [B*N, 1, H, W]
        mask_missing = (gt_depth_reshaped < 1e-4)
        mask_far = (gt_depth_reshaped > max_sky_depth)
        print(gt_depth_reshaped.min(), gt_depth_reshaped.max(), mask_missing.sum(), mask_far.sum(),flush=True)
        # 疑似天空区域：没打到的地方 + 明确是很远的地方
        suspect_mask = mask_missing | mask_far
        # 剔除 sky_head 的高斯，只保留 Dense 部分
        num_sky = gauss_raw.get("num_sky", self.num_sky_anchors)
        dense_gauss_render = {k: v[:, :-num_sky] for k, v in gauss_render.items() if k != "num_sky"}

        # 渲染 Dense 部分的 Alpha (这里只需渲染 ED 或专门的 Alpha 模式以节省性能)
        # 如果你的光栅化器支持直接吐出 transmittance 或 alpha 最好，否则可以用深度渲染通道占位
        _, dense_alpha, _ = self._render_gs(dense_gauss_render, render_w2c, gt_ks, H, W, render_mode='ED') # 或者自定义的 Alpha 模式
        dense_alpha = dense_alpha.reshape(B * N_total, 1, H, W)
        ###### matrixcity depth#############
        # invalid_depth_mask = ~mask_depth
        
        # loss_dense_sparsity = torch.tensor(0.0, device=pred_c2w.device)
        # loss_push_back = torch.tensor(0.0, device=pred_c2w.device)

        # if self.train_stage in [1, 2] and invalid_depth_mask.sum() > 0:
        #     num_sky = gauss_raw.get("num_sky", self.num_sky_anchors)
        #     dense_gauss_render = {k: v[:, :-num_sky] for k, v in gauss_render.items() if k != "num_sky"}
        #     _, dense_alpha, _ = self._render_gs(dense_gauss_render, render_w2c, gt_ks, H, W, render_mode='RGB')
        #     dense_alpha = dense_alpha.reshape(B * N_total, 1, H, W)
            
        #     alpha_in_invalid = dense_alpha[invalid_depth_mask]
        #     loss_dense_sparsity = torch.mean(torch.abs(alpha_in_invalid))
            
        #     z_pred_in_invalid = aligned_depth_map[invalid_depth_mask]
        #     loss_push_back = torch.mean(torch.exp(-z_pred_in_invalid / 10.0))
            
        with torch.no_grad():
            num_viz = min(4, B * N_total)
            rgb_viz = torch.cat([gt_imgs_reshaped[:num_viz], rgb_full[:num_viz]], dim=2) 
            d_pred_viz = aligned_depth_map[:num_viz] / (aligned_depth_map[:num_viz].max() + 1e-5)
            d_gt_viz = gt_depth_reshaped[:num_viz] / (gt_depth_reshaped[:num_viz].max() + 1e-5)
            d_pred_viz = d_pred_viz.repeat(1, 3, 1, 1)
            d_gt_viz = d_gt_viz.repeat(1, 3, 1, 1)
            depth_viz = torch.cat([d_gt_viz, d_pred_viz], dim=2)
            
            final_viz = torch.cat([rgb_viz, depth_viz], dim=3)
            torchvision.utils.save_image(final_viz, f"debug_output/step_{batch_idx}_stage_{self.train_stage}.png")

        if self.train_stage in [1, 2]:
            loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
            loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)
            
            mask_depth = (gt_depth_reshaped > 1e-4) & (gt_depth_reshaped < 58982.4)
            if self.lambda_depth > 0 and mask_depth.sum() > 10:
                loss_depth = F.l1_loss(aligned_depth_map[mask_depth], gt_depth_reshaped[mask_depth])
            ###### matrixcity depth#############
            
            loss_opacity_decay = torch.tensor(0.0, device=pred_c2w.device)
            if suspect_mask.sum() > 0:
                print("yes")
                alpha_in_suspect = dense_alpha[suspect_mask]
                # 施加 L1 惩罚，迫使这些高斯变透明
                loss_opacity_decay = torch.mean(torch.abs(alpha_in_suspect))
            # if self.train_stage == 1:
            #     # ==========================================================
            #     # [修改点 6] Pose loss 对所有 N_total 张相机位姿生效
            #     # ==========================================================
            #     # loss_pose, t_err, r_err = self.camera_loss_fn(pred_c2w, gt_c2w, scale_opt.detach())
            #     # details['pose_trans_err'] = t_err
            #     # details['pose_rot_err'] = r_err

            #     rgb_sub = rgb_full.view(B, N_total, 3, H, W)[:, sub_idx].reshape(B * N_sub, 3, H, W)
            #     gt_imgs_sub = gt_imgs[:, sub_idx].reshape(B * N_sub, 3, H, W)
            #     pixel_error = torch.abs(rgb_sub - gt_imgs_sub).mean(dim=1, keepdim=True).detach()
            #     valid_target = (pixel_error < 0.1).float() 
            #     dense_conf_logits = pred['conf'].reshape(B * N_sub, 1, H, W)
            #     loss_conf = F.binary_cross_entropy_with_logits(dense_conf_logits, valid_target)

        elif self.train_stage == 3:
            rgb_sub = rgb_full.view(B, N_total, 3, H, W)[:, sub_idx].reshape(B * N_sub, 3, H, W)
            gt_imgs_sub = gt_imgs[:, sub_idx].reshape(B * N_sub, 3, H, W)
            pixel_error = torch.abs(rgb_sub - gt_imgs_sub).mean(dim=1, keepdim=True).detach()
            valid_target = (pixel_error < 0.1).float() 
            dense_conf_logits = pred['conf'].reshape(B * N_sub, 1, H, W)
            loss_conf = F.binary_cross_entropy_with_logits(dense_conf_logits, valid_target)
            
        # lambda_sparsity = 0.05 
        # lambda_push_back = 0.02
        final_loss = (
            self.lambda_rgb * loss_rgb + 
            self.lambda_ssim * loss_ssim + 
            self.lambda_depth * loss_depth +
            # self.lambda_pose * loss_pose + 
            # self.lambda_scale * loss_scale + 
            0.1 * loss_opacity_decay +
            self.lambda_pts * loss_pts 
            # loss_conf
            # lambda_sparsity * loss_dense_sparsity + 
            # lambda_push_back * loss_push_back       
        )

        if final_loss == 0.0:
             final_loss = (pred_local_pts.sum() * 0.0)

        details.update({
            "loss_rgb": loss_rgb, "loss_ssim": loss_ssim, 
            "loss_depth": loss_depth, "loss_scale": loss_scale, "loss_pts": loss_pts, "loss_opacity_decay": loss_opacity_decay,
            "loss_conf": loss_conf, "total_loss": final_loss
        })

        return final_loss, details