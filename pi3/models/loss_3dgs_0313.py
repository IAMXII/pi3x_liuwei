import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips # [保留上一次修改] 导入 lpips

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
        
        self.train_stage = int(train_stage) 
        self.local_align_res = local_align_res
        
        self.train_conf = train_conf 
        self.num_sky_anchors = num_sky_anchors
        self.camera_loss_fn = CameraPoseLoss()

        self.lpips_loss_fn = lpips.LPIPS(net='alex')
        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False

    def prepare_gt(self, gt):
        imgs = torch.stack([view['img'] for view in gt], dim=1)
        gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)

        B, N, H, W = gt_depths.shape
        device = gt_depths.device

        masks = (gt_depths > 1e-4).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        grid_x = grid_x.expand(B, N, -1, -1).float()
        grid_y = grid_y.expand(B, N, -1, -1).float()

        fx = gt_ks[..., 0, 0].view(B, N, 1, 1)
        fy = gt_ks[..., 1, 1].view(B, N, 1, 1)
        cx = gt_ks[..., 0, 2].view(B, N, 1, 1)
        cy = gt_ks[..., 1, 2].view(B, N, 1, 1)

        local_x = (grid_x - cx) * gt_depths / fx
        local_y = (grid_y - cy) * gt_depths / fy
        local_z = gt_depths
        gt_local_pts_raw = torch.stack([local_x, local_y, local_z], dim=-1)

        gt_local_pts_h = homogenize_points(gt_local_pts_raw).view(B, N, -1, 4).transpose(2, 3)
        gt_pts = torch.matmul(poses, gt_local_pts_h).transpose(2, 3).reshape(B, N, H, W, 4)[..., :3] 

        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        valid_batch = masks.view(B, -1).sum(dim=-1) > 0 
        
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone() 
            
            mask_bool = masks[valid_batch].squeeze(-1) 
            all_pts[~mask_bool] = 0
            
            all_pts = all_pts.reshape(B_, -1, 3)
            all_dis = all_pts.norm(dim=-1)      
            
            num_valid_pts = mask_bool.view(B_, -1).float().sum(dim=-1) 
            
            norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8)
            norm_factor = norm_factor.clamp_min(1e-4)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
            gt_depths[valid_batch] /= norm_factor[..., None, None, None]

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
        sub_idx = torch.arange(0, N_total, 1, device=gt['imgs'].device)
        N_sub = len(sub_idx)

        gt_sub_mask = {'masks': gt['masks'][:, sub_idx]}
        pred = self.normalize_pred(pred, gt_sub_mask) 
        
        gt_ks, gt_c2w, gt_imgs, gt_depths, valid_masks = gt['gt_ks'], gt['gt_c2w'], gt['imgs'], gt['gt_depths'], gt['masks']
        gt_local_pts = gt['gt_local_pts']
        
        valid_masks_sub = valid_masks[:, sub_idx].squeeze(-1)
        gt_local_pts_sub = gt_local_pts[:, sub_idx]

        gauss_raw = pred['gaussians']
        pred_c2w = pred['camera_poses']
        
        pred_local_pts = torch.clamp(pred['local_points'], min=-1e4, max=1e4)

        loss_rgb = loss_ssim = loss_depth = loss_pose = loss_conf = loss_scale = loss_pts = loss_lpips = torch.tensor(0.0, device=pred_c2w.device)
        details = {}

        scale_opt = torch.ones((B,), device=pred_c2w.device)
        
        if self.train_stage in [1, 2]:
            weights_ = gt_local_pts_sub[..., 2].clamp_min(1e-3)
            weights_ = 1 / (weights_ + 1e-6)
            
            xyz_pred_ROE = self.prepare_ROE(pred_local_pts.reshape(B, N_sub, H, W, 3), valid_masks_sub, target_size=self.local_align_res)
            with torch.no_grad():
                xyz_gt_ROE = self.prepare_ROE(gt_local_pts_sub.reshape(B, N_sub, H, W, 3), valid_masks_sub, target_size=self.local_align_res)
                xyz_w_ROE = self.prepare_ROE((weights_[..., None]).reshape(B, N_sub, H, W, 1), valid_masks_sub, target_size=self.local_align_res)[..., 0]
            
            # [说明] 这里的 scale_opt 必须保留计算，尽管 pts 冻结了，
            # 但算出的 scale_opt 后续用来缩放 gauss_render 以对齐 GT，进而才能计算出正确的 rgb 和 depth loss
            scale_opt = align_points_scale(xyz_pred_ROE, xyz_gt_ROE, xyz_w_ROE)

            # ==========================================================
            # [修改点] 删除了下方计算 loss_pts 的模块块
            # ==========================================================
            # 删除了 F.l1_loss(aligned_local_pts[valid_masks_sub], gt_local_pts_sub[valid_masks_sub])

        gauss_render = {k: v for k, v in gauss_raw.items()} 
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

            self.lpips_loss_fn.to(rgb_full.device)
            rgb_full_norm = rgb_full * 2.0 - 1.0
            gt_imgs_norm = gt_imgs_reshaped * 2.0 - 1.0
            lpips_val = self.lpips_loss_fn(rgb_full_norm, gt_imgs_norm)
            loss_lpips = lpips_val.mean()
            
            mask_depth = (gt_depth_reshaped > 1e-4)
            if self.lambda_depth > 0 and mask_depth.sum() > 10:
                loss_depth = F.l1_loss(aligned_depth_map[mask_depth], gt_depth_reshaped[mask_depth])

        elif self.train_stage == 3:
            rgb_sub = rgb_full.view(B, N_total, 3, H, W)[:, sub_idx].reshape(B * N_sub, 3, H, W)
            gt_imgs_sub = gt_imgs[:, sub_idx].reshape(B * N_sub, 3, H, W)
            pixel_error = torch.abs(rgb_sub - gt_imgs_sub).mean(dim=1, keepdim=True).detach()
            valid_target = (pixel_error < 0.1).float() 
            dense_conf_logits = pred['conf'].reshape(B * N_sub, 1, H, W)
            loss_conf = F.binary_cross_entropy_with_logits(dense_conf_logits, valid_target)
            
        final_loss = (
            self.lambda_rgb * loss_rgb + 
            self.lambda_ssim * loss_ssim + 
            self.lambda_depth * loss_depth +
            # ==========================================================
            # [修改点] 从最终损失汇总中去掉了 self.lambda_pts * loss_pts 
            # ==========================================================
            0.2 * loss_lpips 
        )

        if final_loss == 0.0:
             final_loss = (pred_local_pts.sum() * 0.0)

        details.update({
            "loss_rgb": loss_rgb, "loss_ssim": loss_ssim, 
            "loss_depth": loss_depth, "loss_scale": loss_scale, "loss_pts": loss_pts, # [说明] 依然传回 0.0 防止外界记录报错
            "loss_conf": loss_conf, "loss_lpips": loss_lpips,
            "total_loss": final_loss
        })

        return final_loss, details