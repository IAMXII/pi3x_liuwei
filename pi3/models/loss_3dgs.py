import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim


def se3_inverse(T):
    """SE(3) Matrix Inverse"""
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
            self,
            lambda_rgb=1.0,
            lambda_ssim=0.5,
            lambda_depth=0.2,
            lambda_repulsion=0.01,
            lambda_sparsity=0.005,
            lambda_consist=0.05,
            lambda_pose=0.1,
            warmup_ratio=0.5
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_repulsion = lambda_repulsion
        self.lambda_sparsity = lambda_sparsity
        self.lambda_consist = lambda_consist
        self.lambda_pose = lambda_pose
        self.warmup_ratio = warmup_ratio

    def prepare_gt(self, gt):
        """
        Simplified GT preparation:
        Only stacks depthmap, img, camerapose, intrinsics, and mask.
        No coordinate alignment or normalization is performed here.
        """
        # Stack lists into tensors [B, N, ...]
        gt_imgs = torch.stack([view['img'] for view in gt], dim=1)
        gt_depths = torch.stack([view['depthmap'] for view in gt], dim=1)
        gt_c2w = torch.stack([view['camera_pose'] for view in gt], dim=1)
        gt_ks = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)

        return dict(
            imgs=gt_imgs,
            gt_depths=gt_depths,
            gt_c2w=gt_c2w,
            gt_ks=gt_ks,
            masks=masks
        )

    def _render_gs(self, gaussians, w2c, ks, H, W, num_gaussians=None, render_mode='RGB'):
        # 1. Get properties
        means = gaussians["xyz"]
        quats = gaussians["rotation"]
        scales = gaussians["scale"]
        opacities = gaussians["opacity"]
        colors = gaussians["color"]

        B = means.shape[0]

        # 2. Handle num_gaussians (Ragged Batch Masking)
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

        # 3. Rasterization
        return rasterization(
            means=means.contiguous(),
            quats=quats.contiguous(),
            scales=scales.contiguous(),
            opacities=opacities.squeeze(-1).contiguous(),
            colors=colors.contiguous(),
            viewmats=w2c,
            Ks=ks,  # Note: gsplat argument is usually ks or Ks depending on version, keeping your code
            width=W, height=H,
            render_mode=render_mode,
            packed=False
        )

    def _calc_repulsion_loss(self, xyz, num_near, k=4, max_points=1024):
        B = xyz.shape[0]
        loss = 0.0

        for b in range(B):
            N_valid = int(num_near[b].item())
            if N_valid < k + 1: continue

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

    def forward(self, pred, gt_raw, current_epoch=None, total_epochs=None, enable_pose_loss=False):
        """
        Args:
            pred: Model prediction output (must contain 'gaussians' and 'camera_poses')
            gt_raw: Ground Truth data
        """
        # 1. Prepare GT (Simplified)
        gt = self.prepare_gt(gt_raw)

        B, N, C, H, W = gt['imgs'].shape
        gt_ks = gt['gt_ks']
        gt_c2w = gt['gt_c2w']
        gt_imgs = gt['imgs']
        gt_depths = gt['gt_depths']
        masks = gt['masks']

        gauss = pred['gaussians']
        pred_c2w = pred['camera_poses']  # Expected shape [B, N, 4, 4]
        num_near = gauss.get('num_near', None)

        # 2. Prepare Render View Matrix (W2C) from PREDICTED Poses
        # NOTE: Rasterizer needs W2C, but model usually predicts C2W.
        pred_w2c = se3_inverse(pred_c2w)

        # 3. Render RGB using Predicted Poses
        rgb_full, _, _ = self._render_gs(
            gauss, pred_w2c, gt_ks, H, W,
            num_gaussians=num_near, render_mode='RGB'
        )

        # 4. Render Depth using Predicted Poses
        depth_near = None
        if self.lambda_depth > 0 and num_near is not None:
            depth_near_map, _, _ = self._render_gs(
                gauss, pred_w2c, gt_ks, H, W,
                num_gaussians=num_near, render_mode='ED'
            )
            depth_near = depth_near_map

        # 5. Compute Basic Losses (RGB, SSIM, Depth)
        rgb_full = rgb_full.reshape(B * N, H, W, 3).permute(0, 3, 1, 2)
        gt_imgs = gt_imgs.reshape(B * N, 3, H, W)

        loss_rgb = F.l1_loss(rgb_full, gt_imgs)
        loss_ssim = 1.0 - ssim(rgb_full, gt_imgs, data_range=1.0)

        loss_depth = torch.tensor(0.0, device=rgb_full.device)
        depth_scale_scalar = 1.0  # For logging

        if depth_near is not None:
            # Flatten tensors for easier processing
            target_depth = gt_depths.reshape(-1)
            pred_depth = depth_near.reshape(-1)

            # Mask for valid GT depth (GT > 1e-4)
            valid_mask = (target_depth > 1e-4)

            # --- [Modified] Logarithmic Depth Loss ---
            if valid_mask.sum() > 10:  # Ensure enough points for stability
                t_masked = target_depth[valid_mask]
                p_masked = pred_depth[valid_mask]
                
                # 1. Log Transform
                # Apply log(x + epsilon) to prevent NaN and handle scale
                # Log loss treats multiplicative error as additive error: log(p/t) = log(p) - log(t)
                epsilon = 1e-7
                log_pred = torch.log(p_masked + epsilon)
                log_target = torch.log(t_masked + epsilon)
                
                # 2. Compute L1 Loss in Log Space
                # loss_depth = F.l1_loss(log_pred, log_target)
                log_pred_centered = log_pred - log_pred.mean()
                log_target_centered = log_target - log_target.mean()

                loss_depth = F.l1_loss(log_pred_centered, log_target_centered)
                # Optional: Calculate scale just for visualization/logging purposes
                # (Since we are using Log loss, we don't strictly align the scale for the loss itself anymore, 
                # but knowing the scale factor is useful for debugging)
                with torch.no_grad():
                    dot_pt = (p_masked * t_masked).sum()
                    dot_pp = (p_masked.pow(2)).sum()
                    scale = dot_pt / (dot_pp + 1e-8)
                    depth_scale_scalar = scale.item()
            # -------------------------------------------

        # 6. Compute Camera Pose Loss (Direct Supervision)
        loss_pose_R = torch.tensor(0.0, device=rgb_full.device)
        loss_pose_T = torch.tensor(0.0, device=rgb_full.device)

        if self.lambda_pose > 0 and enable_pose_loss:
            # Rotation L1
            loss_pose_R = F.l1_loss(pred_c2w[..., :3, :3], gt_c2w[..., :3, :3])
            # Translation L1 (Absolute coordinates)
            loss_pose_T = F.l1_loss(pred_c2w[..., :3, 3], gt_c2w[..., :3, 3])

        # 7. Regularization
        loss_repulsion = torch.tensor(0.0, device=rgb_full.device)
        loss_sparsity = torch.tensor(0.0, device=rgb_full.device)
        loss_consist = torch.tensor(0.0, device=rgb_full.device)

        enable_regularization = True
        if current_epoch is not None and total_epochs is not None:
            if current_epoch < (total_epochs * self.warmup_ratio):
                enable_regularization = False

        if enable_regularization:
            # Repulsion
            if self.lambda_repulsion > 0 and num_near is not None:
                if torch.rand(1).item() < 0.1:
                    loss_repulsion = self._calc_repulsion_loss(gauss['xyz'], num_near) * 10.0

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

        # 8. Total Loss
        loss_pose_total = self.lambda_pose * (loss_pose_R + loss_pose_T)

        final_loss = (
                self.lambda_rgb * loss_rgb +
                self.lambda_ssim * loss_ssim +
                self.lambda_depth * loss_depth +
                loss_pose_total +
                self.lambda_repulsion * loss_repulsion +
                self.lambda_sparsity * loss_sparsity +
                self.lambda_consist * loss_consist
        )

        details = {
            "loss_gs_rgb": loss_rgb,
            "loss_gs_ssim": loss_ssim,
            "loss_gs_depth": loss_depth,
            "depth_scale": depth_scale_scalar,  # 增加了这个便于监控缩放比例
            "loss_pose_R": loss_pose_R,
            "loss_pose_T": loss_pose_T,
            "loss_repulsion": loss_repulsion,
            "loss_sparsity": loss_sparsity,
            "loss_consist": loss_consist,
            "total_loss": final_loss,
            "reg_enabled": float(enable_regularization)
        }

        return final_loss, details