import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim

import lpips

from ..utils.alignment import align_points_scale
from ..utils.geometry import homogenize_points


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
    """
    GT-supervised 3DGS loss for MidAir.

    Compared with loss_3dgs_1:
    - uses GT masks for normalization
    - uses GT camera pose/intrinsics for rendering the supervision views
    - uses GT depth instead of predicted pseudo depth as the depth target
    """

    def __init__(
        self,
        lambda_rgb=1,
        lambda_ssim=0.3,
        lambda_depth=0.5,
        lambda_pose=0.2,
        lambda_scale=0.1,
        train_stage=1,
        local_align_res=4096,
        train_conf=False,
        num_sky_anchors=8196,
        lpips_downsample=0.5,
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_pose = lambda_pose
        self.lambda_scale = lambda_scale
        self.lambda_lpips = 0.1

        self.train_stage = int(train_stage)
        self.local_align_res = local_align_res

        self.train_conf = train_conf
        self.num_sky_anchors = num_sky_anchors
        self.lpips_downsample = min(max(float(lpips_downsample), 0.1), 1.0)
        self.lpips_loss_fn = lpips.LPIPS(net='alex').eval()
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
            indexing='ij',
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

        # Move GT into the first-view coordinate frame to match the training convention.
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        valid_batch = masks.view(B, -1).sum(dim=-1) > 0
        if valid_batch.any():
            B_valid = int(valid_batch.sum().item())
            all_pts = gt_pts[valid_batch].clone()
            mask_bool = masks[valid_batch].squeeze(-1)
            all_pts[~mask_bool] = 0

            all_pts = all_pts.reshape(B_valid, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            num_valid_pts = mask_bool.view(B_valid, -1).float().sum(dim=-1)

            norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8)
            norm_factor = norm_factor.clamp_min(1e-4)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]
            gt_depths[valid_batch] /= norm_factor[..., None, None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum(
            'bnij, bnhwj -> bnhwi',
            extrinsics,
            homogenize_points(gt_pts),
        )[..., :3]

        return dict(
            imgs=imgs,
            gt_ks=gt_ks,
            gt_depths=gt_depths,
            gt_local_pts=gt_local_pts,
            masks=masks,
            gt_c2w=poses,
        )

    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, _, _, _, _ = local_points.shape
        masks = gt['masks']

        mask_bool = masks.squeeze(-1)

        all_pts = local_points.clone()
        all_pts[~mask_bool] = 0

        all_pts = all_pts.reshape(B, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        num_valid_pts = mask_bool.view(B, -1).float().sum(dim=-1)

        norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8)
        norm_factor = norm_factor.clamp_min(1e-4)

        pred['local_points'] = local_points / norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred['camera_poses'] = camera_poses_normalized

        if 'gaussians' in pred:
            pred['gaussians']['xyz'] = pred['gaussians']['xyz'] / norm_factor.view(B, 1, 1)
            pred['gaussians']['scale'] = pred['gaussians']['scale'] / norm_factor.view(B, 1, 1)

        return pred

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, _, _, _, C = pts.shape
        output = []
        for i in range(B):
            valid_pts = pts[i][mask[i]]
            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')
                valid_pts = valid_pts.squeeze(0).permute(1, 0)
            else:
                valid_pts = torch.ones((target_size, C), device=pts.device, dtype=pts.dtype)
            output.append(valid_pts)
        return torch.stack(output, dim=0)

    def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
        return rasterization(
            means=gaussians['xyz'],
            quats=gaussians['rotation'],
            scales=gaussians['scale'],
            opacities=gaussians['opacity'].squeeze(-1),
            colors=gaussians['color'],
            viewmats=w2c,
            Ks=ks,
            width=W,
            height=H,
            render_mode=render_mode,
            packed=True,
        )

    def forward(self, pred, gt_raw, batch_idx=0, current_epoch=None, total_epochs=None, **kwargs):
        gt = self.prepare_gt(gt_raw)

        B, N_total, _, H, W = gt['imgs'].shape
        if next(self.lpips_loss_fn.parameters()).device != gt['imgs'].device:
            self.lpips_loss_fn = self.lpips_loss_fn.to(gt['imgs'].device)

        pred = self.normalize_pred(pred, {'masks': gt['masks']})

        gt_ks = gt['gt_ks']
        gt_c2w = gt['gt_c2w']
        gt_imgs = gt['imgs']
        gt_depths = gt['gt_depths']
        gt_local_pts = gt['gt_local_pts']
        valid_masks = gt['masks']

        gauss_raw = pred['gaussians']
        pred_local_pts = pred['local_points'].clamp(min=-1e4, max=1e4)

        loss_rgb = torch.tensor(0.0, device=gt_imgs.device)
        loss_ssim = torch.tensor(0.0, device=gt_imgs.device)
        loss_depth = torch.tensor(0.0, device=gt_imgs.device)
        loss_lpips = torch.tensor(0.0, device=gt_imgs.device)
        details = {}

        sub_idx = torch.arange(0, N_total, 1, device=gt_imgs.device)
        valid_masks_sub = valid_masks[:, sub_idx].squeeze(-1)
        gt_local_pts_sub = gt_local_pts[:, sub_idx]

        scale_opt = torch.ones((B,), device=gt_imgs.device, dtype=gt_imgs.dtype)
        if valid_masks_sub.any():
            weights = gt_local_pts_sub[..., 2].clamp_min(1e-3)
            weights = 1.0 / (weights + 1e-6)

            xyz_pred_roe = self.prepare_ROE(pred_local_pts[:, sub_idx], valid_masks_sub, target_size=self.local_align_res)
            with torch.no_grad():
                xyz_gt_roe = self.prepare_ROE(gt_local_pts_sub, valid_masks_sub, target_size=self.local_align_res)
                xyz_w_roe = self.prepare_ROE(weights[..., None], valid_masks_sub, target_size=self.local_align_res)[..., 0]

            scale_opt = align_points_scale(xyz_pred_roe, xyz_gt_roe, xyz_w_roe)

        gauss_render = {k: v for k, v in gauss_raw.items()}
        detach_scale = scale_opt.detach().view(B, 1, 1)
        gauss_render['xyz'] = gauss_raw['xyz'] * detach_scale
        gauss_render['scale'] = gauss_raw['scale'] * detach_scale

        render_w2c = se3_inverse(gt_c2w)

        render_out_rgb, _, _ = self._render_gs(gauss_render, render_w2c, gt_ks, H, W, render_mode='RGB')
        rgb_full = render_out_rgb[..., :3].reshape(B * N_total, H, W, 3).permute(0, 3, 1, 2)
        del render_out_rgb

        gt_imgs_reshaped = gt_imgs.reshape(B * N_total, 3, H, W)

        if self.train_stage in [1, 2, 3]:
            loss_rgb = F.mse_loss(rgb_full, gt_imgs_reshaped)
            loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)

            if self.lambda_lpips > 0:
                if self.lpips_downsample < 1.0:
                    lp_rgb = F.interpolate(
                        rgb_full,
                        scale_factor=self.lpips_downsample,
                        mode='bilinear',
                        align_corners=False,
                        antialias=True,
                    )
                    lp_gt = F.interpolate(
                        gt_imgs_reshaped,
                        scale_factor=self.lpips_downsample,
                        mode='bilinear',
                        align_corners=False,
                        antialias=True,
                    )
                    loss_lpips = self.lpips_loss_fn(lp_rgb, lp_gt, normalize=True).mean()
                    del lp_rgb, lp_gt
                else:
                    loss_lpips = self.lpips_loss_fn(rgb_full, gt_imgs_reshaped, normalize=True).mean()

        del rgb_full, gt_imgs_reshaped

        need_depth_loss = (
            self.train_stage in [1, 2, 3]
            and self.lambda_depth > 0
            and bool(valid_masks.any().item())
        )
        if need_depth_loss:
            render_out_depth, _, _ = self._render_gs(gauss_render, render_w2c, gt_ks, H, W, render_mode='ED')
            depth_map = render_out_depth[..., 0:1].reshape(B * N_total, H, W, 1).permute(0, 3, 1, 2)
            del render_out_depth

            gt_depth_reshaped = gt_depths.reshape(B * N_total, 1, H, W)
            mask_depth = valid_masks.reshape(B * N_total, 1, H, W)
            if mask_depth.sum() > 10:
                loss_depth = F.l1_loss(depth_map[mask_depth], gt_depth_reshaped[mask_depth])
            del depth_map, gt_depth_reshaped, mask_depth

        final_loss = (
            self.lambda_rgb * loss_rgb
            + self.lambda_ssim * loss_ssim
            + self.lambda_depth * loss_depth
            + self.lambda_lpips * loss_lpips
        )

        if final_loss == 0.0:
            final_loss = pred_local_pts.sum() * 0.0

        details.update({
            'loss_rgb': loss_rgb,
            'loss_ssim': loss_ssim,
            'loss_depth': loss_depth,
            'loss_lpips': loss_lpips,
            'scale_opt_mean': scale_opt.detach().mean(),
            'total_loss': final_loss,
        })

        return final_loss, details
