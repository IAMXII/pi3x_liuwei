import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torchmetrics.functional import structural_similarity_index_measure as ssim

import lpips

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


def sanitize_quaternions(quats):
    quats = torch.nan_to_num(quats.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norms = quats.norm(dim=-1, keepdim=True)
    default = torch.zeros_like(quats)
    default[..., 0] = 1.0
    quats = torch.where(norms > 1e-8, quats / norms.clamp_min(1e-8), default)
    return quats


def sanitize_gaussians_for_render(gaussians, xyz_clip=256.0, scale_min=1e-6, scale_max=10.0):
    sanitized = {k: v for k, v in gaussians.items()}

    xyz = torch.nan_to_num(gaussians["xyz"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    xyz = xyz.clamp(min=-xyz_clip, max=xyz_clip)

    scale = torch.nan_to_num(gaussians["scale"].float(), nan=scale_min, posinf=scale_max, neginf=scale_min)
    scale = scale.clamp(min=scale_min, max=scale_max)

    opacity = torch.nan_to_num(gaussians["opacity"].float(), nan=0.0, posinf=1.0, neginf=0.0)
    opacity = opacity.clamp(min=0.0, max=1.0)

    color = torch.nan_to_num(gaussians["color"].float(), nan=0.0, posinf=1.0, neginf=0.0)
    color = color.clamp(min=0.0, max=1.0)

    sanitized["xyz"] = xyz
    sanitized["rotation"] = sanitize_quaternions(gaussians["rotation"])
    sanitized["scale"] = scale
    sanitized["opacity"] = opacity
    sanitized["color"] = color
    return sanitized


class Pi3LossGS(nn.Module):
    """
    Appearance transfer loss with dataset-aware supervision:
    - MidAir: RGB + paired RGB + scale-aligned GT depth.
    - NeRF-OSR: RGB + paired RGB + global style-statistics, no depth loss.
    """

    def __init__(
        self,
        lambda_rgb=1.0,
        lambda_ssim=0.2,
        lambda_depth=0.25,
        lambda_lpips=0.05,
        lambda_paired_rgb=0.75,
        lambda_paired_ssim=0.15,
        lambda_paired_lpips=0.05,
        lambda_style_consistency=0.02,
        lambda_style_reg=0.01,
        lambda_style_moment=0.20,
        train_stage=1,
        local_align_res=4096,
        train_conf=False,
        num_sky_anchors=8196,
        lpips_downsample=0.5,
        pred_mask_conf_threshold=0.15,
        pred_mask_edge_rtol=0.03,
        min_scale=1e-2,
        max_scale=100.0,
        render_view_chunk_size=1,
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_lpips = lambda_lpips
        self.lambda_paired_rgb = lambda_paired_rgb
        self.lambda_paired_ssim = lambda_paired_ssim
        self.lambda_paired_lpips = lambda_paired_lpips
        self.lambda_style_consistency = lambda_style_consistency
        self.lambda_style_reg = lambda_style_reg
        self.lambda_style_moment = lambda_style_moment

        self.train_stage = int(train_stage)
        self.local_align_res = local_align_res
        self.train_conf = train_conf
        self.num_sky_anchors = num_sky_anchors
        self.lpips_downsample = min(max(float(lpips_downsample), 0.1), 1.0)
        self.pred_mask_conf_threshold = float(pred_mask_conf_threshold)
        self.pred_mask_edge_rtol = float(pred_mask_edge_rtol)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.render_view_chunk_size = max(1, int(render_view_chunk_size))

        self.lpips_loss_fn = lpips.LPIPS(net="alex").eval()
        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False

    @staticmethod
    def _sanitize_scalar_loss(value):
        if not isinstance(value, torch.Tensor):
            return value
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _finite_flag(value, ref_tensor):
        if not isinstance(value, torch.Tensor):
            return ref_tensor.new_tensor(1.0)
        return ref_tensor.new_tensor(float(torch.isfinite(value.detach()).all().item()))

    @staticmethod
    def _safe_absmax(value, ref_tensor):
        if not isinstance(value, torch.Tensor):
            return ref_tensor.new_tensor(0.0)
        safe = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        return safe.abs().max()

    @staticmethod
    def _estimate_pose_scale(poses, eps=1e-4):
        rel_t = poses[..., :3, 3]
        dist = rel_t.norm(dim=-1)
        scale = torch.ones((poses.shape[0],), device=poses.device, dtype=poses.dtype)
        for b in range(poses.shape[0]):
            valid = dist[b] > eps
            if bool(valid.any().item()):
                scale[b] = dist[b][valid].median()
        return scale.clamp_min(eps)

    @staticmethod
    def _image_mean_std(imgs):
        B = imgs.shape[0]
        flat = imgs.permute(0, 2, 1, 3, 4).reshape(B, imgs.shape[2], -1)
        mean = flat.mean(dim=-1)
        std = flat.std(dim=-1, unbiased=False)
        return mean, std

    @staticmethod
    def _find_grad_anchor(pred):
        candidates = []
        for key in [
            "style_code_consistency",
            "style_delta_norm",
            "appearance_color_delta_abs",
            "appearance_opacity_delta_abs",
            "appearance_gate_mean",
            "style_code_norm",
            "environment_code",
        ]:
            value = pred.get(key)
            if isinstance(value, torch.Tensor):
                candidates.append((key, value))

        gaussians_modulated = pred.get("gaussians_modulated")
        if isinstance(gaussians_modulated, dict):
            for key in ["color", "opacity"]:
                value = gaussians_modulated.get(key)
                if isinstance(value, torch.Tensor):
                    candidates.append((f"gaussians_modulated.{key}", value))

        for name, value in candidates:
            if value.requires_grad:
                return name, value
        return None, None

    def prepare_gt(self, gt):
        imgs = torch.stack([view["img"] for view in gt], dim=1)
        gt_depths = torch.stack([view["depthmap"] for view in gt], dim=1)
        poses = torch.stack([view["camera_pose"] for view in gt], dim=1)
        gt_ks = torch.stack([view["camera_intrinsics"] for view in gt], dim=1)

        imgs_paired = None
        if "img_paired" in gt[0] and isinstance(gt[0]["img_paired"], torch.Tensor):
            imgs_paired = torch.stack([view["img_paired"] for view in gt], dim=1)

        imgs_style = None
        if "img_style" in gt[0] and isinstance(gt[0]["img_style"], torch.Tensor):
            imgs_style = torch.stack([view["img_style"] for view in gt], dim=1)

        poses_paired = None
        if "camera_pose_paired" in gt[0] and isinstance(gt[0]["camera_pose_paired"], torch.Tensor):
            poses_paired = torch.stack([view["camera_pose_paired"] for view in gt], dim=1)

        gt_ks_paired = None
        if "camera_intrinsics_paired" in gt[0] and isinstance(gt[0]["camera_intrinsics_paired"], torch.Tensor):
            gt_ks_paired = torch.stack([view["camera_intrinsics_paired"] for view in gt], dim=1)

        B, N, H, W = gt_depths.shape
        device = gt_depths.device
        has_depth = torch.ones((B, N), device=device, dtype=torch.bool)
        if "has_depth" in gt[0]:
            has_depth = torch.stack([view["has_depth"] for view in gt], dim=1).to(device=device).bool()

        masks = ((gt_depths > 1e-4) & has_depth[:, :, None, None]).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
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
        gt_pts = torch.einsum("bij, bnhwj -> bnhwi", w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum("bij, bnjk -> bnik", w2c_target, poses)
        if poses_paired is not None:
            poses_paired = torch.einsum("bij, bnjk -> bnik", w2c_target, poses_paired)

        valid_depth_batch = masks.view(B, -1).any(dim=-1)
        if valid_depth_batch.any():
            B_valid = int(valid_depth_batch.sum().item())
            all_pts = gt_pts[valid_depth_batch].clone()
            mask_bool = masks[valid_depth_batch].squeeze(-1)
            all_pts[~mask_bool] = 0

            all_pts = all_pts.reshape(B_valid, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            num_valid_pts = mask_bool.view(B_valid, -1).float().sum(dim=-1)

            norm_factor = all_dis.sum(dim=-1) / (num_valid_pts + 1e-8)
            norm_factor = norm_factor.clamp_min(1e-4)

            gt_pts[valid_depth_batch] = gt_pts[valid_depth_batch] / norm_factor[..., None, None, None, None]
            poses[valid_depth_batch, ..., :3, 3] /= norm_factor[..., None, None]
            gt_depths[valid_depth_batch] /= norm_factor[..., None, None, None]
            if poses_paired is not None:
                poses_paired[valid_depth_batch, ..., :3, 3] /= norm_factor[..., None, None]

        no_depth_batch = ~valid_depth_batch
        pose_scale = torch.ones((B,), device=device, dtype=gt_depths.dtype)
        if no_depth_batch.any():
            pose_scale_valid = self._estimate_pose_scale(poses[no_depth_batch])
            pose_scale[no_depth_batch] = pose_scale_valid
            poses[no_depth_batch, ..., :3, 3] /= pose_scale_valid[..., None, None]
            if poses_paired is not None:
                poses_paired[no_depth_batch, ..., :3, 3] /= pose_scale_valid[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum(
            "bnij, bnhwj -> bnhwi",
            extrinsics,
            homogenize_points(gt_pts),
        )[..., :3]

        return {
            "imgs": imgs,
            "imgs_paired": imgs_paired,
            "imgs_style": imgs_style,
            "gt_ks": gt_ks,
            "gt_ks_paired": gt_ks_paired,
            "gt_depths": gt_depths,
            "gt_local_pts": gt_local_pts,
            "masks": masks,
            "gt_c2w": poses,
            "gt_c2w_paired": poses_paired,
            "has_depth": has_depth,
            "pose_scale": pose_scale,
        }

    def _build_pred_mask(self, pred):
        local_points = pred["local_points"]
        depth = local_points[..., 2]
        positive_depth = torch.isfinite(depth) & (depth > 1e-6)
        non_edge = ~depth_edge(depth, rtol=self.pred_mask_edge_rtol)

        conf_logits = pred.get("conf")
        if isinstance(conf_logits, torch.Tensor) and conf_logits.ndim == 5 and conf_logits.shape[:4] == local_points.shape[:4]:
            conf_mask = torch.sigmoid(conf_logits[..., 0]) > self.pred_mask_conf_threshold
            return positive_depth & non_edge & conf_mask
        return positive_depth & non_edge

    def normalize_pred(self, pred, gt):
        local_points = pred["local_points"]
        camera_poses = pred["camera_poses"]
        B, _, _, _, _ = local_points.shape
        gt_mask = gt["masks"].squeeze(-1)
        pred_mask = self._build_pred_mask(pred)

        mask_bool = gt_mask.clone()
        no_depth_batch = ~mask_bool.view(B, -1).any(dim=-1)
        if no_depth_batch.any():
            mask_bool[no_depth_batch] = pred_mask[no_depth_batch]

        still_empty = ~mask_bool.view(B, -1).any(dim=-1)
        if still_empty.any():
            fallback = torch.isfinite(local_points[..., 2]) & (local_points[..., 2] > 1e-6)
            mask_bool[still_empty] = fallback[still_empty]

        norm_factor = torch.ones((B,), device=local_points.device, dtype=local_points.dtype)
        for b in range(B):
            valid_pts = local_points[b][mask_bool[b]]
            if valid_pts.numel() == 0:
                continue
            valid_dis = valid_pts.norm(dim=-1)
            norm_factor[b] = valid_dis.median().clamp_min(1e-4)

        pred["local_points"] = local_points / norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred["camera_poses"] = camera_poses_normalized

        if "gaussians" in pred:
            pred["gaussians"]["xyz"] = pred["gaussians"]["xyz"] / norm_factor.view(B, 1, 1)
            pred["gaussians"]["scale"] = pred["gaussians"]["scale"] / norm_factor.view(B, 1, 1)

        if pred.get("gaussians_modulated") is not None:
            pred["gaussians_modulated"]["xyz"] = pred["gaussians_modulated"]["xyz"] / norm_factor.view(B, 1, 1)
            pred["gaussians_modulated"]["scale"] = pred["gaussians_modulated"]["scale"] / norm_factor.view(B, 1, 1)

        return pred

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, _, _, _, C = pts.shape
        output = []
        for i in range(B):
            valid_pts = pts[i][mask[i]]
            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)
                valid_pts = F.interpolate(valid_pts, size=target_size, mode="nearest")
                valid_pts = valid_pts.squeeze(0).permute(1, 0)
            else:
                valid_pts = torch.ones((target_size, C), device=pts.device, dtype=pts.dtype)
            output.append(valid_pts)
        return torch.stack(output, dim=0)

    def _render_gs(self, gaussians, w2c, ks, H, W, render_mode="RGB"):
        gaussians = sanitize_gaussians_for_render(gaussians)
        w2c = torch.nan_to_num(w2c.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ks = torch.nan_to_num(ks.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ks[..., 0, 0] = ks[..., 0, 0].clamp_min(1e-6)
        ks[..., 1, 1] = ks[..., 1, 1].clamp_min(1e-6)
        ks[..., 2, 2] = 1.0
        return rasterization(
            means=gaussians["xyz"],
            quats=gaussians["rotation"],
            scales=gaussians["scale"],
            opacities=gaussians["opacity"].squeeze(-1),
            colors=gaussians["color"],
            viewmats=w2c,
            Ks=ks,
            width=W,
            height=H,
            render_mode=render_mode,
            packed=True,
        )

    def _render_rgb_losses_chunked(
        self,
        gaussians,
        w2c,
        ks,
        target_imgs,
        H,
        W,
        use_lpips,
        collect_moments=False,
    ):
        B, N_total, _, _, _ = target_imgs.shape
        loss_rgb = target_imgs.new_tensor(0.0)
        loss_ssim = target_imgs.new_tensor(0.0)
        loss_lpips = target_imgs.new_tensor(0.0)
        render_finite = target_imgs.new_tensor(1.0)
        prefilter_finite = target_imgs.new_tensor(1.0)

        moment_sum = target_imgs.new_zeros((B, 3))
        moment_sumsq = target_imgs.new_zeros((B, 3))
        moment_count = target_imgs.new_zeros((B, 1))

        for start in range(0, N_total, self.render_view_chunk_size):
            end = min(start + self.render_view_chunk_size, N_total)
            n_chunk = end - start
            weight = float(n_chunk) / float(N_total)

            render_out, _, _ = self._render_gs(
                gaussians,
                w2c[:, start:end],
                ks[:, start:end],
                H,
                W,
                render_mode="RGB",
            )
            render_finite = torch.minimum(render_finite, self._finite_flag(render_out, target_imgs))

            rgb = render_out[..., :3].reshape(B * n_chunk, H, W, 3).permute(0, 3, 1, 2)
            prefilter_finite = torch.minimum(prefilter_finite, self._finite_flag(rgb, target_imgs))
            rgb = torch.nan_to_num(rgb.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            target = target_imgs[:, start:end].reshape(B * n_chunk, 3, H, W)

            loss_rgb = loss_rgb + weight * self._sanitize_scalar_loss(F.l1_loss(rgb, target))
            loss_ssim = loss_ssim + weight * self._sanitize_scalar_loss(1.0 - ssim(rgb, target, data_range=1.0))
            if use_lpips:
                loss_lpips = loss_lpips + weight * self._sanitize_scalar_loss(self._lpips(rgb, target))

            if collect_moments:
                rgb_bn = rgb.reshape(B, n_chunk, 3, H, W)
                moment_sum = moment_sum + rgb_bn.sum(dim=(1, 3, 4))
                moment_sumsq = moment_sumsq + rgb_bn.square().sum(dim=(1, 3, 4))
                moment_count = moment_count + rgb.new_full((B, 1), float(n_chunk * H * W))

            del render_out, rgb, target

        moments = None
        if collect_moments:
            mean = moment_sum / moment_count.clamp_min(1.0)
            var = moment_sumsq / moment_count.clamp_min(1.0) - mean.square()
            std = var.clamp_min(0.0).sqrt()
            moments = (mean, std)

        return loss_rgb, loss_ssim, loss_lpips, render_finite, prefilter_finite, moments

    def _render_depth_loss_chunked(self, gaussians, w2c, ks, gt_depths, valid_masks, H, W):
        B, N_total, _, _, _ = valid_masks.shape
        render_finite = gt_depths.new_tensor(1.0)
        depth_sum = gt_depths.new_tensor(0.0)
        depth_count = gt_depths.new_tensor(0.0)

        for start in range(0, N_total, self.render_view_chunk_size):
            end = min(start + self.render_view_chunk_size, N_total)
            n_chunk = end - start

            render_out, _, _ = self._render_gs(
                gaussians,
                w2c[:, start:end],
                ks[:, start:end],
                H,
                W,
                render_mode="ED",
            )
            render_finite = torch.minimum(render_finite, self._finite_flag(render_out, gt_depths))

            depth_map = render_out[..., 0:1].reshape(B * n_chunk, H, W, 1).permute(0, 3, 1, 2)
            depth_map = torch.nan_to_num(depth_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
            gt_depth = gt_depths[:, start:end].reshape(B * n_chunk, 1, H, W)
            mask = valid_masks[:, start:end].reshape(B * n_chunk, 1, H, W)

            if mask.sum() > 0:
                depth_sum = depth_sum + (depth_map[mask] - gt_depth[mask]).abs().sum()
                depth_count = depth_count + mask.sum().to(depth_sum.dtype)

            del render_out, depth_map, gt_depth, mask

        if depth_count > 10:
            return self._sanitize_scalar_loss(depth_sum / depth_count.clamp_min(1.0)), render_finite
        return gt_depths.new_tensor(0.0), render_finite

    def _lpips(self, pred_img, gt_img):
        if self.lambda_lpips <= 0 and self.lambda_paired_lpips <= 0:
            return pred_img.new_tensor(0.0)

        if self.lpips_downsample < 1.0:
            pred_img = F.interpolate(
                pred_img,
                scale_factor=self.lpips_downsample,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            gt_img = F.interpolate(
                gt_img,
                scale_factor=self.lpips_downsample,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return self.lpips_loss_fn(pred_img, gt_img, normalize=True).mean()

    def _apply_scale_alignment(self, gaussians, scale_opt):
        aligned = {k: v for k, v in gaussians.items()}
        safe_scale = torch.nan_to_num(scale_opt.detach().float(), nan=1.0, posinf=1.0, neginf=1.0)
        safe_scale = safe_scale.clamp(min=self.min_scale, max=self.max_scale)
        detach_scale = safe_scale.view(scale_opt.shape[0], 1, 1)
        aligned["xyz"] = gaussians["xyz"] * detach_scale
        aligned["scale"] = gaussians["scale"] * detach_scale
        return aligned

    def _camera_translation_scale(self, pred_c2w, gt_c2w, eps=1e-6):
        pred_t = pred_c2w[..., :3, 3]
        gt_t = gt_c2w[..., :3, 3]

        pred_t = pred_t - pred_t[:, :1, :]
        gt_t = gt_t - gt_t[:, :1, :]

        pred_norm = pred_t.norm(dim=-1)
        gt_norm = gt_t.norm(dim=-1)

        ratio = torch.ones(pred_c2w.shape[0], device=pred_c2w.device, dtype=pred_c2w.dtype)
        for b in range(pred_c2w.shape[0]):
            valid = (pred_norm[b] > 1e-4) & (gt_norm[b] > 1e-4)
            if bool(valid.any().item()):
                candidates = gt_norm[b][valid] / pred_norm[b][valid].clamp_min(eps)
                candidates = torch.nan_to_num(candidates, nan=1.0, posinf=1.0, neginf=1.0)
                ratio[b] = candidates.median()
        return ratio.clamp(min=self.min_scale, max=self.max_scale)

    def forward(self, pred, gt_raw, batch_idx=0, current_epoch=None, total_epochs=None, **kwargs):
        gt = self.prepare_gt(gt_raw)

        if next(self.lpips_loss_fn.parameters()).device != gt["imgs"].device:
            self.lpips_loss_fn = self.lpips_loss_fn.to(gt["imgs"].device)

        pred = self.normalize_pred(pred, gt)

        B, N_total, _, H, W = gt["imgs"].shape
        gt_ks = gt["gt_ks"]
        gt_ks_paired = gt["gt_ks_paired"]
        gt_c2w = gt["gt_c2w"]
        gt_c2w_paired = gt["gt_c2w_paired"]
        gt_imgs = gt["imgs"]
        gt_imgs_paired = gt["imgs_paired"]
        gt_imgs_style = gt["imgs_style"]
        gt_depths = gt["gt_depths"]
        gt_local_pts = gt["gt_local_pts"]
        valid_masks = gt["masks"]
        gaussians_modulated = pred.get("gaussians_modulated")

        if gaussians_modulated is None or gt_imgs_paired is None:
            raise RuntimeError(
                "Appearance training requires paired/style supervision, but this batch is missing "
                "`img_paired` / `img_style`. Use `data=example_midair_nerfosr`."
            )

        pred_local_pts = pred["local_points"].clamp(min=-1e4, max=1e4)
        valid_masks_sub = valid_masks.squeeze(-1)

        scale_opt = torch.ones((B,), device=gt_imgs.device, dtype=gt_imgs.dtype)
        scale_opt_raw = scale_opt.clone()

        valid_depth_batch = valid_masks_sub.view(B, -1).any(dim=-1)
        if valid_depth_batch.any():
            weights = gt_local_pts[valid_depth_batch][..., 2].clamp_min(1e-3)
            weights = 1.0 / (weights + 1e-6)

            xyz_pred_roe = self.prepare_ROE(
                pred_local_pts[valid_depth_batch],
                valid_masks_sub[valid_depth_batch],
                target_size=self.local_align_res,
            )
            with torch.no_grad():
                xyz_gt_roe = self.prepare_ROE(
                    gt_local_pts[valid_depth_batch],
                    valid_masks_sub[valid_depth_batch],
                    target_size=self.local_align_res,
                )
                xyz_w_roe = self.prepare_ROE(
                    weights[..., None],
                    valid_masks_sub[valid_depth_batch],
                    target_size=self.local_align_res,
                )[..., 0]

            scale_opt_valid = align_points_scale(xyz_pred_roe, xyz_gt_roe, xyz_w_roe)
            scale_opt[valid_depth_batch] = scale_opt_valid
            scale_opt_raw[valid_depth_batch] = scale_opt_valid

        no_depth_batch = ~valid_depth_batch
        if bool(no_depth_batch.any().item()):
            cam_scale = self._camera_translation_scale(
                pred["camera_poses"][no_depth_batch],
                gt_c2w[no_depth_batch],
            )
            scale_opt[no_depth_batch] = cam_scale
            scale_opt_raw[no_depth_batch] = cam_scale

        scale_opt = torch.nan_to_num(scale_opt, nan=1.0, posinf=1.0, neginf=1.0).clamp(
            min=self.min_scale,
            max=self.max_scale,
        )

        render_w2c = se3_inverse(gt_c2w)
        gauss_render = self._apply_scale_alignment(pred["gaussians"], scale_opt)

        loss_rgb = gt_imgs.new_tensor(0.0)
        loss_ssim = gt_imgs.new_tensor(0.0)
        loss_depth = gt_imgs.new_tensor(0.0)
        loss_lpips = gt_imgs.new_tensor(0.0)
        loss_rgb_paired = gt_imgs.new_tensor(0.0)
        loss_ssim_paired = gt_imgs.new_tensor(0.0)
        loss_lpips_paired = gt_imgs.new_tensor(0.0)
        loss_style_moment = gt_imgs.new_tensor(0.0)

        (
            loss_rgb,
            loss_ssim,
            loss_lpips,
            render_rgb_finite,
            rgb_full_prefilter_finite,
            _,
        ) = self._render_rgb_losses_chunked(
            gauss_render,
            render_w2c,
            gt_ks,
            gt_imgs,
            H,
            W,
            use_lpips=self.lambda_lpips > 0,
        )

        if self.lambda_depth > 0 and bool(valid_masks.any().item()):
            loss_depth, render_depth_finite = self._render_depth_loss_chunked(
                gauss_render,
                render_w2c,
                gt_ks,
                gt_depths,
                valid_masks,
                H,
                W,
            )
        else:
            render_depth_finite = gt_imgs.new_tensor(1.0)

        gauss_render_mod = self._apply_scale_alignment(gaussians_modulated, scale_opt)
        render_w2c_paired = se3_inverse(gt_c2w_paired) if gt_c2w_paired is not None else render_w2c
        render_ks_paired = gt_ks_paired if gt_ks_paired is not None else gt_ks
        (
            loss_rgb_paired,
            loss_ssim_paired,
            loss_lpips_paired,
            render_rgb_paired_finite,
            rgb_paired_prefilter_finite,
            paired_moments,
        ) = self._render_rgb_losses_chunked(
            gauss_render_mod,
            render_w2c_paired,
            render_ks_paired,
            gt_imgs_paired,
            H,
            W,
            use_lpips=self.lambda_paired_lpips > 0,
            collect_moments=self.lambda_style_moment > 0,
        )

        if self.lambda_style_moment > 0:
            style_ref = gt_imgs_style if gt_imgs_style is not None else gt_imgs_paired
            style_mean, style_std = self._image_mean_std(style_ref)
            render_mean, render_std = paired_moments
            loss_style_moment = self._sanitize_scalar_loss(
                F.l1_loss(render_mean, style_mean) + F.l1_loss(render_std, style_std)
            )

        style_consistency = self._sanitize_scalar_loss(
            pred.get("style_code_consistency", gt_imgs.new_tensor(0.0))
        )
        style_reg = self._sanitize_scalar_loss(
            pred.get("appearance_color_delta_abs", gt_imgs.new_tensor(0.0))
            + pred.get("appearance_opacity_delta_abs", gt_imgs.new_tensor(0.0))
        )

        final_loss = (
            self.lambda_rgb * loss_rgb
            + self.lambda_ssim * loss_ssim
            + self.lambda_depth * loss_depth
            + self.lambda_lpips * loss_lpips
            + self.lambda_paired_rgb * loss_rgb_paired
            + self.lambda_paired_ssim * loss_ssim_paired
            + self.lambda_paired_lpips * loss_lpips_paired
            + self.lambda_style_consistency * style_consistency
            + self.lambda_style_reg * style_reg
            + self.lambda_style_moment * loss_style_moment
        )
        raw_total_loss = final_loss.detach()
        final_loss = torch.nan_to_num(final_loss, nan=0.0, posinf=0.0, neginf=0.0)

        grad_anchor_name, grad_anchor = self._find_grad_anchor(pred)
        if grad_anchor is not None:
            final_loss = final_loss + grad_anchor.reshape(-1)[0] * 0.0

        if torch.is_grad_enabled() and not final_loss.requires_grad:
            raise RuntimeError(
                "Appearance loss lost its gradient path. "
                f"anchor={grad_anchor_name}, final_loss_value={float(final_loss.detach())}"
            )

        details = {
            "loss_rgb": loss_rgb,
            "loss_ssim": loss_ssim,
            "loss_depth": loss_depth,
            "loss_lpips": loss_lpips,
            "loss_rgb_paired": loss_rgb_paired,
            "loss_ssim_paired": loss_ssim_paired,
            "loss_lpips_paired": loss_lpips_paired,
            "loss_style_moment": loss_style_moment,
            "style_code_consistency": style_consistency,
            "style_delta_norm": pred.get("style_delta_norm", gt_imgs.new_tensor(0.0)),
            "appearance_color_delta_abs": pred.get("appearance_color_delta_abs", gt_imgs.new_tensor(0.0)),
            "appearance_opacity_delta_abs": pred.get("appearance_opacity_delta_abs", gt_imgs.new_tensor(0.0)),
            "appearance_gate_mean": pred.get("appearance_gate_mean", gt_imgs.new_tensor(0.0)),
            "style_code_norm": pred.get("style_code_norm", gt_imgs.new_tensor(0.0)),
            "source_style_code_norm": pred.get("source_style_code_norm", gt_imgs.new_tensor(0.0)),
            "target_style_code_norm": pred.get("target_style_code_norm", gt_imgs.new_tensor(0.0)),
            "scale_opt_mean": scale_opt.detach().mean(),
            "scale_opt_raw_finite": self._finite_flag(scale_opt_raw, gt_imgs),
            "scale_opt_raw_absmax": self._safe_absmax(scale_opt_raw, gt_imgs),
            "gt_imgs_finite": self._finite_flag(gt_imgs, gt_imgs),
            "gt_imgs_paired_finite": self._finite_flag(gt_imgs_paired, gt_imgs),
            "gt_ks_finite": self._finite_flag(gt_ks, gt_imgs),
            "gt_ks_paired_finite": self._finite_flag(gt_ks_paired, gt_imgs),
            "gt_c2w_finite": self._finite_flag(gt_c2w, gt_imgs),
            "gt_c2w_paired_finite": self._finite_flag(gt_c2w_paired, gt_imgs),
            "base_gaussians_xyz_absmax": self._safe_absmax(pred["gaussians"]["xyz"], gt_imgs),
            "base_gaussians_scale_absmax": self._safe_absmax(pred["gaussians"]["scale"], gt_imgs),
            "mod_gaussians_color_absmax": self._safe_absmax(gaussians_modulated["color"], gt_imgs),
            "mod_gaussians_opacity_absmax": self._safe_absmax(gaussians_modulated["opacity"], gt_imgs),
            "render_rgb_finite": render_rgb_finite,
            "rgb_full_prefilter_finite": rgb_full_prefilter_finite,
            "render_depth_finite": render_depth_finite,
            "render_rgb_paired_finite": render_rgb_paired_finite,
            "rgb_paired_prefilter_finite": rgb_paired_prefilter_finite,
            "raw_total_loss_finite": self._finite_flag(raw_total_loss, gt_imgs),
            "raw_total_loss_abs": self._safe_absmax(raw_total_loss, gt_imgs),
            "loss_has_grad": gt_imgs.new_tensor(float(final_loss.requires_grad)),
            "total_loss": final_loss,
        }
        return final_loss, details
