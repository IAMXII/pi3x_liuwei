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
            warmup_ratio=0.2
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

    def forward(self, pred, gt_raw, current_epoch=None, total_epochs=None, enable_pose_loss=False, depth_only_ratio=0, rgb_depth_ratio=0.5):
        """
        Args:
            pred: Model prediction output (must contain 'gaussians' and 'camera_poses')
            gt_raw: Ground Truth data
            current_epoch: Current training epoch
            total_epochs: Total training epochs
            enable_pose_loss: Whether to enable pose supervision
            depth_only_ratio: 仅训练深度 Loss 的 Epoch 占比 (默认 0.2，即前 20%)
            rgb_depth_ratio: 训练 RGB + SSIM + Depth 的 Epoch 占比截止点 (默认 0.4，即 20%~40%)
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
        pred_w2c = se3_inverse(pred_c2w)
        
        # --- 阶段控制逻辑 (Curriculum Learning) ---
        progress = 1.0  # Default to full training if epochs are not provided
        if current_epoch is not None and total_epochs is not None and total_epochs > 0:
            progress = current_epoch / total_epochs

        # 根据传入的参数控制阶段
        # Phase 1: 0 ~ depth_only_ratio - Only Depth
        # Phase 2: depth_only_ratio ~ rgb_depth_ratio - RGB + SSIM + Depth
        # Phase 3: >= rgb_depth_ratio - RGB + SSIM + Depth + Pose + Regularization
        phase_depth_only = progress < depth_only_ratio
        phase_rgb_depth = depth_only_ratio <= progress < rgb_depth_ratio
        phase_all = progress >= rgb_depth_ratio
        
        # Initialize all losses to 0.0 safely
        device = pred_c2w.device
        loss_rgb = torch.tensor(0.0, device=device)
        loss_ssim = torch.tensor(0.0, device=device)
        loss_depth = torch.tensor(0.0, device=device)
        loss_pose_R = torch.tensor(0.0, device=device)
        loss_pose_T = torch.tensor(0.0, device=device)
        loss_repulsion = torch.tensor(0.0, device=device)
        loss_sparsity = torch.tensor(0.0, device=device)
        loss_consist = torch.tensor(0.0, device=device)
        depth_scale_scalar = 1.0

        # ==========================================
        # 3. 深度渲染与 Loss 计算 (所有阶段均开启)
        # ==========================================
        if self.lambda_depth > 0 and num_near is not None:
            depth_near_map, _, _ = self._render_gs(
                gauss, pred_w2c, gt_ks, H, W,
                num_gaussians=num_near, render_mode='ED'
            )
            
            target_depth = gt_depths.reshape(-1)
            pred_depth = depth_near_map.reshape(-1)
            valid_mask = (target_depth > 1e-4)

            # [Modified] Logarithmic Depth Loss
            if valid_mask.sum() > 10:  
                t_masked = target_depth[valid_mask]
                p_masked = pred_depth[valid_mask]
                
                epsilon = 1e-7
                log_pred = torch.log(p_masked + epsilon)
                log_target = torch.log(t_masked + epsilon)
                
                log_pred_centered = log_pred - log_pred.mean()
                log_target_centered = log_target - log_target.mean()

                loss_depth = F.l1_loss(log_pred_centered, log_target_centered)
                
                with torch.no_grad():
                    dot_pt = (p_masked * t_masked).sum()
                    dot_pp = (p_masked.pow(2)).sum()
                    scale = dot_pt / (dot_pp + 1e-8)
                    depth_scale_scalar = scale.item()

        # ==========================================
        # 4. RGB 渲染与 Loss 计算 (突破 depth_only_ratio 时开启)
        # ==========================================
        if not phase_depth_only:
            rgb_full, _, _ = self._render_gs(
                gauss, pred_w2c, gt_ks, H, W,
                num_gaussians=num_near, render_mode='RGB'
            )
            rgb_full = rgb_full.reshape(B * N, H, W, 3).permute(0, 3, 1, 2)
            gt_imgs_reshaped = gt_imgs.reshape(B * N, 3, H, W)
            # # ===================================================================
            # # 🔍 [DEBUG] 检查图像值域匹配度
            # # ===================================================================
            # with torch.no_grad():
            #     gt_min, gt_max = gt_imgs.min().item(), gt_imgs.max().item()
            #     pred_min, pred_max = rgb_full.min().item(), rgb_full.max().item()
                
            #     # 允许极微小的浮点误差 (1e-4)，如果超出则报警
            #     if gt_min < -1e-4 or gt_max > 1.0001:
            #         print("\n" + "!"*60, flush=True)
            #         print("🚨 致命警告: 发现 GT 图像值域异常！这会导致 SSIM 失效！", flush=True)
            #         print(f"👉 GT 图像 (gt_imgs) 值域范围:   [{gt_min:.4f}, {gt_max:.4f}]",flush=True)
            #         print(f"👉 渲染图像 (rgb_full) 值域范围: [{pred_min:.4f}, {pred_max:.4f}]",flush=True)
                    
            #         if gt_min < 0:
            #             print("💡 诊断: 你的 GT 图像似乎包含了负数！可能是因为 DataLoader 里的",flush=True)
            #             print("   Normalize(mean=[0.485...], std=[0.229...]) 操作污染了 GT 数据。",flush=True)
            #         elif gt_max > 2.0:
            #             print("💡 诊断: 你的 GT 图像可能还是 [0, 255] 的 uint8/float 格式，未归一化！",flush=True)
            #         print("!"*60 + "\n", flush=True)
            # # ===================================================================
            loss_rgb = F.l1_loss(rgb_full, gt_imgs_reshaped)
            loss_ssim = 1.0 - ssim(rgb_full, gt_imgs_reshaped, data_range=1.0)

        # ==========================================
        # 5. Pose 与 正则化 Loss 计算 (达到 rgb_depth_ratio 时开启)
        # ==========================================
        if phase_all:
            # Camera Pose Loss
            if self.lambda_pose > 0 and enable_pose_loss:
                loss_pose_R = F.l1_loss(pred_c2w[..., :3, :3], gt_c2w[..., :3, :3])
                loss_pose_T = F.l1_loss(pred_c2w[..., :3, 3], gt_c2w[..., :3, 3])

            # Regularization
            if self.lambda_repulsion > 0 and num_near is not None:
                if torch.rand(1).item() < 0.1:
                    loss_repulsion = self._calc_repulsion_loss(gauss['xyz'], num_near) * 10.0

            if self.lambda_sparsity > 0:
                opacity = gauss['opacity']
                total_sparsity = 0.0
                for b in range(B):
                    n = int(num_near[b].item())
                    if n > 0:
                        total_sparsity += opacity[b, :n].mean()
                loss_sparsity = total_sparsity / B

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

        # 6. Total Loss Accumulation
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

        # Details Dictionary for Logging
        details = {
            "loss_gs_rgb": loss_rgb,
            "loss_gs_ssim": loss_ssim,
            "loss_gs_depth": loss_depth,
            "depth_scale": depth_scale_scalar, 
            "loss_pose_R": loss_pose_R,
            "loss_pose_T": loss_pose_T,
            "loss_repulsion": loss_repulsion,
            "loss_sparsity": loss_sparsity,
            "loss_consist": loss_consist,
            "total_loss": final_loss,
            "phase_depth_only": float(phase_depth_only),
            "phase_rgb_depth": float(phase_rgb_depth),
            "phase_all": float(phase_all)
        }

        return final_loss, details