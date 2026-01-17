from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)
    

class ContextTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
    ):
        super().__init__()

        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim)

        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope, 
                # cross_attn_class=MemEffCrossAttentionRope,
                attn_class=FlashAttentionRope, 
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, context, xpos=None, ypos=None):
        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class AnchorGaussianHead(nn.Module):
#     """
#     Feed-forward 3DGS Head using Anchor-based Competitive Allocation.
#     Replaces LinearPts3d.
#     """
#
#     def __init__(self,
#                  embed_dim,
#                  num_anchors=4096,
#                  num_sky_anchors=1024,
#                  patch_size=16):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_anchors = num_anchors
#         self.num_sky_anchors = num_sky_anchors
#
#         # 1. Learnable Anchors (in Canonical Unit Space)
#         # Position: (x, y, z) for object
#         self.anchor_pos = nn.Parameter(torch.rand(num_anchors, 3) * 2 - 1)
#         # Sky: (theta, phi) for background (infinite depth)
#         self.sky_anchor_dir = nn.Parameter(torch.randn(num_sky_anchors, 3))
#
#         # Learnable Queries for Slot Attention
#         self.anchor_query = nn.Parameter(torch.randn(num_anchors + num_sky_anchors, embed_dim))
#
#         # 2. Feature Upsampler (Token -> Feature Map)
#         # 将 DINOv2 的 Patch Token 上采样，恢复一定的空间分辨率以便采样
#         self.feat_upsampler = nn.Sequential(
#             nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
#             nn.BatchNorm2d(embed_dim // 2),
#             nn.ReLU(),
#             nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
#         )
#         self.norm = nn.LayerNorm(embed_dim)
#
#         # 3. Slot Attention Components
#         self.temperature = embed_dim ** -0.5
#
#         # 4. Gating Network (For Staticity)
#         # Input: Feature Variance (1) + Feature Mean (D) -> Output: Gate Score (1)
#         self.gating_mlp = nn.Sequential(
#             nn.Linear(embed_dim + 1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#
#         # 5. Decoders (Feature -> 3DGS Attributes)
#         # Output: d_xyz(3), rotation(4), scaling(3), opacity(1), rgb(3) or sh(3*k)
#         self.geo_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 3 + 4 + 3 + 1))
#         self.color_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 3))  # simple RGB for now
#
#     def forward(self, tokens, camera_poses, img_shape):
#         """
#         tokens: [B, S, D] (S = N_views * H_patch * W_patch)
#         camera_poses: [B, N_views, 4, 4] (World-to-Camera or Camera-to-World)
#         img_shape: (H, W)
#         """
#         B, S, D = tokens.shape
#         H, W = img_shape
#         N_views = camera_poses.shape[1]
#
#         # --- 1. Reshape & Upsample Features ---
#         # [B, S, D] -> [B*N, D, H_p, W_p]
#         h_p, w_p = H // self.patch_size, W // self.patch_size
#         feats = tokens.view(B, N_views, h_p, w_p, D).view(B * N_views, h_p, w_p, D)
#         feats = feats.permute(0, 3, 1, 2)  # [BN, D, Hp, Wp]
#
#         # Upsample: 14x14 -> 28x28 (example) for better sampling precision
#         feats_high = self.feat_upsampler(feats)  # [BN, D, H_feat, W_feat]
#         H_f, W_f = feats_high.shape[-2:]
#
#         # --- 2. Anchor Projection (Geometry-aware Sampling) ---
#         # 我们需要将 3D Anchor 投影到每个视角的特征图上
#
#         # Expand anchors to batch
#         # Obj: [B, M_obj, 3]
#         anchors_obj = self.anchor_pos.unsqueeze(0).expand(B, -1, -1)
#         # Sky: [B, M_sky, 3] (Directions)
#         anchors_sky = F.normalize(self.sky_anchor_dir, dim=-1).unsqueeze(0).expand(B, -1, -1)
#
#         # Prepare for sampling
#         sampled_features = []  # Will store features from all views for each anchor
#
#         # Loop over views (Can be vectorized but loop is clearer for logic)
#         for v in range(N_views):
#             # Pose: Assuming Camera-to-World (T_cw). Need World-to-Camera (T_wc) for projection
#             pose = camera_poses[:, v]  # [B, 4, 4]
#             inv_pose = torch.inverse(pose)  # [B, 4, 4] T_wc
#
#             R = inv_pose[:, :3, :3]
#             T = inv_pose[:, :3, 3:]
#
#             # --- Project Object Anchors ---
#             # P_cam = R * P_world + T
#             p_cam = torch.matmul(R, anchors_obj.transpose(1, 2)) + T  # [B, 3, M]
#             z = p_cam[:, 2:3, :] + 1e-5
#             xy = p_cam[:, :2, :] / z  # Perspective division
#
#             # Normalize to [-1, 1] for grid_sample (Assuming FOV approx 90 or calibrated)
#             # In practice, you need Intrinsic K. Here we assume normalized coords or learnable K.
#             # Simplified: assuming standard range.
#             grid_obj = xy.transpose(1, 2).unsqueeze(1)  # [B, 1, M, 2]
#
#             # --- Project Sky Anchors ---
#             # Only Rotation matters for infinite depth
#             p_sky_cam = torch.matmul(R, anchors_sky.transpose(1, 2))  # [B, 3, M]
#             z_sky = p_sky_cam[:, 2:3, :] + 1e-5
#             xy_sky = p_sky_cam[:, :2, :] / z_sky
#             grid_sky = xy_sky.transpose(1, 2).unsqueeze(1)  # [B, 1, M_sky, 2]
#
#             # --- Sampling ---
#             # grid_sample treats values outside [-1, 1] as zeros (padding) -> Occlusion handling
#             curr_feat = feats_high[B * v: B * (v + 1)]  # [B, D, H, W]
#
#             feat_obj = F.grid_sample(curr_feat, grid_obj, align_corners=True).squeeze(2)  # [B, D, M_obj]
#             feat_sky = F.grid_sample(curr_feat, grid_sky, align_corners=True).squeeze(2)  # [B, D, M_sky]
#
#             # Concat object and sky features
#             feat_all = torch.cat([feat_obj, feat_sky], dim=2)  # [B, D, M_total]
#             sampled_features.append(feat_all.permute(0, 2, 1))  # [B, M_total, D]
#
#         # Stack views: [B, N_views, M_total, D]
#         multi_view_feats = torch.stack(sampled_features, dim=1)
#
#         # --- 3. Feature Aggregation & Static Gating ---
#
#         # Calculate Variance across views (measure of multi-view consistency)
#         # [B, M_total, D]
#         feat_mean = multi_view_feats.mean(dim=1)
#         feat_var = multi_view_feats.var(dim=1).mean(dim=-1, keepdim=True)  # [B, M_total, 1] average variance scaler
#
#         # Gating Score: Low variance (consistent) -> 1.0, High variance (dynamic) -> 0.0
#         gate_score = self.gating_mlp(torch.cat([feat_mean, feat_var], dim=-1))  # [B, M_total, 1]
#
#         # --- 4. Anchor-Slot Allocation (Competitive) ---
#         # Instead of simple mean, use the Query to attend to views
#         # Here we simplify: The Anchors compete for information from the aggregated mean
#         # Or more complex: Slot Attention between Learnable Query and Sampled Features
#
#         # Let's add the learnable query residual to the sampled features
#         slots = feat_mean + self.anchor_query.unsqueeze(0)  # [B, M, D]
#
#         # --- 5. Decode Gaussian Parameters ---
#         geo_out = self.geo_head(slots)
#
#         d_xyz = torch.tanh(geo_out[..., :3]) * 0.1  # Small offset from anchor center
#         rot = F.normalize(geo_out[..., 3:7], dim=-1)  # Quaternion
#         scale = torch.sigmoid(geo_out[..., 7:10]) * 0.1  # Small scale init
#         opacity = torch.sigmoid(geo_out[..., 10:11])
#
#         color = torch.sigmoid(self.color_head(slots))
#
#         # --- 6. Apply Logic ---
#
#         # A. Apply Gating (Point 2: Staticity)
#         final_opacity = opacity * gate_score
#
#         # B. Apply Anchor Position
#         base_xyz = torch.cat([anchors_obj, anchors_sky], dim=1)  # [B, M, 3]
#         final_xyz = base_xyz + d_xyz
#
#         return {
#             "xyz": final_xyz,
#             "opacity": final_opacity,
#             "scale": scale,
#             "rotation": rot,
#             "color": color,
#             "gate_score": gate_score  # For Loss monitoring
#         }

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorGaussianHead(nn.Module):
    """
    Phase 2 (Refined):
    - 显式区分近景(Object)与远景(Sky)
    - 将 Sky Anchors 映射到超大半径球面上
    - 返回 num_near 以便区分处理
    """

    def __init__(self,
                 embed_dim,
                 num_sky_anchors=1024,
                 patch_size=14,
                 K=4,
                 sky_radius=1000.0):  # 增加天空球半径
        super().__init__()
        self.patch_size = patch_size
        self.num_sky_anchors = num_sky_anchors
        self.K = K
        self.sky_radius = sky_radius

        # 1. Position Encoder (XYZ -> Query)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 2. Sky Components
        # 这里的 dir 存储单位向量方向
        self.sky_anchor_dir = nn.Parameter(torch.randn(num_sky_anchors, 3))
        self.sky_anchor_query = nn.Parameter(torch.randn(num_sky_anchors, embed_dim))

        # 3. Feature Upsampler
        self.feat_upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        )

        # 4. Heads
        self.gating_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.geo_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 11 * self.K)
        )
        self.color_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 3 * self.K)
        )

    def forward(self, tokens, camera_poses, img_shape, selected_anchors):
        B, S, D = tokens.shape
        H, W = img_shape
        N_views = camera_poses.shape[1]
        M_obj = selected_anchors.shape[1]  # 动态获取近景数量

        # --- 1. Prepare Anchors ---
        # 近景：直接使用
        anchors_obj = selected_anchors

        # 远景：归一化方向并推向无穷远球面 (sky_radius)
        sky_dirs = F.normalize(self.sky_anchor_dir, dim=-1)  # [M_sky, 3]
        anchors_sky = (sky_dirs * self.sky_radius).unsqueeze(0).expand(B, -1, -1)  # [B, M_sky, 3]

        # --- 2. Encode to Query ---
        query_obj = self.pos_encoder(anchors_obj)
        # 天空编码：可以考虑传入单位方向而非绝对坐标，以增加数值稳定性
        query_sky = self.pos_encoder(sky_dirs.unsqueeze(0).expand(B, -1, -1))
        # 或者使用预定义的 sky_anchor_query 进行结合
        query_sky = query_sky + self.sky_anchor_query.unsqueeze(0).expand(B, -1, -1)

        # --- 3. Feature Sampling ---
        h_p, w_p = H // self.patch_size, W // self.patch_size
        feats = tokens.view(B, N_views, h_p, w_p, D).view(B * N_views, h_p, w_p, D).permute(0, 3, 1, 2)
        feats_high = self.feat_upsampler(feats)

        sampled_features = []
        for v in range(N_views):
            pose = camera_poses[:, v]
            inv_pose = torch.inverse(pose)
            R, T = inv_pose[:, :3, :3], inv_pose[:, :3, 3:]

            # Project Object (Translation + Rotation)
            p_cam_obj = torch.matmul(R, anchors_obj.transpose(1, 2)) + T
            grid_obj = (p_cam_obj[:, :2, :] / (p_cam_obj[:, 2:3, :] + 1e-5)).transpose(1, 2).unsqueeze(1)

            # Project Sky (Only Rotation - 远景忽略平移 T)
            p_cam_sky = torch.matmul(R, anchors_sky.transpose(1, 2))
            grid_sky = (p_cam_sky[:, :2, :] / (p_cam_sky[:, 2:3, :] + 1e-5)).transpose(1, 2).unsqueeze(1)

            curr_feat = feats_high[B * v: B * (v + 1)]
            feat_obj = F.grid_sample(curr_feat, grid_obj, align_corners=True, padding_mode='border').squeeze(2)
            feat_sky = F.grid_sample(curr_feat, grid_sky, align_corners=True, padding_mode='border').squeeze(2)

            sampled_features.append(torch.cat([feat_obj, feat_sky], dim=2).permute(0, 2, 1))

        # --- 4. Aggregation & Slot ---
        multi_view_feats = torch.stack(sampled_features, dim=1)
        feat_mean = multi_view_feats.mean(dim=1)
        feat_var = multi_view_feats.var(dim=1).mean(dim=-1, keepdim=True)

        gate_score = self.gating_mlp(torch.cat([feat_mean, feat_var], dim=-1))

        query_total = torch.cat([query_obj, query_sky], dim=1)
        slots = feat_mean + query_total

        # --- 5. Decoding ---
        M_total = slots.shape[1]
        geo_raw = self.geo_head(slots).view(B, M_total, self.K, 11)
        color_raw = self.color_head(slots).view(B, M_total, self.K, 3)

        base_xyz = torch.cat([anchors_obj, anchors_sky], dim=1).unsqueeze(2).expand(-1, -1, self.K, -1)

        # 偏移限制：近景可以有微小偏移，天空则保持原位或极小修正
        d_xyz = torch.tanh(geo_raw[..., :3]) * 0.1
        final_xyz = base_xyz + d_xyz

        # 其他属性计算
        rot = F.normalize(geo_raw[..., 3:7], dim=-1)
        scale = torch.sigmoid(geo_raw[..., 7:10]) * 0.05
        # 如果是天空，Scale 应该相对较大以覆盖视野
        sky_mask = torch.ones((B, M_total, self.K, 1), device=slots.device)
        sky_mask[:, :M_obj] = 0.0  # 标记天空部分
        scale = scale * (1.0 + sky_mask * 10.0)  # 让天空 Gaussian 变大

        opacity = torch.sigmoid(geo_raw[..., 10:11])
        final_color = torch.sigmoid(color_raw)
        final_opacity = opacity * gate_score.unsqueeze(2)

        # 确定近景 Gaussian 的总数 (每个 Anchor 产生 K 个)
        num_near = M_obj * self.K

        return {
            "xyz": final_xyz.reshape(B, -1, 3),
            "opacity": final_opacity.reshape(B, -1, 1),
            "scale": scale.reshape(B, -1, 3),
            "rotation": rot.reshape(B, -1, 4),
            "color": final_color.reshape(B, -1, 3),
            "num_near": torch.full((B,), num_near, dtype=torch.long, device=slots.device)
        }