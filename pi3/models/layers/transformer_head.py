from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch


class TransformerDecoder(nn.Module):
    # ... (保持原样不变) ...
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


class LinearPts3d(nn.Module):
    # ... (保持原样不变) ...
    def __init__(self, patch_size, dec_embed_dim, output_dim=3, ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dec_embed_dim, (output_dim) * self.patch_size ** 2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
        return feat.permute(0, 2, 3, 1)


class ContextTransformerDecoder(nn.Module):
    # ... (保持原样不变) ...
    def __init__(self, in_dim, out_dim, dec_embed_dim=512, depth=5, dec_num_heads=8, mlp_ratio=4, rope=None):
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


# ==========================================
# New / Modified Classes for 3DGS
# ==========================================

class SlotAttention(nn.Module):
    """Simple Slot Attention for aggregating multi-view features."""

    def __init__(self, dim, num_heads=4, qkv_bias=False, eps=1e-6):
        super().__init__()
        self.norm_slots = nn.LayerNorm(dim, eps=eps)
        self.norm_feat = nn.LayerNorm(dim, eps=eps)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim // num_heads) ** -0.5
        self.num_heads = num_heads

    def forward(self, slots, features):
        # slots (Query): [B, M, D]
        # features (Key/Val): [B, M, N_views, D]
        B, M, N_views, D = features.shape

        slots_norm = self.norm_slots(slots)
        feat_norm = self.norm_feat(features)

        q = self.to_q(slots_norm).view(B, M, 1, self.num_heads, -1)
        k = self.to_k(feat_norm).view(B, M, N_views, self.num_heads, -1)
        v = self.to_v(feat_norm).view(B, M, N_views, self.num_heads, -1)

        q = q.permute(0, 1, 3, 2, 4)  # [B, M, H, 1, d]
        k = k.permute(0, 1, 3, 4, 2)  # [B, M, H, d, N]
        v = v.permute(0, 1, 3, 2, 4)  # [B, M, H, 1, N] (logic wise)

        attn = (q @ k) * self.scale  # [B, M, H, 1, N]
        attn = attn.softmax(dim=-1)

        # Aggregate
        x = (attn @ v.transpose(-1, -2)).transpose(2, 3).reshape(B, M, D)
        return x


class AnchorGaussianHead(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_sky_anchors=1024,
                 patch_size=14,
                 K=4,
                 sky_radius=100.0):  # Sky radius
        super().__init__()
        self.patch_size = patch_size
        self.num_sky_anchors = num_sky_anchors
        self.K = K
        self.sky_radius = sky_radius

        # 1. Position Encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim)
        )

        # 2. Sky Params
        self.sky_anchor_dir = nn.Parameter(torch.randn(num_sky_anchors, 3))
        self.sky_anchor_query = nn.Parameter(torch.randn(num_sky_anchors, embed_dim))

        # 3. Feature Upsampler (Restore resolution for better sampling)
        self.feat_upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2), nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        )

        # 4. Attention & Gating
        self.slot_attention = SlotAttention(embed_dim)

        # Input: [Mean, Var] -> Gating
        self.gating_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # 5. Output Heads
        self.geo_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 11 * self.K))
        self.color_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 3 * self.K))

    def forward(self, tokens, camera_poses, img_shape, selected_anchors):
        """
        tokens: [B, S, D] (Backbone/Decoder output)
        """
        B = camera_poses.shape[0]
        N_views = camera_poses.shape[1]

        # --- 1. Prepare Anchors (Dual Stream) ---
        anchors_obj = selected_anchors  # [B, M_obj, 3]

        sky_dirs = F.normalize(self.sky_anchor_dir, dim=-1)
        anchors_sky = (sky_dirs * self.sky_radius).unsqueeze(0).expand(B, -1, -1)  # [B, M_sky, 3]

        query_obj = self.pos_encoder(anchors_obj)
        query_sky = self.pos_encoder(sky_dirs.unsqueeze(0).expand(B, -1, -1)) + \
                    self.sky_anchor_query.unsqueeze(0).expand(B, -1, -1)
        query_total = torch.cat([query_obj, query_sky], dim=1)  # [B, M_total, D]

        # --- 2. Feature Handling ---
        H, W = img_shape
        h_p, w_p = H // self.patch_size, W // self.patch_size

        # Upsample Features
        feats = tokens.view(B, N_views, h_p, w_p, -1).permute(0, 1, 4, 2, 3)  # [B, N, D, H, W]
        feats = feats.reshape(B * N_views, -1, h_p, w_p)
        feats_high = self.feat_upsampler(feats)  # [BN, D, H*2, W*2]

        # --- 3. Sampling (Projection) ---
        sampled_features = []
        anchors_all = torch.cat([anchors_obj, anchors_sky], dim=1)  # [B, M_tot, 3]

        for v in range(N_views):
            pose = camera_poses[:, v]  # [B, 4, 4]
            inv_pose = torch.inverse(pose)  # W2C
            R, T = inv_pose[:, :3, :3], inv_pose[:, :3, 3:]

            # Project
            p_cam = torch.matmul(R, anchors_all.transpose(1, 2)) + T
            depth = p_cam[:, 2:3, :] + 1e-5
            uv = p_cam[:, :2, :] / depth
            grid = uv.transpose(1, 2).unsqueeze(1)  # [B, 1, M, 2]

            # Sample
            curr_feat = feats_high[B * v: B * (v + 1)]
            sampled = F.grid_sample(curr_feat, grid, align_corners=True, padding_mode='border')
            sampled_features.append(sampled.squeeze(2).permute(0, 2, 1))

        # [B, M, N, D]
        multi_view_feats = torch.stack(sampled_features, dim=2)

        # --- 4. Gating & Slot Attention ---
        feat_mean = multi_view_feats.mean(dim=2)
        feat_var = multi_view_feats.var(dim=2).mean(dim=-1, keepdim=True)  # Variance for gating

        gate_score = self.gating_mlp(torch.cat([feat_mean, feat_var], dim=-1))  # [B, M, 1]

        # Slot Attention: Query=Anchor, Key/Val=Features
        slots = self.slot_attention(query_total, multi_view_feats)
        slots = slots + query_total  # Residual

        # --- 5. Decode ---
        M_total = slots.shape[1]
        geo_raw = self.geo_head(slots).view(B, M_total, self.K, 11)
        color_raw = self.color_head(slots).view(B, M_total, self.K, 3)

        base_xyz = anchors_all.unsqueeze(2).expand(-1, -1, self.K, -1)

        d_xyz = torch.tanh(geo_raw[..., :3]) * 0.1
        final_xyz = base_xyz + d_xyz
        rot = F.normalize(geo_raw[..., 3:7], dim=-1)

        # Scale handling
        scale = torch.sigmoid(geo_raw[..., 7:10]) * 0.05
        # Sky scale boost
        is_sky = torch.zeros((B, M_total, self.K, 1), device=slots.device)
        is_sky[:, anchors_obj.shape[1]:] = 1.0
        scale = scale * (1.0 + is_sky * 50.0)

        opacity = torch.sigmoid(geo_raw[..., 10:11])
        final_color = torch.sigmoid(color_raw)

        # Apply Gating
        final_opacity = opacity * gate_score.unsqueeze(2)

        return {
            "xyz": final_xyz.reshape(B, -1, 3),
            "opacity": final_opacity.reshape(B, -1, 1),
            "scale": scale.reshape(B, -1, 3),
            "rotation": rot.reshape(B, -1, 4),
            "color": final_color.reshape(B, -1, 3),
            "gate_score": gate_score,
            "num_near": torch.full((B,), anchors_obj.shape[1] * self.K, dtype=torch.long, device=slots.device)
        }