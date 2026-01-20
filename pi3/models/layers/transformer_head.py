from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch


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


# ... (保持 GlobalContextModulation, TransformerDecoder, LinearPts3d 等类不变) ...

# [Innovation B] 全局上下文调制模块
class GlobalContextModulation(nn.Module):
    def __init__(self, dim, global_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_global = nn.Linear(global_dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, local_feats, global_tokens):
        # local_feats: [B, M, C]
        # global_tokens: [B, N_views, L_g, C_g] -> need pooling
        B, M, C = local_feats.shape

        # 简单聚合全局信息 (Average Pooling over views and tokens)
        if global_tokens.dim() == 4:
            g_feat = global_tokens.flatten(1, 2).mean(dim=1)  # [B, C_g]
        else:
            g_feat = global_tokens.mean(dim=1)

        g_feat = self.proj_global(g_feat).unsqueeze(1).expand(-1, M, -1)  # [B, M, C]

        # 门控融合
        concat = torch.cat([self.norm(local_feats), g_feat], dim=-1)
        gate = self.gate(concat)

        return self.proj_out(local_feats * gate + g_feat * (1 - gate))


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
        # x = (attn @ v.transpose(-1, -2)).transpose(2, 3).reshape(B, M, D)
        x = (attn @ v).transpose(2, 3).reshape(B, M, D)
        return x


class AnchorGaussianHead(nn.Module):
    def __init__(self,
                 embed_dim,
                 in_channels=2048,  # <--- [FIX] 新增参数：输入特征的通道数
                 num_sky_anchors=1024,
                 patch_size=14,
                 K=4,
                 sky_radius=100.0,
                 global_dim=1024):
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

        # 3. Feature Upsampler
        # [FIX] 关键修改：第一层 ConvTranspose2d 的输入通道改为 in_channels (2048)
        self.feat_upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, embed_dim // 2, kernel_size=2, stride=2),
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

        # [Innovation A] Split Probability Head
        self.split_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # [Innovation B] Global Context Modulator
        self.global_modulator = GlobalContextModulation(embed_dim, global_dim)

        # 5. Output Heads
        self.geo_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 11 * self.K))
        self.color_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 3 * self.K))

    def forward(self, tokens, camera_poses, img_shape, selected_anchors,
                global_tokens=None, anchor_confidence=None):
        """
        tokens: [B, S, D_in] (Backbone output, D_in is usually 2048)
        """
        B = camera_poses.shape[0]
        N_views = camera_poses.shape[1]

        # --- 1. Prepare Anchors ---
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

        # tokens shape [B, N*h*w, 2048] -> reshape to image layout
        # Upsample Features: [B*N, 2048, h, w] -> [B*N, embed_dim, h*2, w*2]
        feats = tokens.view(B, N_views, h_p, w_p, -1).permute(0, 1, 4, 2, 3)  # [B, N, C_in, H, W]
        feats = feats.reshape(B * N_views, -1, h_p, w_p)
        
        # 这里之前报错是因为 feats 是 2048 通道，而 conv 期望 1024
        # 现在修改了 __init__，应该匹配了
        feats_high = self.feat_upsampler(feats)  # [BN, D_embed, H*2, W*2]

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

        # Slot Attention
        slots = self.slot_attention(query_total, multi_view_feats)
        slots = slots + query_total  # Residual

        # [Innovation B] Global Context Modulation
        if global_tokens is not None:
            slots = self.global_modulator(slots, global_tokens)

        # --- 5. Decode ---
        M_total = slots.shape[1]
        geo_raw = self.geo_head(slots).view(B, M_total, self.K, 11)
        color_raw = self.color_head(slots).view(B, M_total, self.K, 3)

        # [Innovation A] Split Probability
        split_prob = self.split_head(slots).view(B, M_total, 1, 1)

        base_xyz = anchors_all.unsqueeze(2).expand(-1, -1, self.K, -1)

        d_xyz = torch.tanh(geo_raw[..., :3]) * 0.1
        final_xyz = base_xyz + d_xyz
        rot = F.normalize(geo_raw[..., 3:7], dim=-1)

        scale = torch.sigmoid(geo_raw[..., 7:10]) * 0.05
        # Sky scale boost
        is_sky = torch.zeros((B, M_total, self.K, 1), device=slots.device)
        is_sky[:, anchors_obj.shape[1]:] = 1.0
        scale = scale * (1.0 + is_sky * 50.0)

        # Apply Split Logic
        split_mask = torch.ones_like(scale)
        split_mask[:, :, 1:, :] = split_prob
        scale = scale * split_mask

        opacity = torch.sigmoid(geo_raw[..., 10:11])
        final_color = torch.sigmoid(color_raw)

        # Apply Gating
        final_opacity = opacity * gate_score.unsqueeze(2)

        # [Innovation D] Uncertainty-guided Pruning
        if anchor_confidence is not None:
            sky_conf = torch.ones((B, self.num_sky_anchors, 1), device=slots.device) * 10.0
            full_conf = torch.cat([anchor_confidence, sky_conf], dim=1)
            conf_prob = torch.sigmoid(full_conf).unsqueeze(2).expand(-1, -1, self.K, 1)
            final_opacity = final_opacity * conf_prob

        return {
            "xyz": final_xyz.reshape(B, -1, 3),
            "opacity": final_opacity.reshape(B, -1, 1),
            "scale": scale.reshape(B, -1, 3),
            "rotation": rot.reshape(B, -1, 4),
            "color": final_color.reshape(B, -1, 3),
            "gate_score": gate_score,
            "num_near": torch.full((B,), anchors_obj.shape[1] * self.K, dtype=torch.long, device=slots.device),
            "base_anchors": base_xyz.reshape(B, -1, 3)
        }