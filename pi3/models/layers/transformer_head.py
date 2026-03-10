############################################## light ######################################################

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
            # [优化] 如果显存充裕，这里也可以去掉 checkpoint
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


class AmbientLightFiLM(nn.Module):
    """利用 Global Tokens 预测全局光照参数，专门调制颜色特征 (FiLM)"""
    def __init__(self, dim, global_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film_gen = nn.Sequential(
            nn.Linear(global_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2)
        )

    def forward(self, local_feats, global_tokens):
        # 提取纯粹的全局环境向量
        if global_tokens.dim() == 4:
            g_feat = global_tokens.flatten(1, 2).mean(dim=1) 
        else:
            g_feat = global_tokens.mean(dim=1)
            
        # 生成全局光照的仿射参数
        film_params = self.film_gen(g_feat)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # 对局部特征进行特征级仿射调制
        gamma = gamma.unsqueeze(1).expand(-1, local_feats.shape[1], -1)
        beta = beta.unsqueeze(1).expand(-1, local_feats.shape[1], -1)
        
        return self.norm(local_feats) * (1.0 + gamma) + beta

class ViewWeighting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 极轻量级的 MLP：输入 [多视角特征 + 视角方向余弦]，输出标量权重
        self.mlp = nn.Sequential(
            nn.Linear(dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, feats, view_dirs, valid_mask):
        """
        feats: [B, M, V, D] 多视角特征
        view_dirs: [B, M, V, 3] Anchor 到相机的观测方向
        valid_mask: [B, M, V] 视锥体内的有效掩码
        """
        # 拼接特征和观测方向
        cat_feat = torch.cat([feats, view_dirs], dim=-1)
        
        # 预测原始权重: [B, M, V]
        scores = self.mlp(cat_feat).squeeze(-1)
        
        # 将视锥体外的视角权重设为负无穷，Softmax 后变为 0
        scores = scores.masked_fill(~valid_mask, -1e9)
        
        # 在 V 维度上计算权重
        weights = F.softmax(scores, dim=-1)
        
        # 加权求和: [B, M, D]
        agg_feat = (feats * weights.unsqueeze(-1)).sum(dim=2)
        return agg_feat, weights
    
class AnchorGaussianHead(nn.Module):
    def __init__(self,
                 embed_dim,
                 in_channels=2048,
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

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim)
        )

        self.sky_anchor_dir = nn.Parameter(torch.randn(num_sky_anchors, 3))
        self.sky_anchor_query = nn.Parameter(torch.randn(num_sky_anchors, embed_dim))

        self.feat_upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2), nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        )

        self.gating_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        self.split_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # 引入 FiLM 调制器
        self.global_modulator = AmbientLightFiLM(embed_dim, global_dim)

        self.geo_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 11 * self.K))
        self.color_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 3 * self.K))
        self.view_weighter = ViewWeighting(embed_dim)

    def _compact_batch(self, tensors_dict, mask, B):
        valid_counts = mask.sum(dim=1) 
        max_valid = int(valid_counts.max().item())
        
        if max_valid == 0:
            max_valid = 1
            mask[:, 0] = True
            valid_counts[:] = 1

        sorted_idxs = torch.argsort(mask.int(), dim=1, descending=True, stable=True)
        gather_idxs = sorted_idxs[:, :max_valid]

        out_dict = {}
        for name, tensor in tensors_dict.items():
            C = tensor.shape[-1]
            expanded_idxs = gather_idxs.unsqueeze(-1).expand(B, max_valid, C)
            compacted = torch.gather(tensor, 1, expanded_idxs)
            out_dict[name] = compacted
            
        return out_dict, valid_counts

    def _decode_heavy(self, slots, global_tokens=None):
        # 1. 几何不受全局光照影响
        geo_raw = self.geo_head(slots)
        split_prob = self.split_head(slots)
        
        # 2. 颜色特征接受全局光照的 FiLM 调制
        if global_tokens is not None:
            color_slots = self.global_modulator(slots, global_tokens)
        else:
            color_slots = slots
            
        color_raw = self.color_head(color_slots)
        return geo_raw, color_raw, split_prob

    def forward(self, tokens, camera_poses, intrinsics, img_shape, selected_anchors,
                global_tokens=None, anchor_confidence=None):
        B = camera_poses.shape[0]
        N_views = camera_poses.shape[1]

        anchors_obj = selected_anchors
        sky_dirs = F.normalize(self.sky_anchor_dir, dim=-1)
        anchors_sky = (sky_dirs * self.sky_radius).unsqueeze(0).expand(B, -1, -1)
        
        query_obj = self.pos_encoder(anchors_obj)
        query_sky = self.pos_encoder(sky_dirs.unsqueeze(0).expand(B, -1, -1)) + \
                    self.sky_anchor_query.unsqueeze(0).expand(B, -1, -1)
        query_total = torch.cat([query_obj, query_sky], dim=1)

        H, W = img_shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        
        # ==========================================================
        # [修复 2080 通道报错] 剔除 DINOv2 的 4 个 Register Tokens
        num_spatial_patches = h_p * w_p
        pure_spatial_tokens = tokens[:, -num_spatial_patches:, :]
        # ==========================================================
        
        feats = pure_spatial_tokens.view(B, N_views, h_p, w_p, -1).permute(0, 1, 4, 2, 3)
        feats = feats.reshape(B * N_views, -1, h_p, w_p)
        feats_high = self.feat_upsampler(feats)

        # sampled_features = []
        anchors_all = torch.cat([anchors_obj, anchors_sky], dim=1)
        
        sampled_features = []
        valid_masks = []
        view_dirs = []

        for v in range(N_views):
            pose = camera_poses[:, v]
            K_mat = intrinsics[:, v]
            inv_pose = torch.inverse(pose)
            R, T = inv_pose[:, :3, :3], inv_pose[:, :3, 3:]
            
            # 1. 转换回当前相机的物理坐标
            p_cam = torch.matmul(R, anchors_all.transpose(1, 2)) + T
            depth = p_cam[:, 2:3, :] + 1e-5
            
            # 2. 投影到像素坐标系
            p_pixel = torch.matmul(K_mat[:, :2, :3], p_cam)
            u_pixel = p_pixel[:, 0:1, :] / depth
            v_pixel = p_pixel[:, 1:2, :] / depth
            
            # 3. 归一化到 [-1, 1]
            u_norm = (u_pixel / W) * 2.0 - 1.0
            v_norm = (v_pixel / H) * 2.0 - 1.0
            
            # === 新增：计算 Frustum Mask (是否在视野内且在相机前方) ===
            # 允许稍微越界一点特征 (例如 1.1)
            in_frustum = (u_norm >= -1.1) & (u_norm <= 1.1) & \
                        (v_norm >= -1.1) & (v_norm <= 1.1) & \
                        (depth > 0.05)
            # valid_masks.append(in_frustum.squeeze(1).transpose(1, 2)) # [B, M, 1]
            valid_masks.append(in_frustum.transpose(1, 2)) # [B, M, 1]
            # === 新增：计算观测方向 ===
            # 相机中心在世界坐标系的位置: -R^T * T
            cam_center = -torch.matmul(R.transpose(1, 2), T) # [B, 3, 1]
            # 观测方向向量: 从 相机 指向 Anchor
            dir_vec = anchors_all - cam_center.transpose(1, 2)
            dir_vec = F.normalize(dir_vec, p=2, dim=-1) # [B, M, 3]
            view_dirs.append(dir_vec)

            # ... 原有的 grid_sample 逻辑 ...
            grid = torch.cat([u_norm, v_norm], dim=1).transpose(1, 2).unsqueeze(1)
            curr_feat = feats_high[B * v: B * (v + 1)]
            sampled = F.grid_sample(curr_feat, grid, align_corners=True, padding_mode='zeros')
            sampled_features.append(sampled.squeeze(2).permute(0, 2, 1))
        # valid_masks = torch.stack(valid_masks, dim=2).squeeze(-1) # [B, M, V]
        # 堆叠张量
        multi_view_feats = torch.stack(sampled_features, dim=2) # [B, M, V, D]
        valid_masks = torch.stack(valid_masks, dim=2).squeeze(-1) # [B, M, V]
        view_dirs = torch.stack(view_dirs, dim=2)                 # [B, M, V, 3]

        feat_mean = multi_view_feats.mean(dim=2)
        feat_var = multi_view_feats.var(dim=2).mean(dim=-1, keepdim=True)
        
        # 引入特征方差惩罚，压制动态物体
        variance_penalty = torch.exp(-2.0 * feat_var)
        gate_score = self.gating_mlp(torch.cat([feat_mean, feat_var], dim=-1))
        gate_score = gate_score * variance_penalty
        valid_counts = valid_masks.sum(dim=-1)
        valid_masks = torch.where(valid_counts.unsqueeze(-1) == 0, torch.ones_like(valid_masks), valid_masks)
        slots, view_weights = self.view_weighter(multi_view_feats, view_dirs, valid_masks)
        slots = slots + query_total

        M_total = slots.shape[1]
        
        # 传入 global_tokens 进行着色调制
        geo_raw, color_raw, split_prob = self._decode_heavy(slots, global_tokens)

        geo_raw = geo_raw.view(B, M_total, self.K, 11)
        color_raw = color_raw.view(B, M_total, self.K, 3)
        split_prob = split_prob.view(B, M_total, 1, 1)

        base_xyz = anchors_all.unsqueeze(2).expand(-1, -1, self.K, -1)
        d_xyz = torch.tanh(geo_raw[..., :3]) * 0.5
        final_xyz = base_xyz + d_xyz
        rot = F.normalize(geo_raw[..., 3:7], dim=-1)
        
        scale_base = F.softplus(geo_raw[..., 7:10]) 
        scale = scale_base * 0.05 + 0.005  
        is_sky = torch.zeros((B, M_total, self.K, 1), device=slots.device)
        is_sky[:, anchors_obj.shape[1]:] = 1.0
        scale = scale * (1.0 + is_sky * 50.0)

        opacity = torch.sigmoid(geo_raw[..., 10:11])
        final_color = torch.sigmoid(color_raw)
        final_opacity = opacity * gate_score.unsqueeze(2)

        if anchor_confidence is not None:
            sky_conf = torch.ones((B, self.num_sky_anchors, 1), device=slots.device) * 10.0
            full_conf = torch.cat([anchor_confidence, sky_conf], dim=1)
            conf_prob = torch.sigmoid(full_conf).unsqueeze(2).expand(-1, -1, self.K, 1)
            final_opacity = final_opacity * conf_prob
            
        split_prob_expanded = torch.ones_like(final_opacity) 
        split_prob_expanded[:, :, 1:, :] = split_prob
        
        scale_mod = torch.ones_like(scale)
        scale_mod[:, :, 1:, :] = split_prob
        scale = scale * scale_mod

        if self.training:
            epsilon = 0.01 
            final_opacity = final_opacity * (split_prob_expanded + epsilon)
            final_opacity = torch.clamp(final_opacity, max=1.0)
            flat_xyz = final_xyz.reshape(B, -1, 3)
            flat_opacity = final_opacity.reshape(B, -1, 1)
            flat_scale = scale.reshape(B, -1, 3)
            flat_rot = rot.reshape(B, -1, 4)
            flat_color = final_color.reshape(B, -1, 3)
            flat_base = base_xyz.reshape(B, -1, 3) 
            
            num_near = torch.full((B,), anchors_obj.shape[1]*self.K, dtype=torch.long, device=slots.device)
            
            compacted_dict = {
                "xyz": flat_xyz,
                "opacity": flat_opacity,
                "scale": flat_scale,
                "rotation": flat_rot,
                "color": flat_color,
                "base_anchors": flat_base
            }
        else:
            split_mask = split_prob_expanded > 0.4 
            
            flat_xyz = final_xyz.reshape(B, -1, 3)
            flat_opacity = final_opacity.reshape(B, -1, 1)
            flat_scale = scale.reshape(B, -1, 3)
            flat_rot = rot.reshape(B, -1, 4)
            flat_color = final_color.reshape(B, -1, 3)
            flat_base = base_xyz.reshape(B, -1, 3) 
            flat_mask = split_mask.reshape(B, -1)
            num_near = torch.full((B,), anchors_obj.shape[1]*self.K, dtype=torch.long, device=slots.device)
    
            tensors_to_compact = {
                "xyz": flat_xyz,
                "opacity": flat_opacity,
                "scale": flat_scale,
                "rotation": flat_rot,
                "color": flat_color,
                "base_anchors": flat_base
            }
            
            compacted_dict, num_near = self._compact_batch(tensors_to_compact, flat_mask, B)

        compacted_dict["num_near"] = num_near
        compacted_dict["gate_score"] = gate_score 
        return compacted_dict