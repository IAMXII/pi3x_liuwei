import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
import sys
import math
import torch.nn.functional as F
import torchvision.transforms as T  
from safetensors.torch import load_file

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, depth_edge
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.conv_head import ConvHead
from .layers.transformer_head import AppearanceModulationHead, TransformerDecoder, ConvPts3dHead, ConvDenseGaussianHead, \
    SkyGaussianHead
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14_reg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


class MemDebug:
    def __init__(self, name="Model", active=True):
        self.name = name
        self.active = active
        self.last_mem = 0
        if self.active and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.last_mem = torch.cuda.memory_allocated()

    def step(self, tag):
        if not self.active or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        current = torch.cuda.memory_allocated()
        self.last_mem = current


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dim=-1)


def matrix_to_quaternion(matrix):
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    tr = m00 + m11 + m22

    cond1 = (tr > 0).unsqueeze(-1)
    cond2 = ((m00 > m11) & (m00 > m22)).unsqueeze(-1)
    cond3 = (m11 > m22).unsqueeze(-1)

    safe_tr = torch.sqrt(torch.clamp(tr + 1.0, min=1e-6)).unsqueeze(-1)
    safe_m00 = torch.sqrt(torch.clamp(m00 - m11 - m22 + 1.0, min=1e-6)).unsqueeze(-1)
    safe_m11 = torch.sqrt(torch.clamp(m11 - m00 - m22 + 1.0, min=1e-6)).unsqueeze(-1)
    safe_m22 = torch.sqrt(torch.clamp(m22 - m00 - m11 + 1.0, min=1e-6)).unsqueeze(-1)

    q1 = torch.stack([tr + 1.0, m21 - m12, m02 - m20, m10 - m01], dim=-1) * 0.5 / safe_tr
    q2 = torch.stack([m21 - m12, m00 - m11 - m22 + 1.0, m10 + m01, m02 + m20], dim=-1) * 0.5 / safe_m00
    q3 = torch.stack([m02 - m20, m10 + m01, m11 - m00 - m22 + 1.0, m21 + m12], dim=-1) * 0.5 / safe_m11
    q4 = torch.stack([m10 - m01, m02 + m20, m21 + m12, m22 - m00 - m11 + 1.0], dim=-1) * 0.5 / safe_m22

    q = torch.where(cond1, q1, torch.where(cond2, q2, torch.where(cond3, q3, q4)))
    q = F.normalize(q, dim=-1)

    return q

class SpatialIlluminationEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=128): # 维度降低，防过拟合
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            
            # 输出特征图大小为原图的 1/8，保留了基本的空间光照分布
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        # x shape: [B*N, 3, H, W]
        # 不再压缩到 128x128，直接提取以保留与当前视角的对应关系
        return self.net(x) # 输出形状: [B*N, out_dim, H/8, W/8]

# =========================================================================
# [重构] 空间外观调制头 (Spatial Appearance Head)
# =========================================================================
class SpatialAppearanceHead(nn.Module):
    def __init__(self, light_dim=128):
        super().__init__()
        # 只依靠插值后的光照特征图来预测 Gamma 和 Beta
        # 去掉 gs_h 的纠缠，强制解耦：颜色改变只能由 light_map 驱动
        self.modulator = nn.Sequential(
            nn.Conv2d(light_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, kernel_size=1) # 输出 3 通道 Gamma, 3 通道 Beta
        )

    def forward(self, light_map_spatial):
        # light_map_spatial: [B*N, light_dim, H, W]
        out = self.modulator(light_map_spatial)
        return out.permute(0, 2, 3, 1) # 转换回 [B*N, H, W, 6] 格式

class Pi3_3DGS(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=False,
            freeze_encoder=True,
            train_conf=False,
            train_cam=False,
            train_geo=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt="outputs/pi3_lowres_free/ckpts/best_model/model.safetensors",
            anchors_per_view=100000,
            num_sky_anchors=8196,
            K=8,
            debug_mem=False,
            train_stage=1,
            max_dense_gaussians=1000000
    ):
        super().__init__()
        self.debug_mem = debug_mem
        self.patch_size = 14
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        self.train_stage = train_stage
        self.max_dense_gaussians = max_dense_gaussians
        self.anchors_per_view = anchors_per_view

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim, dec_num_heads, mlp_ratio, dec_depth = 384, 6, 4, 24
        elif decoder_size == 'base':
            dec_embed_dim, dec_num_heads, mlp_ratio, dec_depth = 768, 12, 4, 24
        elif decoder_size == 'large':
            dec_embed_dim, dec_num_heads, mlp_ratio, dec_depth = 1024, 16, 4, 36
        else:
            raise NotImplementedError

        self.dec_embed_dim = dec_embed_dim

        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                ffn_layer=Mlp, init_values=0.01, qk_norm=True,
                attn_class=FlashAttentionRope, rope=self.rope
            ) for _ in range(dec_depth)])

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Heads & Sub-Decoders
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16, 
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = ConvHead(
            num_features=4,
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=[2, 1],
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

        ## --------------- Camera ---------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16, 
            out_dim=512,
            rope=self.rope,
        )
        self.camera_head = CameraHead(dim=512)

        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = ConvPts3dHead(patch_size=14, dec_embed_dim=1024, dim_out=[1])

        self.gs_decoder = TransformerDecoder(in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16,
                                             out_dim=1024, rope=self.rope)
        self.gs_head = ConvDenseGaussianHead(patch_size=14, dec_embed_dim=1024, dim_out=[4, 3, 1, 3])

        # ====== [修改: 实例化轻量级的光照编码器] ======
        # 注意：不要把它加入 freeze_all_params，让它跟着一起训练！
        self.light_dim = 128
        self.appearance_encoder = SpatialIlluminationEncoder(in_channels=3, out_dim=self.light_dim)
        self.appearance_head = SpatialAppearanceHead(light_dim=self.light_dim)

        # 孪生一致性增强
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        # ========================================================

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ----------------------
        #   VGGT Weight Loading
        # ----------------------
        if load_vggt:
            vggt_weight = load_file('ckpts/pi3/model_pi3x.safetensors')

            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''): vggt_weight[k] for k in
                               list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''): vggt_weight[k] for k in
                               list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx) * 2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''): vggt_weight[k] for k in
                                     list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx) * 2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))
            
            cam_dec_weight = {k.replace('camera_decoder.', ''): v for k, v in vggt_weight.items() if
                              k.startswith('camera_decoder.')}
            if cam_dec_weight:
                print("Loading camera_decoder", self.camera_decoder.load_state_dict(cam_dec_weight, strict=False))

            cam_head_weight = {k.replace('camera_head.', ''): v for k, v in vggt_weight.items() if
                               k.startswith('camera_head.')}
            if cam_head_weight:
                print("Loading camera_head", self.camera_head.load_state_dict(cam_head_weight, strict=False))

            pt_dec_weight = {k.replace('point_decoder.', ''): v for k, v in vggt_weight.items() if
                             k.startswith('point_decoder.')}
            if pt_dec_weight:
                print("Loading point_decoder", self.point_decoder.load_state_dict(pt_dec_weight, strict=False))

            pt_head_weight = {k.replace('point_head.', ''): v for k, v in vggt_weight.items() if
                              k.startswith('point_head.')}
            if pt_head_weight:
                print("Loading point_head", self.point_head.load_state_dict(pt_head_weight, strict=False))

        if ckpt is not None:
            if ckpt.endswith(".safetensors"):
                checkpoint = load_file(ckpt, device="cpu")
            else:
                checkpoint = torch.load(ckpt, map_location="cpu")

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()

        if freeze_encoder:
            freeze_all_params([self.encoder])
            print('Freezing the main DINO encoder.')

        self._set_stage_gradients()

    def _set_stage_gradients(self):
        if self.train_stage == 2:
            freeze_all_params([self.camera_decoder, self.camera_head])
            freeze_all_params([self.conf_decoder, self.conf_head])
            freeze_all_params([self.decoder])
        elif self.train_stage == 3:
            freeze_all_params([
                self.decoder, self.camera_decoder, self.camera_head
            ])
        elif self.train_stage == 1:
            freeze_all_params([self.decoder])
            freeze_all_params([self.camera_decoder, self.camera_head])
            freeze_all_params([self.conf_decoder, self.conf_head])
            freeze_all_params([self.point_decoder, self.point_head])
    def _extract_spatial_light(self, imgs_norm, B, N_total, H_target, W_target):
        """辅助函数：提取光照并双线性插值到原图分辨率"""
        imgs_flat = imgs_norm.reshape(B * N_total, 3, imgs_norm.shape[-2], imgs_norm.shape[-1])
        light_map = self.appearance_encoder(imgs_flat) # [B*N, C, H/8, W/8]
        # 插值对齐到高斯的密集网格尺寸 (H, W)
        light_map_up = F.interpolate(light_map, size=(H_target, W_target), mode='bilinear', align_corners=False)
        return light_map_up
    def decode(self, hidden, N, H, W, mem_debug=None):
        BN, hw, _ = hidden.shape
        B = BN // N
        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            if i % 2 == 0:
                pos_curr = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos_curr = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos_curr, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos_curr)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        if mem_debug: mem_debug.step("Shared Decoder")
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def _extract_light_code(self, imgs_norm, B, N_total):
        """辅助函数：通过自建的 CNN 提取光照特征"""
        imgs_flat = imgs_norm.reshape(B * N_total, 3, imgs_norm.shape[-2], imgs_norm.shape[-1])
        light_code = self.appearance_encoder(imgs_flat)
        return light_code

    def forward(self, imgs,imgs_paired=None, intrinsics=None, chunk_size=30000):
        mem = MemDebug(active=self.debug_mem)
        B, N_total, C, H, W = imgs.shape

        imgs = (imgs - self.image_mean) / self.image_std
        imgs_flat = imgs.reshape(B * N_total, C, H, W)

        # 1. 几何支路提取 (仅使用主图 imgs，坚决不碰 imgs_paired)
        hidden_dict = self.encoder(imgs_flat, is_training=True)
        hidden = hidden_dict["x_norm_patchtokens"] if isinstance(hidden_dict, dict) else hidden_dict

        # [修改] 提取主图的空间光照特征
        light_map_original = self._extract_spatial_light(imgs, B, N_total, H, W)

        mem.step("Encoder")

        # ==========================================
        # 2. 几何与属性解码 
        # ==========================================
        hidden, pos = self.decode(hidden, N_total, H, W, mem_debug=mem)

        patch_h, patch_w = H // 14, W // 14
        cam_h = self.camera_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
        camera_poses = self.camera_head(cam_h, patch_h, patch_w).reshape(B, N_total, 4, 4)
        mem.step("Camera Decoder")

        sub_idx = torch.arange(0, N_total, 1, device=imgs.device)
        N_sub = len(sub_idx)
        hw = hidden.shape[1]

        hidden_sub = hidden.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)
        pos_sub = pos.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)

        point_h = self.point_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        local_xyz_raw = self.point_head(point_h.float(), patch_h=patch_h, patch_w=patch_w)

        gs_h = self.gs_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        gs_attrs = self.gs_head([gs_h], (H, W)).reshape(B, N_sub, H, W, 11)

        conf_logits = torch.zeros((B, N_sub, H, W, 1), device=hidden.device)
        mem.step("Geometry & Attributes")

        camera_poses_sub = camera_poses[:, sub_idx]

        xy, z = local_xyz_raw[0].reshape(B, N_sub, H, W, -1), torch.exp(local_xyz_raw[1].reshape(B, N_sub, H, W, -1))
        local_pts = torch.cat([xy * z, z], dim=-1)
        local_pts_h = homogenize_points(local_pts).view(B, N_sub, -1, 4).transpose(2, 3)
        global_pts = torch.matmul(camera_poses_sub, local_pts_h).transpose(2, 3).reshape(B, N_sub, H, W, 4)[..., :3]

        local_rot = F.normalize(gs_attrs[..., 0:4], dim=-1)
        scale = torch.exp(torch.clamp(gs_attrs[..., 4:7], min=-10.0, max=5.0)) * 0.01
        opacity = torch.sigmoid(gs_attrs[..., 7:8])

        # ==========================================
        # 3. 原图外观调制
        # ==========================================
        base_color_logits = gs_attrs[..., 8:11]
        
        # 切片对齐 sub_idx
        light_map_sub = light_map_original.view(B, N_total, self.light_dim, H, W)[:, sub_idx].reshape(B * N_sub, self.light_dim, H, W)

        # 直接使用 SpatialHead
        app_out = self.appearance_head(light_map_sub).reshape(B, N_sub, H, W, 6)
        gamma = 1.0 + app_out[..., 0:3]
        beta = app_out[..., 3:6]
        
        toned_color_logits = gamma * base_color_logits + beta
        color = torch.sigmoid(toned_color_logits)

        cam_quats_sub = matrix_to_quaternion(camera_poses_sub[..., :3, :3]).view(B, N_sub, 1, 1, 4).expand(-1, -1, H, W,
                                                                                                           -1)
        global_rot = F.normalize(quat_mult(cam_quats_sub, local_rot), dim=-1)

        d_xyz = global_pts.reshape(B, -1, 3)
        d_rot = global_rot.reshape(B, -1, 4)
        d_scale = scale.reshape(B, -1, 3)
        d_opacity = opacity.reshape(B, -1, 1)
        d_color = color.reshape(B, -1, 3)
        d_conf = conf_logits.reshape(B, -1, 1)

        # ==========================================
        # 4. Jitter 支路 (可训练的光照编码器)
        # ==========================================
        d_color_paired = None
        imgs_paired_gt = None
        illum_cos_sim = None
        illum_l2_dist = None
        if self.train_stage in [1, 2] and imgs_paired is not None:
            imgs_paired_gt = imgs_paired.view(B, N_total, C, H, W)[:, sub_idx]
            imgs_paired_norm = (imgs_paired - self.image_mean) / self.image_std
            
            # 提取图 B 的空间光照特征 (参与反向传播)
            light_map_paired = self._extract_spatial_light(imgs_paired_norm, B, N_total, H, W)
            light_map_paired_sub = light_map_paired.view(B, N_total, self.light_dim, H, W)[:, sub_idx].reshape(B * N_sub, self.light_dim, H, W)
            illum_cos_sim = F.cosine_similarity(light_map_sub.mean(dim=[2, 3]), light_map_paired_sub.mean(dim=[2, 3]), dim=-1).mean()
            # print(f"🚀 [Debug] Original vs Paired Illumination (CLIP) Cosine Similarity: {illum_cos_sim:.4f}")
            illum_l2_dist = F.mse_loss(light_map_sub.mean(dim=[2, 3]), light_map_paired_sub.mean(dim=[2, 3]))
            # print(f"🚀 [Debug] Original vs Paired Illumination (CLIP) L2 Distance: {illum_l2_dist:.4f}")
            # 【核心安全机制】：必须 detach base_color_logits
            # 强迫网络意识到几何和底色没变，只是外部光照环境变了
            app_out_paired = self.appearance_head(light_map_paired_sub).reshape(B, N_sub, H, W, 6)

            gamma_paired = 1.0 + app_out_paired[..., 0:3]
            beta_paired = app_out_paired[..., 3:6]

            toned_color_logits_paired = gamma_paired * base_color_logits.detach() + beta_paired
            color_paired = torch.sigmoid(toned_color_logits_paired)
            d_color_paired = color_paired.reshape(B, -1, 3)

        # ==========================================
        # 5. Top-K 过滤与字典拼装
        # ==========================================
        num_dense = d_xyz.shape[1]
        limit_gaussians = int(self.anchors_per_view * math.sqrt(N_sub))
        if self.max_dense_gaussians is not None:
            limit_gaussians = min(limit_gaussians, self.max_dense_gaussians)

        K_target = min(num_dense, limit_gaussians)

        if K_target < num_dense or self.train_stage == 3:
            _, topk_indices = torch.topk(d_opacity.squeeze(-1), k=K_target, dim=1)

            def filter_topk(tensor):
                C = tensor.shape[-1]
                expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
                return torch.gather(tensor, 1, expanded_indices)

            d_xyz = filter_topk(d_xyz)
            d_rot = filter_topk(d_rot)
            d_scale = filter_topk(d_scale)
            d_opacity = filter_topk(d_opacity)
            d_conf = filter_topk(d_conf)
            d_color = filter_topk(d_color)
            if d_color_paired is not None:
                d_color_paired = filter_topk(d_color_paired)

            mem.step("Dynamic Probability & Top-K Filtering")

        gaussians = {
            "xyz": d_xyz, "rotation": d_rot, "scale": d_scale,
            "opacity": d_opacity, "color": d_color, "conf": d_conf, "num_sky": 0
        }

        gaussians_paired = None
        if d_color_paired is not None:
            gaussians_paired = {
                "xyz": d_xyz, "rotation": d_rot, "scale": d_scale,
                "opacity": d_opacity, "color": d_color_paired, "conf": d_conf, "num_sky": 0
            }

        return dict(
            gaussians=gaussians,
            camera_poses=camera_poses,
            local_points=local_pts,
            conf=conf_logits,
            gaussians_paired=gaussians_paired, # [重命名返回]
            imgs_paired_gt=imgs_paired_gt,      # [重命名返回]
            illum_cos_sim=illum_cos_sim,
            illum_l2_dist=illum_l2_dist
        )


# import torch
# import torch.nn as nn
# from functools import partial
# from copy import deepcopy
# from torch.utils.checkpoint import checkpoint
# from safetensors.torch import load_file
# import sys
# import math
# import torch.nn.functional as F

# # [新增] 引入 CLIPVisionModel (注意这里不需要 WithProjection，直接要 hidden_states)
# from transformers import CLIPVisionModel

# from .dinov2.layers import Mlp
# from ..utils.geometry import homogenize_points, depth_edge
# from .layers.pos_embed import RoPE2D, PositionGetter
# from .layers.block import BlockRope
# from .layers.attention import FlashAttentionRope
# from .layers.conv_head import ConvHead
# from .layers.transformer_head import AppearanceModulationHead, TransformerDecoder, ConvPts3dHead, ConvDenseGaussianHead, \
#     SkyGaussianHead
# from .layers.camera_head import CameraHead
# from .dinov2.hub.backbones import dinov2_vitl14_reg

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


# def freeze_all_params(modules):
#     for module in modules:
#         try:
#             for n, param in module.named_parameters():
#                 param.requires_grad = False
#         except AttributeError:
#             module.requires_grad = False


# class MemDebug:
#     def __init__(self, name="Model", active=True):
#         self.name = name
#         self.active = active
#         self.last_mem = 0
#         if self.active and torch.cuda.is_available():
#             torch.cuda.reset_peak_memory_stats()
#             self.last_mem = torch.cuda.memory_allocated()

#     def step(self, tag):
#         if not self.active or not torch.cuda.is_available():
#             return
#         torch.cuda.synchronize()
#         current = torch.cuda.memory_allocated()
#         self.last_mem = current


# def quat_mult(q1, q2):
#     w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
#     w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
#     return torch.stack([
#         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
#     ], dim=-1)


# def matrix_to_quaternion(matrix):
#     m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
#     m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
#     m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
#     tr = m00 + m11 + m22

#     cond1 = (tr > 0).unsqueeze(-1)
#     cond2 = ((m00 > m11) & (m00 > m22)).unsqueeze(-1)
#     cond3 = (m11 > m22).unsqueeze(-1)

#     safe_tr = torch.sqrt(torch.clamp(tr + 1.0, min=1e-6)).unsqueeze(-1)
#     safe_m00 = torch.sqrt(torch.clamp(m00 - m11 - m22 + 1.0, min=1e-6)).unsqueeze(-1)
#     safe_m11 = torch.sqrt(torch.clamp(m11 - m00 - m22 + 1.0, min=1e-6)).unsqueeze(-1)
#     safe_m22 = torch.sqrt(torch.clamp(m22 - m00 - m11 + 1.0, min=1e-6)).unsqueeze(-1)

#     q1 = torch.stack([tr + 1.0, m21 - m12, m02 - m20, m10 - m01], dim=-1) * 0.5 / safe_tr
#     q2 = torch.stack([m21 - m12, m00 - m11 - m22 + 1.0, m10 + m01, m02 + m20], dim=-1) * 0.5 / safe_m00
#     q3 = torch.stack([m02 - m20, m10 + m01, m11 - m00 - m22 + 1.0, m21 + m12], dim=-1) * 0.5 / safe_m11
#     q4 = torch.stack([m10 - m01, m02 + m20, m21 + m12, m22 - m00 - m11 + 1.0], dim=-1) * 0.5 / safe_m22

#     q = torch.where(cond1, q1, torch.where(cond2, q2, torch.where(cond3, q3, q4)))
#     q = F.normalize(q, dim=-1)

#     return q

# # =========================================================================
# # [重构] 空间外观调制头 (Spatial Appearance Head)
# # 只接 CLIP 的 768 维空间特征，大幅降维并预测局部 Gamma / Beta
# # =========================================================================
# class SpatialAppearanceHead(nn.Module):
#     def __init__(self, light_dim=768):
#         super().__init__()
#         self.modulator = nn.Sequential(
#             nn.Conv2d(light_dim, 128, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 6, kernel_size=1) # 输出 3 通道 Gamma, 3 通道 Beta
#         )

#     def forward(self, light_map_spatial):
#         # light_map_spatial: [B*N, light_dim, H, W]
#         out = self.modulator(light_map_spatial)
#         return out.permute(0, 2, 3, 1) # 转换回 [B*N, H, W, 6] 格式


# class Pi3_3DGS(nn.Module):
#     def __init__(
#             self,
#             pos_type='rope100',
#             decoder_size='large',
#             load_vggt=False,
#             freeze_encoder=True,
#             train_conf=False,
#             train_cam=False,
#             train_geo=False,
#             num_dec_blk_not_to_checkpoint=4,
#             ckpt="outputs/pi3_lowres_free/ckpts/best_model/model.safetensors",
#             anchors_per_view=100000,
#             num_sky_anchors=8196,
#             K=8,
#             debug_mem=False,
#             train_stage=1,
#             max_dense_gaussians=1000000
#     ):
#         super().__init__()
#         self.debug_mem = debug_mem
#         self.patch_size = 14
#         self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

#         self.train_stage = train_stage
#         self.max_dense_gaussians = max_dense_gaussians
#         self.anchors_per_view = anchors_per_view

#         # ----------------------
#         #        Encoder
#         # ----------------------
#         self.encoder = dinov2_vitl14_reg(pretrained=False)
#         del self.encoder.mask_token

#         # ----------------------
#         #  Positonal Encoding
#         # ----------------------
#         self.pos_type = pos_type if pos_type is not None else 'none'
#         self.rope = None
#         if self.pos_type.startswith('rope'):
#             if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D")
#             freq = float(self.pos_type[len('rope'):])
#             self.rope = RoPE2D(freq=freq)
#             self.position_getter = PositionGetter()
#         else:
#             raise NotImplementedError

#         # ----------------------
#         #        Decoder
#         # ----------------------
#         if decoder_size == 'small':
#             dec_embed_dim, dec_num_heads, mlp_ratio, dec_depth = 384, 6, 4, 24
#         elif decoder_size == 'base':
#             dec_embed_dim, dec_num_heads, mlp_ratio, dec_depth = 768, 12, 4, 24
#         elif decoder_size == 'large':
#             dec_embed_dim, dec_num_heads, mlp_ratio, dec_depth = 1024, 16, 4, 36
#         else:
#             raise NotImplementedError

#         self.dec_embed_dim = dec_embed_dim

#         self.decoder = nn.ModuleList([
#             BlockRope(
#                 dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
#                 qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
#                 ffn_layer=Mlp, init_values=0.01, qk_norm=True,
#                 attn_class=FlashAttentionRope, rope=self.rope
#             ) for _ in range(dec_depth)])

#         # ----------------------
#         #     Register_token
#         # ----------------------
#         num_register_tokens = 5
#         self.patch_start_idx = num_register_tokens
#         self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
#         nn.init.normal_(self.register_token, std=1e-6)

#         # ----------------------
#         #  Heads & Sub-Decoders
#         # ----------------------
#         self.point_decoder = TransformerDecoder(
#             in_dim=2 * self.dec_embed_dim,
#             dec_embed_dim=1024,
#             dec_num_heads=16, 
#             out_dim=1024,
#             rope=self.rope,
#         )
#         self.point_head = ConvHead(
#             num_features=4,
#             dim_in=dec_embed_dim,
#             projects=nn.Identity(),
#             dim_out=[2, 1],
#             dim_proj=1024,
#             dim_upsample=[256, 128, 64],
#             dim_times_res_block_hidden=2,
#             num_res_blocks=2,
#             res_block_norm='group_norm',
#             last_res_blocks=0,
#             last_conv_channels=32,
#             last_conv_size=1,
#             using_uv=True
#         )

#         ## --------------- Camera ---------------
#         self.camera_decoder = TransformerDecoder(
#             in_dim=2 * self.dec_embed_dim,
#             dec_embed_dim=1024,
#             dec_num_heads=16, 
#             out_dim=512,
#             rope=self.rope,
#         )
#         self.camera_head = CameraHead(dim=512)

#         self.conf_decoder = deepcopy(self.point_decoder)
#         self.conf_head = ConvPts3dHead(patch_size=14, dec_embed_dim=1024, dim_out=[1])

#         self.gs_decoder = TransformerDecoder(in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16,
#                                              out_dim=1024, rope=self.rope)
#         self.gs_head = ConvDenseGaussianHead(patch_size=14, dec_embed_dim=1024, dim_out=[4, 3, 1, 3])

#         # ====== [修改: 实例化 CLIP 光照提取器] ======
#         self.light_dim = 768 # CLIP ViT-Base 的特征维度
#         self.appearance_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
#         freeze_all_params([self.appearance_encoder]) # 必须冻结 CLIP，严防特征坍塌
        
#         self.appearance_head = SpatialAppearanceHead(light_dim=self.light_dim)
        
#         # CLIP 特有的归一化参数
#         self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
#         self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
#         # ========================================================

#         self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#         # ----------------------
#         #   VGGT Weight Loading
#         # ----------------------
#         # ... (保留了你原有的加载权重逻辑)
#         if load_vggt:
#             vggt_weight = load_file('ckpts/pi3/model_pi3x.safetensors')
#             vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''): vggt_weight[k] for k in
#                                list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
#             print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

#             vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''): vggt_weight[k] for k in
#                                list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
#             vggt_dec_weight1 = {}
#             for k in list(vggt_dec_weight.keys()):
#                 idx = k.split('.')[0]
#                 other = k[len(idx):]
#                 vggt_dec_weight1[f'{int(idx) * 2 + 1}{other}'] = vggt_dec_weight[k]
#             vggt_dec_weight = vggt_dec_weight1

#             vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''): vggt_weight[k] for k in
#                                      list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
#             for k in list(vggt_dec_weight_frame.keys()):
#                 idx = k.split('.')[0]
#                 other = k[len(idx):]
#                 vggt_dec_weight[f'{int(idx) * 2}{other}'] = vggt_dec_weight_frame[k]

#             print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))
            
#             cam_dec_weight = {k.replace('camera_decoder.', ''): v for k, v in vggt_weight.items() if k.startswith('camera_decoder.')}
#             if cam_dec_weight: print("Loading camera_decoder", self.camera_decoder.load_state_dict(cam_dec_weight, strict=False))

#             cam_head_weight = {k.replace('camera_head.', ''): v for k, v in vggt_weight.items() if k.startswith('camera_head.')}
#             if cam_head_weight: print("Loading camera_head", self.camera_head.load_state_dict(cam_head_weight, strict=False))

#             pt_dec_weight = {k.replace('point_decoder.', ''): v for k, v in vggt_weight.items() if k.startswith('point_decoder.')}
#             if pt_dec_weight: print("Loading point_decoder", self.point_decoder.load_state_dict(pt_dec_weight, strict=False))

#             pt_head_weight = {k.replace('point_head.', ''): v for k, v in vggt_weight.items() if k.startswith('point_head.')}
#             if pt_head_weight: print("Loading point_head", self.point_head.load_state_dict(pt_head_weight, strict=False))

#         if ckpt is not None:
#             if ckpt.endswith(".safetensors"):
#                 checkpoint = load_file(ckpt, device="cpu")
#             else:
#                 checkpoint = torch.load(ckpt, map_location="cpu")
#             res = self.load_state_dict(checkpoint, strict=False)
#             print(f'[Pi3] Load checkpoints from {ckpt}: {res}')
#             del checkpoint
#             torch.cuda.empty_cache()

#         if freeze_encoder:
#             freeze_all_params([self.encoder])
#             print('Freezing the main DINO encoder.')

#         self._set_stage_gradients()

#     def _set_stage_gradients(self):
#         if self.train_stage == 2:
#             freeze_all_params([self.camera_decoder, self.camera_head])
#             freeze_all_params([self.conf_decoder, self.conf_head])
#             freeze_all_params([self.decoder])
#         elif self.train_stage == 3:
#             freeze_all_params([
#                 self.decoder, self.camera_decoder, self.camera_head
#             ])
#         elif self.train_stage == 1:
#             freeze_all_params([self.decoder])
#             freeze_all_params([self.camera_decoder, self.camera_head])
#             freeze_all_params([self.conf_decoder, self.conf_head])
#             freeze_all_params([self.point_decoder, self.point_head])

#     def _extract_spatial_light(self, imgs_raw, B, N_total, H_target, W_target):
#         """辅助函数：通过 CLIP 提取 Patch Tokens，还原为空间特征图并插值"""
#         imgs_flat = imgs_raw.reshape(B * N_total, 3, imgs_raw.shape[-2], imgs_raw.shape[-1])
        
#         # 调整到 CLIP 期望的 224x224 尺寸
#         imgs_resized = F.interpolate(imgs_flat, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # 应用 CLIP 专用的归一化
#         imgs_clip = (imgs_resized - self.clip_mean) / self.clip_std

#         with torch.no_grad(): # 确保不阻断其他支路，但这里自身不产出梯度
#             outputs = self.appearance_encoder(pixel_values=imgs_clip, output_hidden_states=True)
#             # 取最后一层隐藏状态，形状: [B*N, 197, 768] (ViT-Base, 196个patch + 1个cls)
#             hidden_states = outputs.hidden_states[-1] 

#         # 剔除 CLS Token
#         patch_tokens = hidden_states[:, 1:, :] # 形状: [B*N, 196, 768]
        
#         # 恢复空间 2D 结构 (14x14)
#         light_map = patch_tokens.permute(0, 2, 1).reshape(B * N_total, self.light_dim, 14, 14)

#         # 插值放大到高斯渲染器的物理网格尺寸 (H, W)
#         light_map_up = F.interpolate(light_map, size=(H_target, W_target), mode='bilinear', align_corners=False)
#         return light_map_up

#     def decode(self, hidden, N, H, W, mem_debug=None):
#         # ... (保留你原有的 decode 逻辑，不做修改) ...
#         BN, hw, _ = hidden.shape
#         B = BN // N
#         final_output = []
#         hidden = hidden.reshape(B * N, hw, -1)
#         register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
#         hidden = torch.cat([register_token, hidden], dim=1)
#         hw = hidden.shape[1]
#         if self.pos_type.startswith('rope'):
#             pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
#         if self.patch_start_idx > 0:
#             pos = pos + 1
#             pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
#             pos = torch.cat([pos_special, pos], dim=1)
#         for i in range(len(self.decoder)):
#             blk = self.decoder[i]
#             if i % 2 == 0:
#                 pos_curr = pos.reshape(B * N, hw, -1)
#                 hidden = hidden.reshape(B * N, hw, -1)
#             else:
#                 pos_curr = pos.reshape(B, N * hw, -1)
#                 hidden = hidden.reshape(B, N * hw, -1)
#             if i >= self.num_dec_blk_not_to_checkpoint and self.training:
#                 hidden = checkpoint(blk, hidden, xpos=pos_curr, use_reentrant=False)
#             else:
#                 hidden = blk(hidden, xpos=pos_curr)
#             if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
#                 final_output.append(hidden.reshape(B * N, hw, -1))
#         if mem_debug: mem_debug.step("Shared Decoder")
#         return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

#     def forward(self, imgs, imgs_paired=None, intrinsics=None, chunk_size=30000):
#         mem = MemDebug(active=self.debug_mem)
#         B, N_total, C, H, W = imgs.shape

#         # 保留未归一化的原始图像用于 CLIP 提取
#         imgs_raw = imgs.clone() 

#         # DINO 的标准化
#         imgs = (imgs - self.image_mean) / self.image_std
#         imgs_flat = imgs.reshape(B * N_total, C, H, W)

#         # 1. 几何支路提取 (主图 DINO 特征)
#         hidden_dict = self.encoder(imgs_flat, is_training=True)
#         hidden = hidden_dict["x_norm_patchtokens"] if isinstance(hidden_dict, dict) else hidden_dict

#         # [修改] 提取主图的 CLIP 空间光照特征 (传入 imgs_raw)
#         light_map_original = self._extract_spatial_light(imgs_raw, B, N_total, H, W)

#         mem.step("Encoder")

#         # ==========================================
#         # 2. 几何与属性解码 
#         # ==========================================
#         hidden, pos = self.decode(hidden, N_total, H, W, mem_debug=mem)

#         patch_h, patch_w = H // 14, W // 14
#         cam_h = self.camera_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
#         camera_poses = self.camera_head(cam_h, patch_h, patch_w).reshape(B, N_total, 4, 4)
#         mem.step("Camera Decoder")

#         sub_idx = torch.arange(0, N_total, 1, device=imgs.device)
#         N_sub = len(sub_idx)
#         hw = hidden.shape[1]

#         hidden_sub = hidden.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)
#         pos_sub = pos.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)

#         point_h = self.point_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
#         local_xyz_raw = self.point_head(point_h.float(), patch_h=patch_h, patch_w=patch_w)

#         gs_h = self.gs_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
#         gs_attrs = self.gs_head([gs_h], (H, W)).reshape(B, N_sub, H, W, 11)

#         conf_logits = torch.zeros((B, N_sub, H, W, 1), device=hidden.device)
#         mem.step("Geometry & Attributes")

#         camera_poses_sub = camera_poses[:, sub_idx]

#         xy, z = local_xyz_raw[0].reshape(B, N_sub, H, W, -1), torch.exp(local_xyz_raw[1].reshape(B, N_sub, H, W, -1))
#         local_pts = torch.cat([xy * z, z], dim=-1)
#         local_pts_h = homogenize_points(local_pts).view(B, N_sub, -1, 4).transpose(2, 3)
#         global_pts = torch.matmul(camera_poses_sub, local_pts_h).transpose(2, 3).reshape(B, N_sub, H, W, 4)[..., :3]

#         local_rot = F.normalize(gs_attrs[..., 0:4], dim=-1)
#         scale = torch.exp(torch.clamp(gs_attrs[..., 4:7], min=-10.0, max=5.0)) * 0.01
#         opacity = torch.sigmoid(gs_attrs[..., 7:8])

#         # ==========================================
#         # 3. 原图外观调制
#         # ==========================================
#         base_color_logits = gs_attrs[..., 8:11]
        
#         # 切片对齐 sub_idx
#         light_map_sub = light_map_original.view(B, N_total, self.light_dim, H, W)[:, sub_idx].reshape(B * N_sub, self.light_dim, H, W)

#         app_out = self.appearance_head(light_map_sub).reshape(B, N_sub, H, W, 6)
        
#         # 【核心安全阀】限制 gamma 和 beta 的幅度，严防光照头接管基础纹理
#         gamma = 1.0 + 0.5 * torch.tanh(app_out[..., 0:3])
#         beta = 0.5 * torch.tanh(app_out[..., 3:6])
        
#         toned_color_logits = gamma * base_color_logits + beta
#         color = torch.sigmoid(toned_color_logits)

#         cam_quats_sub = matrix_to_quaternion(camera_poses_sub[..., :3, :3]).view(B, N_sub, 1, 1, 4).expand(-1, -1, H, W, -1)
#         global_rot = F.normalize(quat_mult(cam_quats_sub, local_rot), dim=-1)

#         d_xyz = global_pts.reshape(B, -1, 3)
#         d_rot = global_rot.reshape(B, -1, 4)
#         d_scale = scale.reshape(B, -1, 3)
#         d_opacity = opacity.reshape(B, -1, 1)
#         d_color = color.reshape(B, -1, 3)
#         d_conf = conf_logits.reshape(B, -1, 1)

#         # ==========================================
#         # 4. 配对天气支路 (Paired Illumination)
#         # ==========================================
#         d_color_paired = None
#         imgs_paired_gt = None
        
#         # [安全初始化监控指标]，防止在没有 paired 数据时报错
#         illum_cos_sim = torch.tensor(1.0, device=imgs.device)
#         illum_l2_dist = torch.tensor(0.0, device=imgs.device)
        
#         if self.train_stage in [1, 2] and imgs_paired is not None:
#             imgs_paired_gt = imgs_paired.view(B, N_total, C, H, W)[:, sub_idx]
            
#             # 提取图 B 的 CLIP 空间光照特征 (传入原始配对图像)
#             light_map_paired = self._extract_spatial_light(imgs_paired, B, N_total, H, W)
#             light_map_paired_sub = light_map_paired.view(B, N_total, self.light_dim, H, W)[:, sub_idx].reshape(B * N_sub, self.light_dim, H, W)
            
#             # 计算全局光照相似度指标 (将空间特征在 HW 维度均值化后再求 CosSim)
#             with torch.no_grad():
#                 illum_cos_sim = F.cosine_similarity(light_map_sub.mean(dim=[2, 3]), light_map_paired_sub.mean(dim=[2, 3]), dim=-1).mean()
#                 illum_l2_dist = F.mse_loss(light_map_sub.mean(dim=[2, 3]), light_map_paired_sub.mean(dim=[2, 3]))
#                 print(f"🚀 [Debug] Original vs Paired Illumination (CLIP) Cosine Similarity: {illum_cos_sim:.4f}")
#                 print(f"🚀 [Debug] Original vs Paired Illumination (CLIP) L2 Distance: {illum_l2_dist:.4f}")

#             # 强迫网络意识到几何和底色没变，只是外部光照环境变了
#             app_out_paired = self.appearance_head(light_map_paired_sub).reshape(B, N_sub, H, W, 6)

#             # 同样加上幅度安全限制
#             gamma_paired = 1.0 + 0.5 * torch.tanh(app_out_paired[..., 0:3])
#             beta_paired = 0.5 * torch.tanh(app_out_paired[..., 3:6])

#             toned_color_logits_paired = gamma_paired * base_color_logits.detach() + beta_paired
#             color_paired = torch.sigmoid(toned_color_logits_paired)
#             d_color_paired = color_paired.reshape(B, -1, 3)

#         # ==========================================
#         # 5. Top-K 过滤与字典拼装
#         # ==========================================
#         num_dense = d_xyz.shape[1]
#         limit_gaussians = int(self.anchors_per_view * math.sqrt(N_sub))
#         if self.max_dense_gaussians is not None:
#             limit_gaussians = min(limit_gaussians, self.max_dense_gaussians)

#         K_target = min(num_dense, limit_gaussians)

#         if K_target < num_dense or self.train_stage == 3:
#             _, topk_indices = torch.topk(d_opacity.squeeze(-1), k=K_target, dim=1)

#             def filter_topk(tensor):
#                 C = tensor.shape[-1]
#                 expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
#                 return torch.gather(tensor, 1, expanded_indices)

#             d_xyz = filter_topk(d_xyz)
#             d_rot = filter_topk(d_rot)
#             d_scale = filter_topk(d_scale)
#             d_opacity = filter_topk(d_opacity)
#             d_conf = filter_topk(d_conf)
#             d_color = filter_topk(d_color)
#             if d_color_paired is not None:
#                 d_color_paired = filter_topk(d_color_paired)

#             mem.step("Dynamic Probability & Top-K Filtering")

#         gaussians = {
#             "xyz": d_xyz, "rotation": d_rot, "scale": d_scale,
#             "opacity": d_opacity, "color": d_color, "conf": d_conf, "num_sky": 0
#         }

#         gaussians_paired = None
#         if d_color_paired is not None:
#             gaussians_paired = {
#                 "xyz": d_xyz, "rotation": d_rot, "scale": d_scale,
#                 "opacity": d_opacity, "color": d_color_paired, "conf": d_conf, "num_sky": 0
#             }

#         return dict(
#             gaussians=gaussians,
#             camera_poses=camera_poses,
#             local_points=local_pts,
#             conf=conf_logits,
#             gaussians_paired=gaussians_paired,
#             imgs_paired_gt=imgs_paired_gt,
#             illum_cos_sim=illum_cos_sim,
#             illum_l2_dist=illum_l2_dist
#         )