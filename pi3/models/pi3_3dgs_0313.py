import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
import sys
import math
import torch.nn.functional as F
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, depth_edge
from .layers.conv_head import ConvHead
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
# 导入更新后的包装头
from .layers.transformer_head import TransformerDecoder, ConvPts3dHead, ConvDenseGaussianHead, SkyGaussianHead
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
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
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

class Pi3_3DGS(nn.Module):
    def __init__(
            self, 
            pos_type='rope100', 
            decoder_size='large', 
            load_vggt=True, 
            freeze_encoder=True,
            train_conf=False, 
            train_cam=False, 
            train_geo=False, 
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None, 
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
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope,
        )
        self.point_head = ConvHead(
            num_features=4, dim_in=dec_embed_dim, projects=nn.Identity(), dim_out=[2, 1],
            dim_proj=1024, dim_upsample=[256, 128, 64], dim_times_res_block_hidden=2,
            num_res_blocks=2, res_block_norm='group_norm', last_res_blocks=0, last_conv_channels=32, last_conv_size=1, using_uv=True
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=512, rope=self.rope,
        )
        self.camera_head = CameraHead(dim=512)

        self.conf_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope,
        )
        self.conf_head = ConvHead(
            num_features=4, dim_in=dec_embed_dim, projects=nn.Identity(), dim_out=[1],
            dim_proj=1024, dim_upsample=[256, 128, 64], dim_times_res_block_hidden=2,
            num_res_blocks=2, res_block_norm='group_norm', last_res_blocks=0, last_conv_channels=32, last_conv_size=1, using_uv=True
        )

        self.gs_decoder = TransformerDecoder(in_dim=2*self.dec_embed_dim, dec_embed_dim=1024,dec_num_heads=16, out_dim=1024, rope=self.rope)
        self.gs_head = ConvDenseGaussianHead(patch_size=14, dec_embed_dim=1024, dim_out=[4, 3, 1, 3])

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ----------------------
        #   Weight Loading
        # ----------------------
        if load_vggt:
            vggt_weight = load_file('ckpts/pi3/model_pi3x.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            self.encoder.load_state_dict(vggt_enc_weight, strict=False)

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            self.decoder.load_state_dict(vggt_dec_weight, strict=False)

        if ckpt is not None:
            checkpoint_data = torch.load(ckpt, weights_only=False, map_location='cpu')
            res = self.load_state_dict(checkpoint_data, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

        if freeze_encoder:
            freeze_all_params([self.encoder])
            print('Freezing the encoder.')

        self._set_stage_gradients()

    def _set_stage_gradients(self):
        # ==========================================================
        # [修改点 1] 强制冻结原始模型的 Camera(Pose) 和 Conf 相关网络参数
        # ==========================================================
        freeze_all_params([self.point_decoder, self.point_head])
        freeze_all_params([self.camera_decoder, self.camera_head])
        freeze_all_params([self.conf_decoder, self.conf_head])
        print('Freezing the Camera (Pose) and Conf modules as requested.')

        # 您的原始 Stage 控制逻辑保留
        if self.train_stage == 2:  # GS only
            freeze_all_params([self.decoder])
        elif self.train_stage == 3: # Conf only
            freeze_all_params([self.decoder])
        elif self.train_stage == 1:
            freeze_all_params([self.decoder])

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

    def forward(self, imgs, intrinsics=None, chunk_size=30000):
        mem = MemDebug(active=self.debug_mem)
        B, N_total, C, H, W = imgs.shape
        
        imgs = (imgs - self.image_mean) / self.image_std
        imgs_flat = imgs.reshape(B * N_total, C, H, W)
        
        hidden = self.encoder(imgs_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        mem.step("Encoder")

        hidden, pos = self.decode(hidden, N_total, H, W, mem_debug=mem)

        # -----------------------------
        # Branches: Camera Pose 
        # -----------------------------
        patch_h, patch_w = H // 14, W // 14
        cam_h = self.camera_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
        camera_poses = self.camera_head(cam_h, patch_h, patch_w).reshape(B, N_total, 4, 4)
        mem.step("Camera Decoder")

        # -----------------------------
        # Geometry & Attributes
        # -----------------------------
        sub_idx = torch.arange(0, N_total, 1, device=imgs.device)
        N_sub = len(sub_idx)
        hw = hidden.shape[1]

        hidden_sub = hidden.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)
        pos_sub = pos.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)

        # [修改点 1] 修复 Point Head 调用，使用显式的 patch_h, patch_w 和 .float()
        point_h = self.point_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        xy, z = self.point_head(point_h.float(), patch_h=patch_h, patch_w=patch_w)
        xy = xy.permute(0, 2, 3, 1).reshape(B, N_sub, H, W, 2)
        z = z.permute(0, 2, 3, 1).reshape(B, N_sub, H, W, 1)
        local_xyz_raw = torch.cat([xy, z], dim=-1) # (B, N_sub, H, W, 3)

        # (保留不变) gs_head 因为是 ConvDenseGaussianHead，可能确实接受 list 和 tuple 
        gs_h = self.gs_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        gs_attrs = self.gs_head([gs_h], (H, W)).reshape(B, N_sub, H, W, 11)

        # [修改点 2] 修复 Conf Head 调用，与 Point Head 同理
        conf_h = self.conf_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        conf_logits = self.conf_head(conf_h.float(), patch_h=patch_h, patch_w=patch_w)[0]
        conf_logits = conf_logits.permute(0, 2, 3, 1).reshape(B, N_sub, H, W, 1)
        
        mem.step("Geometry & Attributes")
        # -----------------------------
        # Transformation
        # -----------------------------
        camera_poses_sub = camera_poses[:, sub_idx]

        xy, z = local_xyz_raw[..., :2], torch.exp(local_xyz_raw[..., 2:3])
        local_pts = torch.cat([xy * z, z], dim=-1)
        local_pts_h = homogenize_points(local_pts).view(B, N_sub, -1, 4).transpose(2, 3)
        global_pts = torch.matmul(camera_poses_sub, local_pts_h).transpose(2, 3).reshape(B, N_sub, H, W, 4)[..., :3]

        local_rot = F.normalize(gs_attrs[..., 0:4], dim=-1)
        scale = torch.exp(torch.clamp(gs_attrs[..., 4:7], min=-10.0, max=5.0)) * 0.01
        opacity = torch.sigmoid(gs_attrs[..., 7:8])
        color = torch.sigmoid(gs_attrs[..., 8:11])

        cam_quats_sub = matrix_to_quaternion(camera_poses_sub[..., :3, :3]).view(B, N_sub, 1, 1, 4).expand(-1, -1, H, W, -1)
        global_rot = F.normalize(quat_mult(cam_quats_sub, local_rot), dim=-1)

        d_xyz = global_pts.reshape(B, -1, 3)
        d_rot = global_rot.reshape(B, -1, 4)
        d_scale = scale.reshape(B, -1, 3)
        d_opacity = opacity.reshape(B, -1, 1)
        d_color = color.reshape(B, -1, 3)
        d_conf = conf_logits.reshape(B, -1, 1)

        # ==========================================================
        # [修改点 2] Conf 使用逻辑重构：先掩模计算数量，再截断至满足最大限制
        # ==========================================================
        limit_gaussians = int(self.anchors_per_view * math.sqrt(N_sub))
        if self.max_dense_gaussians is not None:
            limit_gaussians = min(limit_gaussians, self.max_dense_gaussians)

        conf_prob = torch.sigmoid(d_conf.squeeze(-1))
        conf_threshold = 0.1 
        
        # 步骤 1：先给高斯打掩模，统计满足阈值的有效数量
        valid_mask = conf_prob > conf_threshold
        valid_counts = valid_mask.sum(dim=1)  # 得到每个 Batch 的满足条件的高斯数

        # 步骤 2：判断掩模后的高斯是否满足最大高斯数限制
        # 如果有效数量超标，则截断为 limit_gaussians；未超标则保留该批次最大有效数
        K_target = min(valid_counts.max().item(), limit_gaussians)
        K_target = max(K_target, 1)  # 至少保留 1 个，防止维度崩溃报错

        # 使用 topk 提取前 K_target 个高斯，确保 batch 张量维度一致对齐
        _, topk_indices = torch.topk(conf_prob, k=K_target, dim=1)

        def filter_topk(tensor):
            C = tensor.shape[-1]
            expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(tensor, 1, expanded_indices)

        d_xyz = filter_topk(d_xyz)
        d_rot = filter_topk(d_rot)
        d_scale = filter_topk(d_scale)
        d_opacity = filter_topk(d_opacity)
        d_color = filter_topk(d_color)
        d_conf_filtered = filter_topk(d_conf)
        
        # 步骤 3：真正应用掩模（因为如果一个 Batch 中某个样本有效数量不足 K_target，
        # Top-K 可能会把低于置信度的也选进来，所以必须强行将透明度置 0）
        survived_prob = torch.sigmoid(d_conf_filtered.squeeze(-1))
        invalid_mask = (survived_prob <= conf_threshold).unsqueeze(-1)
        d_opacity = torch.where(invalid_mask, torch.zeros_like(d_opacity), d_opacity)

        d_conf = d_conf_filtered
        mem.step("Masking & Limit Filtering")
        
        gaussians = {
            "xyz": d_xyz,
            "rotation": d_rot,
            "scale": d_scale,
            "opacity": d_opacity,
            "color": d_color,
            "conf": d_conf,
            "num_sky": 0 
        }
        mem.step("GS Concat")

        return dict(
            gaussians=gaussians, 
            camera_poses=camera_poses, 
            local_points=local_pts, 
            conf=conf_logits
        )