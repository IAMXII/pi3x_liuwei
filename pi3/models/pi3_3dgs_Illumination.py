import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
import sys
import math
import torch.nn.functional as F
import torchvision.transforms as T # [新增] 用于光照一致性增强

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, depth_edge
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
        self.point_decoder = TransformerDecoder(in_dim=2*self.dec_embed_dim, dec_embed_dim=1024, out_dim=1024, rope=self.rope)
        self.point_head = ConvPts3dHead(patch_size=14, dec_embed_dim=1024, dim_out=[2, 1])

        self.camera_decoder = TransformerDecoder(in_dim=2*self.dec_embed_dim, dec_embed_dim=1024, out_dim=512, rope=self.rope, use_checkpoint=False)
        self.camera_head = CameraHead(dim=512)

        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = ConvPts3dHead(patch_size=14, dec_embed_dim=1024, dim_out=[1])

        self.gs_decoder = TransformerDecoder(in_dim=2*self.dec_embed_dim, dec_embed_dim=1024, out_dim=1024, rope=self.rope)
        self.gs_head = ConvDenseGaussianHead(patch_size=14, dec_embed_dim=1024, dim_out=[4, 3, 1, 3])

        # ====== [新增: WildGaussians Appearance Modeling] ======
        self.light_proj = nn.Linear(self.dec_embed_dim, 256)
        # 输入: gs_h (1024) + light_code (256) = 1280. 输出: \Delta\gamma (3) + \beta (3) = 6
        self.appearance_head = ConvPts3dHead(patch_size=14, dec_embed_dim=1024 + 256, dim_out=[3, 3])
        
        # 零初始化：保证预训练权重无损过渡
        for param in self.appearance_head.parameters():
            nn.init.zeros_(param)
            
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
            
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

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

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        if ckpt is not None:
            checkpoint_data = torch.load(ckpt, weights_only=False, map_location='cpu')
            res = self.load_state_dict(checkpoint_data, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

        if freeze_encoder:
            freeze_all_params([self.encoder])
            print('Freezing the encoder.')

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
            pass

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
        
        # [新增] 留存一份原始图像用于 Jitter 增强
        imgs_raw = imgs.clone()

        imgs = (imgs - self.image_mean) / self.image_std
        imgs_flat = imgs.reshape(B * N_total, C, H, W)
        
        hidden = self.encoder(imgs_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        mem.step("Encoder")

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
        local_xyz_raw = self.point_head([point_h], (H, W)).reshape(B, N_sub, H, W, 3)

        gs_h = self.gs_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        gs_attrs = self.gs_head([gs_h], (H, W)).reshape(B, N_sub, H, W, 11)

        if self.train_stage == 3:
            conf_h = self.conf_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
            conf_logits = self.conf_head([conf_h], (H, W)).reshape(B, N_sub, H, W, 1)
        else:
            conf_logits = torch.zeros((B, N_sub, H, W, 1), device=hidden.device)
        mem.step("Geometry & Attributes")

        camera_poses_sub = camera_poses[:, sub_idx]

        xy, z = local_xyz_raw[..., :2], torch.exp(local_xyz_raw[..., 2:3])
        local_pts = torch.cat([xy * z, z], dim=-1)
        local_pts_h = homogenize_points(local_pts).view(B, N_sub, -1, 4).transpose(2, 3)
        global_pts = torch.matmul(camera_poses_sub, local_pts_h).transpose(2, 3).reshape(B, N_sub, H, W, 4)[..., :3]

        local_rot = F.normalize(gs_attrs[..., 0:4], dim=-1)
        scale = torch.exp(torch.clamp(gs_attrs[..., 4:7], min=-10.0, max=5.0)) * 0.01
        opacity = torch.sigmoid(gs_attrs[..., 7:8])
        
        # ====== [修改: 颜色解析替换为 Appearance 仿射变换] ======
        base_color_logits = gs_attrs[..., 8:11] 
        
        reg_tokens = hidden_sub[:, :self.patch_start_idx, :]
        light_code = self.light_proj(reg_tokens.mean(dim=1)) 
        light_code_expanded = light_code.unsqueeze(1).expand(-1, hw, -1)
        
        app_in = torch.cat([gs_h, light_code_expanded], dim=-1) 
        app_out = self.appearance_head([app_in], (H, W)).reshape(B, N_sub, H, W, 6)
        
        gamma = torch.exp(app_out[..., 0:3])
        beta = app_out[..., 3:6]
        
        toned_color_logits = gamma * base_color_logits + beta
        color = torch.sigmoid(toned_color_logits) 
        # ========================================================

        cam_quats_sub = matrix_to_quaternion(camera_poses_sub[..., :3, :3]).view(B, N_sub, 1, 1, 4).expand(-1, -1, H, W, -1)
        global_rot = F.normalize(quat_mult(cam_quats_sub, local_rot), dim=-1)

        d_xyz = global_pts.reshape(B, -1, 3)
        d_rot = global_rot.reshape(B, -1, 4)
        d_scale = scale.reshape(B, -1, 3)
        d_opacity = opacity.reshape(B, -1, 1)
        d_color = color.reshape(B, -1, 3)
        d_conf = conf_logits.reshape(B, -1, 1)

        # ====== [新增: Jitter 一致性增强支路] ======
        d_color_jitter = None
        imgs_jitter_gt = None
        
        if self.training and self.train_stage in [1, 2]:
            with torch.no_grad():
                imgs_jitter = self.color_jitter(imgs_raw.view(B * N_total, C, H, W))
                imgs_jitter_gt = imgs_jitter.view(B, N_total, C, H, W)[:, sub_idx]
                imgs_jitter_norm = (imgs_jitter - self.image_mean) / self.image_std
                
            # 仅过 Encoder 抽特征
            hidden_jitter = self.encoder(imgs_jitter_norm, is_training=True)
            if isinstance(hidden_jitter, dict): hidden_jitter = hidden_jitter["x_norm_patchtokens"]
            hidden_jitter_sub = hidden_jitter.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)
            reg_tokens_jitter = hidden_jitter_sub[:, :self.patch_start_idx, :]
            
            # 提 Jitter 光照
            light_code_jitter = self.light_proj(reg_tokens_jitter.mean(dim=1))
            light_jitter_expanded = light_code_jitter.unsqueeze(1).expand(-1, hw, -1)
            
            # 重新调色 (复用原图 gs_h 和 base_color)
            app_in_jitter = torch.cat([gs_h, light_jitter_expanded], dim=-1)
            app_out_jitter = self.appearance_head([app_in_jitter], (H, W)).reshape(B, N_sub, H, W, 6)
            
            gamma_jitter = torch.exp(app_out_jitter[..., 0:3])
            beta_jitter = app_out_jitter[..., 3:6]
            
            toned_color_logits_jitter = gamma_jitter * base_color_logits + beta_jitter
            color_jitter = torch.sigmoid(toned_color_logits_jitter)
            d_color_jitter = color_jitter.reshape(B, -1, 3)
        # ========================================================

        num_dense = d_xyz.shape[1]
        limit_gaussians = int(self.anchors_per_view * math.sqrt(N_sub))  
        if self.max_dense_gaussians is not None:
            limit_gaussians = min(limit_gaussians, self.max_dense_gaussians)

        K_target = min(num_dense, limit_gaussians)

        if K_target < num_dense or self.train_stage == 3:
            if self.train_stage == 3:
                conf_prob = torch.sigmoid(d_conf.squeeze(-1))
                conf_threshold = 0.5 
                max_valid_in_batch = (conf_prob > conf_threshold).sum(dim=1).max().item()
                K_target = max(min(max_valid_in_batch, limit_gaussians), 1) 
                _, topk_indices = torch.topk(conf_prob, k=K_target, dim=1)
            else:
                _, topk_indices = torch.topk(d_opacity.squeeze(-1), k=K_target, dim=1)

            def filter_topk(tensor):
                C = tensor.shape[-1]
                expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
                return torch.gather(tensor, 1, expanded_indices)

            d_xyz = filter_topk(d_xyz)
            d_rot = filter_topk(d_rot)
            d_scale = filter_topk(d_scale)
            d_opacity = filter_topk(d_opacity)
            d_conf_filtered = filter_topk(d_conf)
            
            # [修改] 同步过滤原图颜色和 Jitter 颜色
            d_color = filter_topk(d_color)
            if d_color_jitter is not None:
                d_color_jitter = filter_topk(d_color_jitter)
            
            if self.train_stage == 3:
                survived_prob = torch.sigmoid(d_conf_filtered.squeeze(-1))
                invalid_mask = (survived_prob <= conf_threshold).unsqueeze(-1)
                d_opacity = torch.where(invalid_mask, torch.zeros_like(d_opacity), d_opacity)

            d_conf = d_conf_filtered
            mem.step("Dynamic Probability & Top-K Filtering")

        gaussians = {
            "xyz": d_xyz, "rotation": d_rot, "scale": d_scale,
            "opacity": d_opacity, "color": d_color, "conf": d_conf, "num_sky": 0 
        }
        
        # [新增] 组装 Jitter 高斯字典
        gaussians_jitter = None
        if d_color_jitter is not None:
            gaussians_jitter = {
                "xyz": d_xyz, "rotation": d_rot, "scale": d_scale,
                "opacity": d_opacity, "color": d_color_jitter, "conf": d_conf, "num_sky": 0 
            }
        mem.step("GS Concat")

        return dict(
            gaussians=gaussians, 
            camera_poses=camera_poses, 
            local_points=local_pts, 
            conf=conf_logits,
            gaussians_jitter=gaussians_jitter,  # [新增] 抛出供 Loss 使用
            imgs_jitter_gt=imgs_jitter_gt       # [新增] 抛出供 Loss 使用
        )