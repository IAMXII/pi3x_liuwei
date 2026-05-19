import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
import math
import sys
import math
import torch.nn.functional as F
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, depth_edge
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.conv_head import ConvHead
# 导入更新后的包装头
from .layers.transformer_head import TransformerDecoder, ConvPts3dHead, ConvDenseGaussianHead, SkyGaussianHead
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14_reg
import numpy as np
# import utils3d
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch

def intrinsics_from_focal_center(fx, fy, W, H):
    """
    根据焦距 (fx, fy) 和图像分辨率 (W, H) 构造像素级内参矩阵 K。
    默认主点 (cx, cy) 位于图像正中心 (W/2, H/2)。
    兼容传入标量，并自动对齐 fx 的 Batch 维度。
    """
    if not isinstance(fx, torch.Tensor): 
        fx = torch.tensor(fx)
    if not isinstance(fy, torch.Tensor): 
        fy = torch.tensor(fy, device=fx.device, dtype=fx.dtype)
    
    # 将中心点计算为像素坐标 (通常图像中心是宽高的二分之一)
    cx_val = W / 2.0
    cy_val = H / 2.0
    
    cx_t = torch.full_like(fx, cx_val)
    cy_t = torch.full_like(fx, cy_val)
    
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    
    # 构造像素级 3x3 内参矩阵 K
    K = torch.stack([
        fx*H, zeros, cx_t,
        zeros, fy*W, cy_t,
        zeros, zeros, ones
    ], dim=-1).view(*fx.shape, 3, 3)
    
    return K
def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False

# # 手动实现一个（很简单）
# def intrinsics_from_focal_center(fx, fy, cx, cy):
#     B = fx.shape[0]
#     K = torch.zeros(B, 3, 3, device=fx.device)
#     K[:, 0, 0] = fx
#     K[:, 1, 1] = fy
#     K[:, 0, 2] = cx
#     K[:, 1, 2] = cy
#     K[:, 2, 2] = 1.0
#     return K


def point_cloud_to_depth_map(points_3d, K, H=182, W=336):
    """
    将 3D 点云投影为 2D 深度图。
    
    参数:
        points_3d: torch.Tensor, 形状为 (N, 3)，表示相机坐标系下的点云 (X, Y, Z)
        K: torch.Tensor, 形状为 (3, 3)，相机的内参矩阵
        H: int, 目标深度图的高度
        W: int, 目标深度图的宽度
        
    返回:
        depth_map: torch.Tensor, 形状为 (H, W)，没有点投射到的地方值为 0
    """
    device = points_3d.device
    
    # 1. 剔除相机背后的点 (Z <= 0)
    valid_mask = points_3d[:, 2] > 0
    points = points_3d[valid_mask]
    
    if points.shape[0] == 0:
        return torch.zeros((H, W), device=device)
        
    # 2. 按照 Z 值降序排序 (从远到近)
    # 核心技巧：排序后，在后续的张量赋值中，近处的点会自然覆盖远处的点，实现 Z-Buffer
    sort_idx = torch.argsort(points[:, 2], descending=True)
    points = points[sort_idx]
    
    # 3. 提取 X, Y, Z
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    
    # 4. 根据内参矩阵 K 执行针孔相机投影
    # K 的结构为:
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 计算连续的像素坐标 u, v
    u = (X / Z) * fx + cx
    v = (Y / Z) * fy + cy
    
    # 转换为离散的整数像素坐标 (四舍五入到最近的像素)
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()
    
    # 5. 剔除超出图像 (182x336) 边界的点
    in_bounds_mask = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    
    u_valid = u_int[in_bounds_mask]
    v_valid = v_int[in_bounds_mask]
    Z_valid = Z[in_bounds_mask]
    
    # 6. 初始化并填充深度图
    depth_map = torch.zeros((H, W), device=device, dtype=torch.float32)
    
    # 利用张量的高级索引直接赋值
    depth_map[v_valid, u_valid] = Z_valid
    
    return depth_map
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

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

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
            ckpt="ckpts/pi3/model_pi3x.safetensors", 
            anchors_per_view=100000, 
            num_sky_anchors=8196, 
            K_input = 1000000,                    
            debug_mem=False,
            train_stage=1,
            max_dense_gaussians=1000000
    ):
        super().__init__()
        self.debug_mem = debug_mem
        self.patch_size = 14
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint
        self.K_input = K_input
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
        # 使用替换后的 ConvPts3dHead
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
        )
        # self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.point_head = ConvHead(
                num_features=4, 
                dim_in=dec_embed_dim,
                # projects=nn.Linear(1024, 1024),
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
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
        )
        self.camera_head = CameraHead(dim=512)


        # 同样使用 ConvPts3dHead 预测单个维度的置信度
        self.conf_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
        )
        self.conf_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            # projects=nn.Linear(1024, 1024),
            projects=nn.Identity(),
            dim_out=[1], 
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


        # 使用 ConvDenseGaussianHead 预测高斯的所有属性
        self.gs_decoder = TransformerDecoder(in_dim=2*self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16,out_dim=1024, rope=self.rope)
        self.gs_head = ConvDenseGaussianHead(patch_size=14, dec_embed_dim=1024, dim_out=[4, 3, 1, 3])
        
        # self.sky_head = SkyGaussianHead(
        #     num_sky_anchors=num_sky_anchors, 
        #     in_dim=2 * self.dec_embed_dim
        # )

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ----------------------
        #   VGGT Weight Loading
        # ----------------------
        if load_vggt:
            vggt_weight = load_file('outputs/pi3_lowres_free/ckpts/best_model/model.safetensors')
            
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
            if ckpt.endswith(".safetensors"):
                checkpoint = load_file(ckpt, device="cpu")
            else:
                checkpoint = torch.load(ckpt, map_location="cpu")

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')
            print(f'[Pi3] Load checkpoints from {ckpt}')

            del checkpoint
            torch.cuda.empty_cache()

        if freeze_encoder:
            freeze_all_params([self.encoder])
            print('Freezing the encoder.')

        self._set_stage_gradients()

    def _set_stage_gradients(self):
        if self.train_stage == 2:  # GS only
            freeze_all_params([self.camera_decoder, self.camera_head])
            freeze_all_params([self.conf_decoder, self.conf_head])
            freeze_all_params([self.decoder])
        elif self.train_stage == 3: # Conf only
            freeze_all_params([
                self.decoder, self.camera_decoder, self.camera_head
            ])
        elif self.train_stage == 1:
            freeze_all_params([self.decoder])
            freeze_all_params([self.camera_decoder, self.camera_head])
            freeze_all_params([self.conf_decoder, self.conf_head])
            freeze_all_params([self.point_decoder,self.point_head])
            # pass

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

    def forward(self, imgs, intrinsics=None, chunk_size=30000,global_step=None):
        mem = MemDebug(active=self.debug_mem)
        B, N_total, C, H, W = imgs.shape
        
        # ==========================================================
        # [修改点 1] 所有图像输入 Encoder 并预测位姿
        # ==========================================================
        imgs = (imgs - self.image_mean) / self.image_std
        imgs_flat = imgs.reshape(B * N_total, C, H, W)
        
        hidden = self.encoder(imgs_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        mem.step("Encoder")

        hidden, pos = self.decode(hidden, N_total, H, W, mem_debug=mem)

        # -----------------------------
        # Branches: Camera Pose 对全量数据生效
        # -----------------------------
        patch_h, patch_w = H // 14, W // 14
        cam_h = self.camera_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
        camera_poses = self.camera_head(cam_h, patch_h, patch_w).reshape(B, N_total, 4, 4)
        mem.step("Camera Decoder")

        # ==========================================================
        # [修改点 2] 提取 1/3 等间隔特征用于生成 Gaussian 与 Conf
        # ==========================================================
        sub_idx = torch.arange(0, N_total, 1, device=imgs.device)
        N_sub = len(sub_idx)
        hw = hidden.shape[1]

        # 重塑并提取子集特征
        hidden_sub = hidden.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)
        pos_sub = pos.view(B, N_total, hw, -1)[:, sub_idx].reshape(B * N_sub, hw, -1)

        point_h = self.point_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
        # local_xyz_raw = self.point_head([point_h], (H, W)).reshape(B, N_sub, H, W, 3)
        local_xyz_raw = self.point_head(point_h, patch_h=patch_h, patch_w=patch_w)
        # local_xyz_raw = local_xyz_raw[:, ::2]

        gs_h = self.gs_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        gs_attrs = self.gs_head([gs_h], (H, W)).reshape(B, N_sub, H, W, 11)

        # if self.train_stage == 3:
        #     conf_h = self.conf_decoder(hidden_sub, xpos=pos_sub)[:, self.patch_start_idx:]
        #     conf_logits = self.conf_head([conf_h], (H, W)).reshape(B, N_sub, H, W, 1)
        # else:
        #     # 随便给个不占显存的 dummy tensor，防止后续取值报错
        #     conf_logits = torch.zeros((B, N_sub, H, W, 1), device=hidden.device)
        if self.train_stage in [1, 2]:
            with torch.no_grad():
                ret_conf = self.conf_decoder(hidden, xpos=pos)
                conf = self.conf_head(ret_conf[:, self.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)[0]
        else:
            ret_conf = self.conf_decoder(hidden, xpos=pos)
            conf = self.conf_head(ret_conf[:, self.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)[0]
            
        conf_logits = conf.permute(0, 2, 3, 1).reshape(B, N_total, H, W, -1)
        
        # 极其重要：及时释放厚重的隐层特征，防止驻留显存
        del ret_conf, conf 
        
        mem.step("Geometry & Attributes")

        # ==========================================================
        # [修改点 3] 投影和旋转转换必须使用子集的位姿
        # ==========================================================
        camera_poses_sub = camera_poses[:, sub_idx]

        # 将 Local Point 提升至 Global
        # xy = local_xyz_raw[0].permute(0, 2, 3, 1).reshape(B, N_total, H, W, -1)
        # z = torch.exp(local_xyz_raw[1].permute(0, 2, 3, 1).reshape(B, N_total, H, W, -1))
        
        # # ==========================================================
        # # 【修改 2/2】：根据 conf_mask，将低于 0.1 的点深度向远推 100 倍
        # # 此操作正好发生在组装 local_pts 和赋给高斯属性之前
        # # ==========================================================
        # with torch.no_grad(): 
        #     # 仅仅是推远深度的判别条件，不需要反向传播到 conf_logits
        #     mask_push = torch.sigmoid(conf_logits) < 0.1
        # # print(z.mean(), z.max(), z.min())
        # z = torch.where(mask_push, 1000.0 * z, z)
        # del mask_push  # 释放 mask 占用的显存
        
        # local_pts = torch.cat([xy * z, z], dim=-1)
        # 将 Local Point 提升至 Global (相机坐标系下)
        xy = local_xyz_raw[0].permute(0, 2, 3, 1).reshape(B, N_total, H, W, -1)
        z = torch.exp(local_xyz_raw[1].permute(0, 2, 3, 1).reshape(B, N_total, H, W, -1))
        
        # 1. 先按常规计算所有 local_pts
        local_pts = torch.cat([xy * z, z], dim=-1)
        dx = (xy[..., -1, 0] - xy[..., 0, 0]).mean(dim=-1) / (W - 1)
        
        # dy: 每列最下侧点减去最上侧点，然后对所有列求平均，再除以总跨度 (H - 1)
        dy = (xy[..., -1, :, 1] - xy[..., 0, :, 1]).mean(dim=-1) / (H - 1)
        
        fx = 1.0 / dx
        fy = 1.0 / dy

        # 2. 计算全局平均光心 (质心法)
        # 理论公式: u = x * fx + cx  =>  cx = u - x * fx
        # 我们直接使用整个图像网格的理论中心坐标和预测坐标的全局均值，这样最稳定
        u_mean = (W - 1) / 2.0
        v_mean = (H - 1) / 2.0
        
        # 沿着 H 和 W 维度求平均，得到每个视角全局的 x 和 y 均值
        x_mean = xy[..., 0].mean(dim=(-2, -1))  # Shape: (B, N_sub)
        y_mean = xy[..., 1].mean(dim=(-2, -1))  # Shape: (B, N_sub)
        
        cx = u_mean - x_mean * fx
        cy = v_mean - y_mean * fy

        # 3. 组装内参矩阵 K，目标 shape 为 (B, N_sub, 3, 3)
        K = torch.zeros((B, N_total, 3, 3), device=xy.device, dtype=xy.dtype)

        # 将计算好的参数填入对应的矩阵位置
        K[:, :, 0, 0] = fx
        K[:, :, 1, 1] = fy
        K[:, :, 0, 2] = cx
        K[:, :, 1, 2] = cy
        K[:, :, 2, 2] = 1.0
        # ==========================================================
        # 动态计算 scene_size (基于相机原点的最大距离)
        # ==========================================================
        with torch.no_grad():
            # 计算所有点到相机原点 (0,0,0) 的距离: sqrt(x^2 + y^2 + z^2)
            distances = torch.norm(local_pts, dim=-1) 
            
            # 方法 A (严格最大值): 直接取最远的点作为场景大小
            # scene_size = distances.max() 
            
            # 方法 B (推荐：鲁棒最大值): 取 99% 分位数，过滤掉可能飞到极远处的异常噪点
            scene_size = torch.quantile(distances.float(), 0.8)
            # print(f"Dynamic scene size: {scene_size.item():.2f}")
            # scene_size_f = scene_size / 10.0 
            # local_pts = local_pts / scene_size_f[..., None, None, None]  # 将点云缩放到更合理的范围，防止数值不稳定
            # 目标半径设定为场景大小的 10 倍
            target_radius = 20.0 * scene_size

        # ==========================================================
        # 【修改 2/2】：将 conf < 0.1 的点放置到 10 倍 scene_size 的球面上
        # ==========================================================
        with torch.no_grad(): 
            mask_push = torch.sigmoid(conf_logits) < 0.1
            mask_push_expand = mask_push.expand_as(local_pts)
            
            xy_detached = xy.detach()
            
            # 射线方向 d = (x, y, 1)，模长 |d|
            dir_norm = torch.sqrt(xy_detached[..., 0:1]**2 + xy_detached[..., 1:2]**2 + 1.0)
            
            # 要使最终点距离相机为 target_radius，新的 z = target_radius / |d|
            z_sphere = target_radius / dir_norm
            
            # 组装球面上的点坐标
            sphere_pts = torch.cat([xy_detached * z_sphere, z_sphere], dim=-1)

        # 替换被 push 的点，切断这部分的梯度
        local_pts = torch.where(mask_push_expand, sphere_pts, local_pts)
        
        # 释放内存
        del distances, mask_push, mask_push_expand, xy_detached, dir_norm, z_sphere, sphere_pts
        
        # (B, N, 3) 维度展平
        local_pts = local_pts.reshape(B, N_total, H, W, 3)
        
        # K = K.repeat_interleave(2, dim=1)
        # K = K[:, 0, :, :]
        # local_pts_h = homogenize_points(local_pts).view(B, N_total, -1, 4).transpose(2, 3)
        # global_pts = torch.matmul(camera_poses, local_pts_h).transpose(2, 3).reshape(B, N_total, H, W, 4)[..., :3]
        R_cam = camera_poses[..., :3, :3]       # [B, N_total, 3, 3]
        t_cam = camera_poses[..., :3, 3:4]      # [B, N_total, 3, 1]
        # 展平空间维度，形状变为 [B, N_total, 3, H*W]
        local_pts_flat = local_pts.view(B, N_total, -1, 3).transpose(-1, -2) 
        # R * x + t，再还原回原形状
        global_pts = (torch.matmul(R_cam, local_pts_flat) + t_cam).transpose(-1, -2).reshape(B, N_total, H, W, 3)

        # 解析 Dense 高斯属性
        local_rot = F.normalize(gs_attrs[..., 0:4], dim=-1)
        scale = torch.exp(torch.clamp(gs_attrs[..., 4:7], min=-10.0, max=5.0)) * 0.01
        with torch.no_grad():
            # 提取与 gs_attrs 对齐的 conf 子集，防止未来 sub_idx 发生变化
            conf_logits_sub = conf_logits[:, sub_idx]
            mask_push_scale = torch.sigmoid(conf_logits_sub) < 0.1
            # 将 mask_push_scale 的最后一个维度从 1 扩展到 3，以匹配 scale 的维度
            mask_push_scale = mask_push_scale.expand_as(scale)
            
        # 根据掩码放大 scale 100 倍
        scale = torch.where(mask_push_scale, scale * 50.0, scale)
        del conf_logits_sub, mask_push_scale # 释放显存
        opacity = torch.sigmoid(gs_attrs[..., 7:8])
        color = torch.sigmoid(gs_attrs[..., 8:11])

        # 本地旋转转全局旋转
        cam_quats_sub = matrix_to_quaternion(camera_poses_sub[..., :3, :3]).view(B, N_sub, 1, 1, 4).expand(-1, -1, H, W, -1)
        global_rot = F.normalize(quat_mult(cam_quats_sub, local_rot), dim=-1)

        d_xyz = global_pts[:, sub_idx].reshape(B, -1, 3)
        d_rot = global_rot.reshape(B, -1, 4)
        d_scale = scale.reshape(B, -1, 3)
        d_opacity = opacity.reshape(B, -1, 1)
        d_color = color.reshape(B, -1, 3)
        d_conf = conf_logits.reshape(B, -1, 1)
        # conf_noise = conf_logits[:, sub_idx].reshape(B, -1, 1)

        num_dense = d_xyz.shape[1]
        
        # --- 1. 计算理论下限 K_lower ---
        K_lower = int((0.64 * H * W) * math.sqrt(N_sub))
        if self.max_dense_gaussians is not None:
            K_lower = min(K_lower, self.max_dense_gaussians)
        K_lower = min(num_dense, K_lower)

        # --- 2. 动态 K 策略 ---
        # if self.training and self.train_stage in [1, 2]:
        #     # K_target = torch.randint(K_lower, num_dense + 1, (1,)).item()

        #     log_lower = math.log(K_lower)
        #     log_upper = math.log(num_dense + 1)

        #     # 在对数空间均匀采样
        #     log_k = torch.empty(1).uniform_(log_lower, log_upper)

        #     # 转换回线性空间并转为整数
        #     K_target = int(torch.exp(log_k).item())

        #     # 确保不越界
        #     K_target = max(K_lower, min(num_dense, K_target))
        # else:
        #     K_target = self.K_input
        K_target = K_lower
        # with torch.no_grad():
        #     imgs_sub = imgs[:, sub_idx] 
        #     imgs_gray = imgs_sub.mean(dim=2, keepdim=True) 
        #     dx = torch.abs(imgs_gray[..., :, 1:] - imgs_gray[..., :, :-1])
        #     dy = torch.abs(imgs_gray[..., 1:, :] - imgs_gray[..., :-1, :])
        #     dx = F.pad(dx, (0, 1, 0, 0))
        #     dy = F.pad(dy, (0, 0, 0, 1))
        #     edge_map = (dx + dy).reshape(B, -1) 
        #     edge_map = edge_map / (edge_map.max(dim=1, keepdim=True)[0] + 1e-5)
        if K_target < num_dense or self.train_stage == 3:
            if self.train_stage == 3:
                # conf_prob = torch.sigmoid(d_conf.squeeze(-1))
                # conf_threshold = 0.1 
                # max_valid_in_batch = (conf_prob > conf_threshold).sum(dim=1).max().item()
                # K_target = max(min(max_valid_in_batch, K_target), 1) 
                # _, topk_indices = torch.topk(conf_prob, k=K_target, dim=1)
                pass
            else:
                if self.training:
                    opacity_logits = gs_attrs[..., 7:8].reshape(B, -1)
                    # alpha = 3.0
                    # 基于 conf 的自适应温度与截断 Gumbel 噪声
                    # conf_prob = torch.sigmoid(conf_noise.squeeze(-1)) 
                    # uncertainty = 1.0 - conf_prob 
                    # densification_score = conf_prob + alpha * edge_map
                    
                    # tau_base = 0.2  
                    # gamma = 1.2     
                    # tau_i = tau_base + gamma * uncertainty
                    tau_max = 1.0      
                    tau_min = 0.01      
                    decay_steps = 10000.0 
                    progress = min(global_step / decay_steps, 1.0)
                    tau = tau_max * ((tau_min / tau_max) ** progress)
                    
                    noise = torch.rand_like(opacity_logits)
                    gumbel_noise = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
                    # gumbel_noise = torch.clamp(gumbel_noise, min=-2.0, max=3.0)
                    
                    scores = opacity_logits + gumbel_noise * tau 
                    
                    # ==========================================
                    # 核心修复：Epsilon-Greedy 保底策略
                    # ==========================================
                    # eps = 0.1 # 扣出 15% 的名额无视分数，强行全图随机播撒
                    # K_top = int(K_target * (1.0 - eps))
                    # K_rand = K_target - K_top
                    
                    # 1. 主力部队：选出得分最高的 K_top 个点（主攻有深度监督的前景）
                    _, topk_indices = torch.topk(scores, k=K_target, dim=1)
                    
                    # # 2. 星火部队：在剩下的点中，完全随机抽取 K_rand 个点（强行给天空留种）
                    # # 构造纯随机打分，并将已经选中的主力点分数设为极小值（防止重复选中）
                    # rand_scores = torch.rand_like(scores)
                    # rand_scores.scatter_(1, topk_indices_main, -1e9)
                    # _, topk_indices_rand = torch.topk(rand_scores, k=K_rand, dim=1)
                    
                    # # 3. 会师：合并主力与星火的索引
                    # topk_indices = torch.cat([topk_indices_main, topk_indices_rand], dim=1)
                    # ==========================================
                else:
                    _, topk_indices = torch.topk(d_opacity.squeeze(-1), k=K_target, dim=1)
        

        # ==========================================================
        # 【核心整合】：复合打分 + Gumbel退火 + 10%纯随机保底
        # ==========================================================
        # if K_target < num_dense:
        #     # 复合致密化得分: 基础置信度 + 边缘引导 (alpha=3.0)
        #     alpha = 3.0
        #     densification_score = d_conf + alpha * edge_map
            
        #     if self.training:
        #         # 1. 指数退火参数设定 (基于 100,000 步)
        #         tau_max = 2.0      
        #         tau_min = 0.01      
        #         decay_steps = 20000.0 
        #         progress = min(global_step / decay_steps, 1.0)
        #         tau = tau_max * ((tau_min / tau_max) ** progress)
                
        #         # 2. 注入 Gumbel 噪声
        #         noise = torch.rand_like(densification_score)
        #         gumbel_noise = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
        #         noisy_logits = (densification_score + gumbel_noise) / tau
        #         noisy_logits_squeeze = noisy_logits.squeeze(-1)
                
        #         # 3. 混合调度策略：90% Gumbel 主力 + 10% 纯随机保底
        #         # eps = 0.0 
        #         # K_top = int(K_target * (1.0 - eps))
        #         # K_rand = K_target - K_top
                
        #         #   3.1 主力部队：选出得分最高的前 90%
        #         _, topk_indices = torch.topk(noisy_logits_squeeze, k=K_target, dim=1)
                
        #         # #   3.2 星火侦察兵：剩余点中完全随机抽 10%
        #         # rand_scores = torch.rand_like(noisy_logits_squeeze)
        #         # # 将已选中的主力点得分设为极小值，避免重复抽取
        #         # rand_scores.scatter_(1, topk_indices_main, -1e9)
        #         # _, topk_indices_rand = torch.topk(rand_scores, k=K_rand, dim=1)
                
        #         #   3.3 会师合并
        #         # topk_indices = torch.cat([topk_indices_main, topk_indices_rand], dim=1)
                
        #     else:
        #         # 推理阶段：不加噪声，也不加纯随机，100% 凭绝对得分截断
        #         _, topk_indices = torch.topk(densification_score.squeeze(-1), k=K_target, dim=1)
            # else:
            #     if self.training:
            #         # ==========================================================
            #         # 极速优化版：阈值截断 + 随机探索 (替代 Gumbel TopK)
            #         # ==========================================================
            #         # 提取 Logits，形状为 [B, N]
            #         opacity_logits = gs_attrs[..., 7:8].squeeze(-1).reshape(B, -1)
                    
            #         # 设定存活阈值：Sigmoid(x) > 0.05 等价于 logits > -2.944
            #         # 这是一个合理的“微弱可见”界限
            #         thresh = -2.944 
                    
            #         topk_indices_list = []
            #         for b in range(B):
            #             logits_b = opacity_logits[b]
                        
            #             # 1. 划分优质点池
            #             valid_mask = logits_b > thresh
            #             valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            #             M = valid_idx.shape[0]
                        
            #             if M >= K_target:
            #                 # 优质点过多：执行极速随机下采样 (Dropout regularizer)
            #                 perm = torch.randperm(M, device=logits_b.device)[:K_target]
            #                 final_idx = valid_idx[perm]
            #             else:
            #                 # 优质点不足：从死神经元中随机抽取填补空缺 (Exploration)
            #                 dead_idx = (~valid_mask).nonzero(as_tuple=True)[0]
            #                 need_more = K_target - M
                            
            #                 if dead_idx.shape[0] > need_more:
            #                     perm = torch.randperm(dead_idx.shape[0], device=logits_b.device)[:need_more]
            #                     resurrect_idx = dead_idx[perm]
            #                 else:
            #                     resurrect_idx = dead_idx
                                
            #                 final_idx = torch.cat([valid_idx, resurrect_idx], dim=0)
                            
            #             topk_indices_list.append(final_idx)
                        
            #         topk_indices = torch.stack(topk_indices_list, dim=0)
            #     else:
            #         # 推理时保持确定性硬截断
            #         _, topk_indices = torch.topk(d_opacity.squeeze(-1), k=K_target, dim=1)
            def filter_topk(tensor):
                C = tensor.shape[-1]
                expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
                return torch.gather(tensor, 1, expanded_indices)

            d_xyz = filter_topk(d_xyz)
            d_rot = filter_topk(d_rot)
            d_scale = filter_topk(d_scale)
            d_opacity = filter_topk(d_opacity)
            d_color = filter_topk(d_color)
            d_conf = filter_topk(d_conf)

            # d_conf_filtered = filter_topk(d_conf)
            # d_scale = torch.ones_like(d_scale)  # 这里改成全1，交给后续的置信度来控制显隐
            # d_opacity = torch.ones_like(d_opacity)  # 这里改成全1，交给后续的置信度来控制显隐
            # if self.train_stage == 3:
            #     survived_prob = torch.sigmoid(d_conf_filtered.squeeze(-1))
            #     invalid_mask = (survived_prob <= conf_threshold).unsqueeze(-1)
            #     d_opacity = torch.where(invalid_mask, torch.zeros_like(d_opacity), d_opacity)

            # d_conf = d_conf_filtered
            mem.step("Dynamic Probability & Top-K Filtering")

        # 提取全量场景特征并传入 Sky Head (使用 N_total)
        # global_scene_feat = hidden.view(B, N_total, hw, -1).mean(dim=(1, 2))
        # s_xyz, s_rot, s_scale, s_opacity, s_color, s_conf_sky = self.sky_head(global_scene_feat)
        
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

        # points = local_pts
        # masks = torch.sigmoid(conf_logits[..., 0]) > 0.1
        # original_height, original_width = points.shape[-3:-1]
        # aspect_ratio = original_width / original_height
        # # use recover_focal_shift function from MoGe
        # focal, shift = recover_focal_shift(points, masks)
        # fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5

        # intrinsics_ = intrinsics_from_focal_center(fx, fy, H, W)
        # print(intrinsics_[0])

        # import numpy as np
        # from PIL import Image

        # with torch.no_grad(): # 避免梯度追踪增加显存开销
        #     # 1. 提取第一个 batch, 第一个 view 的点云和内参
        #     # local_pts 形状: (B, N_sub, H, W, 3), K 形状: (B, N_sub, 3, 3)
        #     # 1. 提取第一个 batch, 第一个 view 的点云和内参，并强制转换为 float32
        #     K1 = K[0, 0].float()  # 内参矩阵，形状为 (3, 3)
        #     depth_image = point_cloud_to_depth_map(local_pts[0,0].reshape(-1, 3), K1)
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(depth_image.cpu().numpy(), cmap='plasma')
        #     plt.colorbar(label='Depth (Z)')
        #     plt.title(f'Projected Depth Map')
        #     plt.axis('off')
        #     plt.savefig('projected_depth.png')
        #     print("深度图已生成，大小:", depth_image.shape)
        #         # print(f"Saved projected depth map to {save_path}")
        # depth = local_pts[0, 0,..., 2].detach().cpu().numpy()
        # depth_vis = np.zeros_like(depth)
        # valid_depths = depth
        # d_min, d_max = valid_depths.min(), valid_depths.max()

        # # 归一化整个深度图
        # normalized_depth = (depth - d_min) / (d_max - d_min + 1e-8)

        # # 将归一化后的有效区域赋值给可视化图像，背景保留为0
        # depth_vis = normalized_depth

        # # 保存图片
        # save_file = os.path.join(f'depth_0.png')
        # plt.imsave(save_file, depth_vis, cmap='plasma')

        return dict(
            gaussians=gaussians,
            camera_poses=camera_poses, 
            local_points=local_pts,
            intrinsics=K, 
            conf=conf_logits
        )