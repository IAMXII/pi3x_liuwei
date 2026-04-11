import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from contextlib import nullcontext
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
            num_dec_blk_not_to_checkpoint=0,
            ckpt="ckpts/pi3/model_pi3x.safetensors", 
            anchors_per_view=100000, 
            num_sky_anchors=8196, 
            K_input = 1000000,                    
            debug_mem=False,
            train_stage=1,
            max_dense_gaussians=1000000,
            min_gaussians_per_view=1024,
            pre_topk_per_view=4096,
            keep_ratio_sqrt=0.4,
            keep_ratio_linear=0.1,
            overlap_voxel_size_ratio=0.002,
            overlap_penalty=0.25,
            dir_penalty=0.08,
            dir_bins_azimuth=16,
            dir_bins_elevation=8,
            score_conf_weight=0.35,
            score_scale_weight=0.15,
            view_novelty_tau=0.8,
            view_angle_weight=0.35,
            novelty_bonus=0.2,
            low_conf_push_radius_ratio=20.0,
            low_conf_scale_boost=20.0,
            sparse_scale_voxel_size_ratio=0.03,
            sparse_scale_boost_max=1.8,
            sparse_scale_density_tau=4.0,
            sparse_scale_conf_threshold=0.1,
            scale_min_ratio=5e-5,
            scale_max_ratio=0.8,
            gs_view_stride=1,
            gs_decoder_view_chunk_size=10,
    ):
        super().__init__()
        self.debug_mem = debug_mem
        self.patch_size = 14
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint
        self.K_input = K_input
        self.train_stage = train_stage
        self.max_dense_gaussians = max_dense_gaussians
        self.anchors_per_view = anchors_per_view
        self.min_gaussians_per_view = max(1, int(min_gaussians_per_view))
        self.pre_topk_per_view = max(1, int(pre_topk_per_view))
        self.keep_ratio_sqrt = float(keep_ratio_sqrt)
        self.keep_ratio_linear = max(0.0, float(keep_ratio_linear))
        self.overlap_voxel_size_ratio = float(overlap_voxel_size_ratio)
        self.overlap_penalty = float(overlap_penalty)
        self.dir_penalty = float(dir_penalty)
        self.dir_bins_azimuth = max(2, int(dir_bins_azimuth))
        self.dir_bins_elevation = max(2, int(dir_bins_elevation))
        self.score_conf_weight = float(score_conf_weight)
        self.score_scale_weight = float(score_scale_weight)
        self.view_novelty_tau = float(view_novelty_tau)
        self.view_angle_weight = float(view_angle_weight)
        self.novelty_bonus = float(novelty_bonus)
        self.low_conf_push_radius_ratio = float(low_conf_push_radius_ratio)
        self.low_conf_scale_boost = float(low_conf_scale_boost)
        self.sparse_scale_voxel_size_ratio = float(sparse_scale_voxel_size_ratio)
        self.sparse_scale_boost_max = max(1.0, float(sparse_scale_boost_max))
        self.sparse_scale_density_tau = max(float(sparse_scale_density_tau), 1e-6)
        self.sparse_scale_conf_threshold = min(max(float(sparse_scale_conf_threshold), 0.0), 1.0)
        self.scale_min_ratio = float(scale_min_ratio)
        self.scale_max_ratio = max(float(scale_max_ratio), float(scale_min_ratio) * 1.1)
        self.gs_view_stride = max(1, int(gs_view_stride))
        self.gs_decoder_view_chunk_size = max(1, int(gs_decoder_view_chunk_size))

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
                
            if self.training:
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
        cam_ctx = torch.no_grad() if self.train_stage in [1, 2] else nullcontext()
        with cam_ctx:
            cam_h = self.camera_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
            camera_poses = self.camera_head(cam_h, patch_h, patch_w).reshape(B, N_total, 4, 4)
        mem.step("Camera Decoder")

        # ==========================================================
        # [修改点 2] 提取等间隔特征用于生成 Gaussian 与 Conf
        # ==========================================================
        sub_idx = torch.arange(0, N_total, self.gs_view_stride, device=imgs.device)
        if sub_idx[-1].item() != (N_total - 1):
            sub_idx = torch.unique(torch.cat([sub_idx, sub_idx.new_tensor([N_total - 1])]), sorted=True)
        N_sub = len(sub_idx)
        hw = hidden.shape[1]

        hidden_views = hidden.view(B, N_total, hw, -1)[:, sub_idx].contiguous()
        pos_views = pos.view(B, N_total, hw, -1)[:, sub_idx].contiguous()

        point_ctx = torch.no_grad() if self.train_stage == 1 else nullcontext()
        with point_ctx:
            point_h = self.point_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
            local_xyz_raw = self.point_head(point_h, patch_h=patch_h, patch_w=patch_w)

        conf_ctx = torch.no_grad() if self.train_stage in [1, 2] else nullcontext()
        with conf_ctx:
            ret_conf = self.conf_decoder(hidden, xpos=pos)
            conf = self.conf_head(ret_conf[:, self.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)[0]

        conf_logits = conf.permute(0, 2, 3, 1).reshape(B, N_total, H, W, -1)

        # 极其重要：及时释放厚重的隐层特征，防止驻留显存
        del ret_conf, conf, point_h, hidden, pos, cam_h

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
            scene_size = torch.quantile(distances.float().reshape(B, -1), 0.8, dim=1)
            # print(f"Dynamic scene size: {scene_size.item():.2f}")
            # scene_size_f = scene_size / 10.0 
            # local_pts = local_pts / scene_size_f[..., None, None, None]  # 将点云缩放到更合理的范围，防止数值不稳定
            # 目标半径设定为场景大小的 10 倍
            target_radius = self.low_conf_push_radius_ratio * scene_size.view(B, 1, 1, 1, 1)

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

        # 解析 Dense 高斯属性
        # ==========================================================
        # 1. 解析 Dense 高斯属性 & 天空推远/膨胀逻辑 (完全保留你的逻辑)
        # ==========================================================
        # ==========================================================
        # 1. 解析 Dense 高斯属性 & 天空推远/膨胀逻辑 (完全保留你的逻辑)
        # ==========================================================
        with torch.no_grad():
            # 提取与 gs_attrs 对齐的 conf 子集
            conf_logits_sub = conf_logits[:, sub_idx]
            conf_prob = torch.sigmoid(conf_logits_sub)
            
            mask_push_scale = conf_prob < 0.1
            

        # ==========================================================
        # 2. 动态多级四叉树 (Quadtree-Pro 激进版：最大 32，无 Scale 截断)
        # ==========================================================
        with torch.no_grad():
            z_raw = z.permute(0, 1, 4, 2, 3)
            depth_flat = z_raw[:, sub_idx].reshape(B * N_sub, 1, H, W)
            imgs_sub = imgs[:, sub_idx].reshape(B * N_sub, 3, H, W)

            # --- A. 提取底层特征联合 Score Map ---
            local_mean = F.avg_pool2d(imgs_sub, 7, stride=1, padding=3)
            color_diff = torch.abs(imgs_sub - local_mean).mean(dim=1, keepdim=True)
            color_score = color_diff / (color_diff.amax(dim=(-2, -1), keepdim=True) + 1e-5)

            sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=imgs.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=imgs.device).view(1, 1, 3, 3)
            dx = F.conv2d(depth_flat, sobel_x, padding=1)
            dy = F.conv2d(depth_flat, sobel_y, padding=1)
            depth_edge = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
            
            depth_edge = torch.clamp(depth_edge, max=torch.quantile(depth_edge.float().view(-1), 0.98))
            depth_score = 0.5 * depth_edge / (depth_edge.amax(dim=(-2, -1), keepdim=True) + 1e-5)

            score_map = torch.max(color_score, depth_score)

            # --- B. 计算多级 Scale Map (带自动 Padding) ---
            # 【修改 1】：重新放开到 32 级大格子
            patch_sizes = [32, 16, 8, 4, 2, 1] 
            base_threshold = 0.055  
            relax_factor = 0.3
            
            # 对齐到 32 的倍数
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            score_map_padded = F.pad(score_map, (0, pad_w, 0, pad_h), mode='replicate')
            H_pad, W_pad = H + pad_h, W + pad_w

            covered_mask = torch.zeros((B * N_sub, 1, H_pad, W_pad), dtype=torch.bool, device=imgs.device)
            scale_map_padded = torch.ones((B * N_sub, 1, H_pad, W_pad), dtype=torch.float32, device=imgs.device)

            for i, size in enumerate(patch_sizes):
                if i == len(patch_sizes) - 1:
                    is_flat_expanded = torch.ones((B * N_sub, 1, H_pad, W_pad), dtype=torch.bool, device=imgs.device)
                else:
                    current_threshold = base_threshold * (1.0 + relax_factor * math.log2(size))
                    pooled_score = F.max_pool2d(score_map_padded, kernel_size=size, stride=size)
                    is_flat_expanded = pooled_score.repeat_interleave(size, dim=2).repeat_interleave(size, dim=3) < current_threshold

                active_region = is_flat_expanded & (~covered_mask)
                scale_map_padded = torch.where(active_region, torch.full_like(scale_map_padded, size), scale_map_padded)
                covered_mask = covered_mask | active_region

            scale_map = scale_map_padded[:, :, :H, :W].reshape(B, N_sub, H, W, 1)

            # --- C. 选点逻辑：锚点掩码 (居中对齐) ---
            u_coords = torch.arange(W, device=imgs.device).view(1, 1, 1, W, 1).expand(B, N_sub, H, W, 1)
            v_coords = torch.arange(H, device=imgs.device).view(1, 1, H, 1, 1).expand(B, N_sub, H, W, 1)
            
            offset = (scale_map.long() // 2)
            quad_keep_mask = ((u_coords - offset) % scale_map.long() == 0) & ((v_coords - offset) % scale_map.long() == 0)
            
            quad_keep_mask = quad_keep_mask & (~mask_push_scale)

        # --- D. 提取并应用属性 (极速瘦身：先筛点，再分块解码 GS 属性) ---
        keep_mask = (quad_keep_mask | mask_push_scale).squeeze(-1) # 形状: (B, N_sub, H, W)
        local_pts_sub = local_pts[:, sub_idx]
        del z, xy, local_xyz_raw, z_raw, depth_flat, imgs_sub, score_map, score_map_padded
        del covered_mask, scale_map_padded, u_coords, v_coords, offset, conf_prob
        voxel_size_val = (scene_size.mean() * 0.002).clamp_min(1e-4).item()
        fused_gaussians = []

        # ==========================================================
        # 3. 仅对有效保留点解析属性并执行 3D Soft Voxel Attention Fusion
        #    先筛，再按视角 chunk 解码 GS，避免整块 gs_attrs 常驻显存
        # ==========================================================
        for b in range(B):
            mask_b = keep_mask[b] # [N_sub, H, W]

            # 兜底：防止该批次全图被过滤导致后续报错，强行保留一个点
            if not mask_b.any():
                mask_b[0, 0, 0] = True

        #####

        #     gs_attrs_parts = []
        #     local_pts_parts = []
        #     scale_map_parts = []
        #     conf_parts = []
        #     quad_center_parts = []
        #     low_conf_parts = []
        #     view_idx_parts = []

        #     for start in range(0, N_sub, self.gs_decoder_view_chunk_size):
        #         end = min(start + self.gs_decoder_view_chunk_size, N_sub)
        #         mask_chunk = mask_b[start:end]
        #         if not mask_chunk.any():
        #             continue

        #         hidden_chunk = hidden_views[b, start:end].reshape(end - start, hw, -1)
        #         pos_chunk = pos_views[b, start:end].reshape(end - start, hw, -1)
        #         gs_h_chunk = self.gs_decoder(hidden_chunk, xpos=pos_chunk)[:, self.patch_start_idx:]
        #         gs_attrs_chunk = self.gs_head([gs_h_chunk], (H, W)).reshape(end - start, H, W, 11)

        #         gs_attrs_parts.append(gs_attrs_chunk[mask_chunk])
        #         local_pts_parts.append(local_pts_sub[b, start:end][mask_chunk])
        #         scale_map_parts.append(scale_map[b, start:end][mask_chunk])
        #         conf_parts.append(conf_logits_sub[b, start:end][mask_chunk])
        #         quad_center_parts.append(quad_keep_mask[b, start:end][mask_chunk])
        #         low_conf_parts.append(mask_push_scale[b, start:end][mask_chunk])
        #         view_idx_parts.append(mask_chunk.nonzero(as_tuple=True)[0] + start)

        #         del hidden_chunk, pos_chunk, gs_h_chunk, gs_attrs_chunk, mask_chunk

        #     gs_attrs_v = torch.cat(gs_attrs_parts, dim=0)
        #     local_pts_v = torch.cat(local_pts_parts, dim=0)
        #     scale_map_v = torch.cat(scale_map_parts, dim=0)
        #     conf_v = torch.cat(conf_parts, dim=0)
        #     is_quad_center_v = torch.cat(quad_center_parts, dim=0)
        #     low_conf_v = torch.cat(low_conf_parts, dim=0)
        #     view_idx = torch.cat(view_idx_parts, dim=0)

        #     del gs_attrs_parts, local_pts_parts, scale_map_parts, conf_parts, quad_center_parts, low_conf_parts, view_idx_parts

        #     # ----------------------------------------------------
        #     # 2. 仅对有效点解析基本属性 (不计算无用点的 Sigmoid)
        #     # ----------------------------------------------------
        #     local_rot_v = F.normalize(gs_attrs_v[:, 0:4], dim=-1)
        #     scale_v = torch.exp(torch.clamp(gs_attrs_v[:, 4:7], min=-10.0, max=5.0)) * 0.1
        #     opacity_v = torch.sigmoid(gs_attrs_v[:, 7:8])
        #     color_v = torch.sigmoid(gs_attrs_v[:, 8:11])
        #     scale_v = torch.where(low_conf_v, scale_v * self.low_conf_scale_boost, scale_v)

        #     # 仅对四叉树选中的大格子中心点应用 scale_map 倍率放大
        #     scale_v = torch.where(is_quad_center_v, scale_v * scale_map_v, scale_v)

        #     # ----------------------------------------------------
        #     # 3. 仅对保留的点执行相机位姿到世界坐标系的旋转转换
        #     # 极大地节省了全尺寸张量的 Quat Mult 乘法计算
        #     # ----------------------------------------------------
        #     cam_poses_v = camera_poses_sub[b, view_idx]     # [K, 4, 4]
        #     cam_rot_v = cam_poses_v[:, :3, :3]
        #     cam_trans_v = cam_poses_v[:, :3, 3]
        #     xyz_v = torch.bmm(cam_rot_v, local_pts_v.unsqueeze(-1)).squeeze(-1) + cam_trans_v
        #     cam_quats_v = matrix_to_quaternion(cam_rot_v) # [K, 4]
        #     rot_v = F.normalize(quat_mult(cam_quats_v, local_rot_v), dim=-1)

        #     # ----------------------------------------------------
        #     # 4. 再次剔除掉模型主动预测为透明的废点 (如果存在)
        #     # ----------------------------------------------------
        #     valid_opacity_mask = opacity_v[:, 0] > 0.05
        #     if not valid_opacity_mask.any():
        #         valid_opacity_mask[0] = True

        #     xyz_v = xyz_v[valid_opacity_mask]
        #     # print("xyz_v",xyz_v.shape)
        #     rot_v = rot_v[valid_opacity_mask]
        #     scale_v = scale_v[valid_opacity_mask]
        #     opacity_v = opacity_v[valid_opacity_mask]
        #     color_v = color_v[valid_opacity_mask]
        #     conf_v = conf_v[valid_opacity_mask]

        #     # # ----------------------------------------------------
        #     # # 5. 空间哈希体素软融合 (Voxel Attention Fusion)
        #     # # ----------------------------------------------------
        #     # vox_coords = torch.floor(xyz_v / voxel_size_val).int()
        #     # hash_idx = (vox_coords[:, 0] * 73856093 ^ vox_coords[:, 1] * 19349663 ^ vox_coords[:, 2] * 83492791)
        #     # unique_hashes, inverse_indices = torch.unique(hash_idx, return_inverse=True)
        #     # num_voxels = unique_hashes.size(0)

        #     # safe_conf = torch.clamp(conf_v, max=20.0)
        #     # exp_conf = torch.exp(safe_conf)
        #     # sum_exp = torch.zeros(num_voxels, 1, device=xyz_v.device).index_add_(0, inverse_indices, exp_conf)
        #     # weights = exp_conf / (sum_exp[inverse_indices] + 1e-8)

        #     # # 原地无损聚合
        #     # f_xyz = torch.zeros(num_voxels, 3, device=xyz_v.device).index_add_(0, inverse_indices, xyz_v * weights)
        #     # # print("f_xyz:",num_voxels)
        #     # f_rot = torch.zeros(num_voxels, 4, device=xyz_v.device).index_add_(0, inverse_indices, rot_v * weights)
        #     # f_rot = F.normalize(f_rot, dim=-1)
        #     # f_scale = torch.zeros(num_voxels, 3, device=xyz_v.device).index_add_(0, inverse_indices, scale_v * weights)
        #     # f_opacity = torch.zeros(num_voxels, 1, device=xyz_v.device).index_add_(0, inverse_indices, opacity_v * weights)
        #     # f_color = torch.zeros(num_voxels, 3, device=xyz_v.device).index_add_(0, inverse_indices, color_v * weights)

        #     fused_gaussians.append({
        #         "xyz": xyz_v, "rotation": rot_v, "scale": scale_v,
        #         "opacity": opacity_v, "color": color_v
        #     })
        #     # fused_gaussians.append({
        #     #     "xyz": f_xyz, "rotation": f_rot, "scale": f_scale,
        #     #     "opacity": f_opacity, "color": f_color
        #     # })

        #     del gs_attrs_v, local_pts_v, scale_map_v, conf_v, is_quad_center_v, low_conf_v, view_idx
        #     del local_rot_v, scale_v, opacity_v, color_v, cam_poses_v, cam_rot_v, cam_trans_v, xyz_v, cam_quats_v, rot_v
        #     # del valid_opacity_mask, vox_coords, hash_idx, unique_hashes, inverse_indices, safe_conf, exp_conf, sum_exp, weights
        #     # del f_xyz, f_rot, f_scale, f_opacity, f_color

        # del hidden_views, pos_views, local_pts_sub, conf_logits_sub, mask_push_scale, scale_map, quad_keep_mask, keep_mask

        # mem.step("Voxel Attention Fusion")
        #####
            # --- 优化后的高斯解析逻辑 ---
            b_xyz, b_rot, b_scale, b_opacity, b_color = [], [], [], [], []

            for start in range(0, N_sub, self.gs_decoder_view_chunk_size):
                end = min(start + self.gs_decoder_view_chunk_size, N_sub)
                mask_chunk = mask_b[start:end]
                if not mask_chunk.any():
                    continue

                hidden_chunk = hidden_views[b, start:end].reshape(end - start, hw, -1)
                pos_chunk = pos_views[b, start:end].reshape(end - start, hw, -1)
                gs_h_chunk = self.gs_decoder(hidden_chunk, xpos=pos_chunk)[:, self.patch_start_idx:]
                gs_attrs_chunk = self.gs_head([gs_h_chunk], (H, W)).reshape(end - start, H, W, 11)

                # 1. 提取当前 chunk 内有效四叉树点
                gs_attrs_v = gs_attrs_chunk[mask_chunk]
                local_pts_v = local_pts_sub[b, start:end][mask_chunk]
                scale_map_v = scale_map[b, start:end][mask_chunk]
                is_quad_center_v = quad_keep_mask[b, start:end][mask_chunk]
                low_conf_v = mask_push_scale[b, start:end][mask_chunk]
                view_idx_v = mask_chunk.nonzero(as_tuple=True)[0] + start

                # 及时释放 chunk 级无用显存
                del hidden_chunk, pos_chunk, gs_h_chunk, gs_attrs_chunk, mask_chunk

                # 2. 提前计算 Opacity，执行第一波残酷过滤 (极大幅度降低后续计算量)
                opacity_v = torch.sigmoid(gs_attrs_v[:, 7:8])
                valid_mask = opacity_v[:, 0] > 0.05
                if not valid_mask.any():
                    continue
                
                # 应用过滤
                gs_attrs_v = gs_attrs_v[valid_mask]
                local_pts_v = local_pts_v[valid_mask]
                scale_map_v = scale_map_v[valid_mask]
                is_quad_center_v = is_quad_center_v[valid_mask]
                low_conf_v = low_conf_v[valid_mask]
                view_idx_v = view_idx_v[valid_mask]
                opacity_v = opacity_v[valid_mask]

                # 3. 仅对存活的点解析其余属性并变换
                local_rot_v = F.normalize(gs_attrs_v[:, 0:4], dim=-1)
                scale_v = torch.exp(torch.clamp(gs_attrs_v[:, 4:7], min=-10.0, max=5.0)) * 0.1
                color_v = torch.sigmoid(gs_attrs_v[:, 8:11])

                # 【修复 OOM 广播灾难】：强制转为 (N, 1) 的形状，避免 N x N 维度爆炸
                mask_low = low_conf_v.view(-1, 1)
                mask_quad = is_quad_center_v.view(-1, 1)
                map_scale = scale_map_v.view(-1, 1)

                scale_v = torch.where(mask_low, scale_v * self.low_conf_scale_boost, scale_v)
                scale_v = torch.where(mask_quad, scale_v * map_scale, scale_v)

                cam_poses_v = camera_poses_sub[b, view_idx_v]
                cam_rot_v = cam_poses_v[:, :3, :3]
                cam_trans_v = cam_poses_v[:, :3, 3]

                # 坐标与旋转变换 (此时数据量已锐减)
                xyz_v = torch.bmm(cam_rot_v, local_pts_v.unsqueeze(-1)).squeeze(-1) + cam_trans_v
                cam_quats_v = matrix_to_quaternion(cam_rot_v)
                rot_v = F.normalize(quat_mult(cam_quats_v, local_rot_v), dim=-1)

                # 将存活的点追加到列表
                b_xyz.append(xyz_v)
                b_rot.append(rot_v)
                b_scale.append(scale_v)
                b_opacity.append(opacity_v)
                b_color.append(color_v)

                del gs_attrs_v, local_pts_v, local_rot_v, cam_poses_v, cam_rot_v, cam_trans_v

            # 当前 Batch 视角处理完毕，合并结果兜底
            if len(b_xyz) > 0:
                fused_gaussians.append({
                    "xyz": torch.cat(b_xyz, dim=0),
                    "rotation": torch.cat(b_rot, dim=0),
                    "scale": torch.cat(b_scale, dim=0),
                    "opacity": torch.cat(b_opacity, dim=0),
                    "color": torch.cat(b_color, dim=0)
                })
            else:
                # 极端兜底，防止全被过滤导致维度错误
                fused_gaussians.append({
                    "xyz": torch.zeros((1, 3), device=imgs.device),
                    "rotation": torch.tensor([[1., 0., 0., 0.]], device=imgs.device),
                    "scale": torch.full((1, 3), 1e-5, device=imgs.device),
                    "opacity": torch.zeros((1, 1), device=imgs.device),
                    "color": torch.zeros((1, 3), device=imgs.device)
                })
        # 对齐 Batch 内高斯数量
        max_k = max(g["xyz"].size(0) for g in fused_gaussians)
        d_xyz_out, d_rot_out, d_scale_out, d_opacity_out, d_color_out = [], [], [], [], []
        for g in fused_gaussians:
            pad_len = max_k - g["xyz"].size(0)
            d_xyz_out.append(F.pad(g["xyz"], (0,0, 0,pad_len), value=0.0))
            d_rot_out.append(F.pad(g["rotation"], (0,0, 0,pad_len), value=1.0))
            d_scale_out.append(F.pad(g["scale"], (0,0, 0,pad_len), value=1e-5))
            d_opacity_out.append(F.pad(g["opacity"], (0,0, 0,pad_len), value=0.0))
            d_color_out.append(F.pad(g["color"], (0,0, 0,pad_len), value=0.0))

        gaussians = {
            "xyz": torch.stack(d_xyz_out, dim=0),
            "rotation": torch.stack(d_rot_out, dim=0),
            "scale": torch.stack(d_scale_out, dim=0),
            "opacity": torch.stack(d_opacity_out, dim=0),
            "color": torch.stack(d_color_out, dim=0),
            "num_sky": 0 
        }

        return dict(
            gaussians=gaussians,
            camera_poses=camera_poses, 
            local_points=local_pts,
            intrinsics=K, 
            conf=conf_logits
        )
