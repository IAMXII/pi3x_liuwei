import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from functools import partial
from copy import deepcopy
from contextlib import nullcontext
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


class PixelwiseMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        return y.view(*shape, -1)

class Pi3_3DGS(nn.Module):
    def __init__(
            self, 
            pos_type='rope100', 
            decoder_size='large', 
            load_vggt=False, 
            freeze_encoder=True,
            num_dec_blk_not_to_checkpoint=4,
            ckpt="ckpts/pi3/model_pi3x.safetensors",
            debug_mem=False,
            train_stage=1,
            max_dense_gaussians=1000000,
            min_gaussians_per_view=8192,
            pre_topk_per_view=32768,
            overlap_voxel_size_ratio=0.002,
            dir_bins_azimuth=16,
            dir_bins_elevation=8,
            score_conf_weight=0.35,
            score_scale_weight=0.15,
            low_conf_push_radius_ratio=10.0,
            low_conf_scale_boost=10.0,
            sparse_scale_voxel_size_ratio=0.03,
            sparse_scale_boost_max=1.8,
            sparse_scale_density_tau=4.0,
            sparse_scale_conf_threshold=0.1,
            scale_min_ratio=5e-5,
            scale_max_ratio=0.8,
            num_scale_experts=6,
            scale_expert_priors=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
            gate_hidden_dim=96,
            learned_anchor_thresh=0.35,
            target_gaussians_per_view=65536,
            proposal_teacher_steps=20000,
            num_structural_prototypes=4,
            branches_per_prototype=3,
            prototype_dim=16,
            top_prototypes_per_anchor=2,
            prototype_temperature=0.7,
            gs_view_stride=1,
            gs_decoder_view_chunk_size=8,
            selected_anchor_chunk_size=1024,
    ):
        super().__init__()
        self.debug_mem = debug_mem
        self.patch_size = 14
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint
        self.train_stage = train_stage
        self.max_dense_gaussians = max_dense_gaussians
        self.min_gaussians_per_view = max(1, int(min_gaussians_per_view))
        self.pre_topk_per_view = max(1, int(pre_topk_per_view))
        self.overlap_voxel_size_ratio = float(overlap_voxel_size_ratio)
        self.dir_bins_azimuth = max(2, int(dir_bins_azimuth))
        self.dir_bins_elevation = max(2, int(dir_bins_elevation))
        self.score_conf_weight = float(score_conf_weight)
        self.score_scale_weight = float(score_scale_weight)
        self.low_conf_push_radius_ratio = float(low_conf_push_radius_ratio)
        self.low_conf_scale_boost = float(low_conf_scale_boost)
        self.sparse_scale_voxel_size_ratio = float(sparse_scale_voxel_size_ratio)
        self.sparse_scale_boost_max = max(1.0, float(sparse_scale_boost_max))
        self.sparse_scale_density_tau = max(float(sparse_scale_density_tau), 1e-6)
        self.sparse_scale_conf_threshold = min(max(float(sparse_scale_conf_threshold), 0.0), 1.0)
        self.scale_min_ratio = float(scale_min_ratio)
        self.scale_max_ratio = max(float(scale_max_ratio), float(scale_min_ratio) * 1.1)
        self.num_scale_experts = int(num_scale_experts)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.learned_anchor_thresh = float(learned_anchor_thresh)
        self.target_gaussians_per_view = float(target_gaussians_per_view)
        self.proposal_teacher_steps = int(proposal_teacher_steps)
        self.num_structural_prototypes = int(num_structural_prototypes)
        self.branches_per_prototype = int(branches_per_prototype)
        self.prototype_dim = int(prototype_dim)
        self.top_prototypes_per_anchor = max(1, min(int(top_prototypes_per_anchor), self.num_structural_prototypes))
        self.prototype_temperature = max(float(prototype_temperature), 1e-3)
        self.gs_view_stride = max(1, int(gs_view_stride))
        self.gs_decoder_view_chunk_size = max(1, int(gs_decoder_view_chunk_size))
        self.selected_anchor_chunk_size = max(1, int(selected_anchor_chunk_size))
        scale_priors = torch.tensor(scale_expert_priors, dtype=torch.float32)
        if scale_priors.numel() != self.num_scale_experts:
            raise ValueError("scale_expert_priors must match num_scale_experts")
        self.register_buffer("scale_expert_priors", scale_priors.view(1, 1, 1, 1, self.num_scale_experts, 1))

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

        # 在保留旧 gs_head 的前提下，新增“结构原型驱动的高斯分裂器”。
        # 与 F4Splat 的 densification score 不同，这里不是只预测“要加多少高斯”，
        # 而是先选择结构原型，再由原型分裂出一组具有形状先验的高斯。
        self.anchor_head = PixelwiseMLP(in_dim=15, hidden_dim=self.gate_hidden_dim, out_dim=1)
        self.prototype_logits_head = PixelwiseMLP(in_dim=15, hidden_dim=self.gate_hidden_dim, out_dim=self.num_structural_prototypes)
        self.prototype_feat_head = PixelwiseMLP(in_dim=15, hidden_dim=self.gate_hidden_dim, out_dim=self.prototype_dim)
        self.prototype_delta_head = PixelwiseMLP(in_dim=15 + self.prototype_dim, hidden_dim=self.gate_hidden_dim, out_dim=self.branches_per_prototype * 14)
        self.prototype_branch_gate = nn.Parameter(torch.zeros(self.num_structural_prototypes, self.branches_per_prototype, 1))
        self.prototype_codes = nn.Parameter(torch.randn(self.num_structural_prototypes, self.prototype_dim) * 0.02)
        self.prototype_branch_offsets = nn.Parameter(torch.zeros(self.num_structural_prototypes, self.branches_per_prototype, 3))
        self.prototype_branch_logscale = nn.Parameter(torch.zeros(self.num_structural_prototypes, self.branches_per_prototype, 3))
        self.prototype_branch_rot = nn.Parameter(torch.zeros(self.num_structural_prototypes, self.branches_per_prototype, 4))
        self.prototype_branch_opacity = nn.Parameter(torch.zeros(self.num_structural_prototypes, self.branches_per_prototype, 1))
        self.prototype_branch_color = nn.Parameter(torch.zeros(self.num_structural_prototypes, self.branches_per_prototype, 3))

        if self.num_structural_prototypes >= 4 and self.branches_per_prototype >= 3:
            with torch.no_grad():
                plane = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
                edge = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                corner = torch.tensor([[0.0, 0.0, 0.0], [0.75, 0.75, 0.0], [-0.75, 0.75, 0.0]])
                blob = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, -0.6]])
                self.prototype_branch_offsets[0, :3] = plane
                self.prototype_branch_offsets[1, :3] = edge
                self.prototype_branch_offsets[2, :3] = corner
                self.prototype_branch_offsets[3, :3] = blob

                self.prototype_branch_logscale[0, :3] = torch.log(torch.tensor([[1.6, 0.45, 0.25], [1.2, 0.55, 0.3], [1.6, 0.45, 0.25]]))
                self.prototype_branch_logscale[1, :3] = torch.log(torch.tensor([[0.45, 1.6, 0.25], [0.55, 1.2, 0.3], [0.45, 1.6, 0.25]]))
                self.prototype_branch_logscale[2, :3] = torch.log(torch.tensor([[0.6, 0.6, 0.28], [0.75, 0.75, 0.35], [0.75, 0.75, 0.35]]))
                self.prototype_branch_logscale[3, :3] = torch.log(torch.tensor([[1.2, 1.2, 0.9], [1.0, 1.0, 1.2], [1.0, 1.0, 1.2]]))

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

    def _module_requires_grad(self, module: nn.Module) -> bool:
        return any(param.requires_grad for param in module.parameters())

    def _branch_context(self, *modules: nn.Module):
        if any(self._module_requires_grad(module) for module in modules):
            return nullcontext()
        return torch.no_grad()

    def _decode_gs_attrs_chunked(self, hidden: torch.Tensor, pos: torch.Tensor, B: int, N_total: int, H: int, W: int, sub_idx: torch.Tensor) -> torch.Tensor:
        hw = hidden.shape[1]
        N_sub = int(sub_idx.numel())
        hidden_bn = hidden.view(B, N_total, hw, -1)
        pos_bn = pos.view(B, N_total, hw, -1)
        chunk_size = min(self.gs_decoder_view_chunk_size, N_sub)
        gs_attrs_chunks = []

        with self._branch_context(self.gs_decoder, self.gs_head):
            for start in range(0, N_sub, chunk_size):
                end = min(start + chunk_size, N_sub)
                idx = sub_idx[start:end]
                hidden_chunk = hidden_bn[:, idx].reshape(B * (end - start), hw, -1)
                pos_chunk = pos_bn[:, idx].reshape(B * (end - start), hw, -1)
                gs_h_chunk = self.gs_decoder(hidden_chunk, xpos=pos_chunk)[:, self.patch_start_idx:]
                gs_attrs_chunk = self.gs_head([gs_h_chunk], (H, W)).reshape(B, end - start, H, W, 11)
                gs_attrs_chunks.append(gs_attrs_chunk)
                del hidden_chunk, pos_chunk, gs_h_chunk

        return torch.cat(gs_attrs_chunks, dim=1)

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



    def _teacher_forcing_ratio(self, global_step=None):
        if global_step is None:
            return 1.0 if self.training else 0.0
        if not self.training:
            return 0.0
        return max(0.0, 1.0 - float(global_step) / max(float(self.proposal_teacher_steps), 1.0))

    def _build_scale_targets(self, scale_map: torch.Tensor) -> torch.Tensor:
        # scale_map: [B, N, H, W, 1]
        target_log = torch.log2(scale_map.clamp_min(1.0))
        prior_log = torch.log2(self.scale_expert_priors.view(1, 1, 1, 1, self.num_scale_experts))
        dist = torch.abs(target_log - prior_log)
        return torch.softmax(-2.0 * dist, dim=-1)

    def _safe_logit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logit(x.clamp(1e-4, 1.0 - 1e-4))

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
        with self._branch_context(self.camera_decoder, self.camera_head):
            cam_h = self.camera_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
            camera_poses = self.camera_head(cam_h, patch_h, patch_w).reshape(B, N_total, 4, 4)
        mem.step("Camera Decoder")

        # ==========================================================
        # [修改点 2] 仅对子视图运行 GS 分支，减少显存峰值
        # ==========================================================
        sub_idx = torch.arange(0, N_total, self.gs_view_stride, device=imgs.device)
        N_sub = len(sub_idx)

        with self._branch_context(self.point_decoder, self.point_head):
            point_h = self.point_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
            local_xyz_raw = self.point_head(point_h, patch_h=patch_h, patch_w=patch_w)

        gs_attrs = self._decode_gs_attrs_chunked(hidden, pos, B, N_total, H, W, sub_idx)

        with self._branch_context(self.conf_decoder, self.conf_head):
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
        # z= torch.clamp(z, 1e-3, 1e3)  # 限制深度范围，防止数值不稳定
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
        # ==========================================================
        # 1. 解析 Dense 高斯属性 & 天空推远/膨胀逻辑 (完全保留你的逻辑)
        # ==========================================================
        # ==========================================================
        # 1. 解析 Dense 高斯属性 & 天空推远/膨胀逻辑 (完全保留你的逻辑)
        # ==========================================================
        local_rot = F.normalize(gs_attrs[..., 0:4], dim=-1)
        scale = torch.exp(torch.clamp(gs_attrs[..., 4:7], min=-10.0, max=5.0)) * 0.1
        
        with torch.no_grad():
            # 提取与 gs_attrs 对齐的 conf 子集
            conf_logits_sub = conf_logits[:, sub_idx]
            conf_prob = torch.sigmoid(conf_logits_sub)
            
            mask_push_scale = conf_prob < 0.1
            mask_push_scale = mask_push_scale.expand_as(scale)
            
        # 根据掩码放大天空区域 scale (你的原始逻辑)
        scale = torch.where(mask_push_scale, scale * self.low_conf_scale_boost, scale)

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
            base_threshold = 0.08  
            relax_factor = 0.5
            
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
            
            quad_keep_mask = quad_keep_mask & (~mask_push_scale[..., 0:1])
        # --- D. 保留你原始代码的 keep_mask 作为唯一 base-anchor 选择器 ---
        # 只在 keep_mask 选出的稀疏 base anchors 上做 prototype split，
        # 不再在 dense hidden / dense gs_attrs 上额外做 learned proposal，避免破坏原始稳定主干。
        keep_mask = (quad_keep_mask | mask_push_scale[..., 0:1]).squeeze(-1)
        local_pts_sub = local_pts[:, sub_idx]
        global_pts_sub = global_pts[:, sub_idx]

        del z, xy, local_xyz_raw, z_raw, depth_flat, imgs_sub, score_map, score_map_padded
        del covered_mask, scale_map_padded, u_coords, v_coords, offset, conf_prob

        voxel_size_val = (scene_size.mean() * self.overlap_voxel_size_ratio).clamp_min(1e-5).item()
        fused_gaussians = []
        overlap_losses = []
        proto_ownership_losses = []
        proto_usage_losses = []
        proto_compact_losses = []
        selected_before_fusion = []
        selected_after_fusion = []
        anchor_losses = []
        budget_losses = []
        lowconf_flat_terms = []
        lowconf_coarse_terms = []
        teacher_ratio = self._teacher_forcing_ratio(global_step)

        topk_limit = max(self.min_gaussians_per_view, min(self.pre_topk_per_view, H * W))
        uniform_proto = torch.full((self.num_structural_prototypes,), 1.0 / self.num_structural_prototypes, device=imgs.device)

        for b in range(B):
            batch_fused = {"xyz": [], "rotation": [], "scale": [], "opacity": [], "color": []}
            batch_before = 0.0
            batch_after = 0.0
            batch_budget_loss = []
            batch_overlap = []
            batch_ownership = []
            batch_usage = []
            batch_compact = []

            for n in range(N_sub):
                mask_v = keep_mask[b, n]
                if not mask_v.any():
                    mask_v[0, 0] = True

                flat_idx = torch.nonzero(mask_v.reshape(-1), as_tuple=False).flatten()
                if flat_idx.numel() == 0:
                    flat_idx = torch.zeros(1, device=imgs.device, dtype=torch.long)

                # 仅在 base-anchor 候选内部做预算裁剪；当 pre_topk_per_view 足够大时，等价于不裁。
                if flat_idx.numel() > topk_limit:
                    conf_score = torch.sigmoid(conf_logits_sub[b, n].reshape(-1, 1)[flat_idx])[:, 0]
                    scale_score = torch.log2(scale_map[b, n].reshape(-1, 1)[flat_idx].float().clamp_min(1.0))[:, 0]
                    if scale_score.numel() > 0:
                        scale_score = scale_score / scale_score.max().clamp_min(1.0)
                    cand_score = self.score_conf_weight * conf_score + self.score_scale_weight * scale_score
                    rel_idx = torch.topk(cand_score, k=topk_limit, sorted=False).indices
                    flat_idx = flat_idx[rel_idx]

                y_idx = torch.div(flat_idx, W, rounding_mode='floor')
                x_idx = flat_idx % W

                gs_attrs_v = gs_attrs[b, n, y_idx, x_idx]
                local_pts_v = local_pts_sub[b, n, y_idx, x_idx]
                xyz_v = global_pts_sub[b, n, y_idx, x_idx]
                scale_map_v = scale_map[b, n, y_idx, x_idx]
                conf_logits_v = conf_logits_sub[b, n, y_idx, x_idx]
                conf_prob_v = torch.sigmoid(conf_logits_v)
                is_quad_center_v = quad_keep_mask[b, n, y_idx, x_idx].float()

                local_rot_v = F.normalize(gs_attrs_v[:, 0:4], dim=-1)
                scale_v = scale[b, n, y_idx, x_idx]
                scale_v = torch.where(is_quad_center_v.expand_as(scale_v) > 0.5, scale_v * scale_map_v, scale_v)
                opacity_v = torch.sigmoid(gs_attrs_v[:, 7:8])
                color_v = torch.sigmoid(gs_attrs_v[:, 8:11])

                cam_pose_v = camera_poses_sub[b, n].unsqueeze(0).expand(flat_idx.numel(), -1, -1)
                cam_quat_v = matrix_to_quaternion(cam_pose_v[:, :3, :3])
                rot_v = F.normalize(quat_mult(cam_quat_v, local_rot_v), dim=-1)

                # 结构原型的描述子只建立在已经通过 keep_mask 的 base anchors 上。
                desc_v = torch.cat([gs_attrs_v, local_pts_v, conf_logits_v], dim=-1)
                with torch.cuda.amp.autocast(enabled=False):
                    desc_v_fp32 = desc_v.float()
                    proto_logits_v = self.prototype_logits_head(desc_v_fp32)
                    proto_probs_v = F.softmax(proto_logits_v / self.prototype_temperature, dim=-1)
                    top_proto_prob_v, top_proto_idx_v = torch.topk(proto_probs_v, k=self.top_prototypes_per_anchor, dim=-1)

                candidate_mass_v = (top_proto_prob_v.sum(dim=-1) * self.branches_per_prototype).sum()
                batch_budget_loss.append(((candidate_mass_v / (self.target_gaussians_per_view + 1e-6)) - 1.0) ** 2)

                proto_mean = proto_probs_v.mean(dim=0)
                batch_usage.append(((proto_mean - uniform_proto) ** 2).mean())
                entropy = -(proto_probs_v * torch.log(proto_probs_v.clamp_min(1e-8))).sum(dim=-1)
                batch_compact.append(entropy.mean())

                for chunk_start in range(0, flat_idx.numel(), self.selected_anchor_chunk_size):
                    chunk_end = min(chunk_start + self.selected_anchor_chunk_size, flat_idx.numel())
                    desc_chunk = desc_v[chunk_start:chunk_end]
                    xyz_chunk = xyz_v[chunk_start:chunk_end]
                    scale_chunk = scale_v[chunk_start:chunk_end]
                    rot_chunk = rot_v[chunk_start:chunk_end]
                    opacity_chunk = opacity_v[chunk_start:chunk_end]
                    color_chunk = color_v[chunk_start:chunk_end]
                    cam_quat_chunk = cam_quat_v[chunk_start:chunk_end]
                    conf_prob_chunk = conf_prob_v[chunk_start:chunk_end]
                    top_proto_prob_chunk = top_proto_prob_v[chunk_start:chunk_end]
                    top_proto_idx_chunk = top_proto_idx_v[chunk_start:chunk_end]
                    scale_map_chunk = scale_map_v[chunk_start:chunk_end]

                    with torch.cuda.amp.autocast(enabled=False):
                        desc_chunk_fp32 = desc_chunk.float()
                        desc_code = self.prototype_feat_head(desc_chunk_fp32)

                        repeated_desc = []
                        repeated_code = []
                        proto_id_list = []
                        proto_gate_list = []
                        cam_quat_list = []
                        xyz_anchor_list = []
                        scale_anchor_list = []
                        rot_anchor_list = []
                        opacity_anchor_list = []
                        color_anchor_list = []

                        for pslot in range(self.top_prototypes_per_anchor):
                            pid = top_proto_idx_chunk[:, pslot]
                            pprob = top_proto_prob_chunk[:, pslot:pslot+1]
                            repeated_desc.append(desc_chunk_fp32)
                            repeated_code.append(desc_code + self.prototype_codes[pid].float())
                            proto_id_list.append(pid)
                            proto_gate_list.append(pprob.float())
                            cam_quat_list.append(cam_quat_chunk.float())
                            xyz_anchor_list.append(xyz_chunk.float())
                            scale_anchor_list.append(scale_chunk.float())
                            rot_anchor_list.append(rot_chunk.float())
                            opacity_anchor_list.append(opacity_chunk.float())
                            color_anchor_list.append(color_chunk.float())

                        desc_rep = torch.cat(repeated_desc, dim=0)
                        code_rep = torch.cat(repeated_code, dim=0)
                        proto_ids = torch.cat(proto_id_list, dim=0)
                        proto_gate = torch.cat(proto_gate_list, dim=0)
                        cam_quat_rep = torch.cat(cam_quat_list, dim=0)
                        xyz_anchor = torch.cat(xyz_anchor_list, dim=0)
                        scale_anchor = torch.cat(scale_anchor_list, dim=0)
                        rot_anchor = torch.cat(rot_anchor_list, dim=0)
                        opacity_anchor = torch.cat(opacity_anchor_list, dim=0)
                        color_anchor = torch.cat(color_anchor_list, dim=0)

                        proto_cond = torch.cat([desc_rep, code_rep], dim=-1)
                        proto_delta = self.prototype_delta_head(proto_cond).reshape(-1, self.branches_per_prototype, 14)

                        tpl_offset = self.prototype_branch_offsets[proto_ids].float()
                        tpl_logscale = self.prototype_branch_logscale[proto_ids].float()
                        tpl_rot = self.prototype_branch_rot[proto_ids].float()
                        tpl_opacity = self.prototype_branch_opacity[proto_ids].float()
                        tpl_color = self.prototype_branch_color[proto_ids].float()
                        tpl_gate = torch.sigmoid(self.prototype_branch_gate[proto_ids].float())

                        anchor_scale_mean = scale_anchor.mean(dim=-1, keepdim=True).unsqueeze(1)
                        xyz_delta = (tpl_offset + 0.35 * torch.tanh(proto_delta[..., 0:3])) * anchor_scale_mean
                        rot_local = F.normalize(rot_anchor.unsqueeze(1) + tpl_rot + 0.25 * proto_delta[..., 3:7], dim=-1)
                        scale_branch = scale_anchor.unsqueeze(1) * torch.exp(torch.clamp(tpl_logscale + proto_delta[..., 7:10], min=-2.0, max=2.0))
                        opacity_branch = torch.sigmoid(self._safe_logit(opacity_anchor).unsqueeze(1) + tpl_opacity + proto_delta[..., 10:11])
                        color_branch = torch.sigmoid(self._safe_logit(color_anchor).unsqueeze(1) + tpl_color + proto_delta[..., 11:14])
                        gate_branch = proto_gate.unsqueeze(1) * tpl_gate
                        score_branch = opacity_branch * gate_branch
                        xyz_branch = xyz_anchor.unsqueeze(1) + xyz_delta
                        rot_world = F.normalize(quat_mult(cam_quat_rep.unsqueeze(1), rot_local), dim=-1)

                    xyz_flat = xyz_branch.reshape(-1, 3)
                    rot_flat = rot_world.reshape(-1, 4)
                    scale_flat = scale_branch.reshape(-1, 3).clamp_min(1e-5)
                    opacity_flat = opacity_branch.reshape(-1, 1)
                    color_flat = color_branch.reshape(-1, 3)
                    gate_flat = gate_branch.reshape(-1, 1)
                    score_flat = score_branch.reshape(-1, 1)
                    proto_flat = proto_ids.unsqueeze(1).expand(-1, self.branches_per_prototype).reshape(-1)

                    valid_mask = score_flat[:, 0] > 0.01
                    if not valid_mask.any():
                        valid_mask[0] = True

                    xyz_flat = xyz_flat[valid_mask]
                    rot_flat = rot_flat[valid_mask]
                    scale_flat = scale_flat[valid_mask]
                    opacity_flat = opacity_flat[valid_mask]
                    color_flat = color_flat[valid_mask]
                    gate_flat = gate_flat[valid_mask]
                    score_flat = score_flat[valid_mask]
                    proto_flat = proto_flat[valid_mask]
                    batch_before += float(valid_mask.sum().item())

                    vox_coords = torch.floor(xyz_flat / voxel_size_val).int()
                    hash_idx = (vox_coords[:, 0] * 73856093 ^ vox_coords[:, 1] * 19349663 ^ vox_coords[:, 2] * 83492791)
                    unique_hashes, inverse_indices = torch.unique(hash_idx, return_inverse=True)
                    num_voxels = unique_hashes.size(0)
                    batch_after += float(num_voxels)

                    voxel_gate_mass = torch.zeros(num_voxels, 1, device=xyz_flat.device).index_add_(0, inverse_indices, gate_flat)
                    batch_overlap.append(F.relu(voxel_gate_mass - 1.0).pow(2).mean())

                    proto_onehot = F.one_hot(proto_flat, num_classes=self.num_structural_prototypes).float()
                    voxel_proto_mass = torch.zeros(num_voxels, self.num_structural_prototypes, device=xyz_flat.device).index_add_(0, inverse_indices, gate_flat * proto_onehot)
                    voxel_proto_dist = voxel_proto_mass / voxel_proto_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    voxel_entropy = -(voxel_proto_dist * torch.log(voxel_proto_dist.clamp_min(1e-8))).sum(dim=-1)
                    batch_ownership.append(voxel_entropy.mean())

                    weights_src = score_flat.clamp_min(1e-8)
                    sum_w = torch.zeros(num_voxels, 1, device=xyz_flat.device).index_add_(0, inverse_indices, weights_src)
                    weights = weights_src / (sum_w[inverse_indices] + 1e-8)

                    batch_fused["xyz"].append(torch.zeros(num_voxels, 3, device=xyz_flat.device).index_add_(0, inverse_indices, xyz_flat * weights))
                    batch_fused["rotation"].append(F.normalize(torch.zeros(num_voxels, 4, device=xyz_flat.device).index_add_(0, inverse_indices, rot_flat * weights), dim=-1))
                    batch_fused["scale"].append(torch.zeros(num_voxels, 3, device=xyz_flat.device).index_add_(0, inverse_indices, scale_flat * weights).clamp_min(1e-5))
                    batch_fused["opacity"].append(torch.zeros(num_voxels, 1, device=xyz_flat.device).index_add_(0, inverse_indices, opacity_flat * weights))
                    batch_fused["color"].append(torch.zeros(num_voxels, 3, device=xyz_flat.device).index_add_(0, inverse_indices, color_flat * weights))

            fused_gaussians.append({
                "xyz": torch.cat(batch_fused["xyz"], dim=0) if batch_fused["xyz"] else torch.zeros(1, 3, device=imgs.device),
                "rotation": torch.cat(batch_fused["rotation"], dim=0) if batch_fused["rotation"] else F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=imgs.device), dim=-1),
                "scale": torch.cat(batch_fused["scale"], dim=0) if batch_fused["scale"] else torch.full((1, 3), 1e-5, device=imgs.device),
                "opacity": torch.cat(batch_fused["opacity"], dim=0) if batch_fused["opacity"] else torch.zeros(1, 1, device=imgs.device),
                "color": torch.cat(batch_fused["color"], dim=0) if batch_fused["color"] else torch.zeros(1, 3, device=imgs.device),
            })

            overlap_losses.append(torch.stack(batch_overlap).mean() if batch_overlap else gs_attrs.new_tensor(0.0))
            proto_ownership_losses.append(torch.stack(batch_ownership).mean() if batch_ownership else gs_attrs.new_tensor(0.0))
            proto_usage_losses.append(torch.stack(batch_usage).mean() if batch_usage else gs_attrs.new_tensor(0.0))
            proto_compact_losses.append(torch.stack(batch_compact).mean() if batch_compact else gs_attrs.new_tensor(0.0))
            anchor_losses.append(gs_attrs.new_tensor(0.0))
            budget_losses.append(torch.stack(batch_budget_loss).mean() if batch_budget_loss else gs_attrs.new_tensor(0.0))
            lowconf_flat_terms.append(gs_attrs.new_tensor(0.0))
            lowconf_coarse_terms.append(gs_attrs.new_tensor(0.0))
            selected_before_fusion.append(batch_before)
            selected_after_fusion.append(batch_after)
        mem.step("Prototype Structured Splitting")
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
            "num_sky": 0,
        }

        reg_base = gaussians["xyz"].new_tensor(0.0)
        regularizers = {
            "loss_anchor": torch.stack(anchor_losses).mean() if anchor_losses else reg_base,
            "loss_budget": torch.stack(budget_losses).mean() if budget_losses else reg_base,
            "loss_overlap": torch.stack(overlap_losses).mean() if overlap_losses else reg_base,
            "loss_proto_ownership": torch.stack(proto_ownership_losses).mean() if proto_ownership_losses else reg_base,
            "loss_proto_usage": torch.stack(proto_usage_losses).mean() if proto_usage_losses else reg_base,
            # "loss_proto_compact": torch.stack(proto_compact_losses).mean() if proto_compact_losses else reg_base,
            # "loss_lowconf_flat": torch.stack(lowconf_flat_terms).mean() if lowconf_flat_terms else reg_base,
            # "loss_lowconf_coarse": torch.stack(lowconf_coarse_terms).mean() if lowconf_coarse_terms else reg_base,
            "selected_before_fusion": reg_base.new_tensor(selected_before_fusion).mean() if selected_before_fusion else reg_base,
            "selected_after_fusion": reg_base.new_tensor(selected_after_fusion).mean() if selected_after_fusion else reg_base,
            "teacher_ratio": reg_base.new_tensor(float(teacher_ratio)),
        }

        return dict(
            gaussians=gaussians,
            camera_poses=camera_poses,
            local_points=local_pts,
            intrinsics=K,
            conf=conf_logits,
            regularizers=regularizers,
            selected_gaussians=regularizers["selected_after_fusion"],
        )

