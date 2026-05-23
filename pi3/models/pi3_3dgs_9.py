# Pi3_3DGS_9 第二部分去重（局部竞争）实现简述：
# 1. forward 中每个 batch 的高斯候选合并后，调用 _apply_local_competition。
# 2. 先估计每个高斯的局部尺度：优先用相机位姿、内参和可见像素足迹计算
#    cube_size，并转成 local_radius；不可见或信息不足时退回由预测 scale 得到
#    的 fallback radius。这个尺度只决定 Reduced-3DGS 风格的局部球半径。
# 3. _estimate_color_aware_redundancy 统计局部冗余度：按 Reduced-3DGS 的
#    固定 K 近邻候选（默认 30）计算冗余，不再用 counts/view/hash 作为
#    local competition 的资格条件；唯一额外约束是颜色相近的近邻才计入冗余。
# 4. 将 redundancy_score 通过 mean + lambda * std（且不低于 redundancy_minimum）
#    得到阈值，再归一化成 redundancy_coef。只有 redundancy_coef > 0 的高斯
#    才会进入透明度压制。
# 5. local competition 后半段保持软抑制：opacity *= gate；不做 Reduced-3DGS
#    的物理 prune，也不再做额外 top-k 截断。

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
        fx * H, zeros, cx_t,
        zeros, fy * W, cy_t,
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


def modules_require_grad(modules):
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                return True
    return False


def approx_quantile_lastdim(x, q, max_samples=262144):
    """
    Compute a quantile on the last dimension with bounded memory.
    When the last dimension is too large, sample evenly spaced points first.
    """
    n = x.shape[-1]
    if n <= max_samples:
        return torch.quantile(x, q, dim=-1)

    if max_samples <= 1:
        index = torch.zeros(1, device=x.device, dtype=torch.long)
    else:
        base = torch.arange(max_samples, device=x.device, dtype=torch.long)
        index = (base * (n - 1)) // (max_samples - 1)
    sampled = x.index_select(-1, index)
    return torch.quantile(sampled, q, dim=-1)


def approx_quantile_flat(x, q, max_samples=262144):
    flat = x.reshape(-1)
    if flat.numel() <= max_samples:
        return torch.quantile(flat, q)

    if max_samples <= 1:
        index = torch.zeros(1, device=flat.device, dtype=torch.long)
    else:
        base = torch.arange(max_samples, device=flat.device, dtype=torch.long)
        index = (base * (flat.numel() - 1)) // (max_samples - 1)
    sampled = flat.index_select(0, index)
    return torch.quantile(sampled, q)


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


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None,
                             device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype,
                       device=device)
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
            K_input=1000000,
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
            low_conf_push_max_y_ratio=0.5,
            sparse_scale_voxel_size_ratio=0.03,
            sparse_scale_boost_max=1.8,
            sparse_scale_density_tau=4.0,
            sparse_scale_conf_threshold=0.1,
            scale_min_ratio=5e-5,
            scale_max_ratio=0.8,
            gs_view_stride=1,
            gs_decoder_view_chunk_size=24,
            enable_quadtree=True,
            enable_redundancy_pruning=True,
            redundancy_voxel_size_ratio=0.002,
            redundancy_radius_scale=1.0,
            redundancy_lambda_mercy=0.1,
            redundancy_minimum=3,
            redundancy_opacity_quantile=0.5,
            redundancy_max_prune_ratio=0.5,
            redundancy_min_keep=256,
            redundancy_conf_weight=0.0,
            redundancy_exact_max_points=4096,
            redundancy_use_color=True,
            redundancy_color_threshold=0.08,
            redundancy_color_bins=16,
            redundancy_soft_suppression=True,
            redundancy_suppression_gamma=1.5,
            redundancy_suppression_max=0.85,
            redundancy_neighbor_limit=30,
            # Reduced-3DGS style pixel-footprint neighborhood.
            # pixel_scale is the same role as args.box_size in reduced-3dgs:
            # local cube side = projected_pixel_footprint * pixel_scale,
            # local sphere radius = cube_side * sqrt(3) / 2.
            redundancy_use_pixel_footprint=True,
            redundancy_pixel_scale=1.0,
            redundancy_pixel_footprint_min=1e-5,
            redundancy_pixel_footprint_max_ratio=0.05,
            use_input_intrinsics=False,
            enable_learnable_sampling=True,
            learned_sampling_extra_ratio=0.10,
            learned_sampling_min_extra=128,
            density_gate_temperature=1.0,
            density_gate_min_prob=0.05,
            density_gate_opacity_power=0.5,
            scale_bias_strength=0.5,
            opacity_filter_threshold=0.02,
            enable_local_competition=True,
            competition_voxel_size_ratio=0.002,
            competition_radius_scale=1.5,
            competition_color_bins=16,
            competition_temperature=0.8,
            competition_target_power=0.5,
            competition_strength=0.8,
            competition_min_gate=0.05,
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
        self.low_conf_push_max_y_ratio = min(max(float(low_conf_push_max_y_ratio), 0.0), 1.0)
        self.sparse_scale_voxel_size_ratio = float(sparse_scale_voxel_size_ratio)
        self.sparse_scale_boost_max = max(1.0, float(sparse_scale_boost_max))
        self.sparse_scale_density_tau = max(float(sparse_scale_density_tau), 1e-6)
        self.sparse_scale_conf_threshold = min(max(float(sparse_scale_conf_threshold), 0.0), 1.0)
        self.scale_min_ratio = float(scale_min_ratio)
        self.scale_max_ratio = max(float(scale_max_ratio), float(scale_min_ratio) * 1.1)
        self.gs_view_stride = max(1, int(gs_view_stride))
        self.gs_decoder_view_chunk_size = max(1, int(gs_decoder_view_chunk_size))
        self.enable_quadtree = bool(enable_quadtree)
        self.enable_redundancy_pruning = bool(enable_redundancy_pruning)
        self.redundancy_voxel_size_ratio = float(redundancy_voxel_size_ratio)
        self.redundancy_radius_scale = max(float(redundancy_radius_scale), 0.0)
        self.redundancy_lambda_mercy = float(redundancy_lambda_mercy)
        self.redundancy_minimum = max(1, int(redundancy_minimum))
        self.redundancy_opacity_quantile = min(max(float(redundancy_opacity_quantile), 0.0), 1.0)
        self.redundancy_max_prune_ratio = min(max(float(redundancy_max_prune_ratio), 0.0), 1.0)
        self.redundancy_min_keep = max(1, int(redundancy_min_keep))
        self.redundancy_conf_weight = max(0.0, float(redundancy_conf_weight))
        self.redundancy_exact_max_points = max(0, int(redundancy_exact_max_points))
        self.redundancy_use_color = bool(redundancy_use_color)
        self.redundancy_color_threshold = max(0.0, float(redundancy_color_threshold))
        self.redundancy_color_bins = max(2, int(redundancy_color_bins))
        self.redundancy_soft_suppression = bool(redundancy_soft_suppression)
        self.redundancy_suppression_gamma = max(0.0, float(redundancy_suppression_gamma))
        self.redundancy_suppression_max = min(max(float(redundancy_suppression_max), 0.0), 0.99)
        self.redundancy_neighbor_limit = max(0, int(redundancy_neighbor_limit))
        self.redundancy_use_pixel_footprint = bool(redundancy_use_pixel_footprint)
        self.redundancy_pixel_scale = max(float(redundancy_pixel_scale), 1e-6)
        self.redundancy_pixel_footprint_min = max(float(redundancy_pixel_footprint_min), 1e-8)
        self.redundancy_pixel_footprint_max_ratio = max(float(redundancy_pixel_footprint_max_ratio), 0.0)
        self.use_input_intrinsics = bool(use_input_intrinsics)
        self.enable_learnable_sampling = bool(enable_learnable_sampling)
        self.learned_sampling_extra_ratio = max(0.0, float(learned_sampling_extra_ratio))
        self.learned_sampling_min_extra = max(0, int(learned_sampling_min_extra))
        self.density_gate_temperature = max(float(density_gate_temperature), 1e-4)
        self.density_gate_min_prob = min(max(float(density_gate_min_prob), 0.0), 1.0)
        self.density_gate_opacity_power = max(0.0, float(density_gate_opacity_power))
        self.scale_bias_strength = max(0.0, float(scale_bias_strength))
        self.opacity_filter_threshold = min(max(float(opacity_filter_threshold), 0.0), 1.0)
        self.enable_local_competition = bool(enable_local_competition)
        self.competition_voxel_size_ratio = max(0.0, float(competition_voxel_size_ratio))
        self.competition_radius_scale = max(0.0, float(competition_radius_scale))
        self.competition_color_bins = max(2, int(competition_color_bins))
        self.competition_temperature = max(float(competition_temperature), 1e-4)
        self.competition_target_power = min(max(float(competition_target_power), 0.0), 1.0)
        self.competition_strength = min(max(float(competition_strength), 0.0), 1.0)
        self.competition_min_gate = min(max(float(competition_min_gate), 0.0), 1.0)

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
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
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
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
            out_dim=512,
            rope=self.rope,
        )
        self.camera_head = CameraHead(dim=512)

        # 同样使用 ConvPts3dHead 预测单个维度的置信度
        self.conf_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
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
        self.gs_decoder = TransformerDecoder(in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16,
                                             out_dim=1024, rope=self.rope)
        # 13 dims: rotation(4), scale(3), opacity(1), color(3), density_logit(1), scale_bias(1).
        self.gs_head = ConvDenseGaussianHead(patch_size=14, dec_embed_dim=1024, dim_out=[4, 3, 1, 3, 1, 1])

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

        if ckpt is not None:
            if ckpt.endswith(".safetensors"):
                checkpoint = load_file(ckpt, device="cpu")
            else:
                checkpoint = torch.load(ckpt, map_location="cpu")

            res = self._load_state_dict_flexible(checkpoint)
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
        elif self.train_stage == 3:  # Conf only
            freeze_all_params([
                self.decoder, self.camera_decoder, self.camera_head
            ])
        elif self.train_stage == 1:
            freeze_all_params([self.decoder])
            freeze_all_params([self.camera_decoder, self.camera_head])
            freeze_all_params([self.conf_decoder, self.conf_head])
            freeze_all_params([self.point_decoder, self.point_head])
            # pass

    def _load_state_dict_flexible(self, checkpoint):
        current_state = self.state_dict()
        compatible = {}
        skipped = []
        old_residual_gs_head = any(
            key.startswith("gs_head.conv_head.output_block.6.")
            for key in checkpoint.keys()
        )

        for key, value in checkpoint.items():
            target_key = key
            if old_residual_gs_head and key.startswith("gs_head.conv_head.output_block."):
                key_parts = key.split(".")
                if len(key_parts) > 3 and key_parts[3].isdigit():
                    output_idx = int(key_parts[3])
                    if output_idx == 4:
                        skipped.append((key, tuple(value.shape), "removed_xyz_residual"))
                        continue
                    if output_idx > 4:
                        key_parts[3] = str(output_idx - 1)
                        target_key = ".".join(key_parts)

            if target_key in current_state and current_state[target_key].shape == value.shape:
                compatible[target_key] = value
            else:
                target_shape = tuple(current_state[target_key].shape) if target_key in current_state else None
                skipped.append((key, tuple(value.shape), target_shape))

        res = self.load_state_dict(compatible, strict=False)
        if skipped:
            preview = ", ".join(
                f"{key}: {src}->{dst}" for key, src, dst in skipped[:8]
            )
            suffix = "" if len(skipped) <= 8 else f", ... ({len(skipped)} skipped total)"
            print(f"[Pi3_3DGS_7] Skipped incompatible checkpoint tensors: {preview}{suffix}")
        return res

    def _stats_tensor(self, value, device, dtype):
        return torch.tensor(float(value), device=device, dtype=dtype)

    def _build_learned_support_mask(self, density_logits, proposal_mask):
        if not self.enable_learnable_sampling:
            return torch.zeros_like(proposal_mask, dtype=torch.bool)

        with torch.no_grad():
            proposal_mask = proposal_mask.bool()
            candidate_mask = ~proposal_mask
            candidate_count = int(candidate_mask.sum().item())
            if candidate_count <= 0:
                return torch.zeros_like(proposal_mask, dtype=torch.bool)

            proposal_count = int(proposal_mask.sum().item())
            extra_count = int(round(max(proposal_count, 1) * self.learned_sampling_extra_ratio))
            extra_count = max(extra_count, self.learned_sampling_min_extra)
            extra_count = min(extra_count, candidate_count)
            if extra_count <= 0:
                return torch.zeros_like(proposal_mask, dtype=torch.bool)

            density_score = torch.sigmoid(density_logits.detach() / self.density_gate_temperature)
            candidate_score = density_score.masked_fill(~candidate_mask, -float("inf")).reshape(-1)
            top_idx = torch.topk(candidate_score, k=extra_count, largest=True).indices
            learned_mask = torch.zeros_like(candidate_score, dtype=torch.bool)
            learned_mask[top_idx] = True
            return learned_mask.reshape_as(proposal_mask)

    def _estimate_scale_based_radius(self, gaussian_dict, scene_size_scalar, radius_scale=None):
        """Fallback local radius used when pixel-footprint estimation is unavailable.

        This keeps the old behavior: local radius is derived from predicted
        Gaussian scale, with a scene-size fallback for invalid scales.
        """
        xyz = gaussian_dict["xyz"]
        device = xyz.device
        scale_mean = gaussian_dict["scale"].detach().float().mean(dim=-1)
        finite_scale = torch.isfinite(scale_mean) & (scale_mean > 0)
        if radius_scale is None:
            radius_scale = self.competition_radius_scale
        radius_scale = max(float(radius_scale), 0.0)

        if finite_scale.any() and radius_scale > 0:
            radius = scale_mean.clamp_min(1e-5) * radius_scale
            fallback_value = float(radius[finite_scale].mean().clamp_min(1e-5).item())
            radius = torch.where(
                finite_scale,
                radius,
                torch.full_like(radius, fallback_value),
            )
        else:
            fallback_value = max(
                float(scene_size_scalar.item()) * max(self.competition_voxel_size_ratio, 1e-8),
                1e-5,
            )
            radius = torch.full((xyz.shape[0],), fallback_value, device=device, dtype=torch.float32)
        return radius.clamp_min(1e-5)

    def _estimate_projected_pixel_cube_size(
            self,
            gaussian_dict,
            camera_poses=None,
            intrinsics=None,
            image_hw=None,
            scene_size_scalar=None,
    ):
        """Approximate Reduced-3DGS per-Gaussian cube_size.

        Reduced-3DGS computes the smallest world-space length corresponding to
        one image pixel for each primitive over all visible training cameras.
        In this forward-feed setting we approximate the same quantity from the
        predicted camera-to-world matrices and intrinsics:

            cube_size_i = min_visible_c min(z_i^c / fx_c, z_i^c / fy_c)

        This returns both the cube side length and the local sphere radius used
        for redundancy counting:

            radius_i = cube_size_i * redundancy_pixel_scale * sqrt(3) / 2

        If a Gaussian is not visible in any available camera, we fall back to
        the previous scale-based radius so the method remains robust.
        """
        xyz = gaussian_dict["xyz"]
        K = xyz.shape[0]
        device = xyz.device
        dtype = xyz.dtype
        if scene_size_scalar is None:
            scene_size_scalar = torch.tensor(1.0, device=device, dtype=dtype)

        fallback_radius = self._estimate_scale_based_radius(
            gaussian_dict,
            scene_size_scalar.to(device=device, dtype=dtype),
            radius_scale=self.competition_radius_scale,
        )
        fallback_cube = fallback_radius / (self.redundancy_pixel_scale * math.sqrt(3.0) / 2.0)

        if (
                (not self.redundancy_use_pixel_footprint)
                or camera_poses is None
                or intrinsics is None
                or image_hw is None
                or K == 0
        ):
            return fallback_cube.clamp_min(self.redundancy_pixel_footprint_min), fallback_radius

        H, W = int(image_hw[0]), int(image_hw[1])
        cams = camera_poses.detach().to(device=device, dtype=torch.float32)
        Ks = intrinsics.detach().to(device=device, dtype=torch.float32)
        if cams.ndim != 3 or cams.shape[-2:] != (4, 4) or Ks.ndim != 3 or Ks.shape[-2:] != (3, 3):
            return fallback_cube.clamp_min(self.redundancy_pixel_footprint_min), fallback_radius

        M = min(cams.shape[0], Ks.shape[0])
        if M <= 0:
            return fallback_cube.clamp_min(self.redundancy_pixel_footprint_min), fallback_radius
        cams = cams[:M]
        Ks = Ks[:M]

        xyz_f = xyz.detach().float()
        # camera_poses are used elsewhere as camera-to-world:
        #   x_world = R_c2w x_cam + t_c2w
        # Therefore x_cam = R_c2w^T (x_world - t_c2w).
        R = cams[:, :3, :3]
        t = cams[:, :3, 3]
        pts_cam = torch.matmul(xyz_f.unsqueeze(0) - t[:, None, :], R)  # [M, K, 3]
        z = pts_cam[..., 2]

        fx = Ks[:, 0, 0].abs().clamp_min(1e-6).view(M, 1)
        fy = Ks[:, 1, 1].abs().clamp_min(1e-6).view(M, 1)
        cx = Ks[:, 0, 2].view(M, 1)
        cy = Ks[:, 1, 2].view(M, 1)

        valid_z = z > 1e-6
        safe_z = z.clamp_min(1e-6)
        u = pts_cam[..., 0] / safe_z * fx + cx
        v = pts_cam[..., 1] / safe_z * fy + cy
        visible = valid_z & (u >= 0.0) & (u <= float(W - 1)) & (v >= 0.0) & (v <= float(H - 1))

        # World-space distance corresponding to one pixel at this depth. Since
        # camera rotations preserve length, z/fx and z/fy are also world lengths.
        pixel_world_x = safe_z / fx
        pixel_world_y = safe_z / fy
        pixel_world = torch.minimum(pixel_world_x, pixel_world_y)
        pixel_world = pixel_world.masked_fill(~visible, float("inf"))
        cube_size = pixel_world.min(dim=0).values

        has_visible = torch.isfinite(cube_size)
        cube_size = torch.where(has_visible, cube_size, fallback_cube)
        cube_size = cube_size.clamp_min(self.redundancy_pixel_footprint_min)

        if self.redundancy_pixel_footprint_max_ratio > 0:
            max_cube = float(scene_size_scalar.detach().float().item()) * self.redundancy_pixel_footprint_max_ratio
            max_cube = max(max_cube, self.redundancy_pixel_footprint_min)
            cube_size = cube_size.clamp_max(max_cube)

        radius = cube_size * self.redundancy_pixel_scale * math.sqrt(3.0) / 2.0
        radius = radius.clamp_min(self.redundancy_pixel_footprint_min)
        return cube_size.to(device=device), radius.to(device=device)

    def _local_radius_summary(self, local_radius, fallback_radius):
        """Return a scalar summary of the Reduced-3DGS local radius.

        Redundancy itself uses each Gaussian's own radius whenever available.
        The returned scalar is only a diagnostic/fallback scale, not a grouping
        or filtering hash size.
        """
        if local_radius is None or local_radius.numel() == 0:
            return float(fallback_radius.detach().float().mean().clamp_min(1e-5).item())
        finite = torch.isfinite(local_radius) & (local_radius > 0)
        if finite.any():
            return float(local_radius[finite].detach().float().median().clamp_min(1e-5).item())
        return float(fallback_radius.detach().float().mean().clamp_min(1e-5).item())

    def _apply_local_competition(self, gaussian_dict, source_view, scene_size, camera_poses=None, intrinsics=None, image_hw=None):
        """Reduced-3DGS-style redundancy detection plus soft opacity suppression.

        Redundant Gaussians are selected from the whole Gaussian set by the
        global redundancy score threshold. No source-view, repeated-count, or
        spatial/color hash group is used as an additional eligibility condition.
        The only deviation from Reduced-3DGS is that neighbor intersections must
        also pass the optional color-similarity check.
        """
        xyz = gaussian_dict["xyz"]
        K = xyz.shape[0]
        device, dtype = xyz.device, xyz.dtype

        def add_effective_stats(stats, opacity, before_count=K):
            opacity_flat = opacity.detach().float().reshape(-1)
            active_002 = (opacity_flat > 0.02).float().sum().to(device=device, dtype=dtype)
            active_005 = (opacity_flat > 0.05).float().sum().to(device=device, dtype=dtype)
            stats["physical_count"] = self._stats_tensor(opacity_flat.numel(), device, dtype)
            stats["active_count_opacity_002"] = active_002
            stats["active_count_opacity_005"] = active_005
            stats["opacity_mass"] = opacity_flat.sum().to(device=device, dtype=dtype)
            stats["count_after"] = active_005
            stats["pruned"] = (self._stats_tensor(before_count, device, dtype) - active_005).clamp_min(0.0)
            return stats

        def add_competition_stats(
                stats,
                gate_mean=1.0,
                groups=0,
                candidates=0,
                expected_suppressed=0.0,
                redundancy_threshold=0.0,
                redundancy_coef_mean=0.0,
                redundancy_coef_max=0.0,
                hard_cap_pruned=0,
        ):
            stats["competition_gate_mean"] = self._stats_tensor(gate_mean, device, dtype)
            stats["competition_groups"] = self._stats_tensor(groups, device, dtype)
            stats["competition_candidates"] = self._stats_tensor(candidates, device, dtype)
            stats["competition_expected_suppressed"] = self._stats_tensor(expected_suppressed, device, dtype)
            # Redundancy-coefficient diagnostics. These are produced here because
            # redundancy is now the criterion that decides whether local
            # competition should be activated.
            stats["redundancy_threshold"] = self._stats_tensor(redundancy_threshold, device, dtype)
            stats["redundancy_candidates"] = self._stats_tensor(candidates, device, dtype)
            stats["redundancy_coef_mean"] = self._stats_tensor(redundancy_coef_mean, device, dtype)
            stats["redundancy_coef_max"] = self._stats_tensor(redundancy_coef_max, device, dtype)
            stats["redundancy_gate_mean"] = self._stats_tensor(gate_mean, device, dtype)
            stats["redundancy_expected_suppressed"] = self._stats_tensor(expected_suppressed, device, dtype)
            stats["hard_cap_pruned"] = self._stats_tensor(hard_cap_pruned, device, dtype)
            return stats

        def attach_competition_diagnostics(
                gaussian_dict,
                redundancy=None,
                redundancy_coef=None,
                gate=None,
                active_mask=None,
        ):
            gaussian_dict = dict(gaussian_dict)
            k = gaussian_dict["xyz"].shape[0]
            if redundancy is None:
                redundancy = torch.zeros((k,), device=device, dtype=torch.float32)
            if redundancy_coef is None:
                redundancy_coef = torch.zeros((k,), device=device, dtype=torch.float32)
            if gate is None:
                gate = torch.ones((k,), device=device, dtype=torch.float32)
            if active_mask is None:
                active_mask = torch.zeros((k,), device=device, dtype=torch.bool)

            gaussian_dict["redundancy_score"] = redundancy.detach().to(device=device, dtype=dtype).reshape(k, 1)
            gaussian_dict["redundancy_coef"] = redundancy_coef.detach().to(device=device, dtype=dtype).reshape(k, 1)
            gaussian_dict["competition_gate"] = gate.detach().to(device=device, dtype=dtype).reshape(k, 1)
            gaussian_dict["competition_active"] = active_mask.detach().to(device=device, dtype=dtype).reshape(k, 1)
            return gaussian_dict

        if K == 0:
            stats = self._redundancy_stats(device, dtype, 0)
            stats = add_competition_stats(stats)
            stats = add_effective_stats(stats, gaussian_dict["opacity"])
            gaussian_dict = attach_competition_diagnostics(gaussian_dict)
            return gaussian_dict, stats

        if not self.enable_local_competition or K <= 1:
            stats = self._redundancy_stats(device, dtype, K)
            stats = add_competition_stats(stats)
            stats = add_effective_stats(stats, gaussian_dict["opacity"])
            gaussian_dict = attach_competition_diagnostics(gaussian_dict)
            return gaussian_dict, stats

        with torch.no_grad():
            scene_size_scalar = self._normalize_scene_size(scene_size, 1, device, dtype)[0]
            fallback_radius = self._estimate_scale_based_radius(
                gaussian_dict,
                scene_size_scalar,
                radius_scale=self.competition_radius_scale,
            )
            cube_size, local_radius = self._estimate_projected_pixel_cube_size(
                gaussian_dict,
                camera_poses=camera_poses,
                intrinsics=intrinsics,
                image_hw=image_hw,
                scene_size_scalar=scene_size_scalar,
            )
            # Reduced-3DGS uses cube_size * pixel_scale as the local cube side
            # and half-diagonal as the sphere radius. The scalar below is only
            # reported for diagnostics; redundancy uses local_radius_i directly.
            neighborhood_size = self._local_radius_summary(local_radius, fallback_radius)

            # 1) Reduced-3DGS-style redundancy score, with color similarity gate.
            #    This score is the actual criterion for deciding whether a
            #    Gaussian is redundant enough to participate in competition.
            redundancy = self._estimate_color_aware_redundancy(
                gaussian_dict,
                neighborhood_size,
                local_radius=local_radius,
            )
            mean_redundancy = redundancy.mean()
            std_redundancy = redundancy.std(unbiased=False)
            redundancy_threshold = torch.clamp(
                mean_redundancy + self.redundancy_lambda_mercy * std_redundancy,
                min=float(self.redundancy_minimum),
            )
            max_redundancy = redundancy.max()
            redundancy_coef = (redundancy - redundancy_threshold) / (max_redundancy - redundancy_threshold + 1e-6)
            redundancy_coef = torch.clamp(redundancy_coef, 0.0, 1.0)

            # 2) Match Reduced-3DGS candidate selection: redundant primitives are
            #    exactly those above the global mercy threshold. No counts/view/
            #    hash group can veto this mask.
            active_mask = redundancy > redundancy_threshold
            active_count = int(active_mask.sum().item())
            active_group_count = 0
            redundancy_coef_mean = float(redundancy_coef.mean().item())
            redundancy_coef_max = float(redundancy_coef.max().item())

        if active_count == 0:
            stats = self._redundancy_stats(
                device,
                dtype,
                K,
                threshold=float(redundancy_threshold.item()),
                voxel_size=neighborhood_size,
                mean_redundancy=float(mean_redundancy.item()),
            )
            stats = add_competition_stats(
                stats,
                redundancy_threshold=float(redundancy_threshold.item()),
                redundancy_coef_mean=redundancy_coef_mean,
                redundancy_coef_max=redundancy_coef_max,
            )
            stats["pixel_scale"] = self._stats_tensor(self.redundancy_pixel_scale, device, dtype)
            stats["cube_size_mean"] = cube_size.detach().float().mean().to(device=device, dtype=dtype)
            stats["cube_size_median"] = cube_size.detach().float().median().to(device=device, dtype=dtype)
            stats["local_radius_mean"] = local_radius.detach().float().mean().to(device=device, dtype=dtype)
            stats["local_radius_median"] = local_radius.detach().float().median().to(device=device, dtype=dtype)
            stats = add_effective_stats(stats, gaussian_dict["opacity"])
            gaussian_dict = attach_competition_diagnostics(
                gaussian_dict,
                redundancy=redundancy,
                redundancy_coef=redundancy_coef,
            )
            return gaussian_dict, stats

        # Keep the second half as soft opacity suppression, but drive it only by
        # the Reduced-style redundant set and coefficient. With no local hash
        # group, the strongest redundant point maps to the same floor that the
        # old grouped gate could reach.
        redundancy_weight = redundancy_coef.to(device=device, dtype=torch.float32)
        gate_floor = torch.tensor(float(self.competition_min_gate), device=device, dtype=torch.float32)
        gate = 1.0 - self.competition_strength * redundancy_weight * (1.0 - gate_floor)
        gate = torch.where(active_mask, gate, torch.ones_like(gate))
        gate = gate.clamp(min=self.competition_min_gate, max=1.0)

        gaussian_dict = dict(gaussian_dict)
        gaussian_dict["opacity"] = gaussian_dict["opacity"] * gate.to(dtype=dtype).unsqueeze(-1)
        gaussian_dict = attach_competition_diagnostics(
            gaussian_dict,
            redundancy=redundancy,
            redundancy_coef=redundancy_coef,
            gate=gate,
            active_mask=active_mask,
        )

        # Keep local competition as a soft opacity gate only. Dense/no-quadtree
        # ablations should not be physically top-k truncated after this point.
        hard_cap_pruned = 0

        expected_suppressed = (1.0 - gate.detach().float()).sum()
        stats = self._redundancy_stats(
            device,
            dtype,
            K,
            after_count=gaussian_dict["xyz"].shape[0],
            pruned=float(expected_suppressed.item()),
            candidates=active_count,
            threshold=float(redundancy_threshold.item()),
            voxel_size=neighborhood_size,
            mean_redundancy=float(mean_redundancy.item()),
        )
        stats = add_competition_stats(
            stats,
            gate_mean=float(gate.detach().float().mean().item()),
            groups=active_group_count,
            candidates=active_count,
            expected_suppressed=float(expected_suppressed.item()),
            redundancy_threshold=float(redundancy_threshold.item()),
            redundancy_coef_mean=redundancy_coef_mean,
            redundancy_coef_max=redundancy_coef_max,
            hard_cap_pruned=hard_cap_pruned,
        )
        stats["pixel_scale"] = self._stats_tensor(self.redundancy_pixel_scale, device, dtype)
        stats["cube_size_mean"] = cube_size.detach().float().mean().to(device=device, dtype=dtype)
        stats["cube_size_median"] = cube_size.detach().float().median().to(device=device, dtype=dtype)
        stats["local_radius_mean"] = local_radius.detach().float().mean().to(device=device, dtype=dtype)
        stats["local_radius_median"] = local_radius.detach().float().median().to(device=device, dtype=dtype)
        stats = add_effective_stats(stats, gaussian_dict["opacity"], before_count=K)
        return gaussian_dict, stats

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

    def _normalize_scene_size(self, scene_size, B, device, dtype):
        if not isinstance(scene_size, torch.Tensor):
            scene_size_per_batch = torch.full((B,), float(scene_size), device=device, dtype=dtype)
        else:
            if scene_size.ndim == 0:
                scene_size_per_batch = scene_size.view(1).expand(B)
            else:
                scene_size_per_batch = scene_size.reshape(-1)
                if scene_size_per_batch.numel() == 1:
                    scene_size_per_batch = scene_size_per_batch.expand(B)
                elif scene_size_per_batch.numel() != B:
                    scene_size_per_batch = scene_size_per_batch.mean().view(1).expand(B)

            scene_size_per_batch = scene_size_per_batch.to(device=device, dtype=dtype)

        return scene_size_per_batch.clamp_min(1e-6)

    def _redundancy_stats(self, device, dtype, before_count, after_count=None, pruned=0,
                          candidates=0, threshold=0.0, voxel_size=0.0, mean_redundancy=0.0):
        if after_count is None:
            after_count = before_count
        return {
            "count_before": torch.tensor(float(before_count), device=device, dtype=dtype),
            "count_after": torch.tensor(float(after_count), device=device, dtype=dtype),
            "pruned": torch.tensor(float(pruned), device=device, dtype=dtype),
            "candidates": torch.tensor(float(candidates), device=device, dtype=dtype),
            "threshold": torch.tensor(float(threshold), device=device, dtype=dtype),
            "voxel_size": torch.tensor(float(voxel_size), device=device, dtype=dtype),
            "mean_redundancy": torch.tensor(float(mean_redundancy), device=device, dtype=dtype),
        }

    def _get_redundancy_color(self, gaussian_dict):
        """Return the color used for redundancy grouping.

        Prefer image-space color sampled from the input view (competition_color),
        because it is less affected by the Gaussian color head at early training.
        Fall back to predicted Gaussian RGB when competition_color is unavailable.
        """
        color = gaussian_dict.get("competition_color", None)
        if color is None:
            color = gaussian_dict.get("color", None)
        if color is None:
            return None
        return color.detach().float().clamp(0.0, 1.0)

    def _cap_redundancy_count(self, redundancy):
        """Match reduced-3dgs' bounded-neighbour redundancy score scale.

        The official implementation evaluates a fixed K-neighbour candidate set
        (30 by default) and then adds the center primitive itself, so the raw
        score cannot grow with local voxel density. Our hash path is only an
        approximation, but it should keep the same numerical range.
        """
        if self.redundancy_neighbor_limit <= 0:
            return redundancy
        return redundancy.clamp_max(float(self.redundancy_neighbor_limit + 1))

    def _rotate_by_inverse_quaternion(self, vectors, quaternions):
        """Rotate vectors by the inverse of normalized wxyz quaternions."""
        q = F.normalize(quaternions.detach().float(), dim=-1)
        q_vec = q[..., 1:4]
        q_w = q[..., 0:1]
        uv = torch.cross(q_vec, vectors, dim=-1)
        uuv = torch.cross(q_vec, uv, dim=-1)
        return vectors - 2.0 * q_w * uv + 2.0 * uuv

    def _estimate_color_aware_redundancy(self, gaussian_dict, voxel_size, local_radius=None):
        """Estimate Reduced-3DGS-style local redundancy with an extra color gate.

        Reduced-3DGS first finds a fixed number of nearest spatial neighbours,
        tests whether the local sphere intersects each neighbour ellipsoid, adds
        the center primitive itself, and then propagates the minimum redundancy
        value to intersecting primitives. This implementation follows that flow
        and only adds a color-similarity check to the intersection mask.
        """
        xyz = gaussian_dict["xyz"]
        K = xyz.shape[0]
        device = xyz.device
        pts = xyz.detach().float()
        if K == 0:
            return torch.zeros((0,), device=device, dtype=torch.float32)
        if K == 1:
            return torch.ones((1,), device=device, dtype=torch.float32)

        color = self._get_redundancy_color(gaussian_dict)
        use_color = (
                self.redundancy_use_color
                and color is not None
                and self.redundancy_color_threshold > 0
        )
        scale = gaussian_dict["scale"].detach().float().clamp_min(1e-8)
        rotation = gaussian_dict.get("rotation", None)
        if rotation is not None:
            rotation = rotation.detach().float()

        if local_radius is not None:
            radius = local_radius.detach().float().to(device=device).view(-1).clamp_min(1e-8)
        else:
            radius = torch.full((K,), float(max(voxel_size, 1e-8)), device=device, dtype=torch.float32)

        num_neighbours = int(self.redundancy_neighbor_limit) if self.redundancy_neighbor_limit > 0 else 30
        num_neighbours = min(max(num_neighbours, 1), K - 1)
        raw_redundancy = torch.ones((K,), device=device, dtype=torch.float32)
        min_redundancy = torch.full((K,), float("inf"), device=device, dtype=torch.float32)

        def process_neighbour_indices(center_idx, neighbour_idx):
            if neighbour_idx.numel() == 0:
                return
            center_xyz = pts.index_select(0, center_idx)[:, None, :]
            neighbour_xyz = pts[neighbour_idx]
            delta = center_xyz - neighbour_xyz
            center_radius = radius.index_select(0, center_idx).view(-1, 1, 1)

            neighbour_scale = scale[neighbour_idx]
            if rotation is not None:
                neighbour_rotation = rotation[neighbour_idx]
                local_delta = self._rotate_by_inverse_quaternion(delta, neighbour_rotation)
                expanded_scale = neighbour_scale + center_radius
                intersection_mask = ((local_delta / expanded_scale).square().sum(dim=-1) <= 1.0)
            else:
                neighbour_radius = neighbour_scale.amax(dim=-1)
                distance = torch.linalg.norm(delta, dim=-1)
                intersection_mask = distance <= (center_radius.squeeze(-1) + neighbour_radius)

            if use_color:
                center_color = color.index_select(0, center_idx)[:, None, :]
                neighbour_color = color[neighbour_idx]
                color_mask = torch.linalg.norm(center_color - neighbour_color, dim=-1) <= self.redundancy_color_threshold
                intersection_mask = intersection_mask & color_mask

            score = intersection_mask.float().sum(dim=-1) + 1.0
            raw_redundancy[center_idx] = score

            all_indices = torch.cat([center_idx.view(-1, 1), neighbour_idx], dim=1)
            all_mask = torch.cat(
                [
                    torch.ones((center_idx.numel(), 1), device=device, dtype=torch.bool),
                    intersection_mask,
                ],
                dim=1,
            )
            all_scores = score.view(-1, 1).expand_as(all_indices)
            min_redundancy.scatter_reduce_(
                0,
                all_indices[all_mask],
                all_scores[all_mask],
                reduce="amin",
                include_self=True,
            )

        process_chunk = max(1024, min(65536, 2_000_000 // max(num_neighbours, 1)))
        used_simple_knn = False
        try:
            from simple_knn._C import distIndex2
            if pts.is_cuda:
                _, all_indices = distIndex2(pts.contiguous(), num_neighbours)
                all_indices = all_indices.view(K, num_neighbours).to(device=device, dtype=torch.long)
                for start in range(0, K, process_chunk):
                    end = min(start + process_chunk, K)
                    center_idx = torch.arange(start, end, device=device, dtype=torch.long)
                    process_neighbour_indices(center_idx, all_indices[start:end])
                used_simple_knn = True
        except Exception:
            used_simple_knn = False

        if not used_simple_knn:
            try:
                from scipy.spatial import cKDTree
            except Exception as exc:
                raise RuntimeError(
                    "Pi3_3DGS_9 local competition needs simple_knn._C.distIndex2 "
                    "or scipy.spatial.cKDTree for Reduced-3DGS-style KNN redundancy."
                ) from exc

            pts_cpu = pts.detach().cpu().numpy().astype(np.float32, copy=False)
            tree = cKDTree(pts_cpu)
            query_k = num_neighbours + 1
            for start in range(0, K, process_chunk):
                end = min(start + process_chunk, K)
                query_pts = pts_cpu[start:end]
                try:
                    _, idx_np = tree.query(query_pts, k=query_k, workers=-1)
                except TypeError:
                    _, idx_np = tree.query(query_pts, k=query_k)
                idx_np = np.asarray(idx_np, dtype=np.int64)
                if idx_np.ndim == 1:
                    idx_np = idx_np.reshape(-1, query_k)

                center_np = np.arange(start, end, dtype=np.int64)[:, None]
                not_self = idx_np != center_np
                if np.all(not_self.sum(axis=1) >= num_neighbours):
                    idx_np = idx_np[not_self].reshape(end - start, -1)[:, :num_neighbours]
                else:
                    filtered = np.empty((end - start, num_neighbours), dtype=np.int64)
                    for row in range(end - start):
                        row_idx = idx_np[row][idx_np[row] != start + row]
                        if row_idx.shape[0] < num_neighbours:
                            row_idx = np.pad(row_idx, (0, num_neighbours - row_idx.shape[0]), mode="edge")
                        filtered[row] = row_idx[:num_neighbours]
                    idx_np = filtered

                neighbour_idx = torch.from_numpy(np.ascontiguousarray(idx_np)).to(device=device, dtype=torch.long)
                center_idx = torch.arange(start, end, device=device, dtype=torch.long)
                process_neighbour_indices(center_idx, neighbour_idx)

        redundancy = torch.where(torch.isfinite(min_redundancy), min_redundancy, raw_redundancy)
        return self._cap_redundancy_count(redundancy)

    def _finalize_redundancy_stats(self, stats, gaussian_dict, before_count, device, dtype):
        """Add effective-count statistics after opacity suppression."""
        opacity_flat = gaussian_dict["opacity"].detach().float().reshape(-1)
        active_002 = (opacity_flat > 0.02).float().sum().to(device=device, dtype=dtype)
        active_005 = (opacity_flat > 0.05).float().sum().to(device=device, dtype=dtype)
        stats["physical_count"] = self._stats_tensor(gaussian_dict["xyz"].shape[0], device, dtype)
        stats["active_count_opacity_002"] = active_002
        stats["active_count_opacity_005"] = active_005
        stats["opacity_mass"] = opacity_flat.sum().to(device=device, dtype=dtype)
        stats["count_after"] = active_005
        stats["pruned"] = (self._stats_tensor(before_count, device, dtype) - active_005).clamp_min(0.0)
        return stats

    def _apply_redundancy_mercy(self, gaussian_dict, scene_size):
        """Color-aware redundancy mercy with soft opacity suppression.

        Compared with the original hard-prune version, this keeps the
        Reduced-3DGS local-density framework but changes two things:

        1. Redundancy is counted only for spatially close Gaussians with similar
           color, so boundaries/texture changes are less likely to be treated as
           duplicates.
        2. High-redundancy Gaussians are not removed immediately. Their opacity
           is multiplied by a redundancy-dependent gate, so rendering loss can
           still recover useful Gaussians while truly ambiguous/repeated ones
           naturally fall below the effective opacity threshold.
        """
        xyz = gaussian_dict["xyz"]
        K = xyz.shape[0]
        device, dtype = xyz.device, xyz.dtype

        if K == 0:
            stats = self._redundancy_stats(device, dtype, 0)
            stats["redundancy_gate_mean"] = self._stats_tensor(1.0, device, dtype)
            stats["redundancy_expected_suppressed"] = self._stats_tensor(0.0, device, dtype)
            stats["redundancy_coef_mean"] = self._stats_tensor(0.0, device, dtype)
            stats["redundancy_coef_max"] = self._stats_tensor(0.0, device, dtype)
            stats["hard_cap_pruned"] = self._stats_tensor(0.0, device, dtype)
            return gaussian_dict, stats

        if not self.enable_redundancy_pruning or K <= self.redundancy_minimum:
            stats = self._redundancy_stats(device, dtype, K)
            stats["redundancy_gate_mean"] = self._stats_tensor(1.0, device, dtype)
            stats["redundancy_expected_suppressed"] = self._stats_tensor(0.0, device, dtype)
            stats["redundancy_coef_mean"] = self._stats_tensor(0.0, device, dtype)
            stats["redundancy_coef_max"] = self._stats_tensor(0.0, device, dtype)
            stats["hard_cap_pruned"] = self._stats_tensor(0.0, device, dtype)
            stats = self._finalize_redundancy_stats(stats, gaussian_dict, K, device, dtype)
            return gaussian_dict, stats

        with torch.no_grad():
            scene_size_scalar = self._normalize_scene_size(scene_size, 1, device, dtype)[0]
            scale_mean = gaussian_dict["scale"].detach().float().mean(dim=-1)
            finite_scale = torch.isfinite(scale_mean) & (scale_mean > 0)
            if finite_scale.any() and self.redundancy_radius_scale > 0:
                average_scale = scale_mean[finite_scale].mean()
                voxel_size = float((average_scale * self.redundancy_radius_scale).clamp_min(1e-5).item())
            else:
                voxel_ratio = self.redundancy_voxel_size_ratio
                if voxel_ratio <= 0:
                    voxel_ratio = self.overlap_voxel_size_ratio
                voxel_size = max(float(scene_size_scalar.item()) * max(voxel_ratio, 1e-8), 1e-5)

            redundancy = self._estimate_color_aware_redundancy(gaussian_dict, voxel_size)
            mean_redundancy = redundancy.mean()
            std_redundancy = redundancy.std(unbiased=False)
            threshold = torch.clamp(
                mean_redundancy + self.redundancy_lambda_mercy * std_redundancy,
                min=float(self.redundancy_minimum),
            )

            max_redundancy = redundancy.max()
            redundancy_coef = (redundancy - threshold) / (max_redundancy - threshold + 1e-6)
            redundancy_coef = torch.clamp(redundancy_coef, 0.0, 1.0)
            candidate_mask = redundancy_coef > 0
            candidate_count = int(candidate_mask.sum().item())

            if self.redundancy_soft_suppression and candidate_count > 0:
                gate = torch.exp(-self.redundancy_suppression_gamma * redundancy_coef)
                min_gate = 1.0 - self.redundancy_suppression_max
                gate = gate.clamp(min=min_gate, max=1.0)
            else:
                gate = torch.ones_like(redundancy_coef)

        # Keep this multiplication differentiable with respect to predicted opacity.
        gaussian_dict = dict(gaussian_dict)
        gaussian_dict["opacity"] = gaussian_dict["opacity"] * gate.to(dtype=dtype).view(-1, 1)

        # Optional physical hard cap: only used when the number of produced Gaussians
        # still exceeds max_dense_gaussians. Normal redundancy handling remains soft.
        hard_cap_pruned = 0
        if self.max_dense_gaussians is not None and self.max_dense_gaussians > 0 and K > int(self.max_dense_gaussians):
            with torch.no_grad():
                max_dense = int(self.max_dense_gaussians)
                opacity_quality = gaussian_dict["opacity"].detach().float().squeeze(-1)
                if self.redundancy_conf_weight > 0 and "conf" in gaussian_dict and gaussian_dict["conf"] is not None:
                    conf_quality = torch.sigmoid(gaussian_dict["conf"].detach().float().squeeze(-1))
                    quality = opacity_quality + self.redundancy_conf_weight * conf_quality
                else:
                    quality = opacity_quality
                keep_idx = torch.topk(quality, k=max_dense, largest=True).indices
                keep_mask = torch.zeros(K, device=device, dtype=torch.bool)
                keep_mask[keep_idx] = True
            hard_cap_pruned = int((~keep_mask).sum().item())
            gaussian_dict = {k: v[keep_mask] for k, v in gaussian_dict.items()}

        expected_suppressed = (1.0 - gate.detach().float()).sum()
        stats = self._redundancy_stats(
            device,
            dtype,
            K,
            after_count=gaussian_dict["xyz"].shape[0],
            pruned=float(expected_suppressed.item()),
            candidates=candidate_count,
            threshold=float(threshold.item()),
            voxel_size=voxel_size,
            mean_redundancy=float(mean_redundancy.item()),
        )
        stats["redundancy_gate_mean"] = gate.detach().float().mean().to(device=device, dtype=dtype)
        stats["redundancy_expected_suppressed"] = expected_suppressed.to(device=device, dtype=dtype)
        stats["redundancy_coef_mean"] = redundancy_coef.detach().float().mean().to(device=device, dtype=dtype)
        stats["redundancy_coef_max"] = redundancy_coef.detach().float().max().to(device=device, dtype=dtype)
        stats["hard_cap_pruned"] = self._stats_tensor(hard_cap_pruned, device, dtype)
        stats = self._finalize_redundancy_stats(stats, gaussian_dict, K, device, dtype)
        return gaussian_dict, stats

    def forward(self, imgs, intrinsics=None, chunk_size=30000, global_step=None, return_viz=False):
        mem = MemDebug(active=self.debug_mem)
        B, N_total, C, H, W = imgs.shape
        imgs_raw = imgs

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
        cam_ctx = nullcontext() if modules_require_grad([self.camera_decoder, self.camera_head]) else torch.no_grad()
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
        imgs_raw_sub = imgs_raw[:, sub_idx].contiguous()

        point_ctx = nullcontext() if modules_require_grad([self.point_decoder, self.point_head]) else torch.no_grad()
        with point_ctx:
            point_h = self.point_decoder(hidden, xpos=pos)[:, self.patch_start_idx:]
            local_xyz_raw = self.point_head(point_h, patch_h=patch_h, patch_w=patch_w)

        conf_ctx = nullcontext() if modules_require_grad([self.conf_decoder, self.conf_head]) else torch.no_grad()
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
        local_pts_before_routing_viz = local_pts.detach().clone() if return_viz else None
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
        if self.use_input_intrinsics and intrinsics is not None:
            K = intrinsics.to(device=xy.device, dtype=xy.dtype)
        # ==========================================================
        # 动态计算 scene_size (基于相机原点的最大距离)
        # ==========================================================
        with torch.no_grad():
            # 计算所有点到相机原点 (0,0,0) 的距离: sqrt(x^2 + y^2 + z^2)
            distances = torch.norm(local_pts, dim=-1)

            # 方法 A (严格最大值): 直接取最远的点作为场景大小
            # scene_size = distances.max()

            # 方法 B (推荐：鲁棒最大值): 取分位数，过滤掉可能飞到极远处的异常噪点。
            # 对长序列推理，直接全量 quantile 会触发 tensor-too-large 错误，这里改成有上限采样。
            scene_size = approx_quantile_lastdim(distances.float().reshape(B, -1), 0.8)
            # print(f"Dynamic scene size: {scene_size.item():.2f}")
            # scene_size_f = scene_size / 10.0
            # local_pts = local_pts / scene_size_f[..., None, None, None]  # 将点云缩放到更合理的范围，防止数值不稳定
            # 目标半径设定为场景大小的 10 倍
            target_radius = self.low_conf_push_radius_ratio * scene_size.view(B, 1, 1, 1, 1)

        # ==========================================================
        # 【修改 2/2】：将 conf < 0.1 的点放置到 10 倍 scene_size 的球面上
        # ==========================================================
        with torch.no_grad():
            raw_low_conf_mask = torch.sigmoid(conf_logits) < 0.1
            push_y_limit = max(1, int(round(H * self.low_conf_push_max_y_ratio)))
            push_region_mask = (
                torch.arange(H, device=imgs.device)
                .view(1, 1, H, 1, 1)
                .lt(push_y_limit)
            )
            mask_push = raw_low_conf_mask & push_region_mask
            mask_push_expand = mask_push.expand_as(local_pts)

            xy_detached = xy.detach()

            # 射线方向 d = (x, y, 1)，模长 |d|
            dir_norm = torch.sqrt(xy_detached[..., 0:1] ** 2 + xy_detached[..., 1:2] ** 2 + 1.0)

            # 要使最终点距离相机为 target_radius，新的 z = target_radius / |d|
            z_sphere = target_radius / dir_norm

            # 组装球面上的点坐标
            sphere_pts = torch.cat([xy_detached * z_sphere, z_sphere], dim=-1)

        # 替换被 push 的点，切断这部分的梯度
        local_pts = torch.where(mask_push_expand, sphere_pts, local_pts)

        # 释放内存
        routing_mask_viz = mask_push.detach().clone() if return_viz else None
        raw_low_conf_mask_viz = raw_low_conf_mask.detach().clone() if return_viz else None
        del distances, mask_push, raw_low_conf_mask, push_region_mask, mask_push_expand, xy_detached, dir_norm, z_sphere, sphere_pts

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
            raw_low_conf_sub = conf_prob < 0.1
            push_y_limit = max(1, int(round(H * self.low_conf_push_max_y_ratio)))
            push_region_sub = (
                torch.arange(H, device=imgs.device)
                .view(1, 1, H, 1, 1)
                .lt(push_y_limit)
            )
            mask_push_scale = raw_low_conf_sub & push_region_sub

        # ==========================================================
        # 2. 动态多级四叉树 / 消融用全分辨率候选
        # ==========================================================
        if self.enable_quadtree:
            with torch.no_grad():
                z_raw = z.permute(0, 1, 4, 2, 3)
                depth_flat = z_raw[:, sub_idx].reshape(B * N_sub, 1, H, W)
                imgs_sub = imgs[:, sub_idx].reshape(B * N_sub, 3, H, W)

                # --- A. 提取底层特征联合 Score Map ---
                local_mean = F.avg_pool2d(
                    F.pad(imgs_sub, (3, 3, 3, 3), mode="replicate"),
                    kernel_size=7,
                    stride=1,
                )
                color_diff = torch.abs(imgs_sub - local_mean).mean(dim=1, keepdim=True)
                color_score = color_diff / (color_diff.amax(dim=(-2, -1), keepdim=True) + 1e-5)

                # Depth should raise allocation density at discontinuities/non-planar changes,
                # not on smooth slanted planes such as roads. Use local plane residual and
                # second-order curvature instead of first-order depth gradient.
                plane_kernel = 9
                plane_pad = plane_kernel // 2
                depth_mean = F.avg_pool2d(
                    F.pad(depth_flat, (plane_pad, plane_pad, plane_pad, plane_pad), mode="replicate"),
                    kernel_size=plane_kernel,
                    stride=1,
                )
                depth_plane_residual = torch.abs(depth_flat - depth_mean) / depth_mean.abs().clamp_min(1e-4)

                laplace_kernel = torch.tensor(
                    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                    device=imgs.device,
                    dtype=depth_flat.dtype,
                ).view(1, 1, 3, 3)
                depth_curvature = torch.abs(
                    F.conv2d(F.pad(depth_flat, (1, 1, 1, 1), mode="replicate"), laplace_kernel)
                )
                depth_curvature = depth_curvature / depth_flat.abs().clamp_min(1e-4)

                depth_complexity = torch.maximum(depth_plane_residual, depth_curvature)
                depth_complexity = torch.clamp(depth_complexity, max=approx_quantile_flat(depth_complexity.float(), 0.98))
                depth_score = 0.3 * depth_complexity / (depth_complexity.amax(dim=(-2, -1), keepdim=True) + 1e-5)

                score_map = torch.max(color_score, depth_score)
                score_map_viz = score_map.detach().reshape(B, N_sub, H, W).clone() if return_viz else None
                color_score_viz = color_score.detach().reshape(B, N_sub, H, W).clone() if return_viz else None
                depth_score_viz = depth_score.detach().reshape(B, N_sub, H, W).clone() if return_viz else None

                # --- B. 计算多级 Scale Map (带自动 Padding) ---
                # 【修改 1】：重新放开到 32 级大格子
                patch_sizes = [32, 16, 8, 4, 2, 1]
                base_threshold = 0.1
                relax_factor = 0.2

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
                        is_flat_expanded = pooled_score.repeat_interleave(size, dim=2).repeat_interleave(size,
                                                                                                         dim=3) < current_threshold

                    active_region = is_flat_expanded & (~covered_mask)
                    scale_map_padded = torch.where(active_region, torch.full_like(scale_map_padded, size), scale_map_padded)
                    covered_mask = covered_mask | active_region

                scale_map = scale_map_padded[:, :, :H, :W].reshape(B, N_sub, H, W, 1)

                # --- C. 选点逻辑：锚点掩码 (居中对齐) ---
                u_coords = torch.arange(W, device=imgs.device).view(1, 1, 1, W, 1).expand(B, N_sub, H, W, 1)
                v_coords = torch.arange(H, device=imgs.device).view(1, 1, H, 1, 1).expand(B, N_sub, H, W, 1)

                offset = (scale_map.long() // 2)
                quad_keep_mask = ((u_coords - offset) % scale_map.long() == 0) & (
                            (v_coords - offset) % scale_map.long() == 0)

                quad_keep_mask = quad_keep_mask & (~mask_push_scale)
        else:
            with torch.no_grad():
                keep_shape = (B, N_sub, H, W)
                keep_mask_dense = torch.ones(keep_shape, dtype=torch.bool, device=imgs.device)
                scale_map = torch.ones((*keep_shape, 1), dtype=torch.float32, device=imgs.device)
                quad_keep_mask = torch.zeros((*keep_shape, 1), dtype=torch.bool, device=imgs.device)
                if return_viz:
                    empty_score = torch.zeros(keep_shape, dtype=imgs.dtype, device=imgs.device)
                    score_map_viz = empty_score.detach().clone()
                    color_score_viz = empty_score.detach().clone()
                    depth_score_viz = empty_score.detach().clone()
                else:
                    score_map_viz = color_score_viz = depth_score_viz = None

        # --- D. 提取并应用属性 (极速瘦身：先筛点，再分块解码 GS 属性) ---
        if self.enable_quadtree:
            keep_mask = (quad_keep_mask | mask_push_scale).squeeze(-1)  # 形状: (B, N_sub, H, W)
        else:
            keep_mask = keep_mask_dense
        viz_outputs = None
        if return_viz:
            viz_outputs = {
                "sub_idx": sub_idx.detach().clone(),
                "score_map": score_map_viz,
                "color_score": color_score_viz,
                "depth_score": depth_score_viz,
                "scale_map": scale_map.detach().squeeze(-1).clone(),
                "quad_keep_mask": quad_keep_mask.detach().squeeze(-1).clone(),
                "low_conf_mask": mask_push_scale.detach().squeeze(-1).clone(),
                "raw_low_conf_mask": raw_low_conf_sub.detach().squeeze(-1).clone(),
                "routing_mask": routing_mask_viz.squeeze(-1) if routing_mask_viz is not None else None,
                "raw_low_conf_mask_full": raw_low_conf_mask_viz.squeeze(
                    -1) if raw_low_conf_mask_viz is not None else None,
                "local_points_before_routing": local_pts_before_routing_viz.reshape(B, N_total, H, W, 3),
                "support_mask": keep_mask.detach().clone(),
                "scene_size": scene_size.detach().clone(),
                "low_conf_push_max_y_ratio": imgs.new_tensor(self.low_conf_push_max_y_ratio),
                "conf_threshold": imgs.new_tensor(0.1),
            }
        local_pts_sub = local_pts[:, sub_idx]
        del z, xy, local_xyz_raw, conf_prob, raw_low_conf_sub, push_region_sub
        if self.enable_quadtree:
            del z_raw, depth_flat, imgs_sub, score_map, score_map_padded
            del covered_mask, scale_map_padded, u_coords, v_coords, offset
        else:
            del keep_mask_dense
        voxel_size_val = (scene_size.mean() * 0.002).clamp_min(1e-4).item()
        fused_gaussians = []
        redundancy_stats = []

        # ==========================================================
        # 3. 仅对有效保留点解析属性并执行 3D Soft Voxel Attention Fusion
        #    先筛，再按视角 chunk 解码 GS，避免整块 gs_attrs 常驻显存
        # ==========================================================
        for b in range(B):
            mask_b = keep_mask[b]  # [N_sub, H, W]

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
            b_xyz, b_rot, b_scale, b_opacity, b_color, b_conf, b_source_view, b_comp_color = [], [], [], [], [], [], [], []
            b_proposal_count = 0
            b_learned_extra_count = 0
            b_selected_count = 0
            b_density_sum = 0.0
            b_density_num = 0

            for start in range(0, N_sub, self.gs_decoder_view_chunk_size):
                end = min(start + self.gs_decoder_view_chunk_size, N_sub)
                proposal_mask_chunk = mask_b[start:end]
                if not proposal_mask_chunk.any() and not self.enable_learnable_sampling:
                    continue

                hidden_chunk = hidden_views[b, start:end].reshape(end - start, hw, -1)
                pos_chunk = pos_views[b, start:end].reshape(end - start, hw, -1)
                gs_h_chunk = self.gs_decoder(hidden_chunk, xpos=pos_chunk)[:, self.patch_start_idx:]
                gs_attrs_chunk = self.gs_head([gs_h_chunk], (H, W)).reshape(end - start, H, W, 13)

                density_logits_chunk = gs_attrs_chunk[..., 11]
                learned_mask_chunk = self._build_learned_support_mask(
                    density_logits_chunk, proposal_mask_chunk
                )
                mask_chunk = proposal_mask_chunk | learned_mask_chunk
                if not mask_chunk.any():
                    del hidden_chunk, pos_chunk, gs_h_chunk, gs_attrs_chunk
                    continue

                b_proposal_count += int(proposal_mask_chunk.sum().item())
                b_learned_extra_count += int((learned_mask_chunk & ~proposal_mask_chunk).sum().item())
                b_selected_count += int(mask_chunk.sum().item())

                # 1. 提取当前 chunk 内有效点：heuristic quadtree proposal + learned density extras.
                gs_attrs_v = gs_attrs_chunk[mask_chunk]
                local_pts_v = local_pts_sub[b, start:end][mask_chunk]
                scale_map_v = scale_map[b, start:end][mask_chunk]
                is_quad_center_v = quad_keep_mask[b, start:end][mask_chunk]
                low_conf_v = mask_push_scale[b, start:end][mask_chunk]
                conf_v = conf_logits_sub[b, start:end][mask_chunk]
                comp_color_v = imgs_raw_sub[b, start:end].permute(0, 2, 3, 1)[mask_chunk]
                view_idx_v = mask_chunk.nonzero(as_tuple=True)[0] + start

                # 及时释放 chunk 级无用显存
                del hidden_chunk, pos_chunk, gs_h_chunk, gs_attrs_chunk
                del proposal_mask_chunk, learned_mask_chunk, mask_chunk, density_logits_chunk

                # 2. 先解析 learnable density gate。gate 不只负责候选点补充，
                #    还以可导方式调制 opacity，使密度分配本身能吃到渲染梯度。
                density_prob_v = torch.sigmoid(gs_attrs_v[:, 11:12] / self.density_gate_temperature)
                gated_density = density_prob_v.clamp_min(self.density_gate_min_prob)
                base_opacity_v = torch.sigmoid(gs_attrs_v[:, 7:8])
                opacity_v = base_opacity_v * gated_density.pow(self.density_gate_opacity_power)

                valid_mask = base_opacity_v[:, 0] > self.opacity_filter_threshold
                if not valid_mask.any():
                    fallback_k = min(16, base_opacity_v.shape[0])
                    top_idx = torch.topk(base_opacity_v[:, 0], k=fallback_k, largest=True).indices
                    valid_mask = torch.zeros_like(valid_mask)
                    valid_mask[top_idx] = True

                # 应用过滤
                gs_attrs_v = gs_attrs_v[valid_mask]
                local_pts_v = local_pts_v[valid_mask]
                scale_map_v = scale_map_v[valid_mask]
                is_quad_center_v = is_quad_center_v[valid_mask]
                low_conf_v = low_conf_v[valid_mask]
                conf_v = conf_v[valid_mask]
                comp_color_v = comp_color_v[valid_mask]
                view_idx_v = view_idx_v[valid_mask]
                opacity_v = opacity_v[valid_mask]
                density_prob_v = density_prob_v[valid_mask]

                # 3. 仅对存活的点解析其余属性并变换
                local_rot_v = F.normalize(gs_attrs_v[:, 0:4], dim=-1)
                scale_v = torch.exp(torch.clamp(gs_attrs_v[:, 4:7], min=-10.0, max=5.0)) * 0.1
                color_v = torch.sigmoid(gs_attrs_v[:, 8:11])
                scale_bias_v = torch.tanh(gs_attrs_v[:, 12:13])

                # 【修复 OOM 广播灾难】：强制转为 (N, 1) 的形状，避免 N x N 维度爆炸
                mask_low = low_conf_v.view(-1, 1)
                mask_quad = is_quad_center_v.view(-1, 1)
                map_scale = scale_map_v.view(-1, 1)

                scale_v = torch.where(mask_low, scale_v * self.low_conf_scale_boost, scale_v)
                scale_v = torch.where(mask_quad, scale_v * map_scale, scale_v)
                scale_v = scale_v * torch.exp(self.scale_bias_strength * scale_bias_v)

                b_density_sum += float(density_prob_v.detach().float().sum().item())
                b_density_num += int(density_prob_v.numel())

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
                b_conf.append(conf_v)
                b_source_view.append(view_idx_v)
                b_comp_color.append(comp_color_v)

                del gs_attrs_v, local_pts_v, local_rot_v, cam_poses_v, cam_rot_v, cam_trans_v
                del density_prob_v, scale_bias_v

            # 当前 Batch 视角处理完毕，合并结果兜底
            if len(b_xyz) > 0:
                gaussian_b = {
                    "xyz": torch.cat(b_xyz, dim=0),
                    "rotation": torch.cat(b_rot, dim=0),
                    "scale": torch.cat(b_scale, dim=0),
                    "opacity": torch.cat(b_opacity, dim=0),
                    "color": torch.cat(b_color, dim=0),
                    "conf": torch.cat(b_conf, dim=0),
                    "competition_color": torch.cat(b_comp_color, dim=0),
                    "source_view": torch.cat(b_source_view, dim=0),
                }
                source_view_b = torch.cat(b_source_view, dim=0)
                # Local competition now includes color-aware redundancy scoring,
                # redundancy-coefficient activation, and soft opacity suppression.
                gaussian_b, stats_b = self._apply_local_competition(
                    gaussian_b,
                    source_view_b,
                    scene_size[b],
                    camera_poses=camera_poses_sub[b],
                    intrinsics=K[b, sub_idx],
                    image_hw=(H, W),
                )
                stats_b["proposal_count"] = self._stats_tensor(b_proposal_count, imgs.device, imgs.dtype)
                stats_b["learned_extra_count"] = self._stats_tensor(b_learned_extra_count, imgs.device, imgs.dtype)
                stats_b["selected_count"] = self._stats_tensor(b_selected_count, imgs.device, imgs.dtype)
                stats_b["density_gate_mean"] = self._stats_tensor(
                    b_density_sum / max(b_density_num, 1), imgs.device, imgs.dtype
                )
                stats_b["xyz_residual_norm"] = self._stats_tensor(0.0, imgs.device, imgs.dtype)
                fused_gaussians.append(gaussian_b)
                redundancy_stats.append(stats_b)
            else:
                # 极端兜底，防止全被过滤导致维度错误
                grad_anchor = next(self.gs_decoder.parameters()).flatten()[0] * 0.0
                gaussian_b = {
                    "xyz": torch.zeros((1, 3), device=imgs.device) + grad_anchor,
                    "rotation": torch.tensor([[1., 0., 0., 0.]], device=imgs.device) + grad_anchor,
                    "scale": torch.full((1, 3), 1e-5, device=imgs.device) + grad_anchor,
                    "opacity": torch.zeros((1, 1), device=imgs.device) + grad_anchor,
                    "color": torch.zeros((1, 3), device=imgs.device) + grad_anchor,
                    "conf": torch.full((1, 1), -20.0, device=imgs.device) + grad_anchor,
                    "competition_color": torch.zeros((1, 3), device=imgs.device) + grad_anchor,
                    "source_view": torch.zeros((1,), device=imgs.device, dtype=torch.long),
                }
                source_view_b = torch.zeros((1,), device=imgs.device, dtype=torch.long)
                # Local competition now includes color-aware redundancy scoring,
                # redundancy-coefficient activation, and soft opacity suppression.
                gaussian_b, stats_b = self._apply_local_competition(
                    gaussian_b,
                    source_view_b,
                    scene_size[b],
                    camera_poses=camera_poses_sub[b],
                    intrinsics=K[b, sub_idx],
                    image_hw=(H, W),
                )
                stats_b["proposal_count"] = self._stats_tensor(b_proposal_count, imgs.device, imgs.dtype)
                stats_b["learned_extra_count"] = self._stats_tensor(b_learned_extra_count, imgs.device, imgs.dtype)
                stats_b["selected_count"] = self._stats_tensor(b_selected_count, imgs.device, imgs.dtype)
                stats_b["density_gate_mean"] = self._stats_tensor(0.0, imgs.device, imgs.dtype)
                stats_b["xyz_residual_norm"] = self._stats_tensor(0.0, imgs.device, imgs.dtype)
                fused_gaussians.append(gaussian_b)
                redundancy_stats.append(stats_b)
        # 对齐 Batch 内高斯数量
        max_k = max(g["xyz"].size(0) for g in fused_gaussians)
        d_xyz_out, d_rot_out, d_scale_out, d_opacity_out, d_color_out, d_conf_out = [], [], [], [], [], []
        d_comp_color_out, d_source_view_out = [], []
        d_redundancy_score_out, d_redundancy_coef_out, d_comp_gate_out, d_comp_active_out = [], [], [], []
        for g in fused_gaussians:
            pad_len = max_k - g["xyz"].size(0)
            d_xyz_out.append(F.pad(g["xyz"], (0, 0, 0, pad_len), value=0.0))
            d_rot_out.append(F.pad(g["rotation"], (0, 0, 0, pad_len), value=1.0))
            d_scale_out.append(F.pad(g["scale"], (0, 0, 0, pad_len), value=1e-5))
            d_opacity_out.append(F.pad(g["opacity"], (0, 0, 0, pad_len), value=0.0))
            d_color_out.append(F.pad(g["color"], (0, 0, 0, pad_len), value=0.0))
            d_conf_out.append(F.pad(g["conf"], (0, 0, 0, pad_len), value=-20.0))
            d_comp_color_out.append(F.pad(g["competition_color"], (0, 0, 0, pad_len), value=0.0))
            d_source_view_out.append(F.pad(g["source_view"], (0, pad_len), value=-1))
            d_redundancy_score_out.append(F.pad(g["redundancy_score"], (0, 0, 0, pad_len), value=0.0))
            d_redundancy_coef_out.append(F.pad(g["redundancy_coef"], (0, 0, 0, pad_len), value=0.0))
            d_comp_gate_out.append(F.pad(g["competition_gate"], (0, 0, 0, pad_len), value=1.0))
            d_comp_active_out.append(F.pad(g["competition_active"], (0, 0, 0, pad_len), value=0.0))

        gaussians = {
            "xyz": torch.stack(d_xyz_out, dim=0),
            "rotation": torch.stack(d_rot_out, dim=0),
            "scale": torch.stack(d_scale_out, dim=0),
            "opacity": torch.stack(d_opacity_out, dim=0),
            "color": torch.stack(d_color_out, dim=0),
            "num_sky": 0,
            "conf": torch.stack(d_conf_out, dim=0),
            "competition_color": torch.stack(d_comp_color_out, dim=0),
            "source_view": torch.stack(d_source_view_out, dim=0),
            "redundancy_score": torch.stack(d_redundancy_score_out, dim=0),
            "redundancy_coef": torch.stack(d_redundancy_coef_out, dim=0),
            "competition_gate": torch.stack(d_comp_gate_out, dim=0),
            "competition_active": torch.stack(d_comp_active_out, dim=0),
        }
        gaussian_stats = {
            key: torch.stack([stats[key] for stats in redundancy_stats], dim=0)
            for key in redundancy_stats[0].keys()
        } if len(redundancy_stats) > 0 else {}

        output = dict(
            gaussians=gaussians,
            gaussian_stats=gaussian_stats,
            camera_poses=camera_poses,
            local_points=local_pts,
            intrinsics=K,
            conf=conf_logits
        )
        if return_viz:
            output["routing_viz"] = viz_outputs
        return output
