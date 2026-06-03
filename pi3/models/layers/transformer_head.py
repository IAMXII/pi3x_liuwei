import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint
from .attention import FlashAttentionRope
from .block import BlockRope
from ..dinov2.layers import Mlp
from .conv_head import ConvHead, normalized_view_plane_uv  # 导入全新的卷积头

class TransformerDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dec_embed_dim=512, depth=5, dec_num_heads=8, mlp_ratio=4, rope=None,
                 need_project=True, use_checkpoint=False):
        super().__init__()
        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                ffn_layer=Mlp, init_values=None, qk_norm=False,
                attn_class=FlashAttentionRope, rope=rope
            ) for _ in range(depth)])
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        return self.linear_out(hidden)


class AlternatingViewTransformerDecoder(nn.Module):
    """
    GS decoder with per-view token refinement followed by cross-view fusion.

    This keeps the lightweight Pi3 branch interface while borrowing the
    frame/global alternating-attention idea used by Hunyuan/VGGT-style geometry
    transformers.
    """
    def __init__(self, in_dim, out_dim, dec_embed_dim=1024, depth=5, dec_num_heads=16, mlp_ratio=4, rope=None,
                 need_project=True, use_checkpoint=False):
        super().__init__()
        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint
        self.view_blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                ffn_layer=Mlp, init_values=None, qk_norm=False,
                attn_class=FlashAttentionRope, rope=rope
            ) for _ in range(depth)])
        self.fusion_blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True, drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                ffn_layer=Mlp, init_values=None, qk_norm=False,
                attn_class=FlashAttentionRope, rope=rope
            ) for _ in range(depth)])
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def _run_block(self, block, hidden, xpos=None):
        if self.use_checkpoint and self.training:
            return checkpoint(block, hidden, xpos=xpos, use_reentrant=False)
        return block(hidden, xpos=xpos)

    def forward(self, hidden, xpos=None, batch_size=None, num_views=None):
        hidden = self.projects(hidden)
        flat_views, seq_len, channels = hidden.shape

        if batch_size is None and num_views is None:
            batch_size = 1
            num_views = flat_views
        elif batch_size is None:
            batch_size = flat_views // int(num_views)
        elif num_views is None:
            num_views = flat_views // int(batch_size)

        batch_size = int(batch_size)
        num_views = int(num_views)
        if batch_size * num_views != flat_views:
            raise ValueError(
                f"Cannot reshape {flat_views} GS views into batch_size={batch_size}, num_views={num_views}"
            )

        for view_blk, fusion_blk in zip(self.view_blocks, self.fusion_blocks):
            hidden = self._run_block(view_blk, hidden, xpos=xpos)

            hidden_mv = hidden.reshape(batch_size, num_views * seq_len, channels)
            xpos_mv = None if xpos is None else xpos.reshape(batch_size, num_views * seq_len, -1)
            hidden_mv = self._run_block(fusion_blk, hidden_mv, xpos=xpos_mv)
            hidden = hidden_mv.reshape(flat_views, seq_len, channels)

        return self.linear_out(hidden)


class ConvPts3dHead(nn.Module):
    """
    负责预测 3D 点 (xy, z) 或者置信度 (conf)。
    替代了旧版的 LinearPts3d，使用带有 UV 坐标的 ConvHead 进行上采样解码。
    """
    def __init__(self, patch_size, dec_embed_dim, dim_out=[2, 1]):
        super().__init__()
        self.patch_size = patch_size
        self.conv_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=dim_out, 
            # ====== [修改点: 解除 1024 维度锁定，随输入变化] ======
            dim_proj=dec_embed_dim, 
            # ======================================================
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

    def forward(self, decout, img_shape):
        H, W = img_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        tokens = decout[-1] if isinstance(decout, list) else decout
        
        # ConvHead 需要 float 类型的输入
        out = self.conv_head(tokens, patch_h=patch_h, patch_w=patch_w)
        
        # 如果返回的是列表 (如 xy和z分离预测)，则在通道维度拼接
        if isinstance(out, list):
            feat = torch.cat(out, dim=1)
        else:
            feat = out
            
        # 转换回 [B, H, W, C] 格式，无缝衔接现有 pipeline
        return feat.permute(0, 2, 3, 1)


class ConvDenseGaussianHead(nn.Module):
    """
    负责预测 Dense 图像像素级别的高斯属性。
    输出维度 11: Rotation (4), Scale (3), Opacity (1), Color (3)
    """
    def __init__(self, patch_size, dec_embed_dim, dim_out=[4, 3, 1, 3]):
        super().__init__()
        self.patch_size = patch_size
        self.conv_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=dim_out, 
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

    def forward(self, decout, img_shape):
        H, W = img_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        tokens = decout[-1] if isinstance(decout, list) else decout
        
        # ConvHead 预测多种属性
        out = self.conv_head(tokens, patch_h=patch_h, patch_w=patch_w)
        
        # 将 [rot, scale, opacity, color] 按通道顺序拼接
        feat = torch.cat(out, dim=1)
        
        # 转换回 [B, H, W, 11]
        return feat.permute(0, 2, 3, 1)


class ImageAwareConvDenseGaussianHead(nn.Module):
    """
    Dense Gaussian head with an RGB image skip similar to HunyuanWorld-Mirror.

    Tokens still drive the Gaussian attributes, but a shallow image merger is
    added before the output heads so color/opacity/detail predictions can use
    local texture cues, which is especially helpful for SSIM-oriented training.
    """
    def __init__(self, patch_size, dec_embed_dim, dim_out=[4, 3, 1, 3, 3, 1, 1],
                 image_channels=3, image_gate_init=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.conv_head = ConvHead(
            num_features=4,
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=dim_out,
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
        self.image_merger = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.image_gate = nn.Parameter(torch.tensor(float(image_gate_init)))
        self._init_image_and_residual_paths()

    def _init_image_and_residual_paths(self):
        # Start as a near drop-in replacement for the old head; training can open
        # the image skip and xyz residual branch as useful signal appears.
        nn.init.zeros_(self.image_merger[-1].weight)
        nn.init.zeros_(self.image_merger[-1].bias)
        if isinstance(self.conv_head.output_block, nn.ModuleList) and len(self.conv_head.output_block) > 4:
            residual_last = self.conv_head.output_block[4][-1]
            if isinstance(residual_last, nn.Conv2d):
                nn.init.zeros_(residual_last.weight)
                nn.init.zeros_(residual_last.bias)

    def forward(self, decout, img_shape, image=None):
        H, W = img_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        tokens = decout[-1] if isinstance(decout, list) else decout

        x = self.conv_head.projects(tokens).permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()

        img_h = patch_h * self.patch_size
        img_w = patch_w * self.patch_size
        for block in self.conv_head.upsample_blocks:
            if self.conv_head.using_uv:
                uv = normalized_view_plane_uv(
                    width=x.shape[-1],
                    height=x.shape[-2],
                    aspect_ratio=img_w / img_h,
                    dtype=x.dtype,
                    device=x.device,
                )
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = checkpoint(layer, x, use_reentrant=False)

        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)

        if image is not None:
            if image.dim() == 5:
                image = image.reshape(-1, *image.shape[-3:])
            image = image.to(device=x.device, dtype=x.dtype)
            if image.shape[-2:] != x.shape[-2:]:
                image = F.interpolate(image, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.image_gate.to(dtype=x.dtype) * self.image_merger(image)

        if self.conv_head.using_uv:
            uv = normalized_view_plane_uv(
                width=x.shape[-1],
                height=x.shape[-2],
                aspect_ratio=img_w / img_h,
                dtype=x.dtype,
                device=x.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)

        out = [checkpoint(block, x, use_reentrant=False) for block in self.conv_head.output_block]
        feat = torch.cat(out, dim=1)
        return feat.permute(0, 2, 3, 1)

class AppearanceModulationHead(nn.Module):
    def __init__(self, patch_size=14, dec_embed_dim=1024, light_dim=512): # changed light_dim to match CLIP
        super().__init__()
        # 1. 光照提取器 (可以直接降维，不需要像 DINO 那样池化)
        self.light_extractor = nn.Sequential(
            nn.Linear(light_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256) # 内部使用的特征维度
        )

        # 2. FiLM 生成器
        self.film_scale = nn.Linear(256, dec_embed_dim)
        self.film_shift = nn.Linear(256, dec_embed_dim)

        self.conv_head = ConvPts3dHead(
            patch_size=patch_size, 
            dec_embed_dim=dec_embed_dim, 
            dim_out=[3, 3] 
        )
        self.zero_proj = nn.Linear(6, 6)
        self._init_weights()

    def _init_weights(self):
        # 初始化 FiLM 层为恒等映射
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        # 【核心修正】彻底摒弃去内部寻找 final_conv 的危险做法
        # 只将外部拦截的零投影层置零
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, light_code, gs_h, img_shape):
        # light_code shape: [B*N, 512]
        # 1. 处理 CLIP 提取的全局光照向量
        light_feat = self.light_extractor(light_code) 

        # 2. 生成 FiLM 参数
        scale = self.film_scale(light_feat).unsqueeze(1) 
        shift = self.film_shift(light_feat).unsqueeze(1) 

        # 3. 调制 3D 几何特征
        gs_h_modulated = gs_h * (1.0 + scale) + shift 

        # 4. CNN 上采样
        cnn_out = self.conv_head([gs_h_modulated], img_shape) 

        # 5. 零投影拦截
        app_out = self.zero_proj(cnn_out)

        return app_out
class SkyGaussianHead(nn.Module):
    """
    动态天空高斯头：几何和透明度维持半固定，但颜色由当前场景的图像特征动态预测。
    """
    def __init__(self, num_sky_anchors=8196, sky_radius=1000.0, in_dim=1024):
        super().__init__()
        self.num_sky_anchors = num_sky_anchors
        self.sky_radius = sky_radius
        
        # 几何方向依然固定
        sky_dirs = torch.randn(num_sky_anchors, 3)
        sky_dirs[:, 2] = -torch.abs(sky_dirs[:, 1]) 
        self.register_buffer("sky_dirs", F.normalize(sky_dirs, dim=-1))
        
        self.sky_rotation = nn.Parameter(torch.randn(num_sky_anchors, 4))
        self.sky_scale = nn.Parameter(torch.randn(num_sky_anchors, 3))
        # 透明度可以保留为 Parameter 缓慢学习，或者固定为一个较高值
        self.sky_opacity = nn.Parameter(torch.ones(num_sky_anchors, 1) * 0.5)
        
        # ==========================================================
        # 【核心修改】：移除 nn.Parameter 的 sky_color，替换为 MLP
        # ==========================================================
        self.color_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3) # 输出 RGB 三通道
        )

    def forward(self, global_feat):
        # global_feat 的期望形状为: [B, in_dim]
        B = global_feat.shape[0]
        
        # 1. 动态预测当前场景的基础天空颜色 -> [B, 3]
        base_color = self.color_mlp(global_feat)
        
        xyz = (self.sky_dirs * self.sky_radius).unsqueeze(0).expand(B, -1, -1)
        
        scale = torch.exp(torch.clamp(self.sky_scale, min=-10.0, max=5.0))*0.5
        scale = scale * 51.0 
        scale = scale.unsqueeze(0).expand(B, -1, -1)
        
        rot = F.normalize(self.sky_rotation, dim=-1).unsqueeze(0).expand(B, -1, -1)
        opacity = torch.sigmoid(self.sky_opacity).unsqueeze(0).expand(B, -1, -1)
        
        # 2. 将预测的基础颜色加上 Sigmoid 激活，并扩展到所有的 Sky Anchors
        # 形状变为 -> [B, num_sky_anchors, 3]
        color = torch.sigmoid(base_color).unsqueeze(1).expand(B, self.num_sky_anchors, -1)
        
        conf = torch.ones((B, self.num_sky_anchors, 1), device=xyz.device) * 10.0
        
        return xyz, rot, scale, opacity, color, conf
