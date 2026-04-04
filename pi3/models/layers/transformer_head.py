import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint
from .attention import FlashAttentionRope
from .block import BlockRope
from ..dinov2.layers import Mlp
from .conv_head import ConvHead  # 导入全新的卷积头

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