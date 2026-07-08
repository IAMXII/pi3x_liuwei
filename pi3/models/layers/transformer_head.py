import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint
from .attention import FlashAttentionRope
from .block import BlockRope
from ..dinov2.layers import Mlp
from .conv_head import ConvHead, ResidualConvBlock, normalized_view_plane_uv  # 导入全新的卷积头

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

    def forward(self, hidden, xpos=None, batch_size=None, num_views=None, return_intermediate=False):
        hidden = self.projects(hidden)
        flat_views, seq_len, channels = hidden.shape
        intermediates = []

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
            if return_intermediate:
                intermediates.append(self.linear_out(hidden))

        if return_intermediate:
            return intermediates
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


class DPTFeatureFusionBlock(nn.Module):
    def __init__(self, channels, has_residual=True, res_block_norm='group_norm'):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.residual = ResidualConvBlock(
                channels,
                channels,
                channels * 2,
                activation='relu',
                norm=res_block_norm,
            )
        self.refine = ResidualConvBlock(
            channels,
            channels,
            channels * 2,
            activation='relu',
            norm=res_block_norm,
        )
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, residual=None, size=None):
        if self.has_residual and residual is not None:
            x = x + self.residual(residual)
        x = self.refine(x)
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return self.out_conv(x)


class ImageAwareDPTDenseGaussianHead(nn.Module):
    """
    DPT/refinenet-style dense Gaussian head for V12.

    It consumes intermediate GS decoder tokens, builds a small multi-scale
    pyramid, fuses it back to full resolution, and injects RGB features at the
    end so shape/opacity channels can react to local high-frequency details.
    """
    def __init__(
            self,
            patch_size,
            dec_embed_dim,
            dim_out=[4, 3, 1, 3, 3, 1, 1],
            image_channels=3,
            feature_channels=64,
            head_channels=48,
            res_block_norm='group_norm',
            use_checkpoint=True,
            use_image_branch=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim_out = dim_out
        self.feature_channels = feature_channels
        self.use_checkpoint = bool(use_checkpoint)
        self.use_image_branch = bool(use_image_branch)

        self.projects = nn.ModuleList([
            nn.Conv2d(dec_embed_dim, feature_channels, kernel_size=1)
            for _ in range(4)
        ])
        self.uv_fusers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels + 2, feature_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1),
        ])

        self.refinenet4 = DPTFeatureFusionBlock(feature_channels, has_residual=False, res_block_norm=res_block_norm)
        self.refinenet3 = DPTFeatureFusionBlock(feature_channels, has_residual=True, res_block_norm=res_block_norm)
        self.refinenet2 = DPTFeatureFusionBlock(feature_channels, has_residual=True, res_block_norm=res_block_norm)
        self.refinenet1 = DPTFeatureFusionBlock(feature_channels, has_residual=True, res_block_norm=res_block_norm)

        self.dpt_image_merger = nn.Sequential(
            nn.Conv2d(image_channels, feature_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.image_gate = nn.Parameter(torch.tensor(0.5))
        if not self.use_image_branch:
            self.image_gate.requires_grad_(False)
            for param in self.dpt_image_merger.parameters():
                param.requires_grad = False

        self.output_block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels + 2, head_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True),
                ResidualConvBlock(
                    head_channels,
                    head_channels,
                    head_channels * 2,
                    activation='relu',
                    norm=res_block_norm,
                ),
                nn.Conv2d(head_channels, dim, kernel_size=1, stride=1, padding=0),
            )
            for dim in dim_out
        ])
        self._init_residual_branch()

    def _init_residual_branch(self):
        # Keep center residual and scale-bias neutral at initialization.
        for idx in (4, 6):
            if idx < len(self.output_block):
                final = self.output_block[idx][-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)

    def _tokens_to_map(self, tokens, patch_h, patch_w):
        return tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w).contiguous()

    def _append_uv(self, x, img_h, img_w):
        uv = normalized_view_plane_uv(
            width=x.shape[-1],
            height=x.shape[-2],
            aspect_ratio=img_w / img_h,
            dtype=x.dtype,
            device=x.device,
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, uv], dim=1)

    def _checkpoint_block(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, decout, img_shape, image=None):
        H, W = img_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        img_h, img_w = patch_h * self.patch_size, patch_w * self.patch_size

        if isinstance(decout, (list, tuple)):
            token_list = list(decout)
        else:
            token_list = [decout]
        if len(token_list) < 4:
            token_list = [token_list[0]] * (4 - len(token_list)) + token_list
        token_list = token_list[-4:]

        feats = []
        for tokens, project, uv_fuser, resize in zip(token_list, self.projects, self.uv_fusers, self.resize_layers):
            x = self._tokens_to_map(tokens, patch_h, patch_w)
            x = project(x)
            x = uv_fuser(self._append_uv(x, img_h, img_w))
            x = resize(x)
            feats.append(x)

        layer_1, layer_2, layer_3, layer_4 = feats
        path_4 = self._checkpoint_block(
            lambda x4: self.refinenet4(x4, size=layer_3.shape[-2:]),
            layer_4,
        )
        path_3 = self._checkpoint_block(
            lambda x4, x3: self.refinenet3(x4, x3, size=layer_2.shape[-2:]),
            path_4,
            layer_3,
        )
        path_2 = self._checkpoint_block(
            lambda x3, x2: self.refinenet2(x3, x2, size=layer_1.shape[-2:]),
            path_3,
            layer_2,
        )
        x = self._checkpoint_block(
            lambda x2, x1: self.refinenet1(x2, x1, size=(img_h, img_w)),
            path_2,
            layer_1,
        )

        if self.use_image_branch and image is not None:
            if image.dim() == 5:
                image = image.reshape(-1, *image.shape[-3:])
            image = image.to(device=x.device, dtype=x.dtype)
            if image.shape[-2:] != x.shape[-2:]:
                image = F.interpolate(image, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + self.image_gate.to(dtype=x.dtype) * self.dpt_image_merger(image)

        x = self._append_uv(x, img_h, img_w)
        out = [checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        feat = torch.cat(out, dim=1)
        return feat.permute(0, 2, 3, 1)


class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, num_res_blocks=2,
                 dim_times_res_block_hidden=2, res_block_norm='group_norm'):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * upscale_factor * upscale_factor,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='replicate',
            ),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(
                ResidualConvBlock(
                    out_channels,
                    out_channels,
                    dim_times_res_block_hidden * out_channels,
                    activation="relu",
                    norm=res_block_norm,
                )
                for _ in range(num_res_blocks)
            ),
        )

    def forward(self, x):
        return self.layers(x)


class ImageAwarePixelShuffleDenseGaussianHead(nn.Module):
    """
    V12 dense Gaussian head.

    It replaces additive RGB injection with concat + 1x1 fusion, feeds Sobel
    guidance into the image path, and upsamples patch tokens to pixels with
    PixelShuffle factors 2 and 7, matching the 14px ViT patch size.
    """
    def __init__(self, patch_size, dec_embed_dim, dim_out=[4, 3, 1, 12, 3, 1, 1],
                 image_channels=3, image_feat_channels=64, head_channels=64):
        super().__init__()
        if patch_size != 14:
            raise ValueError("ImageAwarePixelShuffleDenseGaussianHead expects patch_size=14")
        self.patch_size = patch_size
        self.projects = nn.Identity()
        self.using_uv = True
        self.upsample_blocks = nn.ModuleList([
            PixelShuffleUpsampleBlock(
                dec_embed_dim + 2,
                256,
                upscale_factor=2,
                num_res_blocks=2,
                dim_times_res_block_hidden=2,
                res_block_norm='group_norm',
            ),
            PixelShuffleUpsampleBlock(
                256 + 2,
                head_channels,
                upscale_factor=7,
                num_res_blocks=2,
                dim_times_res_block_hidden=2,
                res_block_norm='group_norm',
            ),
        ])
        image_input_channels = image_channels + 4
        self.image_merger = nn.Sequential(
            nn.Conv2d(image_input_channels, image_feat_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_feat_channels, image_feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion_conv = nn.Conv2d(head_channels + image_feat_channels, head_channels, kernel_size=1, stride=1)
        self.output_block = nn.ModuleList([
            self._make_output_block(
                head_channels + 2,
                dim_out_,
                dim_times_res_block_hidden=2,
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                res_block_norm='group_norm',
            )
            for dim_out_ in dim_out
        ])
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self._init_fusion_and_residual_paths(head_channels, image_feat_channels)

    def _make_output_block(self, dim_in, dim_out, dim_times_res_block_hidden,
                           last_res_blocks, last_conv_channels, last_conv_size,
                           res_block_norm):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(
                ResidualConvBlock(
                    last_conv_channels,
                    last_conv_channels,
                    dim_times_res_block_hidden * last_conv_channels,
                    activation='relu',
                    norm=res_block_norm,
                )
                for _ in range(last_res_blocks)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1,
                      padding=last_conv_size // 2, padding_mode='replicate'),
        )

    def _init_fusion_and_residual_paths(self, head_channels, image_feat_channels):
        nn.init.zeros_(self.fusion_conv.weight)
        nn.init.zeros_(self.fusion_conv.bias)
        with torch.no_grad():
            eye = torch.eye(head_channels, dtype=self.fusion_conv.weight.dtype)
            self.fusion_conv.weight[:, :head_channels, 0, 0].copy_(eye)
            image_slice = self.fusion_conv.weight[:, head_channels:head_channels + image_feat_channels, 0, 0]
            image_slice.normal_(mean=0.0, std=0.01)

        if len(self.output_block) > 4:
            residual_last = self.output_block[4][-1]
            if isinstance(residual_last, nn.Conv2d):
                nn.init.zeros_(residual_last.weight)
                nn.init.zeros_(residual_last.bias)

    def _image_guidance(self, image, target_hw, dtype, device):
        image = image.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if image.shape[-2:] != target_hw:
            image = F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)

        gray = (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )
        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)
        grad_x = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), sobel_x)
        grad_y = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), sobel_y)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return torch.cat([image, gray, grad_x, grad_y, grad_mag], dim=1)

    def forward(self, decout, img_shape, image=None):
        H, W = img_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        tokens = decout[-1] if isinstance(decout, list) else decout

        x = self.projects(tokens).permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()
        img_h = patch_h * self.patch_size
        img_w = patch_w * self.patch_size

        for block in self.upsample_blocks:
            if self.using_uv:
                uv = normalized_view_plane_uv(
                    width=x.shape[-1],
                    height=x.shape[-2],
                    aspect_ratio=img_w / img_h,
                    dtype=x.dtype,
                    device=x.device,
                )
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, uv], dim=1)
            x = checkpoint(block, x, use_reentrant=False)

        if image is not None:
            if image.dim() == 5:
                image = image.reshape(-1, *image.shape[-3:])
            img_feat = self.image_merger(self._image_guidance(image, x.shape[-2:], x.dtype, x.device))
            x = self.fusion_conv(torch.cat([x, img_feat], dim=1))

        if self.using_uv:
            uv = normalized_view_plane_uv(
                width=x.shape[-1],
                height=x.shape[-2],
                aspect_ratio=img_w / img_h,
                dtype=x.dtype,
                device=x.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)

        out = [checkpoint(block, x, use_reentrant=False) for block in self.output_block]
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
