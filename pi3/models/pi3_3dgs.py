import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

# 复用原有的层定义
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, depth_edge
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, AnchorGaussianHead
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


class Pi3_3DGS(nn.Module):
    def __init__(
            self,
            # --- Pi3 标准参数 ---
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            train_conf=False,
            train_cam=False,
            train_geo=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            # --- 3DGS 特有参数 ---
            num_anchors=16384,
            num_sky_anchors=1024,
            K=4,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.patch_size = 14

        # ----------------------
        # 1. Encoder (DinoV2)
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        self.embed_dim = self.encoder.embed_dim  # Ensure this attribute exists

        # ----------------------
        # 2. Positional Encoding
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
        # 3. Decoder
        # ----------------------
        if decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36

        self.dec_embed_dim = dec_embed_dim
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])

        # ----------------------
        # 4. Special Tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        # 5. Geometry Heads (Point, Conf, Camera)
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=512, rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        # 6. 3DGS Head (Trainable)
        # ----------------------
        self.gaussian_head = AnchorGaussianHead(
            embed_dim=self.dec_embed_dim,
            num_sky_anchors=num_sky_anchors,
            patch_size=self.patch_size,
            K=K,
            global_dim=self.embed_dim  # [Innovation B] Pass global input dim
        )

        # Utils
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # ----------------------
        # 7. Weight Loading
        # ----------------------
        if load_vggt:
            self._load_vggt_weights()

        if ckpt is not None:
            checkpoint_data = torch.load(ckpt, weights_only=False, map_location='cpu')
            res = self.load_state_dict(checkpoint_data, strict=False)
            print(f'[Pi3-3DGS] Load checkpoints from {ckpt}: {res}')
            del checkpoint_data

        # ----------------------
        # 8. Freeze Logic
        # ----------------------
        self.train_conf = train_conf
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if freeze_encoder:
            print('[Pi3-3DGS] Freezing Encoder')
            freeze_all_params([self.encoder])

        geo_modules = [self.point_decoder, self.point_head, self.register_token]
        if not train_geo:
            print('[Pi3-3DGS] Freezing Geometry Heads (Points)')
            freeze_all_params(geo_modules)

        conf_modules = [self.conf_decoder, self.conf_head]
        if not train_conf:
            print('[Pi3-3DGS] Freezing Confidence Heads')
            freeze_all_params(conf_modules)

        cam_modules = [self.camera_decoder, self.camera_head]
        if not train_cam:
            print('[Pi3-3DGS] Freezing Camera Heads')
            freeze_all_params(cam_modules)

    def _load_vggt_weights(self):
        print("Loading VGGT weights...")
        try:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''): vggt_weight[k] for k in
                               list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            self.encoder.load_state_dict(vggt_enc_weight, strict=False)

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
            self.decoder.load_state_dict(vggt_dec_weight, strict=False)
            print("VGGT weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load VGGT weights: {e}")

    def decode(self, hidden, N, H, W):
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
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def _filter_anchors(self, points_global, conf_logits, local_points, device, B):
        """
        基于置信度和几何连通性筛选 Anchors。
        [Innovation D] 同时返回筛选后的置信度
        """
        mask_conf = torch.sigmoid(conf_logits[..., 0]) > 0.1
        mask_edge = ~depth_edge(local_points[..., 2], rtol=0.03)
        valid_mask = torch.logical_and(mask_conf, mask_edge)

        selected_anchors_list = []
        selected_conf_list = []  # New

        flat_points = points_global.view(B, -1, 3)
        flat_conf = conf_logits.view(B, -1, 1)  # New
        flat_mask = valid_mask.view(B, -1)

        for b in range(B):
            curr_mask = flat_mask[b]
            curr_valid_points = flat_points[b][curr_mask]
            curr_valid_conf = flat_conf[b][curr_mask]  # New

            # Fallback
            if len(curr_valid_points) < 100:
                curr_valid_points = flat_points[b]
                curr_valid_conf = flat_conf[b]

            # Sampling
            num_valid = len(curr_valid_points)
            if num_valid >= self.num_anchors:
                sample_indices = torch.randint(0, num_valid, (self.num_anchors,), device=device)
            else:
                base_indices = torch.arange(num_valid, device=device)
                extra_indices = torch.randint(0, num_valid, (self.num_anchors - num_valid,), device=device)
                sample_indices = torch.cat([base_indices, extra_indices])

            selected_anchors_list.append(curr_valid_points[sample_indices])
            selected_conf_list.append(curr_valid_conf[sample_indices])  # New

        return torch.stack(selected_anchors_list), torch.stack(selected_conf_list)

    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # 1. Encode
        imgs = imgs.reshape(B * N, _, H, W)
        # DinoV2 output usually dict or tensor
        # We need raw tokens to extract global context for Innovation B
        features_dict = self.encoder(imgs, is_training=True)

        # [Innovation B] Extract Global Tokens (CLS + Registers)
        # Assuming `x_norm_clstoken` or similar exists in features_dict or output
        # If output is tensor, usually [B*N, L, C]. Registers are at start.
        if isinstance(features_dict, dict):
            hidden = features_dict["x_norm_patchtokens"]
            # Try to find global info. If not explicit, take mean of patches or CLS if available
            # DinoV2Reg has registers.
            # 假设 registers 在 dict 中或者拼接在 hidden 前面 (取决于实现细节)
            # 这里假设 global 信息需要从 hidden 聚合或 encoder 另有输出
            # 简单起见，我们暂且认为 encoder 输出包含 registers
            # 如果是标准 dinov2_vitl14_reg，它通常返回 patch tokens。
            # 让我们用 patch tokens 的全局池化作为 global context context placeholder
            global_context = hidden.mean(dim=1, keepdim=True)  # [BN, 1, C]
        else:
            # If tensor [BN, L, C]
            hidden = features_dict
            global_context = hidden[:, 0:1, :]  # CLS token assumption

        # Reshape global context for Gaussian Head: [B, N, L_g, C]
        global_tokens = global_context.view(B, N, -1, self.embed_dim)

        # 2. Decode
        hidden, pos = self.decode(hidden, N, H, W)

        # 3. Geometry Branch
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # A. Unproject Points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # B. Confidence
            conf_hidden = conf_hidden.float()
            conf_logits = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # C. Camera Poses
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4,
                                                                                                               4)

            # D. Transform & Filter
            points_global = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            # [Innovation D] Get Confidence along with Anchors
            selected_anchors, selected_conf = self._filter_anchors(
                points_global, conf_logits, local_points, imgs.device, B
            )

        # 4. Gaussian Head
        patch_tokens = hidden[:, self.patch_start_idx:, :]

        gaussians = self.gaussian_head(
            tokens=patch_tokens,
            camera_poses=camera_poses,
            img_shape=(H, W),
            selected_anchors=selected_anchors,
            global_tokens=global_tokens,  # [Innovation B]
            anchor_confidence=selected_conf  # [Innovation D]
        )

        return dict(
            gaussians=gaussians,
            camera_poses=camera_poses,
            points=points_global,
            conf=conf_logits,
            local_points=local_points
        )