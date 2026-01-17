from functools import partial
from copy import deepcopy
from safetensors.torch import load_file


# 假设这些是你项目中已经存在的 import
# from .dinov2.layers import Mlp
# from ..utils.geometry import homogenize_points
# ...

class Pi3_3DGS(nn.Module):
    def __init__(
            self,
            # --- Pi3 Original Params ---
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            ckpt=None,
            # --- New Params ---
            num_anchors=8192,
            num_sky_anchors=1024,
            K=4,
            freeze_camera_head=True,
    ):
        super().__init__()
        self.num_anchors = num_anchors

        # ==========================================================
        # 1. Pi3 Backbone
        # ==========================================================
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # Positional Encoding
        self.pos_type = pos_type
        if self.pos_type.startswith('rope'):
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # Decoder
        if decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36  # Default fallback

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

        # Register Tokens
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ==========================================================
        # 2. Geometry Heads (Point, Conf, Camera) - FROZEN
        # ==========================================================
        # Point Decoder
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # Confidence Decoder (新增：为了过滤点)
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # Camera Decoder
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=512, rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # ==========================================================
        # 3. 3DGS Head (Trainable)
        # ==========================================================
        self.gaussian_head = AnchorGaussianHead(
            embed_dim=self.dec_embed_dim,
            num_sky_anchors=num_sky_anchors,
            patch_size=self.patch_size,
            K=K
        )

        # Utils
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # ==========================================================
        # 4. Weight Loading
        # ==========================================================
        if load_vggt:
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
                print("VGGT weights loaded.")
            except Exception as e:
                print(f"Warning: Failed to load VGGT weights: {e}")

        if ckpt is not None:
            checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}')

        # ==========================================================
        # 5. Freezing Logic
        # ==========================================================
        if freeze_encoder:
            freeze_all_params([self.encoder])

        if freeze_camera_head:
            print("Freezing Pi3 Geometry (Decoder + Point/Conf/Cam Heads)...")
            freeze_all_params([
                self.decoder,
                self.point_decoder,
                self.point_head,
                self.conf_decoder,  # 记得冻结 Conf
                self.conf_head,  # 记得冻结 Conf
                self.camera_decoder,
                self.camera_head,
                self.register_token
            ])

    def decode(self, hidden, N, H, W):
        # ... (Decode 逻辑保持不变) ...
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
            hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # 1. Encode & Decode
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict): hidden = hidden["x_norm_patchtokens"]
        hidden, pos = self.decode(hidden, N, H, W)

        # 2. Frozen Geometry Branch
        with torch.no_grad():
            # A. Camera
            cam_hidden = self.camera_decoder(hidden, xpos=pos).float()
            camera_poses = self.camera_head(cam_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # B. Points & Confidence
            pt_hidden = self.point_decoder(hidden, xpos=pos).float()
            conf_hidden = self.conf_decoder(hidden, xpos=pos).float()

            ret = self.point_head([pt_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)  # [B, N, H, W, 3] (Camera Space)

            conf_logits = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W,
                                                                                                  -1)  # [B, N, H, W, 1]

            # C. Unproject to Global
            points_global = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            # ====================================================
            # 3. Filtering Logic (Based on your snippet)
            # ====================================================
            # Mask generation: [B, N, H, W]
            mask_conf = torch.sigmoid(conf_logits[..., 0]) > 0.1
            mask_edge = ~depth_edge(local_points[..., 2], rtol=0.03)  # Use local z for edge detection
            valid_mask = torch.logical_and(mask_conf, mask_edge)

            # Sampling Anchors
            # Since each batch item has different number of valid points, we handle them carefully.
            selected_anchors_list = []

            # Flatten spatial dims: [B, Total_Points, 3] and [B, Total_Points]
            flat_points = points_global.view(B, -1, 3)
            flat_mask = valid_mask.view(B, -1)

            for b in range(B):
                # Extract valid points for this batch item
                curr_valid_points = flat_points[b][flat_mask[b]]  # [N_valid, 3]

                # Fallback: If no points are valid (rare), use all points or a subset
                if len(curr_valid_points) < 10:
                    curr_valid_points = flat_points[b]

                # Random Sampling with replacement (to ensure constant size [num_anchors])
                # We need exactly self.num_anchors
                sample_indices = torch.randint(0, len(curr_valid_points), (self.num_anchors,), device=imgs.device)
                sampled_anchors = curr_valid_points[sample_indices]  # [num_anchors, 3]

                selected_anchors_list.append(sampled_anchors)

            # Stack: [B, num_anchors, 3]
            selected_anchors = torch.stack(selected_anchors_list)

        # 4. Gaussian Generation
        patch_tokens = hidden[:, self.patch_start_idx:, :]

        gaussians = self.gaussian_head(
            tokens=patch_tokens,
            camera_poses=camera_poses.detach(),
            img_shape=(H, W),
            selected_anchors=selected_anchors.detach()  # Pass filtered anchors
        )

        return {
            "gaussians": gaussians,
            "camera_poses": camera_poses,
        }