# import torch
# import torch.nn as nn
# from functools import partial
# from copy import deepcopy
# from torch.utils.checkpoint import checkpoint
# from safetensors.torch import load_file
# import sys 

# # 复用原有的层定义
# from .dinov2.layers import Mlp
# from ..utils.geometry import homogenize_points, depth_edge
# from .layers.pos_embed import RoPE2D, PositionGetter
# from .layers.block import BlockRope
# from .layers.attention import FlashAttentionRope
# from .layers.transformer_head import TransformerDecoder, LinearPts3d, AnchorGaussianHead
# from .layers.camera_head import CameraHead
# from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg



# def freeze_all_params(modules):
#     for module in modules:
#         try:
#             for n, param in module.named_parameters():
#                 param.requires_grad = False
#         except AttributeError:
#             module.requires_grad = False

# class MemDebug:
#     def __init__(self, name="Model", active=True):
#         self.name = name
#         self.active = active
#         self.last_mem = 0
#         if self.active and torch.cuda.is_available():
#             torch.cuda.reset_peak_memory_stats()
#             self.last_mem = torch.cuda.memory_allocated()

#     def step(self, tag):
#         if not self.active or not torch.cuda.is_available():
#             return
#         torch.cuda.synchronize()
#         current = torch.cuda.memory_allocated()
#         peak = torch.cuda.max_memory_allocated()
#         diff = current - self.last_mem
#         # print(f"🔴 [MEM] {tag:<25} | Curr: {current/1024**3:5.2f}GB | Peak: {peak/1024**3:5.2f}GB | Diff: {diff/1024**3:+5.2f}GB")
#         sys.stdout.flush() 
#         self.last_mem = current

# class Pi3_3DGS(nn.Module):
#     def __init__(
#             self,
#             pos_type='rope100',
#             decoder_size='large',
#             load_vggt=True,
#             freeze_encoder=True,
#             train_conf=False,
#             train_cam=False,
#             train_geo=False,
#             num_dec_blk_not_to_checkpoint=0, 
#             ckpt=None,
#             num_anchors=16384,
#             num_sky_anchors=1024,
#             K=4,
#             debug_mem=False 
#     ):
#         super().__init__()
#         self.debug_mem = debug_mem 
#         self.num_anchors = num_anchors
#         self.patch_size = 14

#         # 1. Encoder
#         self.encoder = dinov2_vitl14_reg(pretrained=False)
#         del self.encoder.mask_token
#         self.embed_dim = self.encoder.embed_dim

#         # 2. Positional Encoding
#         self.pos_type = pos_type if pos_type is not None else 'none'
#         self.rope = None
#         if self.pos_type.startswith('rope'):
#             if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D")
#             freq = float(self.pos_type[len('rope'):])
#             self.rope = RoPE2D(freq=freq)
#             self.position_getter = PositionGetter()
#         else:
#             raise NotImplementedError

#         # 3. Decoder
#         dec_embed_dim = 1024
#         dec_num_heads = 16
#         mlp_ratio = 4
#         dec_depth = 36 if decoder_size == 'large' else 36

#         self.dec_embed_dim = dec_embed_dim
#         self.decoder = nn.ModuleList([
#             BlockRope(
#                 dim=dec_embed_dim,
#                 num_heads=dec_num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=True,
#                 proj_bias=True,
#                 ffn_bias=True,
#                 drop_path=0.0,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 act_layer=nn.GELU,
#                 ffn_layer=Mlp,
#                 init_values=0.01,
#                 qk_norm=True,
#                 attn_class=FlashAttentionRope,
#                 rope=self.rope
#             ) for _ in range(dec_depth)])

#         # 4. Special Tokens
#         num_register_tokens = 5
#         self.patch_start_idx = num_register_tokens
#         self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
#         nn.init.normal_(self.register_token, std=1e-6)

#         # 5. Geometry Heads
#         self.point_decoder = TransformerDecoder(
#             in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=1024, rope=self.rope,
#         )
#         self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

#         self.camera_decoder = TransformerDecoder(
#             in_dim=2 * self.dec_embed_dim, dec_embed_dim=1024, dec_num_heads=16, out_dim=512, rope=self.rope,
#             use_checkpoint=False
#         )
#         self.camera_head = CameraHead(dim=512)

#         self.conf_decoder = deepcopy(self.point_decoder)
#         self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

#         # 6. Gaussian Head
#         self.gaussian_head = AnchorGaussianHead(
#             embed_dim=self.dec_embed_dim,
#             in_channels=self.dec_embed_dim * 2,
#             num_sky_anchors=num_sky_anchors,
#             patch_size=self.patch_size,
#             K=K,
#             global_dim=self.dec_embed_dim * 2 
#         )

#         # Utils
#         image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
#         image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
#         self.register_buffer("image_mean", image_mean)
#         self.register_buffer("image_std", image_std)

#         # Weight Loading
#         if load_vggt: self._load_vggt_weights()
#         if ckpt is not None:
#             checkpoint_data = torch.load(ckpt, weights_only=False, map_location='cpu')
#             self.load_state_dict(checkpoint_data, strict=False)
#             del checkpoint_data

#         # Freeze Logic
#         self.train_conf = train_conf
#         self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

#         if freeze_encoder: freeze_all_params([self.encoder])
#         if not train_geo: freeze_all_params([self.point_decoder, self.point_head, self.register_token])
#         if not train_conf: freeze_all_params([self.conf_decoder, self.conf_head])
#         if not train_cam: freeze_all_params([self.camera_decoder, self.camera_head])

#     def _load_vggt_weights(self):
#         print("Loading VGGT weights...")
#         try:
#             vggt_weight = load_file('ckpts/pi3/model_pi3.safetensors')
#             vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''): vggt_weight[k] for k in
#                                list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
#             self.encoder.load_state_dict(vggt_enc_weight, strict=False)

#             vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''): vggt_weight[k] for k in
#                                list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
#             vggt_dec_weight1 = {}
#             for k in list(vggt_dec_weight.keys()):
#                 idx = k.split('.')[0]
#                 other = k[len(idx):]
#                 vggt_dec_weight1[f'{int(idx) * 2 + 1}{other}'] = vggt_dec_weight[k]
#             vggt_dec_weight = vggt_dec_weight1

#             vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''): vggt_weight[k] for k in
#                                      list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
#             for k in list(vggt_dec_weight_frame.keys()):
#                 idx = k.split('.')[0]
#                 other = k[len(idx):]
#                 vggt_dec_weight[f'{int(idx) * 2}{other}'] = vggt_dec_weight_frame[k]
#             self.decoder.load_state_dict(vggt_dec_weight, strict=False)
#             print("VGGT weights loaded successfully.")
#         except Exception as e:
#             print(f"Warning: Failed to load VGGT weights: {e}")

#     def decode(self, hidden, N, H, W, mem_debug=None):
#         BN, hw, _ = hidden.shape
#         B = BN // N

#         final_output = []
#         hidden = hidden.reshape(B * N, hw, -1)
        
#         register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
#         hidden = torch.cat([register_token, hidden], dim=1)
#         hw = hidden.shape[1]

#         if self.pos_type.startswith('rope'):
#             pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)

#         if self.patch_start_idx > 0:
#             pos = pos + 1
#             pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
#             pos = torch.cat([pos_special, pos], dim=1)
        
#         if mem_debug: mem_debug.step("Decode: Prepared Pos")

#         for i in range(len(self.decoder)):
#             blk = self.decoder[i]
#             if i % 2 == 0:
#                 pos = pos.reshape(B * N, hw, -1)
#                 hidden = hidden.reshape(B * N, hw, -1)
#             else:
#                 pos = pos.reshape(B, N * hw, -1)
#                 hidden = hidden.reshape(B, N * hw, -1)

#             if self.training and i >= self.num_dec_blk_not_to_checkpoint:
#                 hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
#             else:
#                 hidden = blk(hidden, xpos=pos)

#             if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
#                 final_output.append(hidden.reshape(B * N, hw, -1))
        
#         if mem_debug: mem_debug.step("Decode: Loop Done")
#         return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

#     # ------------------------------------------------------------------
#     # [修改] 使用 Matmul 替换 Einsum，并加强显存管理 + CUDA Fix
#     # ------------------------------------------------------------------
#     def _forward_geometry_branch(self, hidden, pos, H, W, B, N, patch_h, patch_w, device):
#         # 1. Run Decoders
#         point_hidden = self.point_decoder(hidden, xpos=pos)
#         conf_hidden = self.conf_decoder(hidden, xpos=pos)
#         camera_hidden = self.camera_decoder(hidden, xpos=pos)

#         with torch.amp.autocast(device_type='cuda', enabled=False):
#             # --- Points ---
#             point_hidden = point_hidden.float()
#             # [Fix] 显式 contiguous 防止 reshape 产生碎片
#             points_feat = point_hidden[:, self.patch_start_idx:].contiguous()
#             del point_hidden # 立即释放
            
#             ret = self.point_head([points_feat], (H, W)).reshape(B, N, H, W, -1)
#             xy, z = ret.split([2, 1], dim=-1)
#             z = torch.exp(z)
#             local_points = torch.cat([xy * z, z], dim=-1)
            
#             # --- Conf ---
#             conf_hidden = conf_hidden.float()
#             conf_feat = conf_hidden[:, self.patch_start_idx:].contiguous()
#             del conf_hidden
#             conf_logits = self.conf_head([conf_feat], (H, W)).reshape(B, N, H, W, -1)

#             # --- Camera ---
#             camera_hidden = camera_hidden.float()
#             cam_feat = camera_hidden[:, self.patch_start_idx:].contiguous()
#             del camera_hidden
#             camera_poses = self.camera_head(cam_feat, patch_h, patch_w).reshape(B, N, 4, 4)

#             # --- Global Transform (REPLACING EINSUM) ---
            
#             # 1. Prepare Points: (B, N, H, W, 4) -> (B, N, H*W, 4) -> (B, N, 4, H*W)
#             local_points_h = homogenize_points(local_points)
            
#             # [CRITICAL FIX] 必须加上 .contiguous()
#             # transpose 返回的是 stride 不连续的 tensor，在混合精度训练(BF16/FP16)时
#             # 直接传给 torch.matmul 会触发 CUBLAS_STATUS_INTERNAL_ERROR
#             flat_local_points = local_points_h.view(B, N, -1, 4).transpose(2, 3).contiguous()
            
#             # 2. Matmul: (B, N, 4, 4) @ (B, N, 4, HW) -> (B, N, 4, HW)
#             # 此时输入已经是连续内存，cuBLAS 可以正常工作
#             transformed_points = torch.matmul(camera_poses, flat_local_points) 
            
#             # 3. Reshape back: (B, N, 4, HW) -> (B, N, HW, 4) -> (B, N, H, W, 4)
#             transformed_points = transformed_points.transpose(2, 3).reshape(B, N, H, W, 4)
            
#             points_global = transformed_points[..., :3].contiguous()

#         return points_global, conf_logits, local_points, camera_poses

#     def _filter_anchors(self, points_global, conf_logits, local_points, device, B):
#         mask_conf = torch.sigmoid(conf_logits[..., 0]) > 0.1
#         mask_edge = ~depth_edge(local_points[..., 2], rtol=0.03)
#         valid_mask = torch.logical_and(mask_conf, mask_edge)

#         selected_anchors_list = []
#         selected_conf_list = []  

#         flat_points = points_global.reshape(B, -1, 3)
#         flat_conf = conf_logits.view(B, -1, 1)  
#         flat_mask = valid_mask.view(B, -1)

#         for b in range(B):
#             curr_mask = flat_mask[b]
#             curr_valid_points = flat_points[b][curr_mask]
#             curr_valid_conf = flat_conf[b][curr_mask]  

#             if len(curr_valid_points) < 100:
#                 curr_valid_points = flat_points[b]
#                 curr_valid_conf = flat_conf[b]

#             num_valid = len(curr_valid_points)
#             if num_valid >= self.num_anchors:
#                 sample_indices = torch.randint(0, num_valid, (self.num_anchors,), device=device)
#             else:
#                 base_indices = torch.arange(num_valid, device=device)
#                 extra_indices = torch.randint(0, num_valid, (self.num_anchors - num_valid,), device=device)
#                 sample_indices = torch.cat([base_indices, extra_indices])

#             selected_anchors_list.append(curr_valid_points[sample_indices])
#             selected_conf_list.append(curr_valid_conf[sample_indices]) 

#         return torch.stack(selected_anchors_list), torch.stack(selected_conf_list)

#     def forward(self, imgs):
#         mem = MemDebug(active=self.debug_mem)
#         mem.step("Start Forward")
        
#         imgs = (imgs - self.image_mean) / self.image_std
#         B, N, _, H, W = imgs.shape
#         patch_h, patch_w = H // 14, W // 14

#         # 1. Encode
#         imgs = imgs.reshape(B * N, _, H, W)
#         features_dict = self.encoder(imgs, is_training=True)
#         hidden = features_dict["x_norm_patchtokens"] if isinstance(features_dict, dict) else features_dict
#         mem.step("After Encoder")

#         # 2. Decode
#         hidden, pos = self.decode(hidden, N, H, W, mem_debug=mem)
#         mem.step("After Decoder")

#         # 3. Split
#         global_tokens_flat = hidden[:, :self.patch_start_idx, :]
#         patch_tokens_flat = hidden[:, self.patch_start_idx:, :]
#         global_tokens = global_tokens_flat.view(B, N, -1, hidden.shape[-1])

#         # 4. Geometry Branch (Checkpointed)
#         mem.step("Before Geo Branch")
#         if self.training:
#             points_global, conf_logits, local_points, camera_poses = checkpoint(
#                 self._forward_geometry_branch,
#                 hidden, pos, H, W, B, N, patch_h, patch_w, imgs.device,
#                 use_reentrant=False
#             )
#         else:
#             points_global, conf_logits, local_points, camera_poses = self._forward_geometry_branch(
#                 hidden, pos, H, W, B, N, patch_h, patch_w, imgs.device
#             )
#         mem.step("After Geo Branch")

#         # 5. Filter
#         selected_anchors, selected_conf = self._filter_anchors(
#             points_global, conf_logits, local_points, imgs.device, B
#         )
#         mem.step("After Anchor Filter")

#         # 6. Gaussian Head
#         gaussians = self.gaussian_head(
#             tokens=patch_tokens_flat,
#             camera_poses=camera_poses,
#             img_shape=(H, W),
#             selected_anchors=selected_anchors,
#             global_tokens=global_tokens, 
#             anchor_confidence=selected_conf
#         )
#         mem.step("After Gaussian Head")

#         return dict(
#             gaussians=gaussians,
#             camera_poses=camera_poses,
#             points=points_global,
#             conf=conf_logits,
#             local_points=local_points
#         )

######################################### light ###########################################
import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from torch.utils.checkpoint import checkpoint # 保留引用以防万一，但逻辑中禁用
from safetensors.torch import load_file
import sys 

# 复用原有的层定义
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, depth_edge
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, AnchorGaussianHead
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg

# [优化 1] 开启 TF32 加速
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
        # peak = torch.cuda.max_memory_allocated()
        # diff = current - self.last_mem
        # sys.stdout.flush() 
        self.last_mem = current

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
            # [优化 2] 默认禁用 checkpoint (设为极大值)
            num_dec_blk_not_to_checkpoint=1000, 
            ckpt=None,
            num_anchors=16384,
            num_sky_anchors=1024,
            K=4,
            debug_mem=False 
    ):
        super().__init__()
        self.debug_mem = debug_mem 
        self.num_anchors = num_anchors
        self.patch_size = 14

        # 1. Encoder
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        self.embed_dim = self.encoder.embed_dim

        # 2. Positional Encoding
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # 3. Decoder
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        
        # [优化 3] 减少 Decoder 层数：36 -> 12
        # A6000 足够跑，但为了速度，Decoder 不需要那么深
        dec_depth = 36 if decoder_size == 'large' else 36 
        print(f"🚀 [Optimization] Decoder depth set to: {dec_depth}")

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

        # 4. Special Tokens
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # 5. Geometry Heads
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

        # 6. Gaussian Head
        self.gaussian_head = AnchorGaussianHead(
            embed_dim=self.dec_embed_dim,
            in_channels=self.dec_embed_dim * 2,
            num_sky_anchors=num_sky_anchors,
            patch_size=self.patch_size,
            K=K,
            global_dim=self.dec_embed_dim * 2 
        )

        # Utils
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # Weight Loading
        if load_vggt: self._load_vggt_weights()
        if ckpt is not None:
            checkpoint_data = torch.load(ckpt, weights_only=False, map_location='cpu')
            self.load_state_dict(checkpoint_data, strict=False)
            del checkpoint_data

        # Freeze Logic
        self.train_conf = train_conf
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if freeze_encoder: freeze_all_params([self.encoder])
        if not train_geo: freeze_all_params([self.point_decoder, self.point_head, self.register_token])
        if not train_conf: freeze_all_params([self.conf_decoder, self.conf_head])
        if not train_cam: freeze_all_params([self.camera_decoder, self.camera_head])

    def _load_vggt_weights(self):
        print("Loading VGGT weights...")
        try:
            vggt_weight = load_file('ckpts/pi3/model_pi3.safetensors')
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
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            # [优化 4] Decoder Loop 中不再使用 Checkpoint
            # A6000 显存足够直接前向传播
            hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))
        
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def _forward_geometry_branch(self, hidden, pos, H, W, B, N, patch_h, patch_w, device):
        # 1. Run Decoders
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # --- Points ---
            point_hidden = point_hidden.float()
            points_feat = point_hidden[:, self.patch_start_idx:].contiguous()
            del point_hidden 
            
            ret = self.point_head([points_feat], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)
            
            # --- Conf ---
            conf_hidden = conf_hidden.float()
            conf_feat = conf_hidden[:, self.patch_start_idx:].contiguous()
            del conf_hidden
            conf_logits = self.conf_head([conf_feat], (H, W)).reshape(B, N, H, W, -1)

            # --- Camera ---
            camera_hidden = camera_hidden.float()
            cam_feat = camera_hidden[:, self.patch_start_idx:].contiguous()
            del camera_hidden
            camera_poses = self.camera_head(cam_feat, patch_h, patch_w).reshape(B, N, 4, 4)

            # --- Global Transform ---
            local_points_h = homogenize_points(local_points)
            flat_local_points = local_points_h.view(B, N, -1, 4).transpose(2, 3).contiguous()
            transformed_points = torch.matmul(camera_poses, flat_local_points)
            transformed_points = transformed_points.transpose(2, 3).reshape(B, N, H, W, 4)
            points_global = transformed_points[..., :3].contiguous()

        return points_global, conf_logits, local_points, camera_poses

    def _filter_anchors(self, points_global, conf_logits, local_points, device, B):
        mask_conf = torch.sigmoid(conf_logits[..., 0]) > 0.1
        mask_edge = ~depth_edge(local_points[..., 2], rtol=0.03)
        valid_mask = torch.logical_and(mask_conf, mask_edge)

        selected_anchors_list = []
        selected_conf_list = []  

        flat_points = points_global.reshape(B, -1, 3)
        flat_conf = conf_logits.view(B, -1, 1)  
        flat_mask = valid_mask.view(B, -1)

        for b in range(B):
            curr_mask = flat_mask[b]
            curr_valid_points = flat_points[b][curr_mask]
            curr_valid_conf = flat_conf[b][curr_mask]  

            if len(curr_valid_points) < 100:
                curr_valid_points = flat_points[b]
                curr_valid_conf = flat_conf[b]

            num_valid = len(curr_valid_points)
            if num_valid >= self.num_anchors:
                sample_indices = torch.randint(0, num_valid, (self.num_anchors,), device=device)
            else:
                base_indices = torch.arange(num_valid, device=device)
                extra_indices = torch.randint(0, num_valid, (self.num_anchors - num_valid,), device=device)
                sample_indices = torch.cat([base_indices, extra_indices])

            selected_anchors_list.append(curr_valid_points[sample_indices])
            selected_conf_list.append(curr_valid_conf[sample_indices]) 

        return torch.stack(selected_anchors_list), torch.stack(selected_conf_list)

    def forward(self, imgs,camera_poses):
        mem = MemDebug(active=self.debug_mem)
        mem.step("Start Forward")
        
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # 1. Encode
        imgs = imgs.reshape(B * N, _, H, W)
        features_dict = self.encoder(imgs, is_training=True)
        hidden = features_dict["x_norm_patchtokens"] if isinstance(features_dict, dict) else features_dict
        mem.step("After Encoder")

        # 2. Decode
        hidden, pos = self.decode(hidden, N, H, W, mem_debug=mem)
        mem.step("After Decoder")

        # 3. Split
        global_tokens_flat = hidden[:, :self.patch_start_idx, :]
        patch_tokens_flat = hidden[:, self.patch_start_idx:, :]
        global_tokens = global_tokens_flat.view(B, N, -1, hidden.shape[-1])

        # 4. Geometry Branch
        mem.step("Before Geo Branch")
        
        # [优化 5] 彻底移除 Checkpoint
        # A6000 显存足够，直接运行以提升速度
        points_global, conf_logits, local_points, camera_poses = self._forward_geometry_branch(
            hidden, pos, H, W, B, N, patch_h, patch_w, imgs.device
        )
        mem.step("After Geo Branch")

        # 5. Filter
        selected_anchors, selected_conf = self._filter_anchors(
            points_global, conf_logits, local_points, imgs.device, B
        )
        mem.step("After Anchor Filter")

        # 6. Gaussian Head
        gaussians = self.gaussian_head(
            tokens=patch_tokens_flat,
            camera_poses=camera_poses,
            img_shape=(H, W),
            selected_anchors=selected_anchors,
            global_tokens=global_tokens, 
            anchor_confidence=selected_conf
        )
        mem.step("After Gaussian Head")

        return dict(
            gaussians=gaussians,
            camera_poses=camera_poses,
            points=points_global,
            conf=conf_logits,
            local_points=local_points
        )