import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pi3_3dgs_1 import Pi3_3DGS as BasePi3_3DGS
from .pi3_3dgs_1 import freeze_all_params, modules_require_grad


def safe_logit(x, eps=1e-4):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def unfreeze_all_params(modules):
    for module in modules:
        try:
            for _, param in module.named_parameters():
                param.requires_grad = True
        except AttributeError:
            module.requires_grad = True


class EnvironmentCodeProjector(nn.Module):
    """
    Compress global image appearance into an environment code that mainly
    carries weather / illumination / daytime cues instead of geometry.
    """

    def __init__(self, token_dim, style_dim=256, hidden_dim=512, stat_dim=6):
        super().__init__()
        in_dim = token_dim + stat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, style_dim),
        )

    def forward(self, pooled_tokens, image_stats):
        return self.net(torch.cat([pooled_tokens, image_stats], dim=-1))


class ModulatedLinear(nn.Module):
    """
    StyleGAN-like modulation / demodulation applied on per-Gaussian features.
    """

    def __init__(self, in_dim, out_dim, style_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.style = nn.Linear(style_dim, in_dim, bias=False)
        nn.init.zeros_(self.style.weight)

    def forward(self, x, style_code):
        style_scale = 1.0 + self.style(style_code)
        x_mod = x * style_scale[:, None, :]

        modulated_weight = self.weight[None, :, :] * style_scale[:, None, :]
        demod = torch.rsqrt(modulated_weight.square().sum(dim=-1) + self.eps)

        out = torch.einsum("bkc,boc->bko", x_mod, self.weight[None, :, :])
        out = out * demod[:, None, :] + self.bias
        return out


class GaussianAppearanceModulationHead(nn.Module):
    """
    Predict color / opacity deltas from canonical Gaussian context and a
    source-to-target environment delta code.
    """

    def __init__(
        self,
        context_dim=14,
        style_dim=256,
        hidden_dim=128,
        max_color_delta=0.45,
        max_opacity_delta=0.20,
    ):
        super().__init__()
        self.max_color_delta = float(max_color_delta)
        self.max_opacity_delta = float(max_opacity_delta)

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mod1 = ModulatedLinear(hidden_dim, hidden_dim, style_dim)
        self.mod2 = ModulatedLinear(hidden_dim, hidden_dim, style_dim)
        self.output = nn.Linear(hidden_dim, 5)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[4] = -4.0

    def forward(self, gaussian_context, style_delta, gaussian_mask=None):
        h = self.context_proj(gaussian_context)
        h = F.silu(self.mod1(h, style_delta))
        h = F.silu(self.mod2(h, style_delta))

        style_strength = torch.tanh(style_delta.norm(dim=-1, keepdim=True))
        raw = self.output(h) * style_strength[:, None, :]

        color_delta = self.max_color_delta * torch.tanh(raw[..., 0:3])
        opacity_delta = self.max_opacity_delta * torch.tanh(raw[..., 3:4])
        gate = torch.sigmoid(raw[..., 4:5])

        if gaussian_mask is not None:
            gate = gate * gaussian_mask
            color_delta = color_delta * gate
            opacity_delta = opacity_delta * gate

        return {
            "color_delta": color_delta,
            "opacity_delta": opacity_delta,
            "gate": gate,
        }


class Pi3_3DGS(BasePi3_3DGS):
    def __init__(
        self,
        style_code_dim=256,
        style_hidden_dim=512,
        appearance_hidden_dim=128,
        style_ref_mode="random_one",
        detach_base_gaussians_for_style=True,
        freeze_base_model=True,
        unfreeze_point_head=True,
        unfreeze_gs_head=True,
        unfreeze_point_decoder=False,
        unfreeze_gs_decoder=False,
        max_color_delta=0.45,
        max_opacity_delta=0.20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        token_dim = getattr(self.encoder, "embed_dim", self.dec_embed_dim)
        self.style_ref_mode = style_ref_mode
        self.detach_base_gaussians_for_style = bool(detach_base_gaussians_for_style)
        self.freeze_base_model = bool(freeze_base_model)
        self.unfreeze_point_head = bool(unfreeze_point_head)
        self.unfreeze_gs_head = bool(unfreeze_gs_head)
        self.unfreeze_point_decoder = bool(unfreeze_point_decoder)
        self.unfreeze_gs_decoder = bool(unfreeze_gs_decoder)

        self.environment_projector = EnvironmentCodeProjector(
            token_dim=token_dim,
            style_dim=style_code_dim,
            hidden_dim=style_hidden_dim,
        )
        self.appearance_head = GaussianAppearanceModulationHead(
            context_dim=14,
            style_dim=style_code_dim,
            hidden_dim=appearance_hidden_dim,
            max_color_delta=max_color_delta,
            max_opacity_delta=max_opacity_delta,
        )

        if self.freeze_base_model:
            self._freeze_base_modules()
        self._unfreeze_selected_modules()

    def _freeze_base_modules(self):
        freeze_all_params([
            self.encoder,
            self.decoder,
            # self.point_decoder,
            
            # self.gs_decoder,
            self.camera_decoder,
            self.camera_head,
            self.conf_decoder,
            self.conf_head,
        ])
        if hasattr(self, "register_token"):
            self.register_token.requires_grad = False

    def _unfreeze_selected_modules(self):
        modules = []
        if self.unfreeze_point_decoder:
            modules.append(self.point_decoder)
        if self.unfreeze_point_head:
            modules.append(self.point_head)
        if self.unfreeze_gs_decoder:
            modules.append(self.gs_decoder)
        if self.unfreeze_gs_head:
            modules.append(self.gs_head)
        if modules:
            unfreeze_all_params(modules)

    def _pool_environment_code(self, per_image_code, mode):
        B, N, _ = per_image_code.shape
        mean_code = per_image_code.mean(dim=1)
        consistency = (per_image_code - mean_code[:, None, :]).square().mean()

        if N == 1:
            return mean_code, consistency, None

        if mode == "first":
            index = torch.zeros(B, dtype=torch.long, device=per_image_code.device)
            return per_image_code[:, 0], consistency, index

        if self.training and mode == "random_one":
            index = torch.randint(0, N, (B,), device=per_image_code.device)
            pooled = per_image_code[torch.arange(B, device=per_image_code.device), index]
            return pooled, consistency, index

        return mean_code, consistency, None

    def _encode_environment_code(self, imgs, prefix, mode):
        if imgs is None:
            return None, {}

        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)

        B, N_ref, C, H, W = imgs.shape
        imgs_norm = (imgs - self.image_mean) / self.image_std
        flat_imgs = imgs_norm.reshape(B * N_ref, C, H, W)

        enc_ctx = nullcontext() if modules_require_grad([self.encoder]) else torch.no_grad()
        with enc_ctx:
            env_hidden = self.encoder(flat_imgs, is_training=True)
            if isinstance(env_hidden, dict):
                env_hidden = env_hidden["x_norm_patchtokens"]

        pooled_tokens = env_hidden.mean(dim=1)
        raw_imgs = imgs.reshape(B * N_ref, C, H, W)
        img_mean = raw_imgs.mean(dim=(-2, -1))
        img_std = raw_imgs.std(dim=(-2, -1), unbiased=False)
        image_stats = torch.cat([img_mean, img_std], dim=-1)

        per_image_code = self.environment_projector(
            pooled_tokens.float(),
            image_stats.float(),
        ).view(B, N_ref, -1)

        pooled_code, consistency, ref_index = self._pool_environment_code(per_image_code, mode)
        stats = {
            f"{prefix}_style_code_consistency": consistency,
            f"{prefix}_style_code_norm": pooled_code.norm(dim=-1).mean(),
        }
        if ref_index is not None:
            stats[f"{prefix}_style_ref_index"] = ref_index
        return pooled_code, stats

    def _build_gaussian_context(self, gaussians):
        xyz = gaussians["xyz"]
        scale = gaussians["scale"].clamp_min(1e-6)
        color = gaussians["color"]
        opacity = gaussians["opacity"]

        scene_radius = xyz.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-4)
        xyz_norm = xyz / scene_radius[..., None]
        xyz_dir = F.normalize(xyz, dim=-1)
        xyz_dir = torch.where(torch.isfinite(xyz_dir), xyz_dir, torch.zeros_like(xyz_dir))
        dist = xyz.norm(dim=-1, keepdim=True) / scene_radius[..., None]
        scale_log = scale.log()
        color_logit = safe_logit(color)
        opacity_logit = safe_logit(opacity)

        return torch.cat(
            [xyz_norm, xyz_dir, dist, scale_log, color_logit, opacity_logit],
            dim=-1,
        )

    def _stylize_gaussians(self, gaussians, style_delta):
        if style_delta is None:
            return None, {}

        base_xyz = gaussians["xyz"].detach() if self.detach_base_gaussians_for_style else gaussians["xyz"]
        base_scale = gaussians["scale"].detach() if self.detach_base_gaussians_for_style else gaussians["scale"]
        base_color = gaussians["color"].detach() if self.detach_base_gaussians_for_style else gaussians["color"]
        base_opacity = gaussians["opacity"].detach() if self.detach_base_gaussians_for_style else gaussians["opacity"]

        gaussian_context = self._build_gaussian_context(
            {
                "xyz": base_xyz,
                "scale": base_scale,
                "color": base_color,
                "opacity": base_opacity,
            }
        )
        gaussian_mask = (gaussians["opacity"] > 1e-6).float()

        appearance_out = self.appearance_head(
            gaussian_context.float(),
            style_delta.float(),
            gaussian_mask=gaussian_mask.float(),
        )

        base_color_logits = safe_logit(base_color.float())
        base_opacity_logits = safe_logit(base_opacity.float())

        styled_color = torch.sigmoid(base_color_logits + appearance_out["color_delta"]).to(gaussians["color"].dtype)
        styled_opacity = torch.sigmoid(base_opacity_logits + appearance_out["opacity_delta"]).to(gaussians["opacity"].dtype)

        stylized = {k: v for k, v in gaussians.items()}
        stylized["color"] = torch.where(
            gaussian_mask.expand_as(styled_color).bool(),
            styled_color,
            gaussians["color"],
        )
        stylized["opacity"] = torch.where(
            gaussian_mask.bool(),
            styled_opacity,
            gaussians["opacity"],
        )

        appearance_stats = {
            "appearance_color_delta_abs": appearance_out["color_delta"].abs().mean(),
            "appearance_opacity_delta_abs": appearance_out["opacity_delta"].abs().mean(),
            "appearance_gate_mean": appearance_out["gate"].mean(),
        }
        return stylized, appearance_stats

    def forward(self, imgs, imgs_paired=None, style_imgs=None, intrinsics=None, chunk_size=30000, global_step=None):
        source_imgs = imgs
        target_imgs = style_imgs if style_imgs is not None else imgs_paired

        base_pred = super().forward(
            imgs,
            intrinsics=intrinsics,
            chunk_size=chunk_size,
            global_step=global_step,
        )

        source_code, source_stats = self._encode_environment_code(
            source_imgs,
            prefix="source",
            mode="mean",
        )
        target_code, target_stats = self._encode_environment_code(
            target_imgs,
            prefix="target",
            mode=self.style_ref_mode,
        )

        if target_code is not None and source_code is not None:
            style_delta = target_code - source_code
        else:
            style_delta = target_code

        gaussians_modulated, appearance_stats = self._stylize_gaussians(
            base_pred["gaussians"],
            style_delta,
        )

        zero = base_pred["gaussians"]["xyz"].new_tensor(0.0)
        base_pred.update(source_stats)
        base_pred.update(target_stats)
        base_pred.update(appearance_stats)
        base_pred["source_environment_code"] = source_code
        base_pred["target_environment_code"] = target_code
        base_pred["environment_code"] = style_delta
        base_pred["style_code_consistency"] = (
            source_stats.get("source_style_code_consistency", zero)
            + target_stats.get("target_style_code_consistency", zero)
        )
        base_pred["style_code_norm"] = target_stats.get("target_style_code_norm", zero)
        base_pred["style_delta_norm"] = style_delta.norm(dim=-1).mean() if style_delta is not None else zero
        base_pred["gaussians_modulated"] = gaussians_modulated
        base_pred["gaussians_paired"] = gaussians_modulated
        return base_pred

    def forward_style_transfer(self, scene_imgs, style_img, intrinsics=None, chunk_size=30000, global_step=None):
        if style_img.ndim == 3:
            style_img = style_img.unsqueeze(0).unsqueeze(0)
        elif style_img.ndim == 4:
            style_img = style_img.unsqueeze(1)
        return self.forward(
            scene_imgs,
            style_imgs=style_img,
            intrinsics=intrinsics,
            chunk_size=chunk_size,
            global_step=global_step,
        )
