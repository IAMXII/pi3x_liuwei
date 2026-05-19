import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .loss_3dgs_1 import Pi3LossGS as BasePi3LossGS


class Pi3LossGSSlots(BasePi3LossGS):
    def __init__(
        self,
        *args,
        lambda_slot_collapse=0.005,
        slot_collapse_max_samples=1024,
        slot_collapse_position_sigma=0.04,
        slot_collapse_color_sigma=0.10,
        slot_collapse_opacity_threshold=0.03,
        lambda_slot_budget=0.0,
        target_active_ratio=1.0,
        slot_budget_temperature=0.02,
        lambda_slot_rate=0.001,
        slot_rate_reference_count=65536,
        lambda_slot_attention_entropy=0.0,
        slot_render_max_gaussians=131072,
        slot_render_min_gaussians=8192,
        slot_render_opacity_threshold=0.01,
        slot_render_view_chunk_size=1,
        slot_render_checkpoint=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_slot_collapse = float(lambda_slot_collapse)
        self.slot_collapse_max_samples = max(2, int(slot_collapse_max_samples))
        self.slot_collapse_position_sigma = max(float(slot_collapse_position_sigma), 1e-6)
        self.slot_collapse_color_sigma = max(float(slot_collapse_color_sigma), 1e-6)
        self.slot_collapse_opacity_threshold = float(slot_collapse_opacity_threshold)
        self.lambda_slot_budget = float(lambda_slot_budget)
        self.target_active_ratio = float(target_active_ratio)
        self.slot_budget_temperature = max(float(slot_budget_temperature), 1e-6)
        self.lambda_slot_rate = float(lambda_slot_rate)
        self.slot_rate_reference_count = max(1.0, float(slot_rate_reference_count))
        self.lambda_slot_attention_entropy = float(lambda_slot_attention_entropy)
        self.slot_render_max_gaussians = max(0, int(slot_render_max_gaussians))
        self.slot_render_min_gaussians = max(1, int(slot_render_min_gaussians))
        self.slot_render_opacity_threshold = max(0.0, float(slot_render_opacity_threshold))
        self.slot_render_view_chunk_size = max(1, int(slot_render_view_chunk_size))
        self.slot_render_checkpoint = bool(slot_render_checkpoint)

    def _render_gs(self, gaussians, w2c, ks, H, W, render_mode='RGB'):
        if w2c.ndim < 4:
            return super()._render_gs(gaussians, w2c, ks, H, W, render_mode=render_mode)

        def render_chunk(w2c_chunk, ks_chunk):
            if self.slot_render_checkpoint and torch.is_grad_enabled():
                def render_fn(xyz, rotation, scale, opacity, color, viewmats, intrinsics):
                    render_gaussians = {
                        "xyz": xyz,
                        "rotation": rotation,
                        "scale": scale,
                        "opacity": opacity,
                        "color": color,
                    }
                    render_out, _, _ = BasePi3LossGS._render_gs(
                        self,
                        render_gaussians,
                        viewmats,
                        intrinsics,
                        H,
                        W,
                        render_mode=render_mode,
                    )
                    return render_out

                return checkpoint(
                    render_fn,
                    gaussians["xyz"],
                    gaussians["rotation"],
                    gaussians["scale"],
                    gaussians["opacity"],
                    gaussians["color"],
                    w2c_chunk,
                    ks_chunk,
                    use_reentrant=False,
                ), None, None

            return BasePi3LossGS._render_gs(
                self,
                gaussians,
                w2c_chunk,
                ks_chunk,
                H,
                W,
                render_mode=render_mode,
            )

        render_chunks = []
        alpha_chunks = []
        meta_out = None
        for start in range(0, w2c.shape[1], self.slot_render_view_chunk_size):
            end = min(start + self.slot_render_view_chunk_size, w2c.shape[1])
            render_out, alpha_out, meta_out = render_chunk(
                w2c[:, start:end].contiguous(),
                ks[:, start:end].contiguous(),
            )
            render_chunks.append(render_out)
            if torch.is_tensor(alpha_out):
                alpha_chunks.append(alpha_out)

        render_out = torch.cat(render_chunks, dim=1)
        alpha_out = torch.cat(alpha_chunks, dim=1) if len(alpha_chunks) == len(render_chunks) else None
        return render_out, alpha_out, meta_out

    def _gather_gaussians(self, gaussians, gather_idx):
        B, K_keep = gather_idx.shape
        gathered = {}
        for key, value in gaussians.items():
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == B:
                if value.shape[1] == gather_idx.shape[1]:
                    gathered[key] = value
                elif value.shape[1] >= int(gather_idx.max().item()) + 1:
                    expand_shape = (B, K_keep) + (1,) * (value.ndim - 2)
                    idx = gather_idx.view(expand_shape).expand(B, K_keep, *value.shape[2:])
                    gathered[key] = torch.gather(value, dim=1, index=idx)
                else:
                    gathered[key] = value
            else:
                gathered[key] = value
        return gathered

    def _select_render_gaussians(self, pred):
        gaussians = pred["gaussians"]
        opacity = gaussians["opacity"].detach().float().squeeze(-1)
        B, K = opacity.shape

        max_keep = self.slot_render_max_gaussians
        if max_keep <= 0 and self.slot_render_opacity_threshold <= 0:
            return pred, opacity.new_full((B,), float(K))

        active_counts = (opacity > self.slot_render_opacity_threshold).sum(dim=1)
        keep_count = int(active_counts.max().item())
        keep_count = max(self.slot_render_min_gaussians, keep_count)
        if max_keep > 0:
            keep_count = min(max_keep, keep_count)
        keep_count = min(K, keep_count)

        if keep_count >= K:
            return pred, opacity.new_full((B,), float(K))

        gather_idx = torch.topk(opacity, k=keep_count, dim=1, largest=True, sorted=False).indices
        pred_render = dict(pred)
        pred_render["gaussians"] = self._gather_gaussians(gaussians, gather_idx)

        stats = dict(pred.get("gaussian_stats", {}))
        device, dtype = opacity.device, opacity.dtype
        stats.update({
            "slot_render_count": torch.full((B,), float(keep_count), device=device, dtype=dtype),
            "slot_render_active_candidates": active_counts.to(device=device, dtype=dtype),
            "slot_render_candidate_count": torch.full((B,), float(K), device=device, dtype=dtype),
            "slot_render_opacity_threshold": torch.full(
                (B,),
                float(self.slot_render_opacity_threshold),
                device=device,
                dtype=dtype,
            ),
        })
        pred_render["gaussian_stats"] = stats
        return pred_render, torch.full((B,), float(keep_count), device=device, dtype=dtype)

    def _slot_collapse_loss(self, gaussians):
        xyz = gaussians["xyz"].float()
        color = gaussians["color"].float()
        opacity = gaussians["opacity"].float().squeeze(-1)
        B, K, _ = xyz.shape
        if K < 2:
            return xyz.sum() * 0.0

        if K > self.slot_collapse_max_samples:
            idx = torch.linspace(
                0,
                K - 1,
                self.slot_collapse_max_samples,
                device=xyz.device,
            ).long()
            xyz = xyz[:, idx]
            color = color[:, idx]
            opacity = opacity[:, idx]

        losses = []
        for b in range(B):
            active = opacity[b] > self.slot_collapse_opacity_threshold
            if int(active.sum().item()) < 2:
                continue

            xyz_b = xyz[b, active]
            color_b = color[b, active]
            opacity_b = opacity[b, active]
            dist_xyz = torch.cdist(xyz_b, xyz_b).pow(2)
            dist_color = torch.cdist(color_b, color_b).pow(2)
            pos_sim = torch.exp(-dist_xyz / (2.0 * self.slot_collapse_position_sigma ** 2))
            color_sim = torch.exp(-dist_color / (2.0 * self.slot_collapse_color_sigma ** 2))
            opacity_pair = opacity_b[:, None] * opacity_b[None, :]
            sim = pos_sim * color_sim * opacity_pair

            upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
            if bool(upper.any().item()):
                losses.append(sim[upper].mean())

        if len(losses) == 0:
            return xyz.sum() * 0.0
        return torch.stack(losses).mean()

    def _slot_budget_loss(self, gaussians):
        opacity = gaussians["opacity"].float().squeeze(-1)
        active_prob = torch.sigmoid(
            (opacity - self.slot_collapse_opacity_threshold) / self.slot_budget_temperature
        )
        return (active_prob.mean() - self.target_active_ratio) ** 2

    def _slot_rate_loss(self, gaussians):
        if "existence" in gaussians:
            active_mass = gaussians["existence"].float().squeeze(-1).sum(dim=1)
        else:
            active_mass = gaussians["opacity"].float().squeeze(-1).sum(dim=1)
        return (active_mass / self.slot_rate_reference_count).mean()

    def forward(self, pred, gt_raw, *args, **kwargs):
        gaussians = pred["gaussians"]
        pred_render, render_count = self._select_render_gaussians(pred)
        base_loss, details = super().forward(pred_render, gt_raw, *args, **kwargs)

        loss_slot_collapse = self._slot_collapse_loss(gaussians)
        loss_slot_budget = self._slot_budget_loss(gaussians)
        loss_slot_rate = self._slot_rate_loss(gaussians)
        loss_slot_attention_entropy = base_loss.new_tensor(0.0)
        if "gaussian_stats" in pred and "slot_attention_entropy" in pred["gaussian_stats"]:
            loss_slot_attention_entropy = pred["gaussian_stats"]["slot_attention_entropy"].float().mean()

        final_loss = (
            base_loss
            + self.lambda_slot_collapse * loss_slot_collapse
            + self.lambda_slot_budget * loss_slot_budget
            + self.lambda_slot_rate * loss_slot_rate
            + self.lambda_slot_attention_entropy * loss_slot_attention_entropy
        )

        details.update({
            "loss_slot_collapse": loss_slot_collapse,
            "loss_slot_budget": loss_slot_budget,
            "loss_slot_rate": loss_slot_rate,
            "loss_slot_attention_entropy": loss_slot_attention_entropy,
            "slot_render_count": render_count.detach().float().mean(),
            "cur_weight_slot_collapse": torch.tensor(self.lambda_slot_collapse, device=base_loss.device),
            "cur_weight_slot_budget": torch.tensor(self.lambda_slot_budget, device=base_loss.device),
            "cur_weight_slot_rate": torch.tensor(self.lambda_slot_rate, device=base_loss.device),
            "cur_weight_slot_attention_entropy": torch.tensor(
                self.lambda_slot_attention_entropy,
                device=base_loss.device,
            ),
            "total_loss": final_loss,
        })
        return final_loss, details


Pi3LossGS = Pi3LossGSSlots
