import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from export_cp_flowchart_assets import load_flowchart_frame_range
from export_cp_quadtree_gaussian_paper_figs import resolve_checkpoint_path
from pi3.models.pi3_3dgs_8 import Pi3_3DGS


def tensor_stats(name, tensor):
    values = tensor.detach().float().reshape(-1).cpu().numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print(f"{name}: no finite values")
        return
    qs = np.percentile(finite, [0, 25, 50, 75, 90, 95, 99, 99.9, 100])
    print(
        f"{name}: mean={finite.mean():.6f} std={finite.std():.6f} "
        f"min={qs[0]:.6f} p25={qs[1]:.6f} p50={qs[2]:.6f} "
        f"p75={qs[3]:.6f} p90={qs[4]:.6f} p95={qs[5]:.6f} "
        f"p99={qs[6]:.6f} p99.9={qs[7]:.6f} max={qs[8]:.6f}"
    )


def scalar_stats(stats, keys):
    for key in keys:
        value = stats.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().mean().cpu().item())
        print(f"{key}: {float(value):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose pi3_3dgs_8 local redundancy on CP frames.")
    parser.add_argument("--data_root", type=str, default="/data/liuwei/dataset/ntu_seq/cp")
    parser.add_argument("--ckpt", type=str, default="outputs/pi3_highres_0506_v6_local_comp/ckpts/best_model/model.safetensors")
    parser.add_argument("--frame_start", type=int, default=800)
    parser.add_argument("--frame_end", type=int, default=900)
    parser.add_argument("--frame_step", type=int, default=10)
    parser.add_argument("--pixel_limit", type=int, default=255000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gs_view_stride", type=int, default=1)
    parser.add_argument("--redundancy_lambda_mercy", type=float, default=None)
    parser.add_argument("--redundancy_neighbor_limit", type=int, default=None)
    parser.add_argument("--enable_local_competition", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    imgs_cpu, _, frame_items, orig_hw, target_hw = load_flowchart_frame_range(
        args.data_root,
        args.frame_start,
        args.frame_end,
        args.frame_step,
        args.pixel_limit,
    )
    print(f"frames: {[item['frame_id'] for item in frame_items]}")
    print(f"resolution: original={orig_hw[0]}x{orig_hw[1]} model={target_hw[0]}x{target_hw[1]}")

    model = Pi3_3DGS(
        pos_type="rope100",
        decoder_size="large",
        ckpt=None,
        debug_mem=False,
        gs_view_stride=args.gs_view_stride,
        enable_local_competition=args.enable_local_competition,
        max_dense_gaussians=0,
        **({
            "redundancy_lambda_mercy": args.redundancy_lambda_mercy,
        } if args.redundancy_lambda_mercy is not None else {}),
        **({
            "redundancy_neighbor_limit": args.redundancy_neighbor_limit,
        } if args.redundancy_neighbor_limit is not None else {}),
    ).to(device).eval()
    ckpt = resolve_checkpoint_path(args.ckpt)
    if ckpt.endswith(".safetensors"):
        weight = load_file(ckpt, device="cpu")
    else:
        weight = torch.load(ckpt, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(weight, strict=False)
    print(f"checkpoint: {ckpt}")
    print(f"load_state_dict: {load_result}")
    del weight

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        amp_context = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    with torch.no_grad():
        with amp_context():
            result = model(imgs_cpu.unsqueeze(0).to(device), return_viz=False)

    gaussians = result["gaussians"]
    stats = result.get("gaussian_stats", {})
    print(f"gaussian_count: {int(gaussians['xyz'].shape[1])}")
    scalar_stats(
        stats,
        [
            "count_before",
            "physical_count",
            "active_count_opacity_002",
            "active_count_opacity_005",
            "mean_redundancy",
            "threshold",
            "redundancy_threshold",
            "voxel_size",
            "cube_size_mean",
            "cube_size_median",
            "local_radius_mean",
            "local_radius_median",
            "competition_candidates",
            "competition_groups",
            "competition_gate_mean",
            "competition_expected_suppressed",
            "redundancy_coef_mean",
            "redundancy_coef_max",
        ],
    )
    tensor_stats("redundancy_score", gaussians["redundancy_score"])
    tensor_stats("redundancy_coef", gaussians["redundancy_coef"])
    tensor_stats("competition_gate", gaussians["competition_gate"])
    tensor_stats("competition_active", gaussians["competition_active"])


if __name__ == "__main__":
    main()
