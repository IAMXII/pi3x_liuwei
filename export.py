import torch
import argparse
import numpy as np
import os
import argparse
import torch
import imageio

from scipy.spatial.transform import Rotation as R
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X


# ============================================================
# Utils
# ============================================================

def save_pointcloud(res, imgs, save_path, conf_thresh=0.1):
    """
    Save world-space point cloud as PLY.
    """
    masks = torch.sigmoid(res['conf'][..., 0]) > conf_thresh
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {save_path}")
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], save_path)


def save_tum_poses(res, save_path="camera_poses.tum"):
    """
    Save camera poses in TUM format (cam2world).
    """
    poses = res['camera_poses'][0]  # (N, 4, 4)

    lines = []
    for i, pose in enumerate(poses):
        pose = pose.cpu().numpy()

        t = pose[:3, 3]
        R_mat = pose[:3, :3]
        q = R.from_matrix(R_mat).as_quat()  # qx qy qz qw

        timestamp = float(i)
        lines.append(
            f"{timestamp:.6f} "
            f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
        )

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] TUM poses saved to {save_path}")


def save_depth_maps(res, save_dir="depth", save_mm=True):
    """
    Save depth maps from local_points[..., z].
    """
    local_points = res['local_points'][0]  # (N, h, w, 3)
    depth = local_points[..., 2]  # (N, h, w)

    os.makedirs(save_dir, exist_ok=True)

    # ---------- visualization depth ----------
    depth_vis = depth.clone()
    depth_vis[depth_vis <= 0] = 0

    valid = depth_vis > 0
    d_min = depth_vis[valid].min()
    d_max = depth_vis[valid].max()

    depth_norm = (depth_vis - d_min) / (d_max - d_min + 1e-6)
    depth_uint8 = (depth_norm * 255).to(torch.uint8)

    for i in range(depth_uint8.shape[0]):
        imageio.imwrite(
            os.path.join(save_dir, f"{i:04d}.png"),
            depth_uint8[i].cpu().numpy()
        )

    print(f"[OK] Depth visualization saved to {save_dir}/")

    # ---------- metric depth (mm) ----------
    if save_mm:
        save_dir_mm = save_dir + "_mm"
        os.makedirs(save_dir_mm, exist_ok=True)

        depth_mm = (depth * 1000).clamp(0, 65535).to(torch.uint16)
        for i in range(depth_mm.shape[0]):
            imageio.imwrite(
                os.path.join(save_dir_mm, f"{i:04d}.png"),
                depth_mm[i].cpu().numpy()
            )

        print(f"[OK] Metric depth (mm) saved to {save_dir_mm}/")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Pi3X inference + export")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Image directory or video path")
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional .npz with poses/depths/intrinsics")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default="outputs")

    args = parser.parse_args()

    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith(".mp4") else 1

    device = torch.device(args.device)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print("[INFO] Loading model...")
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(args.ckpt)
        else:
            state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()

    # ------------------------------------------------------------
    # Load conditions
    # ------------------------------------------------------------
    poses = depths = intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        data = np.load(args.conditions_path)
        poses = data.get("poses", None)
        depths = data.get("depths", None)
        intrinsics = data.get("intrinsics", None)

    conditions = dict(
        poses=poses,
        depths=depths,
        intrinsics=intrinsics
    )

    imgs, conditions = load_multimodal_data(
        args.data_path,
        conditions,
        interval=args.interval,
        device=device
    )

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    print("[INFO] Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        res = model(imgs=imgs, **conditions)

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    save_pointcloud(
        res,
        imgs,
        save_path=os.path.join(args.out_dir, "points.ply")
    )

    save_tum_poses(
        res,
        save_path=os.path.join(args.out_dir, "camera_poses.tum")
    )

    save_depth_maps(
        res,
        save_dir=os.path.join(args.out_dir, "depth"),
        save_mm=True
    )

    print("[DONE] All results exported.")