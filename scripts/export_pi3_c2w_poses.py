#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_3dgs_5 import load_rgb_sequence  # noqa: E402
from scripts.export_alignment_poses import rotmat_to_quat_xyzw  # noqa: E402
from scripts.optimize_ply_gsplat_sh0 import load_or_predict_cameras  # noqa: E402


def save_tum(path, frame_names, c2w):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# frame_index frame_name tx ty tz qx qy qz qw\n")
        for idx, (name, pose) in enumerate(zip(frame_names, c2w)):
            quat = rotmat_to_quat_xyzw(pose[:3, :3])
            tx, ty, tz = pose[:3, 3]
            handle.write(
                f"{idx} {name} {tx:.9g} {ty:.9g} {tz:.9g} "
                f"{quat[0]:.9g} {quat[1]:.9g} {quat[2]:.9g} {quat[3]:.9g}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Export Pi3-predicted c2w camera poses for an RGB sequence.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ckpt",
        default="outputs/pi3_3dgs12_hunyuan_decoder_0620_fix/ckpts/best_model/model.safetensors",
    )
    parser.add_argument(
        "--model-config",
        default="outputs/pi3_3dgs12_hunyuan_decoder_0620_fix/.hydra/config.yaml",
    )
    parser.add_argument("--model-impl", default="_12")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-cache", default=None)
    parser.add_argument("--force-recompute-cameras", action="store_true")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--subset-start", type=int, default=None)
    parser.add_argument("--subset-end", type=int, default=None)
    parser.add_argument("--subset-step", type=int, default=1)
    parser.add_argument("--target-frame-count", type=int, default=None)
    parser.add_argument("--pixel-limit", type=int, default=255000)
    parser.add_argument("--gs-view-stride", type=int, default=1)
    parser.add_argument("--geometry-head-view-chunk-size", type=int, default=120)
    parser.add_argument("--gs-decoder-view-chunk-size", type=int, default=48)
    parser.add_argument("--gs-head-view-chunk-size", type=int, default=1)
    parser.add_argument("--disable-quadtree", action="store_true")
    parser.add_argument("--disable-local-competition", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.camera_cache is None:
        args.camera_cache = str(output_dir / "pi3_predicted_cameras.npz")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    imgs_cpu, frame_items, orig_hw, target_hw = load_rgb_sequence(
        args.data_path,
        interval=args.interval,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        subset_step=args.subset_step,
        pixel_limit=args.pixel_limit,
        target_frame_count=args.target_frame_count,
    )
    if imgs_cpu.numel() == 0:
        raise RuntimeError(f"No RGB frames loaded from {args.data_path}")

    camera_args = SimpleNamespace(
        camera_cache=args.camera_cache,
        force_recompute_cameras=args.force_recompute_cameras,
        ckpt=args.ckpt,
        model_config=args.model_config,
        model_impl=args.model_impl,
        gs_view_stride=args.gs_view_stride,
        disable_quadtree=args.disable_quadtree,
        disable_local_competition=args.disable_local_competition,
        geometry_head_view_chunk_size=args.geometry_head_view_chunk_size,
        gs_decoder_view_chunk_size=args.gs_decoder_view_chunk_size,
        gs_head_view_chunk_size=args.gs_head_view_chunk_size,
    )
    w2c_cpu, K_cpu = load_or_predict_cameras(camera_args, imgs_cpu, frame_items, device)
    w2c = w2c_cpu.numpy().astype(np.float64)
    K = K_cpu.numpy().astype(np.float64)
    c2w = np.linalg.inv(w2c)
    centers = c2w[:, :3, 3]
    frame_stems = np.array([str(item["stem"]) for item in frame_items], dtype=object)
    frame_names = np.array([Path(str(item["path"])).name if item.get("path") else f"{item['stem']}.png" for item in frame_items])
    source_indices = np.array([int(item["source_index"]) for item in frame_items], dtype=np.int32)

    pose_path = output_dir / "pi3_rgbonly_c2w_poses.npz"
    np.savez_compressed(
        pose_path,
        frame_names=frame_names,
        frame_stems=frame_stems,
        source_indices=source_indices,
        w2c=w2c.astype(np.float32),
        c2w=c2w.astype(np.float32),
        K=K.astype(np.float32),
        camera_centers=centers.astype(np.float32),
        orig_hw=np.array(orig_hw, dtype=np.int32),
        target_hw=np.array(target_hw, dtype=np.int32),
        data_path=np.array([str(args.data_path)]),
        camera_cache=np.array([str(args.camera_cache)]),
    )
    save_tum(output_dir / "pi3_rgbonly_c2w_tum.txt", frame_names, c2w)

    summary = {
        "output_dir": str(output_dir),
        "data_path": str(args.data_path),
        "camera_cache": str(args.camera_cache),
        "pose_npz": str(pose_path),
        "pose_tum": str(output_dir / "pi3_rgbonly_c2w_tum.txt"),
        "frame_count": int(len(frame_names)),
        "frame_names_first_last": [str(frame_names[0]), str(frame_names[-1])],
        "source_indices_first_last": [int(source_indices[0]), int(source_indices[-1])],
        "orig_hw": [int(x) for x in orig_hw],
        "target_hw": [int(x) for x in target_hw],
        "ckpt": str(args.ckpt),
        "model_config": str(args.model_config),
        "model_impl": str(args.model_impl),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
