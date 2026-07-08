#!/usr/bin/env python3
import argparse
import csv
import json
import os
import struct
from pathlib import Path

import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def read_next_bytes(handle, num_bytes, fmt):
    return struct.unpack("<" + fmt, handle.read(num_bytes))


def qvec_to_rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(rot):
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as handle:
        num_cameras = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(handle, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.array(read_next_bytes(handle, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as handle:
        num_images = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(handle, 4, "i")[0]
            qvec = np.array(read_next_bytes(handle, 32, "dddd"), dtype=np.float64)
            tvec = np.array(read_next_bytes(handle, 24, "ddd"), dtype=np.float64)
            camera_id = read_next_bytes(handle, 4, "i")[0]

            name = b""
            while True:
                char = handle.read(1)
                if char == b"\x00":
                    break
                name += char
            image_name = name.decode("utf-8")

            num_points2d = read_next_bytes(handle, 8, "Q")[0]
            handle.seek(num_points2d * 24, os.SEEK_CUR)

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = qvec_to_rotmat(qvec)
            w2c[:3, 3] = tvec
            images[image_name] = {
                "image_id": int(image_id),
                "camera_id": int(camera_id),
                "w2c": w2c,
            }
    return images


def camera_to_intrinsics(camera, target_width, target_height):
    model = camera["model"]
    params = camera["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    else:
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    K = np.array([[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    K[0, :] *= float(target_width) / float(camera["width"])
    K[1, :] *= float(target_height) / float(camera["height"])
    return K


def load_colmap_poses(sparse_dir, target_width, target_height):
    sparse_dir = Path(sparse_dir)
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    ordered = sorted(images.items(), key=lambda item: item[0])
    frame_names = np.array([name for name, _ in ordered])
    w2c = np.stack([meta["w2c"] for _, meta in ordered], axis=0)
    c2w = np.linalg.inv(w2c)
    K = np.stack(
        [camera_to_intrinsics(cameras[meta["camera_id"]], target_width, target_height) for _, meta in ordered],
        axis=0,
    )
    centers = c2w[:, :3, 3]
    return frame_names, w2c, c2w, K, centers


def load_pi3_cache(path):
    with np.load(path, allow_pickle=True) as data:
        stems = np.array([str(x) for x in data["frame_names"].tolist()])
        frame_names = np.array([name if Path(name).suffix else f"{name}.png" for name in stems])
        w2c = data["w2c"].astype(np.float64)
        K = data["K"].astype(np.float64)
        height = int(np.asarray(data["height"]).reshape(-1)[0])
        width = int(np.asarray(data["width"]).reshape(-1)[0])
    c2w = np.linalg.inv(w2c)
    centers = c2w[:, :3, 3]
    return frame_names, w2c, c2w, K, centers, width, height


def umeyama_similarity(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    covariance = tgt_centered.T @ src_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    src_variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(singular_values * sign) / src_variance)
    translation = tgt_mean - scale * (rotation @ src_mean)
    transformed = scale * (source @ rotation.T) + translation
    errors = np.linalg.norm(transformed - target, axis=1)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "matrix": matrix,
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "max_error": float(np.max(errors)),
        "errors": errors,
    }


def save_pose_npz(path, frame_names, w2c, c2w, K, centers, source):
    np.savez_compressed(
        path,
        frame_names=frame_names,
        w2c=w2c.astype(np.float32),
        c2w=c2w.astype(np.float32),
        K=K.astype(np.float32),
        camera_centers=centers.astype(np.float32),
        source=np.array([source]),
    )


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


def save_centers_csv(path, frame_names, gs_centers, pi3_centers, errors):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_name",
                "gsplat_colmap_x",
                "gsplat_colmap_y",
                "gsplat_colmap_z",
                "pi3_rgbonly_x",
                "pi3_rgbonly_y",
                "pi3_rgbonly_z",
                "pi3_to_gsplat_center_error",
            ]
        )
        for name, gs_center, pi3_center, error in zip(frame_names, gs_centers, pi3_centers, errors):
            writer.writerow([name, *gs_center.tolist(), *pi3_center.tolist(), float(error)])


def parse_args():
    parser = argparse.ArgumentParser(description="Export paired COLMAP/gsplat and Pi3 poses for alignment.")
    parser.add_argument("--colmap-sparse-dir", default="outputs/colmap_family_mp4_even120_from_scratch/sparse/0")
    parser.add_argument(
        "--pi3-camera-cache",
        default="outputs/gsplat_sh0_opt_family_mp4_even120_fixed_xyz_all_params_from_pi3_ply/pi3_predicted_cameras.npz",
    )
    parser.add_argument("--output-dir", default="outputs/pose_exports_family_mp4_even120_alignment")
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--target-height", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pi3_names, pi3_w2c, pi3_c2w, pi3_K, pi3_centers, pi3_width, pi3_height = load_pi3_cache(args.pi3_camera_cache)
    target_width = args.target_width or pi3_width
    target_height = args.target_height or pi3_height
    gs_names, gs_w2c, gs_c2w, gs_K, gs_centers = load_colmap_poses(
        args.colmap_sparse_dir,
        target_width,
        target_height,
    )

    if list(gs_names) != list(pi3_names):
        raise RuntimeError(
            "COLMAP and Pi3 frame names do not match:\n"
            f"COLMAP first/last={gs_names[0]}/{gs_names[-1]}\n"
            f"Pi3 first/last={pi3_names[0]}/{pi3_names[-1]}"
        )

    sim3 = umeyama_similarity(pi3_centers, gs_centers)
    inverse_matrix = np.linalg.inv(sim3["matrix"])

    save_pose_npz(
        output_dir / "gsplat_colmap_poses.npz",
        gs_names,
        gs_w2c,
        gs_c2w,
        gs_K,
        gs_centers,
        str(args.colmap_sparse_dir),
    )
    save_pose_npz(
        output_dir / "pi3_rgbonly_poses.npz",
        pi3_names,
        pi3_w2c,
        pi3_c2w,
        pi3_K,
        pi3_centers,
        str(args.pi3_camera_cache),
    )
    save_tum(output_dir / "gsplat_colmap_c2w_tum.txt", gs_names, gs_c2w)
    save_tum(output_dir / "pi3_rgbonly_c2w_tum.txt", pi3_names, pi3_c2w)
    save_centers_csv(output_dir / "paired_camera_centers.csv", gs_names, gs_centers, pi3_centers, sim3["errors"])

    transform = {
        "description": "Maps Pi3 rgbonly camera-center/world coordinates into the COLMAP/gsplat coordinate system: x_gsplat = scale * R @ x_pi3 + t.",
        "source": "pi3_rgbonly",
        "target": "gsplat_colmap",
        "frame_count": int(len(gs_names)),
        "scale": sim3["scale"],
        "rotation_3x3": sim3["rotation"].tolist(),
        "translation_3": sim3["translation"].tolist(),
        "matrix_4x4": sim3["matrix"].tolist(),
        "inverse_matrix_4x4": inverse_matrix.tolist(),
        "center_alignment_rmse": sim3["rmse"],
        "center_alignment_mean_error": sim3["mean_error"],
        "center_alignment_median_error": sim3["median_error"],
        "center_alignment_max_error": sim3["max_error"],
    }
    with open(output_dir / "similarity_pi3_to_gsplat.json", "w", encoding="utf-8") as handle:
        json.dump(transform, handle, indent=2)

    summary = {
        "output_dir": str(output_dir),
        "frame_count": int(len(gs_names)),
        "frame_names_first_last": [str(gs_names[0]), str(gs_names[-1])],
        "target_resolution": [int(target_height), int(target_width)],
        "gsplat_colmap_sparse_dir": str(args.colmap_sparse_dir),
        "pi3_camera_cache": str(args.pi3_camera_cache),
        "files": {
            "gsplat_colmap_poses": str(output_dir / "gsplat_colmap_poses.npz"),
            "pi3_rgbonly_poses": str(output_dir / "pi3_rgbonly_poses.npz"),
            "paired_camera_centers": str(output_dir / "paired_camera_centers.csv"),
            "similarity_pi3_to_gsplat": str(output_dir / "similarity_pi3_to_gsplat.json"),
            "gsplat_colmap_tum": str(output_dir / "gsplat_colmap_c2w_tum.txt"),
            "pi3_rgbonly_tum": str(output_dir / "pi3_rgbonly_c2w_tum.txt"),
        },
        "sim3_center_alignment": {
            "scale": sim3["scale"],
            "rmse": sim3["rmse"],
            "mean_error": sim3["mean_error"],
            "median_error": sim3["median_error"],
            "max_error": sim3["max_error"],
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
