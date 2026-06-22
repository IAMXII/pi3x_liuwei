import argparse
import json
import math
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


PINHOLE_MODEL_ID = 1


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def probe_video(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,r_frame_rate,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    nb_frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(stream["duration"]),
        "r_frame_rate": stream["r_frame_rate"],
        "nb_frames": int(nb_frames) if nb_frames and nb_frames != "N/A" else None,
    }


def extract_frames(video_path, image_dir, quality):
    image_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-q:v",
        str(quality),
        str(image_dir / "frame_%06d.JPG"),
    ])


def downsample_images(source_dir, target_dir, factor):
    target_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("*.JPG"))
    for idx, path in enumerate(files, start=1):
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            target_size = (max(1, round(width / factor)), max(1, round(height / factor)))
            resized = image.resize(target_size, Image.Resampling.LANCZOS)
            resized.save(target_dir / path.name, quality=95)
        if idx % 100 == 0 or idx == len(files):
            print(f"downsample x{factor}: {idx}/{len(files)}", flush=True)


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def look_at_c2w(position, target=np.zeros(3, dtype=np.float64)):
    forward = normalize(target - position)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right = normalize(right)
    down = normalize(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack([right, down, forward], axis=1)
    c2w[:3, 3] = position
    return c2w


def rotmat_to_qvec(rotmat):
    r = rotmat
    k = np.array([
        [r[0, 0] - r[1, 1] - r[2, 2], 0.0, 0.0, 0.0],
        [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], 0.0, 0.0],
        [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], 0.0],
        [r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0], r[0, 0] + r[1, 1] + r[2, 2]],
    ], dtype=np.float64) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def synthetic_camera_poses(num_frames, radius=2.0):
    poses = []
    for idx in range(num_frames):
        theta = 2.0 * math.pi * idx / max(num_frames, 1)
        position = np.array([
            radius * math.cos(theta),
            0.25 * math.sin(2.0 * theta),
            radius * math.sin(theta),
        ], dtype=np.float64)
        poses.append(look_at_c2w(position))
    return poses


def write_cameras_bin(path, width, height, focal):
    params = [float(focal), float(focal), width / 2.0, height / 2.0]
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, PINHOLE_MODEL_ID, int(width), int(height)))
        f.write(struct.pack("<dddd", *params))


def write_images_bin(path, image_names, poses):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(image_names)))
        for image_id, (image_name, c2w) in enumerate(zip(image_names, poses), start=1):
            rot_w2c = c2w[:3, :3].T
            tvec = -rot_w2c @ c2w[:3, 3]
            qvec = rotmat_to_qvec(rot_w2c)
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<dddd", *qvec.tolist()))
            f.write(struct.pack("<ddd", *tvec.tolist()))
            f.write(struct.pack("<i", 1))
            f.write(image_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


def write_poses_bounds(path, poses, width, height, focal, near=0.1, far=10.0):
    rows = []
    hwf = np.array([height, width, focal], dtype=np.float64)
    for c2w in poses:
        pose_3x5 = np.concatenate([c2w[:3, :4], hwf[:, None]], axis=1)
        rows.append(np.concatenate([pose_3x5.reshape(-1), np.array([near, far])]))
    np.save(path, np.stack(rows, axis=0))


def write_text_summaries(sparse_dir, width, height, focal, image_names):
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Synthetic camera generated from video frames.\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {focal:.8f} {focal:.8f} {width / 2.0:.8f} {height / 2.0:.8f}\n")
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Synthetic circular camera path. No 2D point observations.\n")
        for image_id, name in enumerate(image_names, start=1):
            f.write(f"# {image_id} {name}\n")
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# Empty synthetic points3D file.\n")


def main():
    parser = argparse.ArgumentParser(description="Create a 360_v2-style scene from a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--dataset_root", default="/data/liuwei/dataset/360_v2")
    parser.add_argument("--scene", default="img_8507")
    parser.add_argument("--jpeg_quality", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing scene directory.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    dataset_root = Path(args.dataset_root)
    scene_root = dataset_root / args.scene
    if scene_root.exists() and any(scene_root.iterdir()) and not args.force:
        raise FileExistsError(f"Scene already exists and is not empty: {scene_root}")

    info = probe_video(video_path)
    print(f"video: {video_path}", flush=True)
    print(f"scene: {scene_root}", flush=True)
    print(f"probe: {info}", flush=True)

    image_dir = scene_root / "images"
    extract_frames(video_path, image_dir, args.jpeg_quality)
    image_names = [path.name for path in sorted(image_dir.glob("*.JPG"))]
    if not image_names:
        raise RuntimeError(f"No frames extracted to {image_dir}")
    if info["nb_frames"] is not None and len(image_names) != info["nb_frames"]:
        raise RuntimeError(f"Expected {info['nb_frames']} frames, extracted {len(image_names)}")
    print(f"extracted frames: {len(image_names)}", flush=True)

    for factor in (2, 4, 8):
        downsample_images(image_dir, scene_root / f"images_{factor}", factor)

    with Image.open(image_dir / image_names[0]) as first:
        width, height = first.size
    focal = 0.8 * max(width, height)
    poses = synthetic_camera_poses(len(image_names))

    sparse_dir = scene_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_bin(sparse_dir / "cameras.bin", width, height, focal)
    write_images_bin(sparse_dir / "images.bin", image_names, poses)
    write_points3d_bin(sparse_dir / "points3D.bin")
    write_text_summaries(sparse_dir, width, height, focal, image_names)
    write_poses_bounds(scene_root / "poses_bounds.npy", poses, width, height, focal)

    metadata = {
        "source_video": str(video_path),
        "scene": args.scene,
        "frame_count": len(image_names),
        "image_width": width,
        "image_height": height,
        "focal": focal,
        "pose_source": "synthetic circular look-at path; not COLMAP/SfM estimated",
    }
    with open(scene_root / "video_scene_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
