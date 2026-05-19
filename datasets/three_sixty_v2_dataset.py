import os
import os.path as osp
import struct

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from utils.basic import colmap_to_opencv_intrinsics


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


def _read_next_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def _qvec_to_rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float32)


def _read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.array(_read_next_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float32)
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_id = _read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(_read_next_bytes(fid, 32, "dddd"), dtype=np.float32)
            tvec = np.array(_read_next_bytes(fid, 24, "ddd"), dtype=np.float32)
            camera_id = _read_next_bytes(fid, 4, "i")[0]

            name = b""
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name += char
            image_name = name.decode("utf-8")

            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, os.SEEK_CUR)

            rot_w2c = _qvec_to_rotmat(qvec)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = rot_w2c.T
            c2w[:3, 3] = -rot_w2c.T @ tvec

            images[image_name] = {
                "image_id": int(image_id),
                "camera_id": int(camera_id),
                "camera_pose": c2w,
            }
    return images


def _camera_to_intrinsics(camera, image_size):
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

    intrinsics = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    intrinsics = colmap_to_opencv_intrinsics(intrinsics)

    image_width, image_height = image_size
    scale_x = image_width / float(camera["width"])
    scale_y = image_height / float(camera["height"])
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    return intrinsics.astype(np.float32)


class ThreeSixtyV2Dataset(BaseDataset):
    """Mip-NeRF 360 v2 / 360_v2 loader using COLMAP sparse binary intrinsics."""

    def __init__(
        self,
        data_root="/data/liuwei/dataset/360_v2",
        mode="train",
        image_dir_name="images_4",
        hold_every=8,
        max_stride=10,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.mode = mode
        self.image_dir_name = image_dir_name
        self.hold_every = int(hold_every)
        self.max_stride = int(max_stride)
        self.verbose = verbose
        self.dataset_label = "360_v2"

        if not osp.exists(self.data_root):
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        self.sequences = []
        self.num_image = {}

        scene_names = sorted([
            name for name in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, name))
        ])

        for scene_name in scene_names:
            scene_path = osp.join(self.data_root, scene_name)
            sparse_dir = osp.join(scene_path, "sparse", "0")
            cameras_path = osp.join(sparse_dir, "cameras.bin")
            images_path = osp.join(sparse_dir, "images.bin")
            image_dir = self._resolve_image_dir(scene_path)

            if image_dir is None or not (osp.exists(cameras_path) and osp.exists(images_path)):
                continue

            try:
                cameras = _read_cameras_binary(cameras_path)
                image_metas = _read_images_binary(images_path)
            except Exception as exc:
                if self.verbose:
                    print(f"[360_v2] Skip {scene_name}: failed to read COLMAP sparse files: {exc}")
                continue

            frames = []
            for image_name, image_meta in image_metas.items():
                image_path = osp.join(image_dir, image_name)
                if not osp.exists(image_path):
                    continue
                if image_meta["camera_id"] not in cameras:
                    continue
                frames.append({
                    "image_name": image_name,
                    "image_id": image_meta["image_id"],
                    "image_path": image_path,
                    "camera_id": image_meta["camera_id"],
                    "camera_pose": image_meta["camera_pose"],
                })

            frames = sorted(frames, key=lambda x: x["image_id"])
            frames = self._split_frames(frames)
            if len(frames) < 1:
                continue

            self.sequences.append({
                "scene_name": scene_name,
                "image_dir": image_dir,
                "cameras": cameras,
                "frames": frames,
            })
            self.num_image[scene_name] = len(frames)

        self.sequences = np.array(self.sequences, dtype=object)

        if self.verbose:
            print(f"[360_v2] Loaded {len(self.sequences)} scenes from {self.data_root} for {self.mode}.")

    def _resolve_image_dir(self, scene_path):
        preferred_dir = osp.join(scene_path, self.image_dir_name)
        if osp.isdir(preferred_dir):
            return preferred_dir

        for candidate in ("images", "images_2", "images_4", "images_8"):
            candidate_dir = osp.join(scene_path, candidate)
            if osp.isdir(candidate_dir):
                return candidate_dir
        return None

    def _split_frames(self, frames):
        if self.mode == "train":
            return [frame for i, frame in enumerate(frames) if i % self.hold_every != 0]
        if self.mode in ("test", "val", "validation"):
            return [frame for i, frame in enumerate(frames) if i % self.hold_every == 0]
        if self.mode == "all":
            return frames
        raise ValueError(f"Unsupported mode: {self.mode}.")

    def __len__(self):
        return len(self.sequences)

    def _sample_frame_indices(self, num_valid, rng):
        frame_num = getattr(self, "frame_num", 1)
        if num_valid < frame_num:
            return sorted(rng.choice(num_valid, frame_num, replace=True).tolist())

        if frame_num <= 1:
            return [int(rng.integers(0, num_valid))]

        available_max_stride = (num_valid - 1) // (frame_num - 1)
        actual_max_stride = min(self.max_stride, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (frame_num - 1) * stride + 1
        start_idx = int(rng.choice(num_valid - window_size + 1))
        return [start_idx + i * stride for i in range(frame_num)]

    def _get_views(self, index, resolution, rng):
        seq = self.sequences[index]
        scene_name = seq["scene_name"]
        frames = seq["frames"]
        cameras = seq["cameras"]

        selected_indices = self._sample_frame_indices(len(frames), rng)
        selected_frames = [frames[i] for i in selected_indices]
        self.this_views_info = {
            "scene": scene_name,
            "frames": [frame["image_name"] for frame in selected_frames],
        }

        views = []
        for frame in selected_frames:
            image_path = frame["image_path"]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_size = image.size
                rgb_image = np.array(image)
            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            intrinsics = _camera_to_intrinsics(cameras[frame["camera_id"]], image_size)
            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                fake_depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=image_path,
            )

            views.append({
                "img": rgb_image,
                "camera_pose": frame["camera_pose"].astype(np.float32),
                "camera_intrinsics": intrinsics.astype(np.float32),
                "dataset": self.dataset_label,
                "label": scene_name,
                "instance": frame["image_name"],
            })

        return views
