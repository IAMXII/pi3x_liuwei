import os
import os.path as osp
import re

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

DEFAULT_WAYMO_INTRINSICS = np.array([
    [974.2257, 0.0, 631.8363],
    [0.0, 968.2562, 638.0002],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def _mode_to_split(mode):
    if mode == "train":
        return "train"
    if mode in ("test", "val", "valid", "validation"):
        return "validation"
    raise ValueError(f"Unsupported mode: {mode}.")


def _normalize_cameras(cameras):
    if cameras is None:
        return None
    if isinstance(cameras, (str, int)):
        cameras = [cameras]

    normalized = []
    for camera in cameras:
        camera = str(camera)
        if camera.startswith("cam"):
            camera = camera[3:]
        normalized.append(camera)
    return tuple(normalized)


def _make_intrinsics(camera_intrinsics):
    if camera_intrinsics is None:
        return DEFAULT_WAYMO_INTRINSICS.copy()

    intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics.copy()

    flat = intrinsics.reshape(-1)
    if flat.size < 4:
        raise ValueError(f"Expected 4 intrinsics values or a 3x3 matrix, got {intrinsics.shape}.")

    fx, fy, cx, cy = flat[:4]
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _frame_key(image_name):
    match = re.match(r"(\d+)_cam(\d+)\.[^.]+$", image_name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


class WaymoDataset(BaseDataset):
    """Waymo exported RGB dataset with fixed camera intrinsics."""

    def __init__(
        self,
        data_root="/data/liuwei/dataset/waymo",
        mode="train",
        image_dir_name="exported_images",
        cameras=None,
        camera_intrinsics=None,
        max_stride=10,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.mode = mode
        self.split = _mode_to_split(mode)
        self.image_dir_name = image_dir_name
        self.cameras = _normalize_cameras(cameras)
        self.camera_intrinsics = _make_intrinsics(camera_intrinsics)
        self.max_stride = int(max_stride)
        self.verbose = verbose
        self.dataset_label = "Waymo"

        self.sequences = []
        self.num_image = {}

        self.images_root = osp.join(data_root, self.split, image_dir_name)
        if not osp.isdir(self.images_root):
            raise FileNotFoundError(f"Waymo images root not found: {self.images_root}")

        self.sequences = self._scan_dataset()

        if self.verbose:
            print(f"[Waymo] Loaded {len(self.sequences)} camera sequences from {self.images_root}.")
        elif len(self.sequences) == 0:
            print(f"[Waymo] CRITICAL: No sequences found in {self.images_root}.")

    def _scan_dataset(self):
        sequences = []
        frame_num = getattr(self, "frame_num", 1)

        segment_names = sorted([
            name for name in os.listdir(self.images_root)
            if osp.isdir(osp.join(self.images_root, name))
        ])

        for segment_name in segment_names:
            segment_dir = osp.join(self.images_root, segment_name)
            frames_by_camera = {}

            for image_name in os.listdir(segment_dir):
                if not image_name.endswith(IMAGE_EXTENSIONS):
                    continue

                parsed = _frame_key(image_name)
                if parsed is None:
                    continue

                timestamp, camera = parsed
                if self.cameras is not None and camera not in self.cameras:
                    continue

                frames_by_camera.setdefault(camera, []).append((timestamp, image_name))

            for camera, frames in sorted(frames_by_camera.items()):
                frames = [image_name for _, image_name in sorted(frames)]
                if len(frames) < frame_num:
                    continue

                seq_name = f"{segment_name}/cam{camera}"
                self.num_image[seq_name] = len(frames)
                sequences.append({
                    "seq_name": seq_name,
                    "segment_name": segment_name,
                    "camera": camera,
                    "segment_dir": segment_dir,
                    "frames": frames,
                })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def _sample_frames(self, frames, rng):
        frame_num = getattr(self, "frame_num", 1)
        if len(frames) < frame_num:
            raise RuntimeError(f"Not enough frames: {len(frames)} < {frame_num}")

        if frame_num == 1:
            return [frames[int(rng.integers(len(frames)))]]

        available_max_stride = (len(frames) - 1) // (frame_num - 1)
        actual_max_stride = min(self.max_stride, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (frame_num - 1) * stride + 1
        start_idx = int(rng.choice(len(frames) - window_size + 1))

        return [frames[start_idx + i * stride] for i in range(frame_num)]

    def _get_views(self, index, resolution, rng):
        sequence = self.sequences[index]
        selected_frames = self._sample_frames(sequence["frames"], rng)

        views = []
        for image_name in selected_frames:
            rgb_path = osp.join(sequence["segment_dir"], image_name)
            if not osp.exists(rgb_path):
                raise FileNotFoundError(f"Missing image: {rgb_path}")

            rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                fake_depthmap,
                self.camera_intrinsics.copy(),
                resolution,
                rng=rng,
                info=rgb_path,
            )

            views.append(dict(
                img=rgb_image,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="Waymo",
                label=sequence["seq_name"],
                instance=image_name,
            ))

        if len(views) < getattr(self, "frame_num", 1):
            raise RuntimeError(f"Failed to load enough frames for {sequence['seq_name']}")

        return views
