import os
import os.path as osp

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


DEFAULT_CAMERA_INTRINSICS = {
    # Raw camera intrinsics from calib_cam_to_cam.txt.
    "image_00": np.array([
        [788.629315, 0.0, 687.158398],
        [0.0, 786.382230, 317.752196],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),
    "image_01": np.array([
        [785.134093, 0.0, 686.437073],
        [0.0, 782.346289, 321.352788],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),
}


DEFAULT_RECT_INTRINSICS = {
    # data_rect images are already rectified to S_rect, so use P_rect[:3, :3].
    "image_00": np.array([
        [552.554261, 0.0, 682.049453],
        [0.0, 552.554261, 238.769549],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),
    "image_01": np.array([
        [552.554261, 0.0, 682.049453],
        [0.0, 552.554261, 238.769549],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),
}


class KITTIDataset(BaseDataset):
    """KITTI-360 RGB-only dataset using cached camera intrinsics."""

    def __init__(
        self,
        data_root="/data/liuwei/dataset/kitti360/KITTI-360/data_2d_raw",
        mode="train",
        cameras=("image_00", "image_01"),
        image_dir_name="data_rect",
        use_rectified_intrinsics=True,
        max_stride=10,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.mode = mode
        self.cameras = tuple(cameras)
        self.image_dir_name = image_dir_name
        self.use_rectified_intrinsics = bool(use_rectified_intrinsics)
        self.max_stride = int(max_stride)
        self.verbose = verbose
        self.dataset_label = "KITTI360"

        self.sequences = []
        self.num_image = {}
        self.camera_intrinsics_cache = (
            DEFAULT_RECT_INTRINSICS if self.use_rectified_intrinsics else DEFAULT_CAMERA_INTRINSICS
        )

        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        drive_dirs = sorted([
            d for d in os.listdir(data_root)
            if osp.isdir(osp.join(data_root, d))
        ])

        for drive_name in drive_dirs:
            drive_path = osp.join(data_root, drive_name)
            for camera_name in self.cameras:
                rgb_dir = osp.join(drive_path, camera_name, self.image_dir_name)
                if not osp.isdir(rgb_dir) or camera_name not in self.camera_intrinsics_cache:
                    continue

                image_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(".png")])
                if not image_files:
                    continue

                if self.mode == "test":
                    split_files = [f for i, f in enumerate(image_files) if i % 10 == 0]
                elif self.mode == "train":
                    split_files = [f for i, f in enumerate(image_files) if i % 10 != 0]
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}.")

                if len(split_files) < getattr(self, "frame_num", 1):
                    continue

                seq_name = f"{drive_name}/{camera_name}"
                self.num_image[seq_name] = len(split_files)
                self.sequences.append({
                    "seq_name": seq_name,
                    "camera_name": camera_name,
                    "rgb_dir": rgb_dir,
                    "frames": split_files,
                })

        if self.verbose:
            print(f"[KITTI360] Loaded {len(self.sequences)} sequences for {self.mode}.")
        elif len(self.sequences) == 0:
            print(f"[KITTI360] CRITICAL: No sequences found in {data_root} for {self.mode}.")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        seq_info = self.sequences[index]
        frames = seq_info["frames"]
        frame_num = getattr(self, "frame_num", 1)

        if len(frames) < frame_num:
            raise RuntimeError(f"Not enough frames in {seq_info['seq_name']}: {len(frames)} < {frame_num}")

        if frame_num > 1:
            available_max_stride = (len(frames) - 1) // (frame_num - 1)
        else:
            available_max_stride = 1

        actual_max_stride = min(self.max_stride, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (frame_num - 1) * stride + 1
        start_idx = int(rng.choice(len(frames) - window_size + 1))
        selected_frames = [frames[start_idx + i * stride] for i in range(frame_num)]

        camera_intrinsics = self.camera_intrinsics_cache[seq_info["camera_name"]]
        views = []
        for image_name in selected_frames:
            rgb_path = osp.join(seq_info["rgb_dir"], image_name)
            if not osp.exists(rgb_path):
                continue

            rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                fake_depthmap,
                camera_intrinsics.copy(),
                resolution,
                rng=rng,
                info=rgb_path,
            )

            views.append(dict(
                img=rgb_image,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset="KITTI360",
                label=seq_info["seq_name"],
                instance=image_name,
            ))

        if len(views) < frame_num:
            raise RuntimeError(f"Failed to load enough frames for {seq_info['seq_name']}")

        return views
