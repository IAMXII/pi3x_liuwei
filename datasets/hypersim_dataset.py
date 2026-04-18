import os
import os.path as osp

import h5py
import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


def load_intrinsics_from_npz(npz_path):
    """Load camera intrinsics from a HyperSim metadata file."""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")

    data = np.load(npz_path)
    if 'intrinsics' in data:
        return data['intrinsics']
    if 'camera_intrinsics' in data:
        return data['camera_intrinsics']
    raise KeyError(f"'intrinsics' not found in {npz_path}")


class HyperSimDataset(BaseDataset):
    """HyperSim dataset loader aligned with the lightweight WildRGBD flow."""

    def __init__(
        self,
        data_root='/data/liuwei/dataset/hypersim_processed',
        mode='train',
        verbose=False,
        mask_bg='rand',
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose
        self.mask_bg = False
        self.dataset_label = 'HyperSim'

        self.sequences = []
        self.num_image = {}

        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        split_path = osp.join(self.data_root, f'cached_metadata_hypersim_{mode}.h5')

        if osp.exists(split_path):
            self._load_sequences_from_split(split_path)
        else:
            self._scan_sequences_from_disk()

        if self.verbose:
            print(f"[HyperSim] Successfully loaded {len(self.sequences)} sequences for mode '{mode}'.")

    @staticmethod
    def _decode_if_bytes(value):
        return value.decode('utf-8') if isinstance(value, bytes) else value

    @staticmethod
    def _frame_index_from_rgb_name(filename):
        stem = osp.basename(filename)
        if not stem.endswith('_rgb.png'):
            raise ValueError(f"Unexpected HyperSim rgb filename: {filename}")
        return int(stem.split('_', 1)[0])

    def _register_sequence(self, scene_rel_path, valid_indices):
        if len(valid_indices) == 0:
            return

        valid_indices = sorted(valid_indices)
        self.sequences.append((scene_rel_path, valid_indices))
        self.num_image[scene_rel_path] = len(valid_indices)

    def _load_sequences_from_split(self, split_path):
        if self.verbose:
            print(f"[HyperSim] Loading split metadata from {split_path}")

        with h5py.File(split_path, 'r') as hf:
            required_keys = ('images', 'scenes', 'scene_img_list')
            missing_keys = [key for key in required_keys if key not in hf]
            if missing_keys:
                raise KeyError(f"Missing keys in {split_path}: {missing_keys}")

            images = hf['images'][:]
            scenes = hf['scenes'][:]
            scene_img_list = hf['scene_img_list'][:]

        for scene_raw, image_ids in zip(scenes, scene_img_list):
            scene_rel_path = self._decode_if_bytes(scene_raw)
            scene_path = osp.join(self.data_root, scene_rel_path)
            if not osp.isdir(scene_path):
                continue

            valid_indices = []
            for image_id in image_ids:
                image_name = self._decode_if_bytes(images[int(image_id)])
                try:
                    frame_idx = self._frame_index_from_rgb_name(image_name)
                except ValueError:
                    continue

                rgb_path = osp.join(scene_path, image_name)
                meta_path = osp.join(scene_path, f'{frame_idx:06d}_cam.npz')
                if osp.exists(rgb_path) and osp.exists(meta_path):
                    valid_indices.append(frame_idx)

            self._register_sequence(scene_rel_path, valid_indices)

    def _scan_sequences_from_disk(self):
        if self.verbose:
            print(f"[HyperSim] Scanning data root: {self.data_root}")

        scenes = sorted(
            d for d in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, d))
        )

        for scene in scenes:
            scene_dir = osp.join(self.data_root, scene)
            cameras = sorted(
                d for d in os.listdir(scene_dir)
                if osp.isdir(osp.join(scene_dir, d))
            )

            for camera in cameras:
                camera_dir = osp.join(scene_dir, camera)
                rgb_files = sorted(
                    f for f in os.listdir(camera_dir)
                    if f.endswith('_rgb.png')
                )

                valid_indices = []
                for rgb_name in rgb_files:
                    try:
                        frame_idx = self._frame_index_from_rgb_name(rgb_name)
                    except ValueError:
                        continue

                    meta_path = osp.join(camera_dir, f'{frame_idx:06d}_cam.npz')
                    if osp.exists(meta_path):
                        valid_indices.append(frame_idx)

                self._register_sequence(osp.join(scene, camera), valid_indices)

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        scene_rel_path, valid_indices = self.sequences[index]
        scene_path = osp.join(self.data_root, scene_rel_path)

        num_valid = len(valid_indices)
        should_replace = num_valid < self.frame_num
        idxs = rng.choice(valid_indices, self.frame_num, replace=should_replace)
        idxs.sort()

        views = []
        for idx in idxs:
            fname = f'{idx:06d}_rgb.png'
            rgb_path = osp.join(scene_path, fname)
            meta_path = osp.join(scene_path, f'{idx:06d}_cam.npz')

            if not osp.exists(rgb_path) or not osp.exists(meta_path):
                continue

            rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
            camera_intrinsics = load_intrinsics_from_npz(meta_path).astype(np.float32)

            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)
            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                fake_depthmap,
                camera_intrinsics.copy(),
                resolution,
                rng=rng,
                info=rgb_path,
            )

            views.append(
                dict(
                    img=rgb_image,
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset='HyperSim',
                    label=scene_rel_path.replace('/', '-'),
                    instance=fname,
                )
            )

        if len(views) != self.frame_num:
            raise FileNotFoundError(f"Failed to load {self.frame_num} views from {scene_rel_path}")

        return views
