import json
import os
import os.path as osp

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def _frame_filename(frame):
    image_path = frame.get('image_path') or frame.get('file_path') or frame.get('filename')
    if image_path:
        return osp.basename(image_path)

    timestamp = frame.get('timestamp_us', frame.get('timestamp'))
    if timestamp is not None:
        return f"{timestamp}.jpg"

    return None


def _raw_intrinsics(frame):
    if 'intrinsics_and_pose_18' in frame:
        return np.asarray(frame['intrinsics_and_pose_18'][:4], dtype=np.float32)

    for key in ('camera_intrinsics', 'intrinsics', 'K'):
        if key in frame:
            return np.asarray(frame[key], dtype=np.float32)

    raise KeyError('No intrinsics found in frame metadata.')


def _denormalize_intrinsics(intrinsics, width, height):
    intrinsics = np.asarray(intrinsics, dtype=np.float32)

    if intrinsics.shape == (3, 3):
        camera_intrinsics = intrinsics.copy()
        if np.nanmax(np.abs(camera_intrinsics[:2])) <= 2.0:
            camera_intrinsics[0, :] *= width
            camera_intrinsics[1, :] *= height
            camera_intrinsics[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return camera_intrinsics.astype(np.float32)

    flat = intrinsics.reshape(-1)
    if flat.size < 4:
        raise ValueError(f'Expected at least 4 intrinsics values, got shape {intrinsics.shape}.')

    fx, fy, cx, cy = flat[:4]
    camera_intrinsics = np.eye(3, dtype=np.float32)
    if np.nanmax(np.abs([fx, fy, cx, cy])) <= 2.0:
        camera_intrinsics[0, 0] = fx * width
        camera_intrinsics[1, 1] = fy * height
        camera_intrinsics[0, 2] = cx * width
        camera_intrinsics[1, 2] = cy * height
    else:
        camera_intrinsics[0, 0] = fx
        camera_intrinsics[1, 1] = fy
        camera_intrinsics[0, 2] = cx
        camera_intrinsics[1, 2] = cy

    return camera_intrinsics


class RealDataset(BaseDataset):
    """RealEstate-style real dataset with RGB frames and normalized intrinsics."""

    def __init__(self, data_root='/data/liuwei/dataset/real', mode='train', max_stride=10, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.mode = mode
        self.max_stride = max_stride
        self.verbose = verbose
        self.dataset_label = 'Real'

        split = 'test' if mode in ('test', 'val') else 'train'
        self.frames_root = osp.join(data_root, 'frames', split)
        self.meta_root = osp.join(data_root, 'metadata_json', split)

        self.sequences = np.array(self._scan_dataset(), dtype=object)

        if len(self.sequences) == 0:
            print(f"[{self.__class__.__name__}] CRITICAL: No sequences found in {data_root} for split {split}.")
        else:
            print(f"[{self.__class__.__name__}] Loaded {len(self.sequences)} sequences for {split}.")

    def _scan_dataset(self):
        if not osp.isdir(self.frames_root):
            raise FileNotFoundError(f"Frames root not found: {self.frames_root}")
        if not osp.isdir(self.meta_root):
            raise FileNotFoundError(f"Metadata root not found: {self.meta_root}")

        sequences = []
        for meta_name in sorted(os.listdir(self.meta_root)):
            if not meta_name.endswith('.json'):
                continue

            clip_id = osp.splitext(meta_name)[0]
            frame_dir = osp.join(self.frames_root, clip_id)
            if not osp.isdir(frame_dir):
                continue

            image_files = {f for f in os.listdir(frame_dir) if f.endswith(IMAGE_EXTENSIONS)}
            if not image_files:
                continue

            meta_path = osp.join(self.meta_root, meta_name)
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: failed to read {meta_path}: {e}")
                continue

            frames = []
            for frame in metadata.get('frames', []):
                image_name = _frame_filename(frame)
                if image_name not in image_files:
                    timestamp = frame.get('timestamp_us', frame.get('timestamp'))
                    if timestamp is None:
                        continue
                    matches = [f"{timestamp}{ext}" for ext in IMAGE_EXTENSIONS if f"{timestamp}{ext}" in image_files]
                    image_name = matches[0] if matches else None

                if image_name is None:
                    continue

                try:
                    intrinsics = _raw_intrinsics(frame)
                except Exception:
                    continue

                frames.append((image_name, intrinsics))

            if len(frames) >= getattr(self, 'frame_num', 1):
                sequences.append({
                    'clip_id': clip_id,
                    'frame_dir': frame_dir,
                    'frames': frames,
                })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def _sample_frames(self, frames, rng):
        frame_num = getattr(self, 'frame_num', 1)
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
        selected_frames = self._sample_frames(sequence['frames'], rng)

        views = []
        for image_name, raw_intrinsics in selected_frames:
            rgb_path = osp.join(sequence['frame_dir'], image_name)
            if not osp.exists(rgb_path):
                raise FileNotFoundError(f"Missing image: {rgb_path}")

            rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
            height, width = rgb_image.shape[:2]
            camera_intrinsics = _denormalize_intrinsics(raw_intrinsics, width, height)
            fake_depthmap = np.zeros((height, width), dtype=np.float32)

            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                fake_depthmap,
                camera_intrinsics,
                resolution,
                rng=rng,
                info=rgb_path,
            )

            views.append(dict(
                img=rgb_image,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='Real',
                label=sequence['clip_id'],
                instance=image_name,
            ))

        if len(views) < getattr(self, 'frame_num', 1):
            raise RuntimeError(f"Failed to load enough frames for {sequence['clip_id']}")

        return views
