# 
import os
import os.path as osp
import numpy as np
from PIL import Image
from datasets.base.base_dataset import BaseDataset

def load_intrinsics_from_npz(npz_path):
    """从 metadata npz 文件仅读取相机内参"""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    data = np.load(npz_path)
    return data['camera_intrinsics']

class MatrixCityDataset(BaseDataset):
    """MatrixCity 数据集加载类（精简版）"""

    def __init__(self, data_root='/data/liuwei/dataset/matrixcity_processed', mode='train', verbose=False, **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose
        self.mask_bg = False
        self.sequences = []
        self.num_image = {}

        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        if self.verbose:
            print(f"[MatrixCity] Scanning for '{self.mode}' split in: {data_root}")

        for root, dirs, files in os.walk(data_root):
            if self.mode in dirs:
                split_path = osp.join(root, self.mode)
                category_name = osp.relpath(root, data_root)
                seq_dirs = sorted([d for d in os.listdir(split_path) if osp.isdir(osp.join(split_path, d))])
                
                for seq_name in seq_dirs:
                    seq_path = osp.join(split_path, seq_name)
                    rgb_dir = osp.join(seq_path, 'rgb')

                    if not osp.exists(rgb_dir):
                        continue

                    image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg'))]
                    image_files.sort()

                    if len(image_files) == 0:
                        continue

                    try:
                        valid_indices = [int(osp.splitext(f)[0]) for f in image_files]
                    except ValueError:
                        valid_indices = list(range(len(image_files)))

                    self.num_image[(category_name, seq_name)] = len(valid_indices)
                    self.sequences.append({
                        'cat': category_name,
                        'seq_name': seq_name,
                        'indices': valid_indices,
                        'seq_path': seq_path
                    })

        if self.verbose:
            print(f"[MatrixCity] Successfully loaded {len(self.sequences)} sequences for mode '{mode}'.")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        seq_info = self.sequences[index]
        cat = seq_info['cat']
        seq_name = seq_info['seq_name']
        valid_indices = seq_info['indices']
        seq_path = seq_info['seq_path']

        num_valid = len(valid_indices)
        MAX_STRIDE = 5  
        
        if self.frame_num > 1:
            available_max_stride = (len(valid_indices) - 1) // (self.frame_num - 1)
        else:
            available_max_stride = 1
            
        actual_max_stride = min(MAX_STRIDE, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (self.frame_num - 1) * stride + 1

        max_start_idx = len(valid_indices) - window_size
        start_idx = int(rng.choice(max_start_idx + 1))

        selected_indices = [start_idx + i * stride for i in range(self.frame_num)]
        idxs = [valid_indices[i] for i in selected_indices]

        views = []
        for idx in idxs:
            fname_base = f"{idx:05d}"
            rgb_name_png = f"{fname_base}.png"
            rgb_name_jpg = f"{fname_base}.jpg"
            
            rgb_path = osp.join(seq_path, 'rgb', rgb_name_png)
            instance_name = rgb_name_png
            if not osp.exists(rgb_path):
                rgb_path = osp.join(seq_path, 'rgb', rgb_name_jpg)
                instance_name = rgb_name_jpg
            
            meta_path = osp.join(seq_path, 'metadata', f"{fname_base}.npz")

            if not osp.exists(rgb_path) or not osp.exists(meta_path):
                continue
            
            rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
            camera_intrinsics = load_intrinsics_from_npz(meta_path).astype(np.float32)

            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image, fake_depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
            )

            views.append(dict(
                img=rgb_image,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='MatrixCity',
                label=f"{cat}-{seq_name}",
                instance=instance_name,
            ))
        return views