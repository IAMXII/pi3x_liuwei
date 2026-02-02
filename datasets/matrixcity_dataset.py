import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2  # 需要安装 opencv-python 以读取 .exr
from datasets.base.base_dataset import BaseDataset

# 设置环境变量以允许 OpenCV 读取 EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_camera_from_npz(npz_path):
    """从 metadata npz 文件读取相机参数"""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    data = np.load(npz_path)

    # 读取 camera_pose 和 camera_intrinsics
    # 假设 npz 中的键名为 'camera_pose' 和 'camera_intrinsics'
    camera_pose = data['camera_pose']  # 4x4
    camera_intrinsics = data['camera_intrinsics']  # 3x3

    return camera_pose, camera_intrinsics


class MatrixCityDataset(BaseDataset):
    """MatrixCity 数据集加载类 (使用 .npz metadata 和 .exr depth)"""

    def __init__(self, data_root, mode='train', verbose=False, **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode  # 'train' or 'test'
        self.verbose = verbose

        # MatrixCity 默认不使用 Mask
        self.mask_bg = False

        self.sequences = []
        self.num_image = {}

        # 1. 确定搜索的基础路径: data_root/train 或 data_root/test
        split_root = osp.join(self.data_root, self.mode)

        if not osp.exists(split_root):
            raise FileNotFoundError(f"Split root not found: {split_root}")

        if self.verbose:
            print(f"[MatrixCity] Scanning sequences in: {split_root}")

        # 2. 扫描类别 (例如: aerial, small_city)
        categories = [d for d in os.listdir(split_root) if osp.isdir(osp.join(split_root, d))]
        categories.sort()

        for cat in categories:
            cat_path = osp.join(split_root, cat)
            # 3. 扫描序列 (例如: seq_00, seq_01)
            seqs = [d for d in os.listdir(cat_path) if osp.isdir(osp.join(cat_path, d))]
            seqs.sort()

            for seq_name in seqs:
                seq_path = osp.join(cat_path, seq_name)
                rgb_dir = osp.join(seq_path, 'rgb')
                depth_dir = osp.join(seq_path, 'depth')
                meta_dir = osp.join(seq_path, 'metadata')

                # 检查基本文件夹是否存在
                if not osp.exists(rgb_dir):
                    if self.verbose:
                        print(f"  - Skipping {cat}/{seq_name}: rgb folder missing.")
                    continue

                # 获取有效图片索引
                # 假设 rgb 文件夹下是 00000.png 或 01000.png 等
                image_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')]
                image_files.sort()

                if len(image_files) == 0:
                    continue

                # 提取索引 (假设文件名是纯数字，例如 01000.png -> 1000)
                try:
                    valid_indices = [int(osp.splitext(f)[0]) for f in image_files]
                except ValueError:
                    # 如果文件名包含非数字字符，则无法解析索引，跳过或自定义处理
                    print(f"  - Warning: Skipping {cat}/{seq_name} due to non-numeric filenames.")
                    continue

                self.num_image[(cat, seq_name)] = len(valid_indices)

                # 存储序列信息
                self.sequences.append({
                    'cat': cat,
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

        # 采样逻辑
        should_replace = num_valid < self.frame_num
        idxs = rng.choice(valid_indices, self.frame_num, replace=should_replace)
        idxs.sort()

        views = []
        for idx in idxs:
            # 尝试匹配文件名格式，通常是 5位 (01000.png)
            # 这里先尝试 5位，如果不存在再尝试 4位，增加鲁棒性
            fname_base = f"{idx:05d}"
            rgb_path = osp.join(seq_path, 'rgb', f"{fname_base}.png")
            if not osp.exists(rgb_path):
                rgb_path = osp.join(seq_path, 'rgb', f"{fname_base}.jpg")

            # 如果5位找不到，尝试4位
            if not osp.exists(rgb_path):
                fname_base = f"{idx:04d}"
                rgb_path = osp.join(seq_path, 'rgb', f"{fname_base}.png")
                if not osp.exists(rgb_path):
                    rgb_path = osp.join(seq_path, 'rgb', f"{fname_base}.jpg")

            # 对应的 depth 和 metadata 路径
            depth_path = osp.join(seq_path, 'depth', f"{fname_base}.exr")
            meta_path = osp.join(seq_path, 'metadata', f"{fname_base}.npz")

            # --- 1. Load RGB ---
            if not osp.exists(rgb_path):
                # 容错处理：如果依然找不到，打印警告并返回全黑图
                if self.verbose:
                    print(f"Warning: Image not found {rgb_path}")
                rgb_image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
            else:
                rgb_image = np.array(Image.open(rgb_path))

            # --- 2. Load Depth (EXR) ---
            if not osp.exists(depth_path):
                if self.verbose:
                    print(f"Warning: Depth not found {depth_path}")
                depthmap = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
            else:
                # 使用 cv2 读取 exr, flag=-1 (IMREAD_UNCHANGED) 保持 float32
                depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

                # EXR 有时是多通道的 (RGB)，深度通常在第一通道
                if depthmap.ndim == 3:
                    depthmap = depthmap[:, :, 0]

                depthmap = depthmap.astype(np.float32)

            # --- 3. Load Camera Parameters from NPZ ---
            if not osp.exists(meta_path):
                if self.verbose:
                    print(f"Warning: Metadata not found {meta_path}")
                # 缺省值
                camera_pose = np.eye(4, dtype=np.float32)
                camera_intrinsics = np.eye(3, dtype=np.float32)
            else:
                try:
                    camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
                except Exception as e:
                    print(f"Error loading {meta_path}: {e}")
                    camera_pose = np.eye(4, dtype=np.float32)
                    camera_intrinsics = np.eye(3, dtype=np.float32)

            # 类型转换
            camera_pose = camera_pose.astype(np.float32)
            camera_intrinsics = camera_intrinsics.astype(np.float32)

            # crop/resize if necessary
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='MatrixCity',
                label=f"{cat}-{seq_name}",
                instance=f"{fname_base}.png",
            ))
        return views