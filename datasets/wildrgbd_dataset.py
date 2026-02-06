import os
import os.path as osp
import numpy as np
from PIL import Image
import json
from datasets.base.base_dataset import BaseDataset


def load_camera_from_npz(npz_path):
    """从 metadata npz 文件读取相机参数"""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    data = np.load(npz_path)
    
    # 修改：直接读取 camera_pose 和 camera_intrinsics
    # 假设 npz 中的键名为 'camera_pose' 和 'camera_intrinsics'
    camera_pose = data['camera_pose']            # 4x4
    camera_intrinsics = data['camera_intrinsics'] # 3x3
    
    return camera_pose, camera_intrinsics


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    """
    保持和 CO3DV2Dataset 相同的转换
    (注意：由于新的 npz 格式直接提供了 Pose 和 Intrinsics，此函数在 _get_views 中不再被调用，
     保留它是为了维持代码结构完整性)
    """
    R = R[None, :, :]
    T = T[None, :]
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    image_size_wh = image_size[:, ::-1]

    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]


class WildRGBDDataset(BaseDataset):
    """WildRGBD 数据集加载类"""

    def __init__(self, data_root='/data/liuwei/dataset/wildrgbd_processed', mode='train', verbose=False, mask_bg='rand', **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose

        # --- 修改：删除了 mask_bg 的逻辑处理 ---
        # 即使传入 mask_bg 参数，本类现在也会忽略它
        self.mask_bg = False 

        self.sequences = []
        self.num_image = {}

        # 1. 扫描 data_root 下的所有类别文件夹
        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        categories = [d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))]
        categories.sort()

        # 2. 根据 mode 构建 json 文件名
        split_filename = f'selected_seqs_{mode}.json'

        if self.verbose:
            print(f"[WildRGBD] Initializing dataset in '{mode}' mode using split file: {split_filename}")

        for cat in categories:
            seq_json_path = osp.join(data_root, cat, split_filename)

            # 如果该类别下没有对应的 split 文件，跳过
            if not osp.exists(seq_json_path):
                if self.verbose:
                    print(f"  - Skipping category '{cat}': {split_filename} not found.")
                continue

            try:
                with open(seq_json_path, 'r') as f:
                    selected_seqs_data = json.load(f)
            except Exception as e:
                print(f"  - Error loading json {seq_json_path}: {e}")
                continue

            # 遍历 JSON 中的 items
            for seq_key, valid_indices in selected_seqs_data.items():
                
                # --- 解析序列名 ---
                if '/' in seq_key:
                    seq_name = seq_key.split('/')[-1]
                else:
                    seq_name = seq_key
                
                if not seq_name.startswith('scene_'):
                    seq_name = f"scene_{seq_name}"

                # 检查路径是否存在
                scene_path = osp.join(data_root, cat, 'scenes', seq_name)
                rgb_dir = osp.join(scene_path, 'rgb')

                if not osp.exists(rgb_dir):
                    continue

                if len(valid_indices) == 0:
                    continue

                self.num_image[(cat, seq_name)] = len(valid_indices)
                self.sequences.append((cat, seq_name, valid_indices))

        if self.verbose:
            print(f"[WildRGBD] Successfully loaded {len(self.sequences)} sequences for mode '{mode}'.")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        cat, seq_name, valid_indices = self.sequences[index]
        scene_path = osp.join(self.data_root, cat, 'scenes', seq_name)

        num_valid = len(valid_indices)

        # 采样逻辑
        should_replace = num_valid < self.frame_num
        idxs = rng.choice(valid_indices, self.frame_num, replace=should_replace)
        idxs.sort()

        # --- 修改：删除了 mask_bg 的随机选择逻辑 ---

        views = []
        for idx in idxs:
            fname = f"{idx:05d}.jpg"
            rgb_path = osp.join(scene_path, 'rgb', fname)
            depth_path = osp.join(scene_path, 'depth', f"{idx:05d}.png")
            # --- 修改：删除了 mask_path 定义 ---
            meta_path = osp.join(scene_path, 'metadata', f"{idx:05d}.npz")

            if not osp.exists(rgb_path):
                # 简单跳过，防止报错
                pass
            if not osp.exists(depth_path):
                pass
            if not osp.exists(meta_path):
                pass

            # load RGB and depth
            rgb_image = np.array(Image.open(rgb_path))
            depthmap = np.array(Image.open(depth_path)).astype(np.float32)

            # --- 加载相机参数 ---
            # 直接获取 pose 和 intrinsics
            camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
            
            # 类型转换
            camera_pose = camera_pose.astype(np.float32)
            camera_intrinsics = camera_intrinsics.astype(np.float32)
            
            # --- 修改：删除了 Mask 读取和处理逻辑 ---
            # 原有的 mask 读取、二值化以及 depthmap *= maskmap 代码已移除

            # crop/resize if necessary
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='WildRGBD',
                label=f"{cat}-{seq_name}",
                instance=fname,
            ))
        return views