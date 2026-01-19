import os
import os.path as osp
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset


def load_camera_from_npz(npz_path):
    """从 metadata npz 文件读取相机参数"""
    data = np.load(npz_path)
    R = data['R']  # 3x3
    T = data['T']  # 3
    focal_length = data['focal_length']  # 2
    principal_point = data['principal_point']  # 2
    return R, T, focal_length, principal_point


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    """保持和 CO3DV2Dataset 相同的转换"""
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

    def __init__(self, data_root, mode='train', verbose=False, mask_bg='rand', **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose

        # 针对 test 模式的特殊处理：如果是测试，通常不希望随机mask背景
        if self.mode == 'test' and mask_bg == 'rand':
            if self.verbose:
                print(
                    "[WildRGBD] Warning: 'rand' mask_bg in test mode. Forcing mask_bg=False (or True) for determinism.")
            self.mask_bg = False  # 或者根据需求改为 True
        else:
            self.mask_bg = mask_bg

        assert self.mask_bg in (True, False, 'rand')

        self.sequences = []
        self.num_image = {}

        # 1. 扫描 data_root 下的所有类别文件夹
        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        categories = [d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))]
        categories.sort()  # 排序保证不同机器加载顺序一致

        # 2. 根据 mode 构建 json 文件名
        # 如果 mode='train', 读取 selected_seqs_train.json
        # 如果 mode='test',  读取 selected_seqs_test.json
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
                    selected_seqs = json.load(f)
            except Exception as e:
                print(f"  - Error loading json {seq_json_path}: {e}")
                continue

            for seq_name in selected_seqs:
                scene_path = osp.join(data_root, cat, 'scenes', seq_name)
                rgb_dir = osp.join(scene_path, 'rgb')

                if not osp.exists(rgb_dir):
                    print(f"  - Warning: RGB dir missing for {cat}/{seq_name}, skipping.")
                    continue

                img_files = sorted(os.listdir(rgb_dir))
                if len(img_files) == 0:
                    continue

                self.num_image[(cat, seq_name)] = len(img_files)
                self.sequences.append((cat, seq_name))

        if self.verbose:
            print(f"[WildRGBD] Successfully loaded {len(self.sequences)} sequences for mode '{mode}'.")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        cat, seq_name = self.sequences[index]
        scene_path = osp.join(self.data_root, cat, 'scenes', seq_name)

        num_img = self.num_image[(cat, seq_name)]

        # 训练时允许重复采样以填满 frame_num，测试时根据具体需求（通常保持一致）
        should_replace = num_img < self.frame_num
        idxs = rng.choice(num_img, self.frame_num, replace=should_replace)

        # 如果是 test 模式，这里可能需要改为固定排序（例如取前N帧），看你的具体需求
        # 如果只是单纯划分数据集，保持 rng.choice 也可以，但最好使用固定的 seed

        # 处理背景 Mask 逻辑
        if self.mask_bg == 'rand':
            mask_bg = rng.choice(2)  # 0 or 1
        else:
            mask_bg = self.mask_bg  # True or False

        views = []
        for idx in idxs:
            fname = f"{idx:05d}.jpg"
            rgb_path = osp.join(scene_path, 'rgb', fname)
            depth_path = osp.join(scene_path, 'depth', f"{idx:05d}.png")
            mask_path = osp.join(scene_path, 'masks', f"{idx:05d}.png")
            meta_path = osp.join(scene_path, 'metadata', f"{idx:05d}.npz")

            # load RGB and depth
            rgb_image = np.array(Image.open(rgb_path))
            depthmap = np.array(Image.open(depth_path)).astype(np.float32)

            # load camera
            R, T, focal, p0 = load_camera_from_npz(meta_path)
            image_size = np.array([rgb_image.shape[0], rgb_image.shape[1]])
            R, tvec, camera_intrinsics = opencv_from_cameras_projection(R, T, focal, p0, image_size)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R
            camera_pose[:3, 3] = tvec
            camera_pose = np.linalg.inv(camera_pose)

            # mask
            if mask_bg:
                if osp.exists(mask_path):
                    maskmap = np.array(Image.open(mask_path)).astype(np.float32)
                    maskmap = (maskmap / 255.0) > 0.1
                    depthmap *= maskmap
                else:
                    # 如果 mask 文件不存在但要求 mask，可以报错或忽略
                    pass

            # crop/resize if necessary
            # 这里的 info=rgb_path 有助于 debug
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