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
    R = data['R']        # 3x3
    T = data['T']        # 3
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
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg

        self.sequences = []
        self.num_image = {}

        # 读取所有 category 的 json 文件
        categories = [d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))]
        for cat in categories:
            seq_json_path = osp.join(data_root, cat, f'selected_seqs_{mode}.json')
            if not osp.exists(seq_json_path):
                continue
            with open(seq_json_path, 'r') as f:
                selected_seqs = json.load(f)

            for seq_name in selected_seqs:
                scene_path = osp.join(data_root, cat, 'scenes', seq_name)
                img_files = sorted(os.listdir(osp.join(scene_path, 'rgb')))

                self.num_image[(cat, seq_name)] = len(img_files)
                self.sequences.append((cat, seq_name))

        if self.verbose:
            print(f"[WildRGBD] Loaded {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        cat, seq_name = self.sequences[index]
        scene_path = osp.join(self.data_root, cat, 'scenes', seq_name)

        num_img = self.num_image[(cat, seq_name)]
        should_replace = num_img < self.frame_num
        idxs = rng.choice(num_img, self.frame_num, replace=should_replace)

        mask_bg = (self.mask_bg is True) or (self.mask_bg == 'rand' and rng.choice(2))

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
                maskmap = np.array(Image.open(mask_path)).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap *= maskmap

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
