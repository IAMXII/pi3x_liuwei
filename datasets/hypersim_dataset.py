import os
import os.path as osp
import numpy as np
from PIL import Image
import h5py
import torch.distributed as dist

# 假设 BaseDataset 在 datasets.base.base_dataset
from datasets.base.base_dataset import BaseDataset


class HyperSimDataset(BaseDataset):
    """
    适配 WildRGBD 框架的 HyperSim 数据集加载类
    """

    def __init__(self, data_root, mode='train', verbose=False, mask_bg='rand', **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose

        # HyperSim 数据通常不需要 mask 处理，保持与 WildRGBD 一致的默认行为
        self.mask_bg = False

        self.sequences = []  # 存储 (scene_path, list_of_frame_indices/names)
        self.num_image = {}

        # --- 1. 元数据缓存逻辑 (保留原 HyperSim 的高效扫描方式) ---
        cache_filename = f"cached_metadata_hypersim_{mode}.h5"
        cache_path = osp.join(self.data_root, cache_filename)

        # 仅主进程进行扫描和写入
        rank = dist.get_rank() if dist.is_initialized() else 0

        if not osp.exists(cache_path) and rank == 0:
            if self.verbose:
                print(f"[HyperSim] Scanning data root: {self.data_root}...")

            # 扫描逻辑：遍历所有场景 -> 遍历所有相机视点(子场景)
            all_scenes = sorted([
                d for d in os.listdir(self.data_root)
                if osp.isdir(osp.join(self.data_root, d))
            ])

            cache_sequences = []  # 临时存储用于写入 h5

            for scene in all_scenes:
                scene_dir = osp.join(self.data_root, scene)
                # HyperSim 结构通常是: root/scene_name/camera_name/
                sub_dirs = sorted([
                    d for d in os.listdir(scene_dir)
                    if osp.isdir(osp.join(scene_dir, d))
                ])

                for sub in sub_dirs:
                    sub_scene_path = osp.join(scene, sub)  # 相对路径，如 ai_001_001/cam_00
                    abs_sub_path = osp.join(self.data_root, sub_scene_path)

                    # 查找该子场景下的所有 RGB 文件
                    # 假设文件名格式类似于 frame.0000.rgb.png 或 0000.png
                    # 这里我们需要获取文件的前缀或索引，以便之后通过 replace 找到 depth 和 cam
                    files = sorted(os.listdir(abs_sub_path))
                    # 筛选 rgb 图片 (HyperSim 原始格式通常包含 'rgb')
                    rgb_files = [f for f in files if 'rgb' in f and f.endswith('.png')]

                    if len(rgb_files) < self.frame_num:
                        continue

                    # 将文件列表转为 string 存入
                    # 格式: sub_scene_relative_path | file_name1,file_name2,...
                    files_str = ",".join(rgb_files)
                    cache_sequences.append(f"{sub_scene_path}|{files_str}")

            # 写入 HDF5
            with h5py.File(cache_path, "w") as hf:
                # 使用变长字符串存储
                dt = h5py.string_dtype(encoding='utf-8')
                hf.create_dataset("sequences", data=np.array(cache_sequences, dtype=object), dtype=dt)

            if self.verbose:
                print(f"[HyperSim] Created cache at {cache_path} with {len(cache_sequences)} sequences.")

        # 等待主进程写完
        if dist.is_initialized():
            dist.barrier()

        # --- 2. 读取缓存并构建 self.sequences ---
        if osp.exists(cache_path):
            if self.verbose:
                print(f"[HyperSim] Loading metadata from {cache_path}")
            with h5py.File(cache_path, "r") as hf:
                raw_data = hf["sequences"][:]

            for item in raw_data:
                item_str = item.decode('utf-8') if isinstance(item, bytes) else item
                sub_scene_path, files_str = item_str.split('|')
                file_list = files_str.split(',')

                # 构建 sequence 条目: (sub_scene_path, file_list)
                # sub_scene_path 类似于 "ai_001_001/cam_00"
                self.sequences.append((sub_scene_path, file_list))
                self.num_image[sub_scene_path] = len(file_list)
        else:
            raise FileNotFoundError(f"Cache file not found and creation failed: {cache_path}")

        if self.verbose:
            print(f"[HyperSim] Successfully loaded {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        # 1. 获取序列信息
        sub_scene_path, file_list = self.sequences[index]
        scene_dir = osp.join(self.data_root, sub_scene_path)

        num_valid = len(file_list)

        # 2. 采样逻辑 (与 WildRGBD 保持一致)
        # 如果图片数量少于需要的帧数，允许重复采样
        should_replace = num_valid < self.frame_num
        # 从 file_list 中选择索引
        selected_indices = rng.choice(num_valid, self.frame_num, replace=should_replace)
        selected_indices.sort()

        views = []
        for idx in selected_indices:
            rgb_filename = file_list[idx]  # 例如 frame.0000.rgb.png

            # 构建路径
            rgb_path = osp.join(scene_dir, rgb_filename)

            # 推断 Depth 和 Cam 路径 (基于 HyperSim 命名习惯)
            # 假设 rgb: frame.0000.rgb.png -> depth: frame.0000.depth.npy
            # 假设 rgb: frame.0000.rgb.png -> cam:   frame.0000.cam.npz
            # 注意：如果你的文件名格式不同，请在此处修改 replace 逻辑
            depth_filename = rgb_filename.replace("rgb.png", "depth.npy")
            cam_filename = rgb_filename.replace("rgb.png", "cam.npz")

            depth_path = osp.join(scene_dir, depth_filename)
            cam_path = osp.join(scene_dir, cam_filename)

            if not osp.exists(rgb_path) or not osp.exists(cam_path):
                # 理论上 init 阶段已经过滤，但防止损坏文件
                print(f"[Warning] Missing file: {rgb_path} or {cam_path}")
                # 简单的 fallback: 复制上一个 view (如果存在)，或者跳过
                if len(views) > 0:
                    views.append(views[-1])
                continue

            # --- 加载数据 ---

            # RGB
            rgb_image = np.array(Image.open(rgb_path))

            # Depth (HyperSim 深度通常包含 inf，需要处理)
            if osp.exists(depth_path):
                depthmap = np.load(depth_path).astype(np.float32)
                depthmap[~np.isfinite(depthmap)] = 0.0  # 处理无穷大/NaN
            else:
                # 如果没有深度图，生成全0占位 (视具体需求而定)
                h, w, _ = rgb_image.shape
                depthmap = np.zeros((h, w), dtype=np.float32)

            # Camera
            # HyperSim 的 npz 通常包含 'pose' (4x4) 和 'intrinsics' (3x3)
            # Pose 通常是 Camera-to-World
            meta_data = np.load(cam_path)
            camera_pose = meta_data['pose'].astype(np.float32)
            camera_intrinsics = meta_data['intrinsics'].astype(np.float32)

            # 调用父类的 crop/resize
            # 注意：传入 info 参数以便 debug
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
            )

            # 构建与 WildRGBD 一致的输出字典
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),  # 统一键名: camera_pose
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='HyperSim',
                label=sub_scene_path.replace('/', '-'),  # label: ai_001_001-cam_00
                instance=rgb_filename,
            ))

        return views