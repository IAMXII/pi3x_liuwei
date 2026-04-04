# import os
# import os.path as osp
# import numpy as np
# from PIL import Image
# import json
# from datasets.base.base_dataset import BaseDataset

# def load_camera_from_npz(npz_path):
#     """从 metadata npz 文件读取相机参数"""
#     if not osp.exists(npz_path):
#         raise FileNotFoundError(f"Metadata file not found: {npz_path}")
#     data = np.load(npz_path)
#     return data['camera_pose'], data['camera_intrinsics']

# class CO3DV2Dataset(BaseDataset):
#     def __init__(self, root_dir='/data/liuwei/dataset/co3dv2_processed', split='train', **kwargs):
#         # 1. 初始化基类
#         super().__init__(**kwargs)
        
#         self.root_dir = root_dir
#         self.split = split
#         self.dataset_label = 'CO3DV2Dataset'

#         # 2. 扫描数据集 (Lazy Loading)
#         self.scans = self._scan_dataset()
#         print(f"[{self.dataset_label}] Index built. Split: {split}. Total sequences: {len(self.scans)}")

#     def _scan_dataset(self):
#         """遍历整个数据集目录，构建有效的序列索引列表"""
#         valid_scans = []
        
#         if not osp.exists(self.root_dir):
#             raise FileNotFoundError(f"Root dir not found: {self.root_dir}")

#         # 获取所有类别
#         categories = sorted([d for d in os.listdir(self.root_dir) 
#                              if osp.isdir(osp.join(self.root_dir, d))])

#         for cat in categories:
#             cat_dir = osp.join(self.root_dir, cat)
#             json_path = osp.join(cat_dir, 'selected_seqs_train.json')
            
#             # 读取该类别的训练集划分配置
#             train_registry = {} 
#             if osp.exists(json_path):
#                 with open(json_path, 'r') as f:
#                     train_registry = json.load(f)

#             # 获取该类别下的所有序列
#             sequences = sorted([d for d in os.listdir(cat_dir) 
#                                 if osp.isdir(osp.join(cat_dir, d))])

#             for seq_name in sequences:
#                 seq_dir = osp.join(cat_dir, seq_name)
#                 image_dir = osp.join(seq_dir, 'images')
                
#                 if not osp.exists(image_dir):
#                     continue

#                 # 获取磁盘上实际存在的所有图片文件
#                 all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
#                 if not all_files:
#                     continue

#                 # --- 核心划分逻辑 (修复 int 报错) ---
#                 seq_train_frames = set()
#                 if seq_name in train_registry:
#                     raw_frames = train_registry[seq_name]
                    
#                     # 修复：处理 json 中可能出现的 int 类型，并尝试匹配 CO3D 的 frame000001 格式
#                     for item in raw_frames:
#                         # 情况 1: 已经是字符串 (e.g. "frame000001.jpg" 或 "frame000001")
#                         if isinstance(item, str):
#                             if item.endswith('.jpg'):
#                                 seq_train_frames.add(item)
#                             else:
#                                 seq_train_frames.add(f"{item}.jpg")
                        
#                         # 情况 2: 是整数 (e.g. 1) -> 尝试转换为 "frame000001.jpg"
#                         elif isinstance(item, int):
#                             # 添加标准 CO3D 格式: frame000001.jpg
#                             seq_train_frames.add(f"frame{item:06d}.jpg")
#                             # 也可以添加简单格式以防万一: 1.jpg
#                             seq_train_frames.add(f"{item}.jpg")

#                 valid_frames = []
#                 if self.split == 'train':
#                     # 训练集：文件名必须在 seq_train_frames 集合中
#                     valid_frames = [f for f in all_files if f in seq_train_frames]
#                 else:
#                     # 测试集：文件名不在 seq_train_frames 中
#                     valid_frames = [f for f in all_files if f not in seq_train_frames]

#                 if len(valid_frames) > 0:
#                     valid_scans.append({
#                         'category': cat,
#                         'sequence': seq_name,
#                         'dir': seq_dir,
#                         'frames': valid_frames
#                     })
        
#         return valid_scans

#     def __len__(self):
#         return len(self.scans)

#     def _get_views(self, idx, resolution, rng):
#         """BaseDataset 调用的数据获取接口"""
#         scan_info = self.scans[idx]
#         seq_dir = scan_info['dir']
#         valid_frames = scan_info['frames']
        
#         image_dir = osp.join(seq_dir, 'images')
#         depth_dir = osp.join(seq_dir, 'depths')

#         # ====== [新增: 提前过滤掉文件不齐全的坏帧] ======
#         actually_valid_frames = []
#         for f in valid_frames:
#             frame_name = osp.splitext(f)[0]
#             rgb_path = osp.join(image_dir, f)
#             meta_path = osp.join(image_dir, f"{frame_name}.npz")
#             # 【修改点 1】: 将 f 替换为 frame_name，防止变成 .jpg.geometric.png
#             depth_path = osp.join(depth_dir, f"{f}.geometric.png") 
            
#             if osp.exists(rgb_path) and osp.exists(meta_path) and osp.exists(depth_path):
#                 actually_valid_frames.append(f)

#         if len(actually_valid_frames) == 0:
#             raise ValueError(f"Sequence {seq_dir} 没有任何包含完整 RGB/Depth/Pose 的帧！")
#         # ===============================================

#         views = []
#         candidate_frames = actually_valid_frames.copy()
#         self._rng.shuffle(candidate_frames)
        
#         if len(candidate_frames) < self.frame_num:
#             candidate_frames = self._rng.choice(actually_valid_frames, size=self.frame_num*2, replace=True).tolist()

#         for img_file in candidate_frames:
#             if len(views) >= self.frame_num:
#                 break 

#             frame_name = osp.splitext(img_file)[0]
#             rgb_path = osp.join(image_dir, img_file)
#             meta_path = osp.join(image_dir, f"{frame_name}.npz")
#             # 【修改点 2】: 同样将 img_file 替换为 frame_name
#             depth_path = osp.join(depth_dir, f"{img_file}.geometric.png") 
            
#             # Load RGB & Depth & Camera
#             rgb_image = Image.open(rgb_path)
#             depthmap = np.array(Image.open(depth_path)).astype(np.float32)
#             camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
#             camera_pose = camera_pose.astype(np.float32)
#             camera_intrinsics = camera_intrinsics.astype(np.float32)

#             # Crop/Resize
#             processed_img, processed_depth, processed_intrinsics = self._crop_resize_if_necessary(
#                 rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=self._rng, info=rgb_path
#             )

#             # 遇到无效深度图跳过
#             if processed_depth.sum() <= 1e-4:
#                 continue 

#             views.append(dict(
#                 img=processed_img,
#                 depthmap=processed_depth,
#                 camera_pose=camera_pose,
#                 camera_intrinsics=processed_intrinsics,
#             ))

#         if len(views) < self.frame_num:
#              raise ValueError(f"{seq_dir} 内部有效帧不足 {self.frame_num} 个！")

#         return views

import os
import os.path as osp
import numpy as np
from PIL import Image
import json
from datasets.base.base_dataset import BaseDataset

def load_intrinsics_from_npz(npz_path):
    """从 metadata npz 文件仅读取相机内参"""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    data = np.load(npz_path)
    return data['camera_intrinsics']

class CO3DV2Dataset(BaseDataset):
    def __init__(self, root_dir='/data/liuwei/dataset/co3dv2_processed', split='train', **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.split = split
        self.dataset_label = 'CO3DV2Dataset'

        self.scans = self._scan_dataset()
        print(f"[{self.dataset_label}] Index built. Split: {split}. Total sequences: {len(self.scans)}")

    def _scan_dataset(self):
        valid_scans = []
        if not osp.exists(self.root_dir):
            raise FileNotFoundError(f"Root dir not found: {self.root_dir}")

        categories = sorted([d for d in os.listdir(self.root_dir) if osp.isdir(osp.join(self.root_dir, d))])

        for cat in categories:
            cat_dir = osp.join(self.root_dir, cat)
            json_path = osp.join(cat_dir, 'selected_seqs_train.json')
            
            train_registry = {} 
            if osp.exists(json_path):
                with open(json_path, 'r') as f:
                    train_registry = json.load(f)

            sequences = sorted([d for d in os.listdir(cat_dir) if osp.isdir(osp.join(cat_dir, d))])

            for seq_name in sequences:
                seq_dir = osp.join(cat_dir, seq_name)
                image_dir = osp.join(seq_dir, 'images')
                
                if not osp.exists(image_dir):
                    continue

                all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
                if not all_files:
                    continue

                seq_train_frames = set()
                if seq_name in train_registry:
                    raw_frames = train_registry[seq_name]
                    for item in raw_frames:
                        if isinstance(item, str):
                            seq_train_frames.add(item if item.endswith('.jpg') else f"{item}.jpg")
                        elif isinstance(item, int):
                            seq_train_frames.add(f"frame{item:06d}.jpg")
                            seq_train_frames.add(f"{item}.jpg")

                valid_frames = []
                if self.split == 'train':
                    valid_frames = [f for f in all_files if f in seq_train_frames]
                else:
                    valid_frames = [f for f in all_files if f not in seq_train_frames]

                if len(valid_frames) > 0:
                    valid_scans.append({
                        'category': cat,
                        'sequence': seq_name,
                        'dir': seq_dir,
                        'frames': valid_frames
                    })
        return valid_scans

    def __len__(self):
        return len(self.scans)

    def _get_views(self, idx, resolution, rng):
        scan_info = self.scans[idx]
        seq_dir = scan_info['dir']
        valid_frames = scan_info['frames']
        image_dir = osp.join(seq_dir, 'images')

        actually_valid_frames = []
        for f in valid_frames:
            frame_name = osp.splitext(f)[0]
            rgb_path = osp.join(image_dir, f)
            meta_path = osp.join(image_dir, f"{frame_name}.npz")
            # 仅检查 RGB 和 Meta 是否存在
            if osp.exists(rgb_path) and osp.exists(meta_path):
                actually_valid_frames.append(f)

        if len(actually_valid_frames) == 0:
            raise ValueError(f"Sequence {seq_dir} 没有任何包含完整 RGB/Pose 的帧！")

        views = []
        candidate_frames = actually_valid_frames.copy()
        self._rng.shuffle(candidate_frames)
        
        if len(candidate_frames) < self.frame_num:
            candidate_frames = self._rng.choice(actually_valid_frames, size=self.frame_num*2, replace=True).tolist()

        for img_file in candidate_frames:
            if len(views) >= self.frame_num:
                break 

            frame_name = osp.splitext(img_file)[0]
            rgb_path = osp.join(image_dir, img_file)
            meta_path = osp.join(image_dir, f"{frame_name}.npz")
            
            rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
            camera_intrinsics = load_intrinsics_from_npz(meta_path).astype(np.float32)

            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            processed_img, _, processed_intrinsics = self._crop_resize_if_necessary(
                rgb_image, fake_depthmap, camera_intrinsics.copy(), resolution, rng=self._rng, info=rgb_path
            )

            views.append(dict(
                img=processed_img,
                camera_intrinsics=processed_intrinsics,
            ))

        if len(views) < self.frame_num:
             raise ValueError(f"{seq_dir} 内部有效帧不足 {self.frame_num} 个！")

        return views