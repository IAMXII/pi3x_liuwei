# import os
# import os.path as osp
# import numpy as np
# from PIL import Image
# import cv2
# from datasets.base.base_dataset import BaseDataset

# class NTUDataset(BaseDataset):
#     """NTU Sequence 数据集加载类"""

#     def __init__(self, data_root='/data/liuwei/dataset/ntu_seq', mode='train', verbose=False, **kwargs):
#         super().__init__(**kwargs)
#         assert data_root is not None
#         self.data_root = data_root
#         self.mode = mode
#         self.verbose = verbose

#         self.mask_bg = False
#         self.sequences = []
#         self.num_image = {}
        
#         # [新增] 用于缓存相机的内参，避免在 get_views 中疯狂读硬盘
#         self.camera_intrinsics_cache = {} 

#         if not osp.exists(data_root):
#             raise FileNotFoundError(f"Data root not found: {data_root}")

#         if self.verbose:
#             print(f"[NTUSeq] Scanning in: {data_root} for [{self.mode}] split")

#         seq_dirs = sorted([d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))])
        
#         for seq_name in seq_dirs:
#             seq_path = osp.join(data_root, seq_name)
#             rgb_dir = osp.join(seq_path, 'rgb')
#             depth_dir = osp.join(seq_path, 'depth')
#             npz_path = osp.join(seq_path, 'camera_data.npz')

#             if not (osp.exists(rgb_dir) and osp.exists(depth_dir) and osp.exists(npz_path)):
#                 continue
            
#             # [新增] 在初始化时就读取并缓存该场景的内参，彻底消灭运行时的 I/O 瓶颈
#             try:
#                 camera_data = np.load(npz_path)
#                 # 存入缓存字典中
#                 self.camera_intrinsics_cache[seq_name] = camera_data['camera_intrinsics'].astype(np.float32)
#             except Exception as e:
#                 print(f"Warning: Failed to load camera_intrinsics from {npz_path}: {e}")
#                 continue # 如果内参坏了，直接跳过这个场景

#             image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')]
#             if len(image_files) == 0:
#                 continue

#             try:
#                 valid_indices = sorted([int(osp.splitext(f)[0]) for f in image_files])
#             except ValueError:
#                 valid_indices = list(range(len(image_files)))

#             if self.mode == 'test':
#                 split_indices = [idx for i, idx in enumerate(valid_indices) if i % 10 == 0]
#             elif self.mode == 'train':
#                 split_indices = [idx for i, idx in enumerate(valid_indices) if i % 10 != 0]
#             else:
#                 raise ValueError(f"Unsupported mode: {self.mode}.")

#             if len(split_indices) == 0:
#                 continue

#             self.num_image[seq_name] = len(split_indices)
            
#             self.sequences.append({
#                 'seq_name': seq_name,
#                 'indices': split_indices,
#                 'seq_path': seq_path
#             })

#         if self.verbose:
#             print(f"[NTUSeq] Successfully loaded {len(self.sequences)} sequences for {self.mode}.")

#     def __len__(self):
#         return len(self.sequences)

#     def _get_views(self, index, resolution, rng):
#         seq_info = self.sequences[index]
#         seq_name = seq_info['seq_name']
#         valid_indices = seq_info['indices']
#         seq_path = seq_info['seq_path']

#         num_valid = len(valid_indices)

#         MAX_STRIDE = 10  
#         if getattr(self, 'frame_num', 1) > 1:
#             available_max_stride = (num_valid - 1) // (self.frame_num - 1)
#         else:
#             available_max_stride = 1
            
#         actual_max_stride = min(MAX_STRIDE, max(1, available_max_stride))
#         stride = int(rng.choice(range(1, actual_max_stride + 1)))
#         window_size = (getattr(self, 'frame_num', 1) - 1) * stride + 1

#         max_start_idx = num_valid - window_size
#         start_idx = int(rng.choice(max_start_idx + 1))

#         selected_indices = [start_idx + i * stride for i in range(getattr(self, 'frame_num', 1))]
#         idxs = [valid_indices[i] for i in selected_indices]

#         # [修改] 直接从内存的字典中极速读取内参，不再进行磁盘 I/O
#         camera_intrinsics = self.camera_intrinsics_cache[seq_name]

#         views = []
#         for idx in idxs:
#             fname_base = f"{idx:06d}"
#             image_name = f"{fname_base}.png"
            
#             rgb_path = osp.join(seq_path, 'rgb', image_name)
#             depth_path = osp.join(seq_path, 'depth', image_name)

#             if not osp.exists(rgb_path):
#                 continue
            
#             img = Image.open(rgb_path).convert("RGB")
#             rgb_image = np.array(img)

#             if not osp.exists(depth_path):
#                 continue
            
#             depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
#             if depthmap is None:
#                 continue
            
#             depthmap = depthmap.astype(np.float32)
#             camera_pose = np.eye(4, dtype=np.float32)

#             rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
#                 rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
#             )

#             views.append(dict(
#                 img=rgb_image,
#                 depthmap=depthmap.astype(np.float32),
#                 camera_pose=camera_pose.astype(np.float32),
#                 camera_intrinsics=intrinsics.astype(np.float32),
#                 dataset='NTUSeq',
#                 label=seq_name,
#                 instance=image_name,
#             ))
            
#         return views

import os
import os.path as osp
import numpy as np
from PIL import Image
from datasets.base.base_dataset import BaseDataset

class NTUDataset(BaseDataset):
    """NTU Sequence 数据集加载类（仅包含RGB和内参的精简版）"""

    def __init__(self, data_root='/data/liuwei/dataset/ntu_seq', mode='train', verbose=False, **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose

        self.mask_bg = False
        self.sequences = []
        self.num_image = {}
        
        # 用于缓存相机的内参，避免疯狂读硬盘
        self.camera_intrinsics_cache = {} 

        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        if self.verbose:
            print(f"[NTUSeq] Scanning in: {data_root} for [{self.mode}] split")

        seq_dirs = sorted([d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))])
        
        for seq_name in seq_dirs:
            seq_path = osp.join(data_root, seq_name)
            rgb_dir = osp.join(seq_path, 'rgb')
            npz_path = osp.join(seq_path, 'camera_data.npz')

            # 仅检查 RGB 和内参文件
            if not (osp.exists(rgb_dir) and osp.exists(npz_path)):
                continue
            
            # 初始化时读取并缓存该场景的内参
            try:
                camera_data = np.load(npz_path)
                self.camera_intrinsics_cache[seq_name] = camera_data['camera_intrinsics'].astype(np.float32)
            except Exception as e:
                print(f"Warning: Failed to load camera_intrinsics from {npz_path}: {e}")
                continue

            image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')]
            if not image_files:
                continue

            try:
                valid_indices = sorted([int(osp.splitext(f)[0]) for f in image_files])
            except ValueError:
                valid_indices = list(range(len(image_files)))

            if self.mode == 'test':
                split_indices = [idx for i, idx in enumerate(valid_indices) if i % 10 == 0]
            elif self.mode == 'train':
                split_indices = [idx for i, idx in enumerate(valid_indices) if i % 10 != 0]
            else:
                raise ValueError(f"Unsupported mode: {self.mode}.")

            if not split_indices:
                continue

            self.num_image[seq_name] = len(split_indices)
            self.sequences.append({
                'seq_name': seq_name,
                'indices': split_indices,
                'seq_path': seq_path
            })

        if self.verbose:
            print(f"[NTUSeq] Successfully loaded {len(self.sequences)} sequences for {self.mode}.")

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        seq_info = self.sequences[index]
        seq_name = seq_info['seq_name']
        valid_indices = seq_info['indices']
        seq_path = seq_info['seq_path']

        num_valid = len(valid_indices)
        frame_num = getattr(self, 'frame_num', 1)

        MAX_STRIDE = 10  
        available_max_stride = (num_valid - 1) // (frame_num - 1) if frame_num > 1 else 1
            
        actual_max_stride = min(MAX_STRIDE, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (frame_num - 1) * stride + 1

        max_start_idx = num_valid - window_size
        start_idx = int(rng.choice(max_start_idx + 1))

        selected_indices = [start_idx + i * stride for i in range(frame_num)]
        idxs = [valid_indices[i] for i in selected_indices]

        # 极速读取内参
        camera_intrinsics = self.camera_intrinsics_cache[seq_name]

        views = []
        for idx in idxs:
            image_name = f"{idx:06d}.png"
            rgb_path = osp.join(seq_path, 'rgb', image_name)

            if not osp.exists(rgb_path):
                continue
            
            img = Image.open(rgb_path).convert("RGB")
            rgb_image = np.array(img)
            # [新增] 构造一个与 RGB 图像宽高相同的全 0 假深度图，类型保持 float32
            fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            # 传入 fake_depthmap，并用 "_" 接收返回的修改后的深度图（直接丢弃）
            rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                rgb_image, fake_depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
            )

            views.append(dict(
                img=rgb_image,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='NTUSeq',
                label=seq_name,
                instance=image_name,
            ))
            
        return views
