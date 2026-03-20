import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from datasets.base.base_dataset import BaseDataset

class NTUDataset(BaseDataset):
    """NTU Sequence 数据集加载类"""

    def __init__(self, data_root='/data/liuwei/dataset/ntu_seq', mode='train', verbose=False, **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode  # 用此参数控制是 'train' 还是 'test'
        self.verbose = verbose

        self.mask_bg = False
        self.sequences = []
        self.num_image = {}

        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        if self.verbose:
            print(f"[NTUSeq] Scanning in: {data_root} for [{self.mode}] split")

        # 遍历根目录下的场景文件夹 (例如 campus, cp, garden)
        seq_dirs = sorted([d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))])
        
        for seq_name in seq_dirs:
            seq_path = osp.join(data_root, seq_name)
            rgb_dir = osp.join(seq_path, 'rgb')
            depth_dir = osp.join(seq_path, 'depth')
            npz_path = osp.join(seq_path, 'camera_data.npz')

            # 检查必要的文件和文件夹是否存在 
            if not (osp.exists(rgb_dir) and osp.exists(depth_dir) and osp.exists(npz_path)):
                continue

            # 扫描 RGB 图片以确定有效索引
            image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')]
            if len(image_files) == 0:
                continue

            # 解析索引 (例如 000000.png -> 0)
            try:
                valid_indices = sorted([int(osp.splitext(f)[0]) for f in image_files])
            except ValueError:
                valid_indices = list(range(len(image_files)))

            # ================= 核心修改部分：Train/Test 划分 =================
            # 使用 enumerate 确保基于实际的图片顺序每隔 10 张取 1 张
            if self.mode == 'test':
                # 取索引为 0, 10, 20... 的帧作为 test
                split_indices = [idx for i, idx in enumerate(valid_indices) if i % 10 == 0]
            elif self.mode == 'train':
                # 取其余的帧作为 train
                split_indices = [idx for i, idx in enumerate(valid_indices) if i % 10 != 0]
            else:
                raise ValueError(f"Unsupported mode: {self.mode}. Expected 'train' or 'test'.")
            # =================================================================

            # 如果划分后该序列在这个 mode 下没有图片了，则跳过
            if len(split_indices) == 0:
                continue

            self.num_image[seq_name] = len(split_indices)
            
            # 存储序列的绝对路径和划分后的有效 indices
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
        # 获取序列信息
        seq_info = self.sequences[index]
        seq_name = seq_info['seq_name']
        valid_indices = seq_info['indices']
        seq_path = seq_info['seq_path']

        num_valid = len(valid_indices)

        # --- 连续轨迹采样逻辑 (保持原有逻辑) ---
        MAX_STRIDE = 10  
        if self.frame_num > 1:
            available_max_stride = (num_valid - 1) // (self.frame_num - 1)
        else:
            available_max_stride = 1
            
        actual_max_stride = min(MAX_STRIDE, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (self.frame_num - 1) * stride + 1

        max_start_idx = num_valid - window_size
        start_idx = int(rng.choice(max_start_idx + 1))

        selected_indices = [start_idx + i * stride for i in range(self.frame_num)]
        idxs = [valid_indices[i] for i in selected_indices]
        # ----------------------------------------

        # --- 加载该场景全局的 Camera Intrinsics ---
        npz_path = osp.join(seq_path, 'camera_data.npz')
        try:
            camera_data = np.load(npz_path)
            camera_intrinsics = camera_data['camera_intrinsics'].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load camera_intrinsics from {npz_path}: {e}")

        views = []
        for idx in idxs:
            # 匹配 6 位数文件名格式 (000000.png)
            fname_base = f"{idx:06d}"
            image_name = f"{fname_base}.png"
            
            rgb_path = osp.join(seq_path, 'rgb', image_name)
            depth_path = osp.join(seq_path, 'depth', image_name)

            # --- 1. Load RGB ---
            if not osp.exists(rgb_path):
                continue
            
            img = Image.open(rgb_path).convert("RGB")
            rgb_image = np.array(img)

            # --- 2. Load Depth (PNG) ---
            if not osp.exists(depth_path):
                continue
            
            # 使用 cv2 读取 PNG 深度图 (通常 PNG 深度是 16-bit)
            depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depthmap is None:
                continue
            
            depthmap = depthmap.astype(np.float32)
            # 注意: 如果你的 PNG 深度图保存时放大了倍数(比如乘了 1000 保存为毫米)，
            # 这里可能需要除以对应的 scale 还原为米: depthmap = depthmap / 1000.0

            # --- 3. 设置 Camera Pose ---
            # 根据要求给一个全 0 的矩阵
            # camera_pose = np.zeros((4, 4), dtype=np.float32)
            # 备用方案（如遇奇异矩阵报错可换用单位阵）：
            camera_pose = np.eye(4, dtype=np.float32)

            # --- 4. Crop/Resize ---
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=rgb_path
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='NTUSeq',
                label=seq_name,
                instance=image_name,
            ))
            
        return views