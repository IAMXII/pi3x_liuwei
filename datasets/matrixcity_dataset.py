import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2  # 新增：用于读取 exr 文件
from datasets.base.base_dataset import BaseDataset

# 设置环境变量以确保 OpenCV 可以正确处理 OpenEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_camera_from_npz(npz_path):
    """从 metadata npz 文件读取相机参数"""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    data = np.load(npz_path)
    
    # 假设 npz 中的键名为 'camera_pose' 和 'camera_intrinsics'
    camera_pose = data['camera_pose']            # 4x4
    camera_intrinsics = data['camera_intrinsics'] # 3x3
    
    return camera_pose, camera_intrinsics


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    """
    保持代码结构完整性保留此函数，逻辑未变
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


class MatrixCityDataset(BaseDataset):
    """MatrixCity 数据集加载类"""

    def __init__(self, data_root='/data/liuwei/dataset/matrixcity_processed', mode='train', verbose=False, **kwargs):
        super().__init__(**kwargs)
        assert data_root is not None
        self.data_root = data_root
        self.mode = mode  # 'train' or 'test'
        self.verbose = verbose

        # MatrixCity 不使用 Mask
        self.mask_bg = False

        self.sequences = []
        self.num_image = {}

        if not osp.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        if self.verbose:
            print(f"[MatrixCity] Scanning for '{self.mode}' split in: {data_root}")

        # --- 1. 递归遍历目录寻找 train/test 文件夹 ---
        # MatrixCity 结构: big_city/aerial/test/seq_00
        # train/test 文件夹可能在任意深度的子目录中
        for root, dirs, files in os.walk(data_root):
            if self.mode in dirs:
                # 找到了 split 文件夹 (例如 .../aerial/test)
                split_path = osp.join(root, self.mode)
                
                # 确定类别名称 (例如 big_city/aerial)
                # 使用从 data_root 到当前目录的相对路径作为类别名
                category_name = osp.relpath(root, data_root)
                
                # 获取该 split 下的所有序列文件夹
                seq_dirs = sorted([d for d in os.listdir(split_path) if osp.isdir(osp.join(split_path, d))])
                
                for seq_name in seq_dirs:
                    seq_path = osp.join(split_path, seq_name)
                    rgb_dir = osp.join(seq_path, 'rgb')

                    # 检查 rgb 文件夹是否存在
                    if not osp.exists(rgb_dir):
                        continue

                    # 扫描图片以确定有效索引
                    # 假设图片格式为 .png 或 .jpg
                    image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg'))]
                    image_files.sort()

                    if len(image_files) == 0:
                        continue

                    # 尝试解析索引 (例如 01000.png -> 1000)
                    try:
                        valid_indices = [int(osp.splitext(f)[0]) for f in image_files]
                    except ValueError:
                        # 如果文件名非纯数字，使用枚举索引
                        valid_indices = list(range(len(image_files)))

                    self.num_image[(category_name, seq_name)] = len(valid_indices)
                    
                    # 存储必要信息：类别，序列名，索引，以及序列的绝对路径
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
        # 获取序列信息
        seq_info = self.sequences[index]
        cat = seq_info['cat']
        seq_name = seq_info['seq_name']
        valid_indices = seq_info['indices']
        seq_path = seq_info['seq_path']

        num_valid = len(valid_indices)

        # 采样逻辑
        should_replace = num_valid < self.frame_num
        # idxs = rng.choice(valid_indices, self.frame_num, replace=should_replace)
        # idxs.sort()

        # --- 针对户外大场景的连续轨迹采样逻辑 ---
        # 设定最大步长 (Stride)。步长决定了相机的 Baseline。
        # 步长太大 -> 失去 Overlap；步长太小 (如 1) -> 视角几乎没变，Baseline 太窄。
        # 经验值：对于 30fps 的轨迹数据集，步长设为 3~10 之间通常比较合适。
        MAX_STRIDE = 5  
        
        # 计算在当前序列长度下，实际允许的最大步长
        if self.frame_num > 1:
            available_max_stride = (len(valid_indices) - 1) // (self.frame_num - 1)
        else:
            available_max_stride = 1
            
        actual_max_stride = min(MAX_STRIDE, max(1, available_max_stride))

        # 随机选择一个步长 (或者你也可以去掉 rng.choice 直接固定 stride = actual_max_stride)
        stride = int(rng.choice(range(1, actual_max_stride + 1)))

        # 根据确定的步长，计算采样窗口的总长度
        window_size = (self.frame_num - 1) * stride + 1

        # 在有效的范围内随机选择起始帧的索引
        max_start_idx = len(valid_indices) - window_size
        start_idx = int(rng.choice(max_start_idx + 1))

        # 生成连续且等距的帧索引
        selected_indices = [start_idx + i * stride for i in range(self.frame_num)]
        # ----------------------------------------
        
        idxs = [valid_indices[i] for i in selected_indices]

        views = []
        for idx in idxs:
            # 构建文件路径
            # 尝试匹配文件名长度，通常是 5 位 (01000.png)
            fname_base = f"{idx:05d}"
            rgb_name_png = f"{fname_base}.png"
            rgb_name_jpg = f"{fname_base}.jpg"
            
            # 检查 RGB 文件
            rgb_path = osp.join(seq_path, 'rgb', rgb_name_png)
            instance_name = rgb_name_png
            if not osp.exists(rgb_path):
                rgb_path = osp.join(seq_path, 'rgb', rgb_name_jpg)
                instance_name = rgb_name_jpg
            
            # 深度图路径 (EXR)
            depth_path = osp.join(seq_path, 'depth', f"{fname_base}.exr")
            
            # Metadata 路径 (NPZ)
            meta_path = osp.join(seq_path, 'metadata', f"{fname_base}.npz")

            # --- 1. Load RGB ---
            if not osp.exists(rgb_path):
                # 简单跳过或返回黑图
                # rgb_image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
                continue
            else:
                # rgb_image = np.array(Image.open(rgb_path))
                img = Image.open(rgb_path).convert("RGB")
                rgb_image = np.array(img)

            # --- 2. Load Depth (EXR) ---
            if not osp.exists(depth_path):
                # 如果没有深度图，返回全0
                # depthmap = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
                continue
            else:
                # 使用 cv2 读取 .exr 文件，flag=-1 保持原始深度数据 (float32)
                depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                
                # 处理多通道 EXR (如果 EXR 包含 RGB，通常深度在第一个通道)
                if depthmap.ndim == 3:
                    depthmap = depthmap[:, :, 0]
                
                depthmap = depthmap.astype(np.float32)
                # print(depthmap.min(), depthmap.max())

            # --- 3. Load Camera Parameters from NPZ ---
            # 直接使用 load_camera_from_npz 读取
            if osp.exists(meta_path):
                camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
            else:
                # 缺省相机参数
                continue
                # camera_pose = np.eye(4, dtype=np.float32)
                # camera_intrinsics = np.eye(3, dtype=np.float32)

            # 类型转换
            camera_pose = camera_pose.astype(np.float32)
            camera_intrinsics = camera_intrinsics.astype(np.float32)

            # --- 4. Crop/Resize ---
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
                instance=instance_name,
            ))
        return views