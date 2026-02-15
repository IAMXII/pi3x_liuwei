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
    return data['camera_pose'], data['camera_intrinsics']

class CO3DV2Dataset(BaseDataset):
    def __init__(self, root_dir='/data/liuwei/dataset/co3dv2_processed', split='train', **kwargs):
        # 1. 初始化基类
        super().__init__(**kwargs)
        
        self.root_dir = root_dir
        self.split = split
        self.dataset_label = 'CO3DV2Dataset'

        # 2. 扫描数据集 (Lazy Loading)
        self.scans = self._scan_dataset()
        print(f"[{self.dataset_label}] Index built. Split: {split}. Total sequences: {len(self.scans)}")

    def _scan_dataset(self):
        """遍历整个数据集目录，构建有效的序列索引列表"""
        valid_scans = []
        
        if not osp.exists(self.root_dir):
            raise FileNotFoundError(f"Root dir not found: {self.root_dir}")

        # 获取所有类别
        categories = sorted([d for d in os.listdir(self.root_dir) 
                             if osp.isdir(osp.join(self.root_dir, d))])

        for cat in categories:
            cat_dir = osp.join(self.root_dir, cat)
            json_path = osp.join(cat_dir, 'selected_seqs_train.json')
            
            # 读取该类别的训练集划分配置
            train_registry = {} 
            if osp.exists(json_path):
                with open(json_path, 'r') as f:
                    train_registry = json.load(f)

            # 获取该类别下的所有序列
            sequences = sorted([d for d in os.listdir(cat_dir) 
                                if osp.isdir(osp.join(cat_dir, d))])

            for seq_name in sequences:
                seq_dir = osp.join(cat_dir, seq_name)
                image_dir = osp.join(seq_dir, 'images')
                
                if not osp.exists(image_dir):
                    continue

                # 获取磁盘上实际存在的所有图片文件
                all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
                if not all_files:
                    continue

                # --- 核心划分逻辑 (修复 int 报错) ---
                seq_train_frames = set()
                if seq_name in train_registry:
                    raw_frames = train_registry[seq_name]
                    
                    # 修复：处理 json 中可能出现的 int 类型，并尝试匹配 CO3D 的 frame000001 格式
                    for item in raw_frames:
                        # 情况 1: 已经是字符串 (e.g. "frame000001.jpg" 或 "frame000001")
                        if isinstance(item, str):
                            if item.endswith('.jpg'):
                                seq_train_frames.add(item)
                            else:
                                seq_train_frames.add(f"{item}.jpg")
                        
                        # 情况 2: 是整数 (e.g. 1) -> 尝试转换为 "frame000001.jpg"
                        elif isinstance(item, int):
                            # 添加标准 CO3D 格式: frame000001.jpg
                            seq_train_frames.add(f"frame{item:06d}.jpg")
                            # 也可以添加简单格式以防万一: 1.jpg
                            seq_train_frames.add(f"{item}.jpg")

                valid_frames = []
                if self.split == 'train':
                    # 训练集：文件名必须在 seq_train_frames 集合中
                    valid_frames = [f for f in all_files if f in seq_train_frames]
                else:
                    # 测试集：文件名不在 seq_train_frames 中
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
        """BaseDataset 调用的数据获取接口"""
        scan_info = self.scans[idx]
        seq_dir = scan_info['dir']
        valid_frames = scan_info['frames']
        
        image_dir = osp.join(seq_dir, 'images')
        depth_dir = osp.join(seq_dir, 'depths')

        # 随机采样
        replace = len(valid_frames) < self.frame_num
        selected_indices = rng.choice(len(valid_frames), size=self.frame_num, replace=replace)
        selected_frames = [valid_frames[i] for i in selected_indices]

        views = []
        for img_file in selected_frames:
            frame_name = osp.splitext(img_file)[0]
            
            rgb_path = osp.join(image_dir, img_file)
            meta_path = osp.join(image_dir, f"{frame_name}.npz")
            depth_path = osp.join(depth_dir, f"{img_file}.geometric.png")
            
            # Load RGB
            rgb_image = Image.open(rgb_path)
            
            # Load Depth
            if osp.exists(depth_path):
                depthmap = np.array(Image.open(depth_path)).astype(np.float32)
            else:
                w, h = rgb_image.size
                depthmap = np.zeros((h, w), dtype=np.float32)

            # Load Camera
            # 这里加个 try-except 防止单个坏文件中断训练
            try:
                camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
            except Exception as e:
                print(f"Warning: Failed to load {meta_path}: {e}")
                # 如果失败，这里需要一种回退机制，或者直接报错重试
                # 为了简单起见，这里抛出错误，让 DataLoader 的 collate_fn 或 BaseDataset 里的重试机制处理
                raise e

            camera_pose = camera_pose.astype(np.float32)
            camera_intrinsics = camera_intrinsics.astype(np.float32)

            # Crop/Resize
            processed_img, processed_depth, processed_intrinsics = self._crop_resize_if_necessary(
                rgb_image, 
                depthmap, 
                camera_intrinsics.copy(), 
                resolution, 
                rng=rng, 
                info=rgb_path
            )

            views.append(dict(
                img=processed_img,
                depthmap=processed_depth,
                camera_pose=camera_pose,
                camera_intrinsics=processed_intrinsics,
            ))

        return views