import os
import os.path as osp
import numpy as np
from PIL import Image
from datasets.base.base_dataset import BaseDataset

class MapFreeDataset(BaseDataset):
    def __init__(self, data_root="/data/liuwei/dataset/mapfree_processed", mode='train', verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.mode = mode
        self.verbose = verbose
        self.sequences = []
        self.num_image = {}

        # 1. 确定 split 目录
        split_dir = osp.join(self.data_root, mode)
        if not osp.exists(split_dir):
            if verbose: print(f"[MapFree] Split directory not found: {split_dir}")
            return

        # 2. 遍历场景 (s00xxx)
        scenes = [d for d in os.listdir(split_dir) if osp.isdir(osp.join(split_dir, d))]
        scenes.sort()

        for scene in scenes:
            scene_path = osp.join(split_dir, scene)
            seqs = [d for d in os.listdir(scene_path) if osp.isdir(osp.join(scene_path, d))]
            seqs.sort()

            for seq in seqs:
                seq_path = osp.join(scene_path, seq)
                color_dir = osp.join(seq_path, 'rgb')
                depth_dir = osp.join(seq_path, 'depth')
                poses_dir = osp.join(seq_path, 'cam')

                if not (osp.exists(color_dir) or osp.exists(depth_dir) or osp.exists(poses_dir)):
                    continue

                # 获取所有 frame_xxxxx.jpg 格式的文件
                frames = [f for f in os.listdir(color_dir) if f.startswith('frame_') and f.endswith('.jpg')]
                frames.sort()

                valid_indices = []
                for f in frames:
                    try:
                        idx = int(f.replace('frame_', '').replace('.jpg', ''))
                        depth_name = f"frame_{idx:05d}.npy"
                        pose_name = f"frame_{idx:05d}.npz"
                        
                        if osp.exists(osp.join(depth_dir, depth_name)) and \
                           osp.exists(osp.join(poses_dir, pose_name)):
                            valid_indices.append(idx)
                    except ValueError: continue
                
                # 如果该序列有效帧太少，直接跳过不加入 sequences
                if len(valid_indices) < getattr(self, 'frame_num', 1):
                    continue

                self.sequences.append((scene, seq, valid_indices))
                self.num_image[f"{scene}/{seq}"] = len(valid_indices)

        if self.verbose:
            print(f"[MapFree] Successfully loaded {len(self.sequences)} sequences.")

        # 关键修改点 1: 显式指定 dtype=object 避免 inhomogeneous shape 报错
        self.sequences = np.array(self.sequences, dtype=object)

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        # 关键修改点 2: 整个读取过程包裹在 try-except 中，若失败则返回空或触发重试
        try:
            scene, seq, valid_indices = self.sequences[index]
            base_path = osp.join(self.data_root, self.mode, scene, seq)
            
            color_dir = osp.join(base_path, 'rgb')
            depth_dir = osp.join(base_path, 'depth')
            poses_dir = osp.join(base_path, 'cam')

            num_valid = len(valid_indices)
            should_replace = num_valid < self.frame_num
            idxs = rng.choice(valid_indices, self.frame_num, replace=should_replace)
            idxs = np.sort(idxs)

            views = []
            for idx in idxs:
                fid = f"frame_{idx:05d}"
                rgb_path = osp.join(color_dir, f"{fid}.jpg")
                depth_path = osp.join(depth_dir, f"{fid}.npy")
                pose_path = osp.join(poses_dir, f"{fid}.npz")

                # 如果文件丢失，视为该序列当前无效
                if not (osp.exists(rgb_path) and osp.exists(depth_path) and osp.exists(pose_path)):
                    raise FileNotFoundError(f"Missing files for {fid}")

                rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
                depthmap = np.load(depth_path).astype(np.float32)

                meta_data = np.load(pose_path, allow_pickle=True)
                # 兼容 MapFree 不同版本的 key
                if 'K' in meta_data: intrinsics = meta_data['K']
                elif 'intrinsic' in meta_data: intrinsics = meta_data['intrinsic']
                else: intrinsics = meta_data['intrin']

                if 'pose' in meta_data: pose = meta_data['pose']
                else: pose = meta_data['camera_pose']

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics.astype(np.float32), resolution, rng=rng, info=rgb_path
                )

                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=pose.astype(np.float32),
                    camera_intrinsics=intrinsics,
                    dataset='MapFree',
                    label=f"{scene}-{seq}",
                    instance=f"{fid}.jpg",
                ))

            # 关键修改点 3: 如果因为异常导致 views 数量不对，确保不返回空列表给 collate_fn
            if len(views) < self.frame_num:
                return None # 或者抛出错误，取决于 BaseDataset 的重试机制

            return views

        except Exception as e:
            if self.verbose:
                print(f"[MapFree] Skipping sequence {index} due to error: {e}")
            # 返回 None 触发 BaseDataset 的重试逻辑（需确保 BaseDataset 有重试判断）
            return None