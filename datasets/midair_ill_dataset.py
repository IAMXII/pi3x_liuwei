import os
import os.path as osp
import numpy as np
from PIL import Image
from datasets.base.base_dataset import BaseDataset

def load_camera_from_npz(npz_path):
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    try:
        data = np.load(npz_path)
        # 确保 key 存在，不同版本 MidAir 处理可能有差异
        if 'camera_pose' in data:
            camera_pose = data['camera_pose']
        elif 'pose' in data:
            camera_pose = data['pose']
        else:
            raise KeyError(f"No pose found in {npz_path}")

        if 'camera_intrinsics' in data:
            camera_intrinsics = data['camera_intrinsics']
        elif 'intrinsics' in data:
            camera_intrinsics = data['intrinsics']
        elif 'K' in data:
            camera_intrinsics = data['K']
        else:
            raise KeyError(f"No intrinsics found in {npz_path}")
            
        return camera_pose, camera_intrinsics
    except Exception as e:
        raise IOError(f"Error loading npz {npz_path}: {e}")

class MidAirDataset(BaseDataset):
    def __init__(self, root_dir='/data/liuwei/dataset/MidAir', mode='train', **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.mode = mode
        
        self.scans = self._scan_dataset()
        
        # 将 scans 转换为 object 类型的 numpy 数组，防止某些元数据长度不一致导致的 numpy 报错
        self.scans = np.array(self.scans, dtype=object)

        if len(self.scans) == 0:
            print(f"[{self.__class__.__name__}] CRITICAL: No sequences found in {root_dir}")
        else:
            print(f"[{self.__class__.__name__}] Loaded {len(self.scans)} sequences.")

    def _scan_dataset(self):
        valid_scans = []
        if not osp.exists(self.root_dir):
            return []

        suffix = '_training' if self.mode == 'train' else '_test'
        target_subsets = [f'Kite{suffix}', f'PLE{suffix}'] 

        for subset in target_subsets:
            subset_dir = osp.join(self.root_dir, subset)
            if not osp.exists(subset_dir): continue
            
            conditions = sorted([d for d in os.listdir(subset_dir) if osp.isdir(osp.join(subset_dir, d))])
            if len(conditions) < 2: 
                continue 
            
            # 使用字典按轨迹的后三位 ID 进行聚合
            # 结构: { "000": {"sunny": "trajectory_0000", "cloudy": "trajectory_1000", ...}, ... }
            traj_map = {}
            traj_frames_map = {}

            for cond in conditions:
                cond_dir = osp.join(subset_dir, cond)
                rgb_base = osp.join(cond_dir, 'color_left')
                depth_base = osp.join(cond_dir, 'depth')
                meta_base = osp.join(cond_dir, 'metadata')

                # 基础路径完整性检查
                if not (osp.exists(rgb_base) and osp.exists(depth_base) and osp.exists(meta_base)):
                    continue
                
                trajs = sorted([d for d in os.listdir(rgb_base) if osp.isdir(osp.join(rgb_base, d))])
                
                for traj in trajs:
                    traj_rgb_dir = osp.join(rgb_base, traj)
                    traj_depth_dir = osp.join(depth_base, traj)
                    traj_meta_dir = osp.join(meta_base, traj)

                    # 严格检查：对于当前天气，RGB、Depth、Meta 必须全部存在
                    if not (osp.exists(traj_depth_dir) and osp.exists(traj_meta_dir)):
                        continue

                    # 提取轨迹的物理 ID (后三位)
                    traj_id = traj[-3:]

                    if traj_id not in traj_map:
                        traj_map[traj_id] = {}
                        traj_frames_map[traj_id] = {}
                    
                    # 记录该天气下，真实的文件夹名称
                    traj_map[traj_id][cond] = traj
                    frame_set = {
                        f for f in os.listdir(traj_rgb_dir)
                        if f.endswith('.JPEG')
                        and osp.exists(osp.join(traj_depth_dir, f"{osp.splitext(f)[0]}.PNG"))
                        and osp.exists(osp.join(traj_meta_dir, f"{osp.splitext(f)[0]}.npz"))
                    }
                    traj_frames_map[traj_id][cond] = frame_set

            # 过滤出至少具备两种有效天气的轨迹，并构建 valid_scans
            for traj_id, valid_cond_dict in traj_map.items():
                if len(valid_cond_dict) >= 2:
                    per_cond_frames = [
                        traj_frames_map[traj_id][cond]
                        for cond in valid_cond_dict.keys()
                        if cond in traj_frames_map[traj_id]
                    ]
                    if len(per_cond_frames) < 2:
                        continue
                    all_frames = sorted(set.intersection(*per_cond_frames))
                    if len(all_frames) >= getattr(self, 'frame_num', 1):
                        valid_scans.append({
                            'subset': subset,
                            'traj_id': traj_id,
                            'conditions': valid_cond_dict, # 这是一个字典！不再是列表
                            'frames': all_frames,
                            'scan_id': f"{subset}/traj_{traj_id}"
                        })
        return valid_scans

    def __len__(self):
        return len(self.scans)

    def _get_views(self, idx, resolution, rng):
        try:
            scan = self.scans[idx]
        except ValueError:
            raise RuntimeError(f"Index {idx} is invalid for MidAirDataset")

        frames = scan['frames']
        if len(frames) < self.frame_num:
             raise RuntimeError(f"Not enough frames in {scan['scan_id']}")

        # 1. 随机抽取两种不同的天气名称
        available_conds = list(scan['conditions'].keys())
        cond_A, cond_B = rng.choice(available_conds, 2, replace=False)
        
        # 2. 获取这两种天气下，对应的真实文件夹名称
        traj_folder_A = scan['conditions'][cond_A]
        traj_folder_B = scan['conditions'][cond_B]
        
        subset_dir = osp.join(self.root_dir, scan['subset'])
        dir_A = osp.join(subset_dir, cond_A)
        dir_B = osp.join(subset_dir, cond_B)
        # print(f"[MidAirDataset] Sampled scan {scan['scan_id']} with conditions: {cond_A} vs {cond_B}")
        # print(dir_B,flush=True)

        # --- 采样帧逻辑保持不变 ---
        MAX_STRIDE = 5  
        if self.frame_num > 1:
            available_max_stride = (len(frames) - 1) // (self.frame_num - 1)
        else:
            available_max_stride = 1
        actual_max_stride = min(MAX_STRIDE, max(1, available_max_stride))
        stride = int(rng.choice(range(1, actual_max_stride + 1)))
        window_size = (self.frame_num - 1) * stride + 1
        max_start_idx = len(frames) - window_size
        start_idx = int(rng.choice(max_start_idx + 1))
        selected_indices = [start_idx + i * stride for i in range(self.frame_num)]
        selected_frames = [frames[i] for i in selected_indices]
        # ------------------------

        views = []
        import copy # 如果文件头部没引，这里记得补上
        
        for frame_file in selected_frames:
            frame_id = osp.splitext(frame_file)[0]
            
            # 使用提取出的真实文件夹名 (traj_folder_A / traj_folder_B) 拼接路径
            rgb_path_A = osp.join(dir_A, 'color_left', traj_folder_A, frame_file)
            depth_path = osp.join(dir_A, 'depth', traj_folder_A, f"{frame_id}.PNG") 
            meta_path = osp.join(dir_A, 'metadata', traj_folder_A, f"{frame_id}.npz")
            
            rgb_path_B = osp.join(dir_B, 'color_left', traj_folder_B, frame_file)

            if not (osp.exists(rgb_path_A) and osp.exists(rgb_path_B) and osp.exists(depth_path) and osp.exists(meta_path)):
                raise FileNotFoundError(f"Missing component for frame {frame_id} in {scan['scan_id']}")

            try:
                rgb_image_A = np.array(Image.open(rgb_path_A).convert('RGB'))
                rgb_image_B = np.array(Image.open(rgb_path_B).convert('RGB'))
                depthmap = np.array(Image.open(depth_path)).astype(np.float32)
                camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
                
                # 拷贝 rng 状态，保证 A 和 B 经历同样的裁剪
                rng_state = copy.deepcopy(rng.bit_generator.state)
                
                rgb_image_A, depthmap_A, intrinsics = self._crop_resize_if_necessary(
                    rgb_image_A, depthmap, camera_intrinsics.astype(np.float32), resolution, rng=rng, info=rgb_path_A
                )
                
                rng.bit_generator.state = rng_state
                rgb_image_B, _, _ = self._crop_resize_if_necessary(
                    rgb_image_B, depthmap, camera_intrinsics.astype(np.float32), resolution, rng=rng, info=rgb_path_B
                )

                views.append(dict(
                    img=rgb_image_A,
                    img_paired=rgb_image_B,
                    img_style=rgb_image_B,
                    depthmap=depthmap_A,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics,
                    camera_pose_paired=camera_pose.astype(np.float32),
                    camera_intrinsics_paired=intrinsics,
                    has_depth=np.bool_(True),
                    dataset='MidAir',
                    label=f"{scan['scan_id']}_{cond_A}_vs_{cond_B}",
                    instance=frame_file,
                ))

            except Exception as e:
                raise RuntimeError(f"Error loading frame {frame_file}: {e}")

        if len(views) < self.frame_num:
            raise RuntimeError(f"Failed to load enough frames for {scan['scan_id']}")

        return views
