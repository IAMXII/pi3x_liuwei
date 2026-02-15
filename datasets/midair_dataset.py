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
        # 根据实际文件夹名称调整，这里假设是 Kite_training 等
        target_subsets = [f'Kite{suffix}', f'PLE{suffix}'] 

        for subset in target_subsets:
            subset_dir = osp.join(self.root_dir, subset)
            if not osp.exists(subset_dir): continue
            
            # 遍历 Condition (sunny, cloudy...)
            conditions = sorted([d for d in os.listdir(subset_dir) if osp.isdir(osp.join(subset_dir, d))])
            
            for cond in conditions:
                cond_dir = osp.join(subset_dir, cond)
                rgb_base = osp.join(cond_dir, 'color_left')
                depth_base = osp.join(cond_dir, 'depth')
                meta_base = osp.join(cond_dir, 'metadata')

                if not (osp.exists(rgb_base) and osp.exists(depth_base) and osp.exists(meta_base)):
                    continue

                trajs = sorted([d for d in os.listdir(rgb_base) if osp.isdir(osp.join(rgb_base, d))])
                
                for traj in trajs:
                    traj_rgb_dir = osp.join(rgb_base, traj)
                    traj_depth_dir = osp.join(depth_base, traj)
                    traj_meta_dir = osp.join(meta_base, traj)

                    if not (osp.exists(traj_depth_dir) and osp.exists(traj_meta_dir)):
                        continue

                    # 只扫描 RGB 确定帧数
                    all_frames = sorted([f for f in os.listdir(traj_rgb_dir) if f.endswith('.JPEG')])
                    
                    if len(all_frames) >= getattr(self, 'frame_num', 1):
                        valid_scans.append({
                            'rgb_dir': traj_rgb_dir,
                            'depth_dir': traj_depth_dir,
                            'meta_dir': traj_meta_dir,
                            'frames': all_frames,
                            'scan_id': f"{subset}/{cond}/{traj}"
                        })
        return valid_scans

    def __len__(self):
        return len(self.scans)

    def _get_views(self, idx, resolution, rng):
        # 1. 安全获取 Scan 信息
        try:
            scan = self.scans[idx]
        except ValueError:
            # 处理可能的 numpy 索引问题
            raise RuntimeError(f"Index {idx} is invalid for MidAirDataset")

        rgb_dir = scan['rgb_dir']
        depth_dir = scan['depth_dir']
        meta_dir = scan['meta_dir']
        frames = scan['frames']

        # 2. 采样帧
        if len(frames) < self.frame_num:
             # 如果某种原因帧数不够（初始化时检查过，但为了安全再次检查）
             raise RuntimeError(f"Not enough frames in {scan['scan_id']}")

        # 随机采样
        selected_indices = rng.choice(len(frames), size=self.frame_num, replace=False)
        selected_indices.sort()
        selected_frames = [frames[i] for i in selected_indices]

        views = []
        for frame_file in selected_frames:
            # 提取 ID (例如 "000001.JPEG" -> "000001")
            frame_id = osp.splitext(frame_file)[0]
            
            rgb_path = osp.join(rgb_dir, frame_file)
            depth_path = osp.join(depth_dir, f"{frame_id}.PNG") # 确认后缀是 PNG 还是 JPEG
            meta_path = osp.join(meta_dir, f"{frame_id}.npz")

            # 3. 严格检查文件是否存在
            if not (osp.exists(rgb_path) and osp.exists(depth_path) and osp.exists(meta_path)):
                # 如果单个文件缺失，抛出异常以跳过整个序列（保持 Batch 完整性）
                raise FileNotFoundError(f"Missing component for frame {frame_id} in {scan['scan_id']}")

            try:
                # Load Image
                rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
                
                # Load Depth
                depthmap = np.array(Image.open(depth_path)).astype(np.float32)

                # Load Camera
                camera_pose, camera_intrinsics = load_camera_from_npz(meta_path)
                
                # Crop & Resize
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, camera_intrinsics.astype(np.float32), resolution, rng=rng, info=rgb_path
                )

                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics,
                    dataset='MidAir',
                    label=scan['scan_id'],
                    instance=frame_file,
                ))

            except Exception as e:
                # 捕获读取过程中的任何错误（如图片损坏）
                raise RuntimeError(f"Error loading frame {frame_file}: {e}")

        # 4. 最终检查 (CRITICAL FIX)
        # 如果 views 数量不对，绝对不能返回 partial list，必须报错让 BaseDataset 重试
        if len(views) < self.frame_num:
            raise RuntimeError(f"Failed to load enough frames for {scan['scan_id']}")

        return views