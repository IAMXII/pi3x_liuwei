import os
import os.path as osp
import numpy as np
from PIL import Image
from datasets.base.base_dataset import BaseDataset

def load_intrinsics_from_npz(npz_path):
    """读取 MidAir 的内参，不再加载外参。"""
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Metadata file not found: {npz_path}")
    try:
        data = np.load(npz_path)
        if 'camera_intrinsics' in data:
            camera_intrinsics = data['camera_intrinsics']
        elif 'intrinsics' in data:
            camera_intrinsics = data['intrinsics']
        elif 'K' in data:
            camera_intrinsics = data['K']
        else:
            raise KeyError(f"No intrinsics found in {npz_path}")

        return camera_intrinsics.astype(np.float32)
    except Exception as e:
        raise IOError(f"Error loading npz {npz_path}: {e}")

class MidAirDataset(BaseDataset):
    def __init__(self, root_dir='/data/liuwei/dataset/MidAir', mode='train', **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.mode = mode
        
        self.scans = self._scan_dataset()
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
            
            for cond in conditions:
                cond_dir = osp.join(subset_dir, cond)
                rgb_base = osp.join(cond_dir, 'color_left')
                meta_base = osp.join(cond_dir, 'metadata')

                if not (osp.exists(rgb_base) and osp.exists(meta_base)):
                    continue

                trajs = sorted([d for d in os.listdir(rgb_base) if osp.isdir(osp.join(rgb_base, d))])
                
                for traj in trajs:
                    traj_rgb_dir = osp.join(rgb_base, traj)
                    traj_meta_dir = osp.join(meta_base, traj)

                    if not osp.exists(traj_meta_dir):
                        continue

                    all_frames = sorted([f for f in os.listdir(traj_rgb_dir) if f.endswith('.JPEG')])
                    
                    if len(all_frames) >= getattr(self, 'frame_num', 1):
                        valid_scans.append({
                            'rgb_dir': traj_rgb_dir,
                            'meta_dir': traj_meta_dir,
                            'frames': all_frames,
                            'scan_id': f"{subset}/{cond}/{traj}"
                        })
        return valid_scans

    def __len__(self):
        return len(self.scans)

    def _get_views(self, idx, resolution, rng):
        try:
            scan = self.scans[idx]
        except ValueError:
            raise RuntimeError(f"Index {idx} is invalid for MidAirDataset")

        rgb_dir = scan['rgb_dir']
        meta_dir = scan['meta_dir']
        frames = scan['frames']

        if len(frames) < self.frame_num:
             raise RuntimeError(f"Not enough frames in {scan['scan_id']}")

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

        views = []
        for frame_file in selected_frames:
            frame_id = osp.splitext(frame_file)[0]
            rgb_path = osp.join(rgb_dir, frame_file)
            meta_path = osp.join(meta_dir, f"{frame_id}.npz")

            if not (osp.exists(rgb_path) and osp.exists(meta_path)):
                raise FileNotFoundError(f"Missing component for frame {frame_id} in {scan['scan_id']}")

            try:
                rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
                camera_intrinsics = load_intrinsics_from_npz(meta_path)
                fake_depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

                rgb_image, _, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, fake_depthmap, camera_intrinsics, resolution, rng=rng, info=rgb_path
                )

                views.append(dict(
                    img=rgb_image,
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset='MidAir',
                    label=scan['scan_id'],
                    instance=frame_file,
                ))

            except Exception as e:
                raise RuntimeError(f"Error loading frame {frame_file}: {e}")

        if len(views) < self.frame_num:
            raise RuntimeError(f"Failed to load enough frames for {scan['scan_id']}")

        return views
