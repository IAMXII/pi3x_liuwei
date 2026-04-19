import os
import os.path as osp

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


def load_matrix_txt(path):
    if not osp.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    array = np.loadtxt(path, dtype=np.float32).reshape(4, 4)
    return array


def pose_distance(c2w_a, c2w_b, trans_weight=1.0):
    rot = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    trace = np.trace(rot)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_dist = np.degrees(np.arccos(val)) / 180.0
    trans_dist = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
    return float(rot_dist + trans_weight * trans_dist)


class NeRFOSRDataset(BaseDataset):
    """
    Illumination-paired NeRF-OSR dataset.

    Source multi-view frames come from one recording session.
    Paired supervision views are pose-matched from another session of the
    same scene, which gives us the "same place, different environment" signal.
    """

    def __init__(
        self,
        root_dir="/home/liuwei/mnt/NeRF-OSR/Data",
        mode="train",
        eval_split="test",
        envmap_preference=("ENV_MAP", "ENV_MAP_CC"),
        trans_weight=1.0,
        **kwargs,
    ):
        super().__init__(mode=mode, **kwargs)
        self.root_dir = root_dir
        self.mode = mode
        self.eval_split = eval_split
        self.envmap_preference = tuple(envmap_preference)
        self.trans_weight = float(trans_weight)
        self.dataset_label = "NeRFOSRDataset"

        self.scans = np.array(self._scan_dataset(), dtype=object)
        if len(self.scans) == 0:
            print(f"[{self.__class__.__name__}] CRITICAL: No sequences found in {root_dir}")
        else:
            print(f"[{self.__class__.__name__}] Loaded {len(self.scans)} scenes.")

    def _resolve_scene_dir(self, scene_root):
        for candidate in ["final", "final_clean"]:
            path = osp.join(scene_root, candidate)
            if osp.isdir(path):
                return path
        return None

    def _scan_dataset(self):
        scans = []
        if not osp.isdir(self.root_dir):
            return scans

        split_name = "train" if self.mode == "train" else self.eval_split
        scene_names = sorted([d for d in os.listdir(self.root_dir) if osp.isdir(osp.join(self.root_dir, d))])

        for scene_name in scene_names:
            scene_root = osp.join(self.root_dir, scene_name)
            scene_dir = self._resolve_scene_dir(scene_root)
            if scene_dir is None:
                continue

            split_dir = osp.join(scene_dir, split_name)
            rgb_dir = osp.join(split_dir, "rgb")
            pose_dir = osp.join(split_dir, "pose")
            intr_dir = osp.join(split_dir, "intrinsics")
            mask_dir = osp.join(split_dir, "mask")
            if not (osp.isdir(rgb_dir) and osp.isdir(pose_dir) and osp.isdir(intr_dir)):
                continue

            session_to_views = {}
            rgb_files = sorted([
                f for f in os.listdir(rgb_dir)
                if osp.isfile(osp.join(rgb_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            for rgb_file in rgb_files:
                stem = osp.splitext(rgb_file)[0]
                session = stem.split("_IMG_", 1)[0] if "_IMG_" in stem else None
                if session is None:
                    continue

                pose_path = osp.join(pose_dir, f"{stem}.txt")
                intr_path = osp.join(intr_dir, f"{stem}.txt")
                mask_path = osp.join(mask_dir, f"{stem}.png")
                if not (osp.isfile(pose_path) and osp.isfile(intr_path)):
                    continue

                pose = load_matrix_txt(pose_path)
                intrinsics = load_matrix_txt(intr_path)[:3, :3]
                view = {
                    "scene": scene_name,
                    "session": session,
                    "stem": stem,
                    "rgb_path": osp.join(rgb_dir, rgb_file),
                    "pose": pose.astype(np.float32),
                    "intrinsics": intrinsics.astype(np.float32),
                    "mask_path": mask_path if osp.isfile(mask_path) else None,
                }
                session_to_views.setdefault(session, []).append(view)

            valid_sessions = {k: v for k, v in session_to_views.items() if len(v) >= getattr(self, "frame_num", 1)}
            if len(valid_sessions) < 2:
                continue

            envmap_dirs = {}
            for env_key in self.envmap_preference:
                env_root = osp.join(scene_dir, env_key)
                if not osp.isdir(env_root):
                    continue
                for session in os.listdir(env_root):
                    session_dir = osp.join(env_root, session)
                    if osp.isdir(session_dir):
                        envmap_dirs.setdefault(session, []).append(session_dir)

            scans.append({
                "scene": scene_name,
                "scene_dir": scene_dir,
                "split": split_name,
                "sessions": valid_sessions,
                "session_names": sorted(valid_sessions.keys()),
                "envmap_dirs": envmap_dirs,
                "scan_id": f"{scene_name}/{split_name}",
            })

        return scans

    def __len__(self):
        return len(self.scans)

    def _select_source_views(self, views, rng):
        num_views = len(views)
        anchor_idx = int(rng.integers(num_views))
        if self.frame_num == 1:
            return [views[anchor_idx]]

        anchor_pose = views[anchor_idx]["pose"]
        dists = np.array([pose_distance(anchor_pose, view["pose"], trans_weight=self.trans_weight) for view in views])
        order = np.argsort(dists)
        selected = [views[int(i)] for i in order[:self.frame_num]]
        return selected

    def _match_view_in_target_session(self, source_view, target_views):
        dists = [pose_distance(source_view["pose"], target_view["pose"], trans_weight=self.trans_weight) for target_view in target_views]
        best_idx = int(np.argmin(np.asarray(dists)))
        return target_views[best_idx]

    def _load_rgb(self, path):
        return np.array(Image.open(path).convert("RGB"))

    def _load_style_image(self, scan, target_session, resolution, fallback_rgb, rng):
        candidates = scan["envmap_dirs"].get(target_session, [])
        if candidates:
            session_dir = candidates[0]
            files = sorted([
                f for f in os.listdir(session_dir)
                if osp.isfile(osp.join(session_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            if files:
                style_path = osp.join(session_dir, files[int(rng.integers(len(files)))])
                style_img = Image.open(style_path).convert("RGB")
                return style_img.resize(tuple(resolution), resample=Image.BICUBIC)
        return fallback_rgb

    def _get_views(self, idx, resolution, rng):
        try:
            scan = self.scans[idx]
        except ValueError:
            raise RuntimeError(f"Index {idx} is invalid for NeRFOSRDataset")

        session_names = list(scan["session_names"])
        if len(session_names) < 2:
            raise RuntimeError(f"Not enough sessions in {scan['scan_id']}")

        source_session, target_session = rng.choice(session_names, 2, replace=False)
        source_views_all = scan["sessions"][source_session]
        target_views_all = scan["sessions"][target_session]
        if len(source_views_all) < self.frame_num or len(target_views_all) == 0:
            raise RuntimeError(f"Insufficient views for pairing in {scan['scan_id']}")

        selected_source_views = self._select_source_views(source_views_all, rng)
        views = []

        style_img_pil = None
        for source_view in selected_source_views:
            target_view = self._match_view_in_target_session(source_view, target_views_all)

            src_rgb = self._load_rgb(source_view["rgb_path"])
            tgt_rgb = self._load_rgb(target_view["rgb_path"])

            src_depth = np.zeros(src_rgb.shape[:2], dtype=np.float32)
            tgt_depth = np.zeros(tgt_rgb.shape[:2], dtype=np.float32)

            src_pose = source_view["pose"].astype(np.float32)
            src_intr = source_view["intrinsics"].astype(np.float32)
            tgt_pose = target_view["pose"].astype(np.float32)
            tgt_intr = target_view["intrinsics"].astype(np.float32)

            src_rgb_crop, src_depth_crop, src_intr_crop = self._crop_resize_if_necessary(
                src_rgb,
                src_depth,
                src_intr,
                resolution,
                rng=rng,
                info=source_view["rgb_path"],
            )

            tgt_rgb_crop, _, tgt_intr_crop = self._crop_resize_if_necessary(
                tgt_rgb,
                tgt_depth,
                tgt_intr,
                resolution,
                rng=rng,
                info=target_view["rgb_path"],
            )

            if style_img_pil is None:
                style_img_pil = self._load_style_image(scan, target_session, resolution, tgt_rgb_crop, rng)

            views.append({
                "img": src_rgb_crop,
                "img_paired": tgt_rgb_crop,
                "img_style": style_img_pil.copy(),
                "depthmap": src_depth_crop.astype(np.float32),
                "camera_pose": src_pose,
                "camera_intrinsics": src_intr_crop.astype(np.float32),
                "camera_pose_paired": tgt_pose,
                "camera_intrinsics_paired": tgt_intr_crop.astype(np.float32),
                "has_depth": np.bool_(False),
                "dataset": "NeRF-OSR",
                "label": f"{scan['scene']}_{source_session}_to_{target_session}",
                "instance": source_view["stem"],
            })

        if len(views) < self.frame_num:
            raise RuntimeError(f"Failed to load enough frames for {scan['scan_id']}")

        return views
