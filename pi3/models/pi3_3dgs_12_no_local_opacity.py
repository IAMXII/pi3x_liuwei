"""Pi3_3DGS_12 variant without local opacity competition.

This keeps the Pi3_3DGS_12 architecture and checkpoint keys unchanged while
disabling the local competition path that softly suppresses opacity.
"""

from .pi3_3dgs_12 import Pi3_3DGS as Pi3_3DGS_12


class Pi3_3DGS(Pi3_3DGS_12):
    def __init__(self, *args, **kwargs):
        kwargs["enable_local_competition"] = False
        kwargs["hard_prune_redundant_gaussians_eval"] = False
        super().__init__(*args, **kwargs)
        self.enable_local_competition = False
        self.hard_prune_redundant_gaussians_eval = False

    def _apply_local_competition(self, gaussian_dict, source_view, scene_size, camera_poses=None, intrinsics=None, image_hw=None):
        self.enable_local_competition = False
        self.hard_prune_redundant_gaussians_eval = False
        return super()._apply_local_competition(
            gaussian_dict,
            source_view,
            scene_size,
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            image_hw=image_hw,
        )
