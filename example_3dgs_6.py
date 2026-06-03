import subprocess
import sys

RUNNER = r"""
import sys

sys.argv[0] = "example_3dgs_6.py"

import torch

torch.cuda.is_available()
torch.cuda.device_count()

from example_3dgs_5 import main


def _force_rgb_all_gaussians(argv):
    # Keep pushed/far support Gaussians in RGB renders. PLY export may still use
    # --ply_opacity_threshold, but RGB rendering must not drop low-opacity
    # pushed support Gaussians via --render_opacity_threshold.
    flag = "--render_opacity_threshold"
    if flag not in argv:
        argv.extend([flag, "0.0"])
        return

    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        argv.append("0.0")
        return

    try:
        value = float(argv[idx + 1])
    except ValueError:
        value = 0.0
    if value > 0.0:
        print(
            "[example_3dgs_6] Overriding --render_opacity_threshold to 0.0 "
            "so pushed/far Gaussians participate in RGB rendering.",
            flush=True,
        )
        argv[idx + 1] = "0.0"


_force_rgb_all_gaussians(sys.argv)
main()
"""

if __name__ == "__main__":
    raise SystemExit(subprocess.run([sys.executable, "-c", RUNNER, *sys.argv[1:]]).returncode)
