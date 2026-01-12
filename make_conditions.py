import os
import numpy as np


def build_intrinsics(
    N: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dtype=np.float32
):
    """
    构建 (N, 3, 3) OpenCV pinhole intrinsics
    """
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    intrinsics = np.repeat(K[None, :, :], N, axis=0)
    return intrinsics


def main():
    """
    示例：
    - 已有 poses.npy  -> (N, 4, 4)
    - 已有 depths.npy -> (N, H, W)
    - 输出 conditions.npz
    """
    # =========================
    # 2. 设置相机内参（OpenCV）
    # =========================
    # ⚠️ 根据你自己的数据修改
    fx = 332.232689
    fy = 332.644823
    cx = 333.058485
    cy = 240.998586
    N = 200
    intrinsics = build_intrinsics(
        N=N,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    print(f"Built intrinsics: {intrinsics.shape}")

    # =========================
    # 3. 保存为 npz
    # =========================
    save_path = "conditions/conditions.npz"

    np.savez(
        save_path,
        intrinsics=intrinsics.astype(np.float32),
    )

    print(f"Saved conditions to: {save_path}")


if __name__ == "__main__":
    main()
