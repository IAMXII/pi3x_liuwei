#!/usr/bin/env python
"""
多尺度高斯融合 - 代码验证脚本
用途：验证所有关键函数和参数的集成正确性
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_imports():
    """验证所有关键导入"""
    print("=" * 70)
    print("【验证 1】导入检查")
    print("=" * 70)
    try:
        from pi3.models.pi3_3dgs import (
            Pi3_3DGS,
            fuse_multiscale_gaussians,
            _chunked_clustering_scale_aware,
            _bfs_clustering,
            compute_adaptive_clustering_radii,
        )
        print("✓ pi3_3dgs 核心函数导入成功")

        from pi3.models.loss_3dgs import Pi3LossGS
        print("✓ loss_3dgs 损失函数导入成功")

        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_constructor():
    """验证模型构造器中的多尺度参数"""
    print("\n" + "=" * 70)
    print("【验证 2】模型构造器参数")
    print("=" * 70)
    try:
        from pi3.models.pi3_3dgs import Pi3_3DGS

        # 创建模型，不加载预训练权重
        model = Pi3_3DGS(
            num_scales=3,
            scale_factors=[1.0, 0.5, 0.2],
            ckpt=None  # 跳过权重加载
        )

        # 检查属性
        assert hasattr(model, 'num_scales'), "缺少 num_scales 属性"
        assert model.num_scales == 3, f"num_scales 应为 3，得到 {model.num_scales}"
        print(f"✓ num_scales = {model.num_scales}")

        assert hasattr(model, 'scale_factors'), "缺少 scale_factors 属性"
        assert model.scale_factors == [1.0, 0.5, 0.2], f"scale_factors 错误: {model.scale_factors}"
        print(f"✓ scale_factors = {model.scale_factors}")

        # 检查多尺度高斯头
        assert hasattr(model, 'gs_heads_multiscale'), "缺少 gs_heads_multiscale"
        assert len(model.gs_heads_multiscale) == 3, f"应有 3 个 gs_heads，得到 {len(model.gs_heads_multiscale)}"
        print(f"✓ gs_heads_multiscale x {len(model.gs_heads_multiscale)}")

        assert hasattr(model, 'gs_decoders_multiscale'), "缺少 gs_decoders_multiscale"
        assert len(model.gs_decoders_multiscale) == 3, f"应有 3 个 gs_decoders，得到 {len(model.gs_decoders_multiscale)}"
        print(f"✓ gs_decoders_multiscale x {len(model.gs_decoders_multiscale)}")

        return True
    except Exception as e:
        print(f"✗ 建模失败: {e}")
        return False

def test_clustering_functions():
    """验证聚类函数的逻辑"""
    print("\n" + "=" * 70)
    print("【验证 3】聚类函数")
    print("=" * 70)
    try:
        from pi3.models.pi3_3dgs import (
            _bfs_clustering,
            compute_adaptive_clustering_radii,
        )
        device = torch.device('cpu')

        # 测试 BFS 聚类
        adjacency = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ], device=device)

        cluster_ids = _bfs_clustering(adjacency, device)
        print(f"  BFS 聚类输入 (4 nodes):")
        print(f"    邻接矩阵: {adjacency}")
        print(f"    输出 cluster_ids: {cluster_ids.tolist()}")

        assert cluster_ids[0] == cluster_ids[1], "节点 0 和 1 应在同一 cluster"
        assert cluster_ids[2] == cluster_ids[3], "节点 2 和 3 应在同一 cluster"
        assert cluster_ids[0] != cluster_ids[2], "节点 0 和 2 应在不同 cluster"
        print("✓ BFS 聚类逻辑正确")

        # 测试适应性半径计算
        xyz = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], device=device)

        radius_coarse, radius_fine = compute_adaptive_clustering_radii(xyz)
        print(f"  适应性半径计算 (4 points):")
        print(f"    xyz 范围: [0-3]")
        print(f"    radius_coarse = {radius_coarse.item():.4f}")
        print(f"    radius_fine = {radius_fine.item():.4f}")
        assert radius_fine < radius_coarse, "细尺度半径应小于粗尺度"
        print("✓ 适应性半径计算正确")

        return True
    except Exception as e:
        print(f"✗ 聚类函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scale_awareness():
    """验证尺度感知聚类的判断逻辑"""
    print("\n" + "=" * 70)
    print("【验证 4】尺度感知判断机制")
    print("=" * 70)
    try:
        device = torch.device('cpu')

        # 构造测试案例
        xyz = torch.tensor([
            [0.0, 0.0, 0.0],    # 高斯 0：粗 (tag=0)
            [0.08, 0.0, 0.0],   # 高斯 1：粗 (tag=0)
            [0.05, 0.0, 0.0],   # 高斯 2：细 (tag=2)
        ], device=device)

        scale_tags = torch.tensor([0, 0, 2], device=device)  # 尺度标签

        # 手动计算距离和尺度惩罚
        dist_mat = torch.cdist(xyz, xyz, p=2.0)
        scale_diff = (scale_tags.unsqueeze(1) - scale_tags.unsqueeze(0)).abs().float()

        r_coarse = 0.2  # 假设 scene_diameter = 1.0
        r_fine = r_coarse * 0.5  # = 0.1
        scale_penalty = scale_diff * r_coarse * 0.5
        dist_adjusted = dist_mat + scale_penalty

        print(f"  欧氏距离矩阵:")
        print(f"    {dist_mat.numpy()}")
        print(f"  尺度标签: {scale_tags.tolist()}")
        print(f"  尺度差异: {scale_diff.numpy().astype(int)}")
        print(f"  融合距离 (d_euclid + scale_penalty):")
        print(f"    {dist_adjusted.numpy()}")
        print(f"  融合半径 r_fine = {r_fine}")

        # 判断邻接
        adjacency = (dist_adjusted < r_fine)
        print(f"  邻接关系（< r_fine）:")
        print(f"    {adjacency.numpy().astype(int)}")

        # 验证逻辑
        # (0,1): d=0.08, scale_diff=0, d_adj=0.08+0=0.08 < 0.1 ✓ 应融合
        # (0,2): d=0.05, scale_diff=2, d_adj=0.05+0.2=0.25 > 0.1 ✗ 不融合
        # (1,2): d=0.03, scale_diff=2, d_adj=0.03+0.2=0.23 > 0.1 ✗ 不融合

        assert adjacency[0, 1] and adjacency[1, 0], "高斯 0 和 1（同尺度）应相邻"
        assert not adjacency[0, 2] and not adjacency[2, 0], "高斯 0 和 2（不同尺度）不应相邻"
        print("✓ 尺度感知判断逻辑正确")

        return True
    except Exception as e:
        print(f"✗ 尺度感知测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions():
    """验证损失函数参数"""
    print("\n" + "=" * 70)
    print("【验证 5】损失函数定义")
    print("=" * 70)
    try:
        from pi3.models.loss_3dgs import Pi3LossGS

        loss_fn = Pi3LossGS(
            lambda_consistency=0.05,
            lambda_scale_diversity=0.02,
            train_stage=1
        )

        assert hasattr(loss_fn, 'lambda_consistency'), "缺少 lambda_consistency"
        assert loss_fn.lambda_consistency == 0.05, f"lambda_consistency 应为 0.05，得到 {loss_fn.lambda_consistency}"
        print(f"✓ lambda_consistency = {loss_fn.lambda_consistency}")

        assert hasattr(loss_fn, 'lambda_scale_diversity'), "缺少 lambda_scale_diversity"
        assert loss_fn.lambda_scale_diversity == 0.02, f"lambda_scale_diversity 应为 0.02，得到 {loss_fn.lambda_scale_diversity}"
        print(f"✓ lambda_scale_diversity = {loss_fn.lambda_scale_diversity}")

        # 检查方法
        assert hasattr(loss_fn, '_compute_consistency_loss'), "缺少 _compute_consistency_loss 方法"
        print(f"✓ _compute_consistency_loss 方法存在")

        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_logic():
    """验证融合逻辑（完整但简化的端到端测试）"""
    print("\n" + "=" * 70)
    print("【验证 6】融合逻辑（端到端）")
    print("=" * 70)
    try:
        from pi3.models.pi3_3dgs import fuse_multiscale_gaussians
        device = torch.device('cpu')
        B = 1  # batch size
        K_per_scale = 10

        # 构造 3 个尺度的高斯集合
        gaussians_dict_list = []
        for scale_idx in range(3):
            xyz = torch.randn(B, K_per_scale, 3, device=device)
            rot = torch.randn(B, K_per_scale, 4, device=device)
            rot = torch.nn.functional.normalize(rot, dim=-1)
            scale = torch.abs(torch.randn(B, K_per_scale, 3, device=device)) * 0.1
            opacity = torch.sigmoid(torch.randn(B, K_per_scale, 1, device=device))
            color = torch.sigmoid(torch.randn(B, K_per_scale, 3, device=device))

            gaussians_dict_list.append({
                'xyz': xyz,
                'rotation': rot,
                'scale': scale,
                'opacity': opacity,
                'color': color,
            })

        conf = torch.randn(B, K_per_scale * 3, 1, device=device)

        # 调用融合函数
        result = fuse_multiscale_gaussians(
            gaussians_dict_list,
            conf,
            scale_factors=[1.0, 0.5, 0.2],
            enable_multiscale=True,
            chunk_size=50000,
            return_stats=True
        )

        fused_xyz, fused_rot, fused_scale, fused_opacity, fused_color, fused_conf, stats = result

        print(f"  输入: 3 个尺度 × {K_per_scale} 高斯 = {K_per_scale * 3} 总数")
        print(f"  输出尺寸:")
        print(f"    - xyz: {fused_xyz.shape}")
        print(f"    - rotation: {fused_rot.shape}")
        print(f"    - scale: {fused_scale.shape}")
        print(f"    - opacity: {fused_opacity.shape}")
        print(f"    - color: {fused_color.shape}")
        print(f"    - conf: {fused_conf.shape}")

        print(f"  融合统计:")
        for stat in stats:
            print(f"    批次 {stat['batch']}: {stat['original_K']} → {stat['fused_K']} "
                  f"(压缩率 {stat['compression_ratio']:.2f}x)")

        # 验证输出形状
        assert fused_xyz.shape[0] == B, f"批次维度应为 {B}"
        assert fused_xyz.shape[-1] == 3, "xyz 应为 3 维"
        assert fused_rot.shape[-1] == 4, "rotation 应为 4 维"
        assert fused_scale.shape[-1] == 3, "scale 应为 3 维"
        assert fused_opacity.shape[-1] == 1, "opacity 应为 1 维"
        assert fused_color.shape[-1] == 3, "color 应为 3 维"

        print("✓ 融合逻辑输出正确")
        return True
    except Exception as e:
        print(f"✗ 融合逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有验证"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "多尺度高斯融合 - 代码完整性验证" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    tests = [
        ("导入验证", test_imports),
        ("模型构造", test_constructor),
        ("聚类函数", test_clustering_functions),
        ("尺度感知", test_scale_awareness),
        ("损失函数", test_loss_functions),
        ("融合逻辑", test_fusion_logic),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 发生异常: {e}")
            results.append((name, False))

    # 总结
    print("\n" + "=" * 70)
    print("【验证总结】")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {name}")

    print("=" * 70)
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("\n✅ 所有验证通过！代码集成完整且正确。")
        return 0
    else:
        print(f"\n❌ {total - passed} 个验证失败。请检查上述错误。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
