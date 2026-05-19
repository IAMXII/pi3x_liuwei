"""
深度指标改进的使用示例

这个脚本展示了如何使用改进后的诊断功能
"""

# 示例 1: 运行完整的评估（包含诊断）
# ==========================================

# 基础运行（会自动输出诊断报告）：
"""
python example_3dgs_1.py \\
  --data_path /data/liuwei/dataset/ntu_seq/cp/rgb \\
  --depth_path /data/liuwei/dataset/ntu_seq/cp/depth \\
  --ckpt ./outputs/pi3_highres_0418/ckpts/best_model/model.safetensors \\
  --output_dir ./output_test_depth \\
  --save_aligned_depth
"""

# 示例 2: 对稀疏深度进行诊断（无需完整推理）
# ============================================

import torch
from analyze_depth_metrics import analyze_sparse_depth

# 假设你已经有了预测深度和 GT 深度
# pred_depths_cpu: List[Tensor]，形状 (H, W)
# gt_depths_cpu: List[Tensor]，形状 (H, W)

def example_standalone_diagnosis():
    """
    单独使用诊断模块分析已有的深度图
    """
    import numpy as np
    
    # 模拟一些深度数据
    print("创建模拟数据...")
    
    # 创建稀疏 GT 深度（激光雷达风格）
    H, W = 480, 640
    gt_depths = []
    pred_depths = []
    
    for i in range(5):  # 5 帧
        # 创建稀疏 GT（只有 1% 的像素有值）
        gt = torch.zeros(H, W)
        
        # 随机放置一些激光雷达点
        n_points = int(H * W * 0.01)  # 1% 稀疏度
        indices = torch.randperm(H * W)[:n_points]
        gt_flat = gt.reshape(-1)
        gt_flat[indices] = torch.rand(n_points) * 80 + 5  # 5-85m 范围
        gt = gt_flat.reshape(H, W)
        
        # 创建预测深度（与 GT 相近但有噪声）
        pred = torch.zeros(H, W)
        pred_flat = pred.reshape(-1)
        pred_flat[indices] = gt_flat[indices] + torch.randn(n_points) * 0.5  # 加 0.5m 噪声
        pred = pred_flat.reshape(H, W)
        
        gt_depths.append(gt)
        pred_depths.append(pred)
    
    print(f"生成 {len(gt_depths)} 帧深度图（H={H}, W={W}）\n")
    
    # 运行诊断
    print("运行诊断...\n")
    analyze_sparse_depth(pred_depths, gt_depths)


# 示例 3: 理解诊断输出
# ====================

def example_interpret_output():
    """
    如何理解诊断输出
    """
    print("""
    
【诊断输出解读指南】

1. 稀疏性分析：
   Frame   0: GT有效 234 ( 1.20%) | 重叠 225 ( 96.15%)
   ├─ GT有效：这一帧有 234 个有效的 GT 像素（占 1.20%）
   ├─ 重叠：其中有 225 个像素也在预测中有效（占 96.15%）
   └─ 意义：如果重叠率太低 (<80%)，说明预测和 GT 不匹配

2. 深度范围分析：
   GT深度范围：0.500 ~ 95.230 m
   ├─ min=0.5m：激光雷达最近点距离
   ├─ max=95.2m：激光雷达最远点距离
   └─ 检查：这个范围是否在激光雷达的有效范围内

3. 绝对误差分析（最重要！）：
   平均绝对误差 (MAE)：0.4523 m  ← 关键指标
   ├─ MAE < 0.3m：优秀
   ├─ 0.3m < MAE < 0.7m：很好
   ├─ 0.7m < MAE < 2.0m：还可以
   └─ MAE > 2.0m：需要改进

4. 对齐诊断：
   中位数对齐 scale：0.941027
   ├─ 0.95-1.05：完美对齐（预测深度尺度正确）
   ├─ 0.90-1.10：很好对齐
   ├─ 0.80-1.20：可以接受
   └─ < 0.5 或 > 2.0：严重问题，需要检查预测模型
    """)


# 示例 4: 比较对齐方式的影响
# ===========================

def example_alignment_comparison():
    """
    演示不同对齐方式的效果
    """
    print("""
    
【对齐方式选择指南】

1. 全局对齐 vs 逐帧对齐：
   
   --alignment_scope global（默认）:
   └─ 使用所有帧的有效点一起计算 scale/shift
      适合：稀疏度正常（>1%），数据相对均匀
      优点：稳定，统计意义强
      缺点：无法处理帧间差异很大的情况
   
   --alignment_scope per_frame:
   └─ 每帧单独计算 scale/shift
      适合：稀疏度很低（<1%），帧间差异大
      优点：灵活，能适应不同场景
      缺点：每帧样本少，可能不稳定

2. 对齐模式选择：
   
   auto（推荐）：
   └─ 自动尝试多种模式，选择最好的
      尝试：median_scale, l2_scale, robust_scale, robust_rel_scale
      可选：--allow_affine 也尝试 affine 变换
   
   median_scale：
   └─ 用中位数，对离群点鲁棒
   
   robust_affine（最强大）：
   └─ 同时优化 scale 和 shift，对复杂情况鲁棒
      但需要足够多的有效点（>1000）


【推荐的运行命令】

情况 1: 稀疏度 > 1%，数据相对均匀
$ python example_3dgs_1.py \\
  --data_path ... --depth_path ... \\
  --alignment_scope global \\
  --alignment_mode auto \\
  --allow_affine

情况 2: 稀疏度 < 1%，或帧间差异大
$ python example_3dgs_1.py \\
  --data_path ... --depth_path ... \\
  --alignment_scope per_frame \\
  --alignment_mode auto

情况 3: 对齐困难，需要更多样本
$ python example_3dgs_1.py \\
  --data_path ... --depth_path ... \\
  --max_alignment_points 500000 \\
  --alignment_mode robust_affine
    """)


if __name__ == "__main__":
    print("="*70)
    print("深度指标改进 - 使用示例")
    print("="*70)
    
    # 运行示例
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "standalone":
            print("\n【示例 1: 单独诊断模式】")
            example_standalone_diagnosis()
        elif sys.argv[1] == "interpret":
            print("\n【示例 2: 输出解读】")
            example_interpret_output()
        elif sys.argv[1] == "alignment":
            print("\n【示例 3: 对齐方式比较】")
            example_alignment_comparison()
    else:
        print("\n使用方法:")
        print("  python example_depth_usage.py standalone    # 运行诊断示例")
        print("  python example_depth_usage.py interpret     # 解读诊断输出")
        print("  python example_depth_usage.py alignment     # 对齐方式说明")
        print("\n或者直接运行改进后的 example_3dgs_1.py，会自动输出诊断报告")
