"""
测试修复后的诊断函数
"""
import torch
from analyze_depth_metrics import analyze_sparse_depth

# 创建模拟数据
print("创建模拟数据...")
H, W = 480, 640
pred_depths = []
gt_depths = []

for i in range(3):
    # 创建稀疏 GT（激光雷达风格，只有 ~1% 的像素有值）
    gt = torch.zeros(H, W)
    n_points = int(H * W * 0.01)  # 1% 稀疏度
    indices = torch.randperm(H * W)[:n_points]
    gt_flat = gt.reshape(-1)
    gt_flat[indices] = torch.rand(n_points) * 80 + 5  # 5-85m
    gt = gt_flat.reshape(H, W)
    
    # 创建预测深度（与 GT 相近，加一点噪声）
    pred = torch.zeros(H, W)
    pred_flat = pred.reshape(-1)
    pred_flat[indices] = gt_flat[indices] + torch.randn(n_points) * 0.5  # ±0.5m 噪声
    pred = pred_flat.reshape(H, W)
    
    gt_depths.append(gt)
    pred_depths.append(pred)

print(f"生成 {len(gt_depths)} 帧测试数据\n")

# 运行诊断
print("运行诊断函数...")
try:
    analyze_sparse_depth(pred_depths, gt_depths)
    print("\n✅ 诊断完成，没有错误！")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
