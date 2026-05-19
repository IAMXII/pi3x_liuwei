# 深度指标快速检查清单

## 🔍 诊断你的深度预测问题

运行改进后的脚本时，按这个清单检查输出：

### Step 1: 稀疏性检查
```
平均稀疏度：1.15%（有效像素占比）
预测-GT重叠率：94.50%
```
- ✅ 稀疏度 > 1% + 重叠率 > 80%：正常
- ⚠️  稀疏度 < 1% 或 重叠率 < 80%：对齐可能不稳健

### Step 2: 深度范围检查
```
GT深度范围：0.500 ~ 95.230 m
GT深度平均：35.450 m，中位数：32.100 m
```
- 检查是否包含激光雷达的有效测距范围
- 检查是否有明显的离群点（min 和 max 差异很大）

### Step 3: 绝对误差检查 ⭐
```
平均绝对误差 (MAE)：0.4523 m  
中位数绝对误差：0.3210 m
RMSE：0.6834 m
```
- **这是最重要的指标** 
- MAE < 0.5m：优秀
- 0.5m < MAE < 1.0m：良好
- MAE > 2.0m：需要改进

### Step 4: 对齐参数检查
```
中位数对齐 scale：0.941027
```
- ✅ 0.9 ~ 1.1：对齐很好
- ⚠️  0.7 ~ 1.3：有偏离，但可接受
- ❌ < 0.5 或 > 2.0：严重问题，预测尺度完全错误

### Step 5: 相对误差理解（辅助参考）
```
平均相对误差 (Abs Rel)：0.015234
```
- 不要单独看这个数字！
- **必须结合 MAE 和深度范围一起理解**
- 示例解读：
  - AbsRel=0.10, 平均深度=30m → 平均误差 ≈ 3m
  - AbsRel=0.05, 平均深度=50m → 平均误差 ≈ 2.5m

---

## 常见问题解答

### Q: 为什么 AbsRel 很高（>0.1）但 MAE 看起来还可以？
**A:** 
- AbsRel=0.12，MAE=0.5m，深度平均=4m
- 说明很多点深度都比较近，0.5m 的误差在近距离显得相对误差大
- 这是正常的，**关键看 MAE**

### Q: 为什么对齐的 scale ≠ 1.0？
**A:** 
- GT 和预测可能存在系统性偏差
- 激光雷达测量本身可能有偏差
- 预测模型可能学到了某种深度缩放
- 只要 scale 接近 1.0（0.9-1.1），就是可以接受的

### Q: 稀疏度很低（< 1%）怎么办？
**A:** 
1. 检查激光雷达是否正确投影到图像上
2. 检查 GT 深度加载是否正确（可视化检查几张）
3. 如果确实很稀疏，考虑用 `--alignment_scope per_frame` 做逐帧对齐
4. 考虑增加 `--max_alignment_points` 的数量进行更稳健的对齐

### Q: 重叠率很低（< 80%）怎么办？
**A:**
- 说明预测深度和 GT 深度的有效区域不一致
- 检查：
  1. 预测模型是否正确
  2. 相机内参是否正确
  3. GT 深度数据是否对齐
- 可能需要增加 `--min_eval_depth` 或 `--max_eval_depth` 过滤

---

## 最常见的三种情况

### 情况 1: 深度看起来很好，数值却很高
```
现象：
  MAE=0.3m（好）
  AbsRel=0.25（很高？）
  
原因：
  平均深度 ≈ 1.2m（很近的深度）
  0.3m / 1.2m = 0.25
  
结论：✅ 这是正常的，MAE 才是关键！
```

### 情况 2: 所有指标都很差
```
现象：
  MAE=5.0m
  dRMSE=8.0m
  scale=2.5（严重偏离）
  
原因：
  1. 预测模型输出有问题
  2. 深度单位不对（比如应该乘以 0.001 但没有）
  3. 内参错误
  
解决：检查 depth_unit_scale 参数，试试 --depth_unit_scale 0.001
```

### 情况 3: 对齐很难收敛
```
现象：
  稀疏度 0.5%（太稀疏）
  重叠率 60%
  
解决：
  1. 尝试 --alignment_scope per_frame（逐帧对齐）
  2. 减少 --max_alignment_points（用全部点）
  3. 检查 GT 深度数据质量
```

---

## 数据检查脚本

如果还是不确定，可以运行这个脚本检查原始 GT 深度：

```python
import numpy as np
import cv2

# 检查 GT 深度文件
depth_path = "/data/liuwei/dataset/ntu_seq/cp/depth/000001.npy"
depth = np.load(depth_path)

print(f"形状：{depth.shape}")
print(f"数据类型：{depth.dtype}")
print(f"有效值范围：{np.min(depth[depth>0]):.3f} ~ {np.max(depth):.3f}")
print(f"有效像素数：{(depth > 0).sum()} / {depth.size}")
print(f"NaN 数量：{np.isnan(depth).sum()}")
print(f"Inf 数量：{np.isinf(depth).sum()}")

# 可视化
valid = (depth > 0) & np.isfinite(depth)
if valid.any():
    d_min, d_max = np.percentile(depth[valid], [2, 98])
    d_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    d_norm = np.clip(d_norm, 0, 1)
    d_vis = (d_norm * 255).astype(np.uint8)
    cv2.imwrite("depth_vis.png", d_vis)
    print("已保存 depth_vis.png")
```

---

## 最后一个提示

**如果你的深度看起来很好但指标高，很可能是：**

1. ✅ **正常现象**（相对误差对稀疏、近距离深度本来就会偏高）
2. 使用 **MAE** 作为主要参考指标
3. 用诊断报告**确认对齐参数没问题**
4. 检查**稀疏度和重叠率**是否正常

祝你调试顺利！ 🚀
