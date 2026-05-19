"""
深度指标诊断脚本 - 检查稀疏激光雷达深度的对齐和计算问题
"""
import numpy as np
import torch


def analyze_sparse_depth(pred_depths, gt_depths, min_eval_depth=1e-5, max_eval_depth=None):
    """
    诊断稀疏深度的对齐和计算问题
    
    Args:
        pred_depths: 预测深度列表 (Tensor)
        gt_depths: GT深度列表 (Tensor)
        min_eval_depth: 最小有效深度
        max_eval_depth: 最大有效深度
    """
    print("\n" + "="*70)
    print("稀疏激光雷达深度诊断报告")
    print("="*70)
    
    # 1. 稀疏性分析
    print("\n【稀疏性分析】")
    total_pixels = 0
    valid_gt_pixels = 0
    valid_pred_pixels = 0
    valid_overlap_pixels = 0
    
    frame_reports = []
    
    for idx, (pred, gt) in enumerate(zip(pred_depths, gt_depths)):
        H, W = gt.shape
        frame_size = H * W
        total_pixels += frame_size
        
        # GT 有效像素
        gt_valid = (torch.isfinite(gt) & (gt > min_eval_depth))
        if max_eval_depth is not None:
            gt_valid &= (gt <= max_eval_depth)
        gt_count = gt_valid.sum().item()
        valid_gt_pixels += gt_count
        
        # 预测有效像素
        pred_valid = (torch.isfinite(pred) & (pred > min_eval_depth))
        pred_count = pred_valid.sum().item()
        valid_pred_pixels += pred_count
        
        # 重叠有效像素
        overlap = gt_valid & pred_valid
        overlap_count = overlap.sum().item()
        valid_overlap_pixels += overlap_count
        
        sparsity_ratio = gt_count / frame_size if frame_size > 0 else 0
        overlap_ratio = overlap_count / gt_count if gt_count > 0 else 0
        
        frame_reports.append({
            "frame": idx,
            "gt_valid": gt_count,
            "pred_valid": pred_count,
            "overlap": overlap_count,
            "sparsity": sparsity_ratio,
            "overlap_ratio": overlap_ratio,
        })
        
        if idx < 3 or idx >= len(pred_depths) - 2:  # 显示前3帧和后2帧
            print(f"  Frame {idx:3d}: GT有效 {gt_count:7d} ({sparsity_ratio*100:5.2f}%) | "
                  f"重叠 {overlap_count:7d} ({overlap_ratio*100:5.2f}%)")
    
    avg_sparsity = valid_gt_pixels / total_pixels if total_pixels > 0 else 0
    overall_overlap = valid_overlap_pixels / valid_gt_pixels if valid_gt_pixels > 0 else 0
    
    print(f"\n  总计：{len(pred_depths)} 帧")
    print(f"  平均稀疏度：{avg_sparsity*100:.2f}%（有效像素占比）")
    print(f"  预测-GT重叠率：{overall_overlap*100:.2f}%")
    
    if avg_sparsity < 0.01:
        print(f"  ⚠️  警告：稀疏度 < 1%，对齐结果可能不稳健！")
    if overall_overlap < 0.8:
        print(f"  ⚠️  警告：重叠率 < 80%，预测和GT匹配度差！")
    
    # 2. 深度范围分析
    print("\n【深度范围分析】")
    all_gt_valid = []
    all_pred_valid = []
    
    for pred, gt in zip(pred_depths, gt_depths):
        gt_valid_mask = torch.isfinite(gt) & (gt > min_eval_depth)
        pred_valid_mask = torch.isfinite(pred) & (pred > min_eval_depth)
        
        if gt_valid_mask.any():
            all_gt_valid.append(gt[gt_valid_mask])
        if pred_valid_mask.any():
            all_pred_valid.append(pred[pred_valid_mask])
    
    if all_gt_valid:
        gt_concat = torch.cat(all_gt_valid)
        print(f"  GT深度范围：{gt_concat.min():.3f} ~ {gt_concat.max():.3f} m")
        print(f"  GT深度平均：{gt_concat.mean():.3f} m，中位数：{gt_concat.median():.3f} m")
        
        # 深度分布
        percentiles = [5, 25, 50, 75, 95]
        print(f"  GT深度百分位数: ", end="")
        for p in percentiles:
            val = torch.quantile(gt_concat, p/100)
            print(f"{p}%={val:.2f}m ", end="")
        print()
    
    if all_pred_valid:
        pred_concat = torch.cat(all_pred_valid)
        print(f"  预测深度范围：{pred_concat.min():.3f} ~ {pred_concat.max():.3f} m")
        print(f"  预测深度平均：{pred_concat.mean():.3f} m，中位数：{pred_concat.median():.3f} m")
        print(f"  预测深度百分位数: ", end="")
        for p in percentiles:
            val = torch.quantile(pred_concat, p/100)
            print(f"{p}%={val:.2f}m ", end="")
        print()
    
    # 3. 绝对误差分析（重要！）
    print("\n【绝对误差分析（更可靠）】")
    abs_errors = []
    rel_errors = []
    
    for pred, gt in zip(pred_depths, gt_depths):
        mask = torch.isfinite(pred) & torch.isfinite(gt)
        mask &= (pred > min_eval_depth) & (gt > min_eval_depth)
        
        if mask.any():
            pred_v = pred[mask]
            gt_v = gt[mask]
            abs_err = torch.abs(pred_v - gt_v)
            rel_err = abs_err / (gt_v + 1e-8)
            
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
    
    if abs_errors:
        abs_concat = torch.cat(abs_errors)
        rel_concat = torch.cat(rel_errors)
        
        # 绝对误差（更直观）
        print(f"  平均绝对误差 (MAE)：{abs_concat.mean():.4f} m")
        print(f"  中位数绝对误差 (Median AE)：{abs_concat.median():.4f} m")
        print(f"  RMSE：{torch.sqrt((abs_concat**2).mean()):.4f} m")
        print(f"  绝对误差百分位: ", end="")
        for p in [10, 25, 50, 75, 90]:
            val = torch.quantile(abs_concat, p/100)
            print(f"{p}%={val:.3f}m ", end="")
        print()
        
        # 相对误差
        print(f"\n  平均相对误差 (Abs Rel)：{rel_concat.mean():.6f}")
        print(f"  中位数相对误差：{rel_concat.median():.6f}")
        print(f"  相对误差百分位: ", end="")
        for p in [10, 25, 50, 75, 90]:
            val = torch.quantile(rel_concat, p/100)
            print(f"{p}%={val:.4f} ", end="")
        print()
        
        if rel_concat.mean() > 0.1:
            print(f"\n  ⚠️  相对误差较高 ({rel_concat.mean():.4f})，可能原因：")
            print(f"     1. 预测深度和GT深度尺度差异大（需要对齐）")
            print(f"     2. 预测质量确实有问题")
            print(f"     3. 对齐参数选择不当")
    
    # 4. 对齐参数诊断
    print("\n【对齐诊断】")
    if all_gt_valid and all_pred_valid:
        # 只使用重叠的有效点
        pred_overlaps = []
        gt_overlaps = []
        frame_scales = []  # 记录每帧的 scale
        
        for frame_idx, (pred, gt) in enumerate(zip(pred_depths, gt_depths)):
            mask = torch.isfinite(pred) & torch.isfinite(gt)
            mask &= (pred > min_eval_depth) & (gt > min_eval_depth)
            if max_eval_depth is not None:
                mask &= (gt <= max_eval_depth)
            if mask.any():
                pred_overlaps.append(pred[mask])
                gt_overlaps.append(gt[mask])
                
                # 计算该帧的 scale
                pred_valid = pred[mask]
                gt_valid = gt[mask]
                frame_scale = torch.median(gt_valid) / (torch.median(pred_valid) + 1e-8)
                frame_scales.append((frame_idx, float(frame_scale.item())))
        
        if not pred_overlaps or not gt_overlaps:
            print("  无重叠的有效点，无法诊断对齐参数")
        else:
            src = torch.cat(pred_overlaps)
            tgt = torch.cat(gt_overlaps)
            
            # 尝试几种对齐方式
            scale_median = torch.median(tgt) / (torch.median(src) + 1e-8)
            scale_mean = tgt.mean() / (src.mean() + 1e-8)
            scale_l2 = torch.sum(src * tgt) / (torch.sum(src * src) + 1e-8)
            
            print(f"  预测深度中位数：{torch.median(src):.3f} m")
            print(f"  GT深度中位数：{torch.median(tgt):.3f} m")
            print(f"  预测深度平均值：{src.mean():.3f} m")
            print(f"  GT深度平均值：{tgt.mean():.3f} m")
            print(f"  → 中位数对齐 scale：{scale_median:.6f}")
            print(f"  → 均值对齐 scale：{scale_mean:.6f}")
            print(f"  → L2对齐 scale：{scale_l2:.6f}")
            
            if frame_scales:
                frame_scales_vals = [s for _, s in frame_scales]
                print(f"\n  逐帧 scale 分析：")
                print(f"    最小：{min(frame_scales_vals):.6f}")
                print(f"    最大：{max(frame_scales_vals):.6f}")
                print(f"    平均：{np.mean(frame_scales_vals):.6f}")
                print(f"    标准差：{np.std(frame_scales_vals):.6f}")
                if np.std(frame_scales_vals) > 0.1:
                    print(f"    ⚠️  警告：帧间 scale 差异很大（std={np.std(frame_scales_vals):.4f}）")
                    print(f"       可能需要使用 --alignment_scope per_frame")
            
            if scale_median < 0.5 or scale_median > 2.0:
                print(f"  ⚠️  Scale 偏离 1.0 很远 ({scale_median:.4f})，预测深度尺度严重偏离！")
            elif scale_median < 0.9 or scale_median > 1.1:
                print(f"  ⚠️  Scale 有一定偏离 ({scale_median:.4f})，可能需要优化训练")
    
    print("\n" + "="*70)
    return frame_reports


def suggest_depth_metrics(pred_depths, gt_depths):
    """
    为稀疏激光雷达深度推荐指标
    """
    print("\n【推荐的指标组合】")
    print("""
    对于稀疏激光雷达深度，建议使用以下组合：
    
    1. **绝对误差 (MAE)**：最直观，单位是米
       - 好处：与物理意义直接对应
       - 坏处：对深度尺度变化敏感
    
    2. **相对误差 (Abs Rel)**：标准 benchmark 指标，但对稀疏数据要谨慎
       - 好处：尺度无关
       - 坏处：深度小时误差被放大
    
    3. **RMSE (Root Mean Square Error)**：考虑大误差的影响
       - MAE < RMSE < 最大误差（平方项压低大值）
    
    4. **accuracy metrics (threshold-based)**：
       - δ < 1.25：pred 在 [0.8*gt, 1.25*gt] 范围内的比例
       - δ < 1.25^2, δ < 1.25^3：提高阈值
    
    5. **稀疏性指标**：
       - 有效像素占比（％）
       - 预测-GT重叠率（％）
    """)


if __name__ == "__main__":
    print("这是一个诊断模块，应在 example_3dgs_1.py 中集成使用")
    print("使用方法：在主程序中导入并调用 analyze_sparse_depth(pred_depths_cpu, gt_depths_cpu)")
