# 多尺度高斯融合 - 快速启动和检查清单

## 快速检查清单（开始训练前）

### ✓ 代码检查（1分钟）

```bash
# 1. 语法检查
python -m py_compile pi3/models/pi3_3dgs.py
python -m py_compile pi3/models/loss_3dgs.py
echo "✓ 语法通过"

# 2. 导入测试
python -c "from pi3.models.pi3_3dgs import fuse_multiscale_gaussians; print('✓ 导入成功')"

# 3. 关键函数存在性检查
grep -n "def fuse_multiscale_gaussians" pi3/models/pi3_3dgs.py
grep -n "self.num_scales" pi3/models/pi3_3dgs.py
grep -n "lambda_consistency" pi3/models/loss_3dgs.py
```

### ✓ 配置检查（2分钟）

检查 `configs/model/pi3_3dgs.yaml` 包含：
```yaml
num_scales: 3                    # ✓ 必须有
scale_factors: [1.0, 0.5, 0.2]  # ✓ 必须有
clustering_chunk_size: 50000    # ✓ 必须有
enable_multiscale: true         # ✓ 必须有
```

检查 `configs/train/train_pi3_highres.yaml` 包含：
```yaml
loss:
  lambda_consistency: 0.05       # ✓ 必须有
  lambda_scale_diversity: 0.02   # ✓ 必须有
```

### ✓ 权重检查（1分钟）

```bash
# 确保有旧模型权重（可选，但推荐）
ls -lh outputs/pi3_highres_0402/ckpts/best_model/model.safetensors

# 如果没有，将使用随机初始化（较慢收敛）
```

---

## 训练启动命令

### 方案 1：从现有权重增量训练（推荐）

```bash
python train.py \
  config=configs/train/train_pi3_highres.yaml \
  model.ckpt="outputs/pi3_highres_0402/ckpts/best_model/model.safetensors" \
  model.num_scales=3 \
  model.scale_factors=[1.0,0.5,0.2] \
  loss.lambda_consistency=0.05 \
  loss.lambda_scale_diversity=0.02 \
  train.optimizer.lr=5e-5 \
  train.num_gpus=1 \
  train.batch_size=1
```

**预期**：
- 第一个 step：初始化新的 3 个 GSHead（~30秒）
- 第2-10 step：前向+反向通过（~2分钟）
- 显存占用：~40GB（A6000 48GB 可安全运行）

### 方案 2：从头训练（如无旧权重）

```bash
python train.py \
  config=configs/train/train_pi3_highres.yaml \
  model.ckpt=null \
  model.num_scales=3 \
  model.scale_factors=[1.0,0.5,0.2] \
  loss.lambda_consistency=0.05 \
  loss.lambda_scale_diversity=0.02 \
  train.optimizer.lr=1e-4 \
  train.num_gpus=1 \
  train.batch_size=1
```

**注意**：学习率应提升至 1e-4（收敛较慢）

### 方案 3：快速验证（单 step）

```bash
# 不更新权重，仅验证前向过程
python -c "
import torch
from pi3.models.pi3_3dgs import Pi3_3DGS

model = Pi3_3DGS(num_scales=3, ckpt=None)
model.eval()

# 虚拟输入
imgs = torch.randn(1, 10, 3, 536, 1008)

with torch.no_grad():
    output = model(imgs)

print(f\"✓ Forward OK\")
print(f\"  输出高斯数：{output['gaussians']['xyz'].shape[1]}\")
print(f\"  压缩率：{output['clustering_log']['cluster_compression_ratio_mean']:.2f}x\")
"
```

---

## 监控指标

### TensorBoard 关键指标

启动 tensorboard：
```bash
tensorboard --logdir=outputs/
```

在 tensorboard 中关注：

#### Loss 相关
```
train/loss_rgb              # RGB 重建损失
train/loss_ssim             # SSIM 损失
train/loss_depth            # 深度损失
train/loss_consistency      # 【新】融合一致性
train/loss_scale_diversity  # 【新】尺度多样性
train/total_loss            # 总损失
```

**正常范围**：
- loss_consistency：从 0.5 逐渐下降至 0.05（表示融合逐渐紧凑）
- loss_scale_diversity：波动在 0.001-0.1（表示多尺度被充分利用）

#### Clustering 相关
```
clustering/cluster_dense_k_before       # 融合前高斯数（应 > 100k）
clustering/cluster_fused_k_mean         # 融合后高斯数（应 < 500）
clustering/cluster_compression_ratio    # 压缩率（应 > 1.5）
clustering/cluster_num_scales           # 应为 3.0
clustering/cluster_chunk_size           # 应为 50000
clustering/cluster_enable_multiscale    # 应为 1.0
```

**异常情况**：
- 压缩率 ≈ 1.0：聚类未生效，检查半径计算
- cluster_num_scales ≠ 3：配置错误
- fused_k_mean > 1000：聚类半径过大，需要减小

---

## 故障排查

### 问题 1：OOM (显存溢出)

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**（从易到难）：

1. **降低 chunk_size**（最快）
   ```bash
   model.clustering_chunk_size=30000  # 从 50000 改为 30000
   ```
   预期显存降低 ~40%

2. **降低 batch_size**
   ```bash
   train.batch_size=1  # 改为更小值（如 0.5, 但通常整数）
   ```

3. **启用梯度检查点**
   ```python
   # 在 trainer.py 中
   model.gradient_checkpointing = True
   ```

4. **使用更小的图像分辨率**（最后手段）
   ```yaml
   # 在 config 中
   image_resolution: [256, 512]  # 默认 [536, 1008]
   ```

### 问题 2：loss_consistency 不下降

**症状**：
```
loss_consistency: 0.45 → 0.44 → 0.43  (下降缓慢)
```

**原因**：lambda_consistency 权重过小

**解决**：

```bash
loss.lambda_consistency=0.1  # 从 0.05 改为 0.1
```

预期：loss_consistency 会更快下降，总损失可能略微增加（同时 RGB 损失应下降）

### 问题 3：cluster_compression_ratio ≈ 1.0

**症状**：融合前后高斯数几乎相同
```
cluster_dense_k_before: 500000
cluster_fused_k_mean: 480000
cluster_compression_ratio: 1.04
```

**原因**：聚类半径计算过小

**诊断**：

```python
# 在模型的 forward 中打印调试信息
from pi3.models.pi3_3dgs import compute_adaptive_clustering_radii
xyz = ...  # 高斯位置
r_coarse, r_fine = compute_adaptive_clustering_radii(xyz)
print(f"radius_coarse: {r_coarse.item():.6f}")
print(f"radius_fine: {r_fine.item():.6f}")
```

**解决**：手动指定 clustering_radius

```bash
model.clustering_radius=0.05  # 单位：米，根据场景调整
```

### 问题 4：数值不稳定（NaN/Inf）

**症状**：
```
RuntimeError: ... contains NaN values
```

**常见原因**：

1. **优化器学习率过高**
   ```bash
   train.optimizer.lr=1e-5  # 降低 10 倍
   ```

2. **融合后的尺度过小**
   - 检查 scale_factors 是否合理
   - 确保 `scale` 经过 exp() 变换

3. **置信度异常**
   - 检查 conf_logits 的值域

**调试**：

```python
# 在损失计算前添加检查
assert not torch.isnan(loss_rgb), "RGB loss 包含 NaN"
assert not torch.isnan(loss_consistency), "Consistency loss 包含 NaN"
```

---

## 性能优化建议

### 1. 微调损失权重（推荐）

基线：
```python
lambda_consistency = 0.05
lambda_scale_diversity = 0.02
```

**如果 SSIM 改进不足**（< 1%）：
```python
lambda_consistency = 0.10  # ↑ 加强融合约束
lambda_scale_diversity = 0.01  # ↓ 减弱多样性约束（鼓励融合）
```

**如果细节丢失**（高纹理区域模糊）：
```python
lambda_consistency = 0.03  # ↓ 放松融合约束
lambda_scale_diversity = 0.05  # ↑ 加强多样性约束（保留细尺度）
```

### 2. 调整尺度因子（高级）

默认：
```python
scale_factors = [1.0, 0.5, 0.2]
```

**为了更多极端尺度对比**：
```python
scale_factors = [1.0, 0.4, 0.1]  # 尺度间隔更大
```

**为了更平滑的尺度过渡**：
```python
scale_factors = [1.0, 0.6, 0.35]  # 尺度间隔更小
```

### 3. 增加尺度数量（进阶）

```python
num_scales = 4
scale_factors = [1.0, 0.5, 0.25, 0.1]
```

**优点**：更细致的多频率表示
**代价**：模型参数 +33%，训练时间 +20%

---

## 验证训练正确性

### Checkpoint 1：基础验证（第 1 step）

期望输出：
```
Step 1:
  loss_rgb: 0.XX      # 初始值，应为 0.1-0.5 范围
  loss_ssim: 0.XX     # 初始值，应为 0.3-0.7 范围
  loss_consistency: 0.XX  # 初始值，应为 0.1-1.0 范围
  total_loss: 0.XX    # 综合值

  cluster_num_scales: 3.0  # ✓ 必须为 3
  cluster_compression_ratio: > 1.2  # ✓ 压缩率应 > 1
  cluster_fused_k_mean: < 200k  # ✓ 融合后数量合理
```

### Checkpoint 2：学习信号（第 10-50 steps）

期望行为：
```
Step 10:  loss ≈ 0.95 × initial_loss
Step 20:  loss ≈ 0.90 × initial_loss
Step 50:  loss ≈ 0.80 × initial_loss
```

如果 loss 不下降或上升，说明学习率设置有问题。

### Checkpoint 3：长期训练（1000+ steps）

期望指标：
```
SSIM (测试集): 0.55 → 0.60+ （+5% 改进）
LPIPS (测试集): 0.25 → 0.23 （-8% 改进）
高斯数: 600k → 150-300k （50-75% 压缩）
```

---

## 关键概念速记

### 尺度因子应用

```python
# 记住这个公式
scale_raw_output = model.gs_head(...)  # [B, N, H, W, 3]
scale_actual = exp(scale_raw_output) × 0.02 × scale_factors[scale_idx]
#                                                  ↑ 关键乘法
```

### 融合距离公式

```
d_fuse = sqrt((x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2)
       + |scale_tag_i - scale_tag_j| × r_coarse × 0.5

同尺度 (tag_i == tag_j):
  d_fuse = d_euclid + 0  = d_euclid

跨尺度 (tag_i ≠ tag_j):
  d_fuse = d_euclid + r_coarse × 0.5  (强大排斥）
```

### 聚类判断

```python
if d_fuse < r_fine:
    # 融合：归入同一 cluster
    cluster_id[i] = cluster_id[j]
else:
    # 不融合：保持分离
    pass
```

---

## 常见问题 (FAQ)

**Q: 为什么我的图像分辨率很小，还要设置 num_scales=3？**

A: 多尺度针对的是高斯本身的大小（空间范围），不是图像分辨率。即使图像很小，高斯仍然有大有小。

**Q: 能用 num_scales=2 吗？**

A: 可以。改为：
```yaml
num_scales: 2
scale_factors: [1.0, 0.2]
```
会减少 33% 的参数，但多样性约束变弱。

**Q: 显存原来只用 30GB，现在增加到 38GB 了，还能继续训练吗？**

A: 可以。30-40GB 在 A6000 (48GB) 上是安全的。如果接近 45GB，考虑跳过关键 Q9。

**Q: 如何知道融合是否有效？**

A: 看这三个指标：
1. cluster_compression_ratio > 1.5
2. cluster_fused_k_mean < 0.1 × cluster_dense_k_before
3. loss_consistency > 0（说明在最小化融合内的不一致）

**Q: 旧模型是否被完全覆盖？**

A: 否。新模型 strict=False 加载，意味着：
- 旧的 Encoder/Decoder/其他 Head 权重保留
- 旧的单个 GSHead 权重被忽略（因为现在有 3 个）
- 新的 3 个 GSHead 随机初始化

**Q: 能否在推理时禁用多尺度？**

A: 可以。推理时设置：
```python
model.num_scales = 1  # fallback to first scale only
```
但不推荐，因为融合的高斯本身就是多尺度的最优结果。

---

## 总结

| 任务 | 时间 | 命令 |
|------|------|------|
| 验证代码 | 1 分钟 | `python -m py_compile ...` |
| 验证配置 | 1 分钟 | 检查 yaml 文件 |
| 快速测试 | 2 分钟 | `python verify_multiscale_integration.py` |
| 启动训练 | 30 秒 | `python train.py ...` |
| 监控训练 | 持续 | tensorboard |

**开始训练**：
```bash
# 推荐命令（已包含所有参数）
python train.py \
  config=configs/train/train_pi3_highres.yaml \
  model.ckpt="$(ls -td outputs/pi3_highres_*/ckpts/best_model/model.safetensors | head -1)" \
  model.num_scales=3 \
  loss.lambda_consistency=0.05 \
  loss.lambda_scale_diversity=0.02 \
  train.optimizer.lr=5e-5
```

预期：**第一个 epoch 内，SSIM 应有 2-3% 的改进。**

