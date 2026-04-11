# 多尺度高斯融合完整指南

## 目录
1. [核心概念](#核心概念)
2. [融合流程详解](#融合流程详解)
3. [融合判断机制](#融合判断机制)
4. [显存分析](#显存分析)
5. [集成检查清单](#集成检查清单)

---

## 核心概念

### 多尺度高斯框架

```
输入图像序列 (N视角, H×W分辨率)
    ↓
共享 Encoder + 共享 Decoder
    ↓
3个独立高斯头（粗、中、细）
    ↓
3组高斯集合（尺度因子：1.0, 0.5, 0.2）
    ↓
【核心】尺度感知聚类融合
    ↓
融合后的高斯集合（冗余消除，细节保留）
```

### 问题定义

| 问题 | 原因 | 症状 |
|------|------|------|
| **Overlap 冗余** | 不同视角拍摄同一区域 | 相同位置多份高斯，计算浪费 |
| **前进视角冗余** | 相邻帧视角变化小 | 相邻帧的高斯重叠 |
| **环绕视角冗余** | 360度多角度拍摄 | 同位置不同视角的高斯重复 |
| **TopK欠覆盖** | 单尺度高斯难以表达多频率 | 细节与大结构不能兼得 |

### 解决方案

```
多尺度融合 = 多尺度生成 + 尺度感知聚类 + 自适应融合
```

---

## 融合流程详解

### 流程概览（完整尺度）

```
【第0步】输入准备
  输入: [B, N_total, C, H, W] 图像序列
       B = batch size
       N_total = 总视角数（通常10-16）
       C = RGB通道数
       H, W = 图像分辨率(536×1008 → 182×336 patch化)

【第1步】编码阶段
  encoder(images) → hidden [B*N_total, hw, 1024]
  其中 hw = (182/14) × (336/14) = 13 × 24 = 312

  >>> 代码位置：pi3_3dgs.py line 1055
  >>> hidden = self.encoder(imgs_flat, is_training=True)

【第2步】解码阶段（共享）
  decoder(hidden) → hidden_post [B*N_total, hw, 2048]
  （最后两层输出拼接）

  >>> 代码位置：pi3_3dgs.py line 1060
  >>> hidden, pos = self.decode(hidden, N_total, H, W, mem_debug=mem)

【第3步】多尺度高斯生成（关键！）
  FOR scale_idx IN [0, 1, 2]:
    # 选择第 scale_idx 个decoder和head
    h = gs_decoders_multiscale[scale_idx](hidden_sub, xpos=pos_sub)
    # shape: [B*N_sub, hw, 1024]

    attrs = gs_heads_multiscale[scale_idx]([h], (H, W))
    # shape: [B, N_sub, H, W, 11]
    # 内容: [4维旋转, 3维log_scale, 1维log_opacity, 3维颜色]

    gs_attrs_list.append(attrs)

  >>> 代码位置：pi3_3dgs.py lines 1089-1092

【第4步】属性提取与全局坐标变换
  FOR scale_idx IN [0, 1, 2]:
    # 从 attrs[scale_idx] 提取各属性
    rot_scale = normalize(attrs[..., 0:4])     # [B, N_sub, H, W, 4]
    scale_raw = exp(clamp(attrs[..., 4:7]))   # [B, N_sub, H, W, 3]

    # 【关键】应用尺度因子
    scale_scale = scale_raw * 0.02 * scale_factors[scale_idx]
    # 例如尺度0：scale_scale = scale_raw * 0.02 * 1.0
    #      尺度1：scale_scale = scale_raw * 0.02 * 0.5
    #      尺度2：scale_scale = scale_raw * 0.02 * 0.2

    opacity = sigmoid(attrs[..., 7:8])         # [B, N_sub, H, W, 1]
    color = sigmoid(attrs[..., 8:11])          # [B, N_sub, H, W, 3]

    # 变换到全局坐标系
    cam_rot_quat = matrix_to_quaternion(R_cam_sub)  # [B, N_sub, 1, 1, 4]
    global_rot = normalize(quat_mult(cam_rot_quat, rot_scale))

    # 位置变换：P_global = R @ P_local + t
    local_pts_flat = local_pts_sub.reshape(B, N_sub, -1, 3).transpose(-1, -2)
    global_pts = (R_cam_sub @ local_pts_flat + t_cam_sub).transpose(-1, -2)

    # 展平为 [B, K_s, 3] 格式（K_s = N_sub × H × W）
    xyz_scale = global_pts.reshape(B, -1, 3)
    scale_scale_flat = scale_scale.reshape(B, -1, 3)
    rot_scale_flat = global_rot.reshape(B, -1, 4)
    opacity_flat = opacity.reshape(B, -1, 1)
    color_flat = color.reshape(B, -1, 3)

    # 添加到字典列表
    gaussians_dict_list.append({
      'xyz': xyz_scale,            # [B, K_s, 3]
      'rotation': rot_scale_flat,  # [B, K_s, 4]
      'scale': scale_scale_flat,   # [B, K_s, 3]
      'opacity': opacity_flat,     # [B, K_s, 1]
      'color': color_flat,         # [B, K_s, 3]
    })

  >>> 代码位置：pi3_3dgs.py lines 1247-1287

【第5步】呼叫融合函数（多尺度聚类）
  outputs = fuse_multiscale_gaussians(
    gaussians_dict_list,           # 3个尺度的高斯字典
    conf_logits.reshape(B, -1, 1), # [B, K_total, 1]置信度
    scale_factors=[1.0, 0.5, 0.2],
    clustering_radius=None,        # 自适应计算
    enable_multiscale=True,
    chunk_size=50000,              # 显存限制
    return_stats=True
  )

  return: (fused_xyz, fused_rot, fused_scale, fused_opacity,
           fused_color, fused_conf, clustering_stats)

  >>> 代码位置：pi3_3dgs.py lines 1290-1299

【第6步】渲染与损失计算
  # 融合后仅保留 K' ≈ 200-400 个高斯
  gaussians_render = {
    'xyz': fused_xyz,              # [B, K', 3]
    'rotation': fused_rot,         # [B, K', 4]
    'scale': fused_scale,          # [B, K', 3]
    'opacity': fused_opacity,      # [B, K', 1]
    'color': fused_color,          # [B, K', 3]
  }

  render_output = rasterization(gaussians_render, ...)

  loss = λ_rgb·L_rgb + λ_ssim·L_ssim + λ_depth·L_depth
       + λ_consistency·L_consistency + λ_diversity·L_diversity
```

---

## 融合判断机制

### 核心判断：两个高斯是否应该融合？

#### 算法 1：基础距离判断（全密度情况）

```python
# 代码位置：pi3_3dgs.py lines 153-160

# 【输入】
xyz_b: [K, 3]  # 所有高斯的位置
scale_tags_b: [K]  # 尺度标签 [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2]
radius_fine: float  # 融合半径，自适应计算 = scene_diameter × 0.05

# 【步骤 1】计算欧氏距离
dist_mat = torch.cdist(xyz_b, xyz_b, p=2.0)  # [K, K]
# dist_mat[i, j] = ||xyz_b[i] - xyz_b[j]||₂

# 【步骤 2】计算尺度差异
scale_diff = (scale_tags_b.unsqueeze(1) - scale_tags_b.unsqueeze(0)).abs().float()
# scale_diff[i, j] ∈ {0, 1, 2}
# 值的含义：
#   0 = 同尺度（都是粗 or 都是中 or 都是细）
#   1 = 相邻尺度（粗与中，或中与细）
#   2 = 最远尺度（粗与细）

# 【步骤 3】计算尺度惩罚（排斥势）
radius_coarse = scene_diameter × 0.2  # 约 0.2m
scale_penalty = scale_diff * radius_coarse * 0.5
# 例如 scale_diff=2，radius_coarse=0.2：
#   scale_penalty = 2 × 0.2 × 0.5 = 0.2m

# 【步骤 4】融合距离
dist_adjusted = dist_mat + scale_penalty
# 【关键公式】
# d_fuse(i, j) = d_euclid(i, j) + λ_s × I(s_i ≠ s_j)
# 中文：融合距离 = 欧氏距离 + 尺度惩罚

# 【步骤 5】邻接判断
adjacency = (dist_adjusted < radius_fine)
# 如果融合距离 < 半径 → 相邻 → 应该融合

# 【步骤 6】聚类（BFS连通分量)
cluster_ids = _bfs_clustering(adjacency)
# 在同一连通分量中的高斯被分到同一个cluster
```

**判断流程图**：

```
两个高斯 G_i 和 G_j
  ↓
计算欧氏距离 d_ij = ||xyz_i - xyz_j||₂
  ↓
获取尺度标签 s_i, s_j ∈ {0, 1, 2}
  ↓
计算尺度差异 Δs = |s_i - s_j|
  ↓
计算尺度惩罚 p = Δs × r_coarse × 0.5
  ↓
融合距离 d_fuse = d_ij + p
  ↓
判断：d_fuse < r_fine?
  ↓
─────────────────────
│                   │
YES               NO
│                 │
融合            不融合
│                 │
─────────────────────
```

**具体例子**：

```
例子 1：同尺度高斯（都是细，scale_tag=2）
  d_ij = 0.08m
  Δs = |2 - 2| = 0
  p = 0 × 0.2 × 0.5 = 0
  d_fuse = 0.08 + 0 = 0.08m
  r_fine ≈ 0.1m
  判断：0.08 < 0.1? YES → 融合！✓

例子 2：相邻尺度高斯（一个粗，一个中，scale_tag=0,1）
  d_ij = 0.08m （很近）
  Δs = |0 - 1| = 1
  p = 1 × 0.2 × 0.5 = 0.1m
  d_fuse = 0.08 + 0.1 = 0.18m
  r_fine ≈ 0.1m
  判断：0.18 < 0.1? NO → 不融合！✓
  （即使它们物理很近，不同尺度也被保持）

例子 3：最远尺度高斯（粗与细，scale_tag=0,2）
  d_ij = 0.05m （很近）
  Δs = |0 - 2| = 2
  p = 2 × 0.2 × 0.5 = 0.2m
  d_fuse = 0.05 + 0.2 = 0.25m
  r_fine ≈ 0.1m
  判断：0.25 < 0.1? NO → 不融合！✓
  （强制保留多尺度多样性）
```

#### 算法 2：分块聚类（高密度情况）

当 K > chunk_size (50000) 时，使用 `_chunked_clustering_scale_aware`

```python
# 代码位置：pi3_3dgs.py lines 280-344

# 【PHASE 1】分块内聚类

num_chunks = ceil(K / chunk_size)  # 例如 K=600k，chunk_size=50k → 12个chunks

FOR chunk_idx IN [0, num_chunks):
  # 提取第 chunk_idx 个chunk
  start = chunk_idx × chunk_size
  end = min(start + chunk_size, K)
  xyz_chunk = xyz_b[start:end]       # [~50k, 3]
  scale_tags_chunk = scale_tags_b[start:end]  # [~50k]

  # 在单个chunk内应用【算法 1】的完整流程
  dist_chunk = cdist(xyz_chunk, xyz_chunk)   # [50k, 50k]
  scale_diff_chunk = |scale_tags_chunk[i] - scale_tags_chunk[j]|
  scale_penalty_chunk = scale_diff_chunk × r_coarse × 0.5
  dist_adjusted_chunk = dist_chunk + scale_penalty_chunk

  adjacency_chunk = (dist_adjusted_chunk < radius_fine)
  cluster_ids_local = _bfs_clustering(adjacency_chunk)  # [~50k]

  # 全局化cluster ID（避免多个chunks之间的ID冲突）
  IF chunk_idx > 0:
    cluster_ids_local += current_max_cluster_id

  all_cluster_ids.append(cluster_ids_local)
```

**PHASE 1 的关键点**：
- 每个chunk内部的聚类是完整的（遵循【算法 1】）
- 但不同chunks之间的高斯无法互相看到
- 可能导致本该融合的跨chunk高斯被误分

```
╔═══════════════════════════════════════════════════════════════╗
║      Chunk 0                         Chunk 1                 ║
║  ┌──────────────────┐            ┌──────────────────┐        ║
║  │   G₁ (x=5.0)     │            │   G₂ (x=5.15)    │        ║
║  │   cluster_id=0   │  boundary  │   cluster_id=1   │        ║
║  │   gap: 0.15m     │←――――――――――→│   等待Phase 2    │        ║
║  └──────────────────┘            └──────────────────┘        ║
╚═══════════════════════════════════════════════════════════════╝

PHASE 1 结果：【错误】两个本应融合的同尺度高斯被分到了不同cluster
```

#### 算法 3：边界修复（跨chunk聚类）

PHASE 2 修复 PHASE 1 的缺陷

```python
# 代码位置：pi3_3dgs.py lines 310-343

# 【PHASE 2】Cross-Chunk Boundary Handling

IF num_chunks > 1:
  FOR chunk_idx IN [0, num_chunks-1):
    # 定义两个chunk的交界区域
    start_curr = chunk_idx × chunk_size
    end_curr = (chunk_idx + 1) × chunk_size
    start_next = end_curr
    end_next = min(start_next + chunk_size, K)

    # 【关键】只检查边界附近的1000个点，不是全部
    xyz_curr_boundary = xyz_b[max(start_curr, end_curr - 1000):end_curr]
    xyz_next_boundary = xyz_b[start_next:min(start_next + 1000, end_next)]
    # shape: 每个 ≤ [1000, 3]

    scale_tags_curr = scale_tags_b[...]  # 对应的尺度标签
    scale_tags_next = scale_tags_b[...]

    # 跨chunk的距离矩阵（极低显存！）
    dist_boundary = cdist(xyz_curr_boundary, xyz_next_boundary)
    # shape: [≤1000, ≤1000] × 4B = ≤4MB

    # 【CRUCIAL】只考虑同尺度的跨chunk融合！
    same_scale = (scale_tags_curr[i] == scale_tags_next[j]) for all i, j
    # same_scale[i, j] = True if s_i == s_j, False otherwise

    boundary_adjacency = (dist_boundary < radius_fine) & same_scale
    # 【关键判断】
    # 两个条件都要满足：
    #   1. 物理距离 < 融合半径
    #   2. 尺度标签相同（防止跨尺度融合）

    # 找出需要合并的cluster对
    pairs = boundary_adjacency.nonzero(as_tuple=True)

    FOR (curr_idx, next_idx) IN pairs:
      # 映射回全局索引
      curr_global_idx = max(start_curr, end_curr - 1000) + curr_idx
      next_global_idx = start_next + next_idx

      # 获取它们当前的cluster ID
      curr_cluster_id = all_cluster_ids[chunk_idx][curr_global_idx - start_curr]
      next_cluster_id = all_cluster_ids[chunk_idx + 1][next_global_idx - start_next]

      # 【并查集操作】如果尚未合并，执行合并
      IF curr_cluster_id != next_cluster_id:
        old_id = next_cluster_id
        new_id = curr_cluster_id

        # 重新标记 chunk_idx+1 中所有属于 old_id 的点
        all_cluster_ids[chunk_idx+1][all_cluster_ids[chunk_idx+1] == old_id] = new_id
```

**PHASE 2 的判断机制**：

```
跨边界的两个高斯 G_curr (chunk 0) 和 G_next (chunk 1)
  ↓
计算欧氏距离 d_ij = ||xyz_curr - xyz_next||₂
  ↓
获取尺度标签 s_curr, s_next
  ↓
判断 1：d_ij < r_fine?
  └─→ YES：进行判断 2
  └─→ NO：放弃（太远）
      ↓
判断 2：s_curr == s_next?（同尺度？）
  └─→ YES：融合！合并它们的cluster
  └─→ NO：放弃（跨尺度，保持分离）
      ↓
【结果】只有同尺度且相近的跨chunk高斯被融合
```

**具体例子**：

```
设置：chunk边界处有4个高斯
  G_0 = {xyz: (5.0, 6.0, 7.0), scale_tag: 0（粗）}，chunk 0末尾
  G_1 = {xyz: (5.08, 6.0, 7.0), scale_tag: 0（粗）}，chunk 1开头
  G_2 = {xyz: (5.12, 6.0, 7.0), scale_tag: 2（细）}，chunk 1开头
  G_3 = {xyz: (4.98, 6.0, 7.0), scale_tag: 1（中）}，chunk 0末尾

PHASE 1 后（chunk内聚类）：
  Chunk 0 clustering:
    d(G_0, G_3) = 0.02m < r_fine → cluster_0
    cluster_ids = [... 0, 1, ...]  (G_0=cluster 0, G_3=cluster 1)

  Chunk 1 clustering:
    d(G_1, G_2) = 0.04m < r_fine → cluster_2 (新开辟)
    cluster_ids = [... 2, 2, ...]  (G_1, G_2 都是cluster 2)

PHASE 2 边界检查：
  检查对 1：(G_0, G_1)
    d_ij = ||（5.0,6.0,7.0）-(5.08,6.0,7.0)|| = 0.08m < r_fine=0.1m ✓
    same_scale? 0 == 0? Yes ✓
    → merge(cluster_0, cluster_2) ✓
    结果：G_0, G_1, G_2 都被标记为同一cluster（但这里有误，因为G_2是细尺度）

  【等等，这有问题】让我重新梳理：

  PHASE 1 内：
  Chunk 1中 G_1(粗, tag=0) 和 G_2(细, tag=2) 的聚类判断：
    d_ij = 0.04m
    Δs = |0 - 2| = 2
    p = 2 × 0.2 × 0.5 = 0.2m
    d_fuse = 0.04 + 0.2 = 0.24m
    0.24 > 0.1? YES → 不聚类！
    → 它们在Chunk1中是不同的cluster

  PHASE 2：
  检查对 1：(G_0, G_1)
    度量：仅物理距离 0.08m
    同尺度：0 == 0? Yes
    → merge ✓

  检查对 2：(G_0, G_2)
    度量：物理距离 0.12m
    同尺度：0 == 2? No
    → 不merge ✓

  【正确】G_0 和 G_1 融合，G_2 保持独立
```

### 融合判断的完整决策树

```
INPUT: 两个高斯 G_i 和 G_j

┌─────────────────────────────────────────────────────┐
│  计算欧氏距离 d = ||xyz_i - xyz_j||₂               │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────v────────┐
        │  d < r_fine?    │
        └────────┬────────┘
         YES     │     NO
         │       │      │
    ┌────v───┐   │  ┌──v──────┐
    │   继续  │   │  │ 不融合   │
    └────┬───┘   │  └─────────┘
         │       │
  ┌──────v──────────────────┐
  │ 获取尺度标签 s_i, s_j    │
  └──────┬──────────────────┘
         │
    ┌────v────────┐
    │ s_i == s_j? │
    └────┬────────┘
  YES    │    NO
  │      │     │
  │   ┌──v──────┐
  │   │ 不融合   │ （保留多尺度多样性）
  │   └─────────┘
  │
┌─v──────────────────────┐
│ 【融合】合并cluster ID  │
│ 共同位置 = weighted avg │
│ 共同尺度 = max(同尺度)  │
│ 共同透明度 = multi-view │
└────────────────────────┘
```

---

## 显存分析

### 内存使用分解

```
【模型权重总量】～ 1.5 GB（固定）
  Encoder (DINOv2):        ～ 600 MB
  Decoder (36 blocks):     ～ 800 MB
  其他heads:               ～ 100 MB

【GS分支参数增长】
  原来：1个GSHead + 1个GSDecoder    = ～ 65 MB
  现在：3个GSHead + 3个GSDecoder   = ～ 195 MB
  增长：130 MB （+10% of A6000）

【Inference 时激活值】
  image input:             [1, 10, 3, 536, 1008] = 16 GB （显然超大）
  hidden features:         [B*N, hw, 1024] = 245 MB
  × 3 scale decoders:      = 735 MB （peak）

【聚类临时显存峰值】（【关键】）

  PHASE 1 分块：
    chunk_size = 50000
    距离矩阵 = [50k, 50k] × 4B = 10 GB
    × 3尺度集合 = 分别处理，peak 10GB

  PHASE 2 边界：
    boundary = [≤1000, ≤1000] × 4B = ≤ 4 MB

  总峰值 ～ 12 GB （聚类）

【示例】A6000 (48GB) 分配：
  ┌─────────────────────────────────────┐
  │ GPU Memory 48 GB                    │
  ├─────────────────────────────────────┤
  │ 模型权重         ～5GB              │
  │ 梯度缓冲         ～8GB              │
  │ 激活值           ～10GB             │
  │ 聚类-距离矩阵    ～12GB（峰值）     │
  │ 聚类-其他        ～2GB              │
  │ pymc/overhead    ～5GB              │
  ├─────────────────────────────────────┤
  │ 总计～42GB （安全范围内）           │
  └─────────────────────────────────────┘

【优化建议】
  如果接近OOM：
    1. 降低 batch_size
    2. 降低 clustering_chunk_size from 50k to 30k
    3. 使用 gradient_checkpointing
```

### 显存配置

```yaml
# configs/model/pi3_3dgs.yaml

clustering_chunk_size: 50000        # 50k for A6000 (48GB)
# 降至 30000 可减少峰值显存
# 提升至 80000 可加速但需更多显存

num_scales: 3                        # 必须=3
scale_factors: [1.0, 0.5, 0.2]      # 可调

loss:
  lambda_consistency: 0.05           # 融合内一致性
  lambda_scale_diversity: 0.02       # 尺度多样性约束
```

---

## 代码集成检查清单

### 文件：`pi3/models/pi3_3dgs.py`

- [ ] Line 33-57: `_bfs_clustering()` 函数已定义
  - 检查：返回cluster_ids

- [ ] Line 60-68: `compute_adaptive_clustering_radii()` 函数已定义
  - 检查：r_coarse, r_fine 计算正确

- [ ] Line 71-278: `fuse_multiscale_gaussians()` 函数已定义
  - [ ] Line 95-97: 初始化 device, B, num_scales
  - [ ] Line 103: conf_probs 计算
  - [ ] Line 105-137: 高斯聚合（所有尺度拼接）
  - [ ] Line 140-144: 自适应半径计算
  - [ ] Line 148-160: 尺度感知距离与聚类
  - [ ] Line 162-170: 聚类统计
  - [ ] Line 176-226: Intra-cluster融合
  - [ ] Line 246-271: 批次对齐

- [ ] Line 280-344: `_chunked_clustering_scale_aware()` 函数已定义
  - [ ] Line 288-308: PHASE 1 - 分块内聚类
  - [ ] Line 310-343: PHASE 2 - 边界融合

- [ ] Line 758-798: Pi3_3DGS.__init__()
  - [ ] Line 779-781: num_scales, scale_factors 参数
  - [ ] Line 920-927: 多尺度GSHead/Decoder ModuleList

- [ ] Line 1089-1095: 多尺度高斯生成循环
  - [ ] 检查：gs_attrs_list 包含3个张量

- [ ] Line 1247-1287: 高斯属性提取与全局变换
  - [ ] 检查：scale_factors[scale_idx] 应用

- [ ] Line 1290-1299: fuse_multiscale_gaussians() 调用
  - [ ] 检查：返回7个输出（包括clustering_stats）

### 文件：`pi3/models/loss_3dgs.py`

- [ ] Line 438-445: Pi3LossGS.__init__() 新参数
  - [ ] `lambda_consistency: 0.05`
  - [ ] `lambda_scale_diversity: 0.02`

- [ ] Line 575-616: `_compute_consistency_loss()` 函数
  - [ ] 检查：scale 和 rotation 一致性损失

- [ ] Line 618-657: `_compute_scale_diversity_loss()` 函数（可选）
  - [ ] 检查：尺度多样性约束

- [ ] Line 708: 损失初始化
  - [ ] loss_consistency, loss_scale_diversity 初始化

- [ ] Line 749-772: 损失组合
  - [ ] self.lambda_consistency 权重
  - [ ] self.lambda_scale_diversity 权重
  - [ ] clustering_log 集成

### 文件：`configs/model/pi3_3dgs.yaml`

- [ ] num_scales: 3
- [ ] scale_factors: [1.0, 0.5, 0.2]
- [ ] clustering_chunk_size: 50000 （或 30000）
- [ ] enable_multiscale: true

### 文件：`configs/train/train_pi3_highres.yaml`

- [ ] loss.lambda_consistency: 0.05
- [ ] loss.lambda_scale_diversity: 0.02

---

## 验证方法

### 1. 语法检查

```bash
python -m py_compile pi3/models/pi3_3dgs.py
python -m py_compile pi3/models/loss_3dgs.py
```

### 2. 导入测试

```bash
python -c "from pi3.models.pi3_3dgs import Pi3_3DGS, fuse_multiscale_gaussians; print('✓ Import OK')"
```

### 3. 前向传播测试（无梯度）

```python
import torch
from pi3.models.pi3_3dgs import Pi3_3DGS

model = Pi3_3DGS(num_scales=3, scale_factors=[1.0, 0.5, 0.2])
model.eval()

# 虚拟输入
imgs = torch.randn(1, 10, 3, 536, 1008)  # 1个batch, 10视角

with torch.no_grad():
    output = model(imgs)

assert 'gaussians' in output
assert 'clustering_log' in output
assert output['gaussians']['xyz'].shape[1] < 600000  # 融合后更少
print(f"✓ Forward OK - 高斯数：{output['gaussians']['xyz'].shape[1]}")
```

### 4. 融合验证

```python
# 检查融合统计
clustering_log = output['clustering_log']
print(f"原始高斯数：{clustering_log['cluster_dense_k_before']}")
print(f"融合后高斯数：{clustering_log['cluster_fused_k_mean']}")
print(f"压缩率：{clustering_log['cluster_compression_ratio_mean']:.1%}")
print(f"尺度数：{clustering_log['cluster_num_scales']}")

assert clustering_log['cluster_num_scales'] == 3, "Should have 3 scales"
assert clustering_log['cluster_compression_ratio_mean'] > 1.0, "Should compress"
```

### 5. 聚类判断验证

```python
from pi3.models.pi3_3dgs import fuse_multiscale_gaussians

# 构造测试案例
gaussians_test = [
  {  # 尺度 0（粗）
    'xyz': torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]).unsqueeze(0),
    'rotation': torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]).unsqueeze(0),
    'scale': torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]).unsqueeze(0),
    'opacity': torch.ones(1, 2, 1),
    'color': torch.ones(1, 2, 3),
  },
  {  # 尺度 1（中）
    'xyz': torch.tensor([[0.05, 0.0, 0.0]]).unsqueeze(0),
    'rotation': torch.tensor([[1.0, 0, 0, 0]]).unsqueeze(0),
    'scale': torch.tensor([[0.05, 0.05, 0.05]]).unsqueeze(0),
    'opacity': torch.ones(1, 1, 1),
    'color': torch.ones(1, 1, 3),
  },
]

conf = torch.zeros(1, 3, 1)

result = fuse_multiscale_gaussians(
  gaussians_test, conf, [1.0, 0.5],
  enable_multiscale=True, return_stats=True
)

fused_xyz, _, _, _, _, _, stats = result
print(f"聚类统计：{stats}")
# 预期：0.0和0.08（同尺度）应该融合，0.05（不同尺度）应保持分离
```

---

## 总结

| 步骤 | 关键操作 | 代码位置 | 融合决策 |
|------|---------|--------|--------|
| 生成 | 3×高斯头 | line 1089-1095 | N/A |
| 提取 | 属性→全局 | line 1247-1287 | N/A |
| 融合 | 尺度感知聚类 | line 71-278 | 【公式】d_fuse = d_euclid + λ_s×I(Δs) |
| 分块 | 低显存聚类 | line 280-344 | PHASE1: 块内，PHASE2: 同尺度跨界 |
| 损失 | 一致性+多样性 | loss_3dgs.py | λ_consistency, λ_diversity |
| 渲染 | 融合高斯集合 | line 1332-1340 | K' ≈ 200-400个 |

**核心判断**：
```
融合 = (d_euclid < r_fine) AND (s_i == s_j OR 相同块内)
     = (物理相近) AND (同尺度或同块内)
```

**显存预算** (A6000 48GB):
- 参数：+500MB
- 激活：+1GB
- 聚类峰值：12GB （可控）
- 总增长：～15% （安全）
