"""
可微的软局部竞争模块

核心设计：
1. 不使用离散哈希分组（不可微）
2. 基于高斯相似度计算连续的竞争门控
3. 梯度完全流回 opacity/xyz/scale
4. 同时处理多尺度冗余（粗高斯 vs 细高斯）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLocalCompetition(nn.Module):
    """
    可微的软竞争机制
    
    核心思想：
    - 同一竞争组内的高斯（位置+颜色相似）形成软竞争
    - 相似度高 → 竞争强 → opacity被压低
    - 但这个过程完全可微，梯度流回到生成网络
    
    参数说明：
    - position_radius: 位置相似度的衰减半径（高斯核sigma）
    - color_distance_threshold: 颜色距离阈值（超过则认为不同颜色）
    - temperature: 竞争软性参数（越低越硬）
    - target_power: 竞争强度目标幂次
    - strength: 整体竞争强度乘子
    - min_gate: 最低允许的门控值
    """
    
    def __init__(
        self,
        position_radius: float = 0.1,           # 相邻高斯的位置衰减距离
        color_distance_threshold: float = 0.1,  # 颜色相似度阈值
        temperature: float = 1.0,                # 竞争门控的温度参数
        target_power: float = 0.5,               # 根据竞争组大小调整强度
        strength: float = 1.0,                   # 整体竞争强度
        min_gate: float = 0.05,                  # 最小门控值
        use_scale_aware: bool = True,            # 是否使用scale感知的相似度
        optimization_mode: str = "full",         # 优化模式：full/topk/sampling/chunking
        optimization_param: float = 0.5,         # 优化参数：topk数(0-1)或采样率或chunk比例
    ):
        super().__init__()
        self.position_radius = float(position_radius)
        self.color_distance_threshold = float(color_distance_threshold)
        self.temperature = max(float(temperature), 1e-4)
        self.target_power = float(target_power)
        self.strength = float(strength)
        self.min_gate = float(min_gate)
        self.use_scale_aware = bool(use_scale_aware)
        
        # ===== 优化参数 =====
        self.optimization_mode = str(optimization_mode).lower()
        assert self.optimization_mode in ["full", "topk", "sampling", "chunking"], \
            f"optimization_mode must be one of: full/topk/sampling/chunking, got {self.optimization_mode}"
        self.optimization_param = float(optimization_param)
        
        # 根据mode解释optimization_param
        if self.optimization_mode == "topk":
            # optimization_param = k / K (比例)，或者直接是K
            self.topk_ratio = self.optimization_param
        elif self.optimization_mode == "sampling":
            # optimization_param = 采样率（0-1）
            self.sampling_rate = max(min(self.optimization_param, 1.0), 0.01)
        elif self.optimization_mode == "chunking":
            # optimization_param = chunk大小的比例（0-1）
            self.chunk_ratio = max(min(self.optimization_param, 1.0), 0.1)
    
    def _compute_position_similarity(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        计算高斯之间的位置相似度（高斯核）
        
        Input:  xyz [K, 3]
        Output: similarity [K, K]  (高斯值，exp(-dist^2 / sigma^2))
        """
        K = xyz.shape[0]
        
        # 计算两两距离 [K, K]
        diff = xyz.unsqueeze(1) - xyz.unsqueeze(0)  # [K, 1, 3] - [1, K, 3] = [K, K, 3]
        distances = torch.norm(diff, dim=-1)         # [K, K]
        
        # 高斯核：exp(-dist^2 / (2 * sigma^2))
        # sigma = position_radius
        sigma = self.position_radius + 1e-8
        similarity = torch.exp(-distances ** 2 / (2 * sigma ** 2))  # [K, K]
        
        return similarity
    
    def _compute_color_similarity(self, color: torch.Tensor) -> torch.Tensor:
        """
        计算高斯之间的颜色相似度（余弦相似度）
        
        Input:  color [K, 3]  范围 [0, 1]
        Output: mask [K, K]  bool，True表示颜色足够接近
        """
        K = color.shape[0]
        
        # 余弦相似度
        color_norm = F.normalize(color, dim=-1, p=2)  # [K, 3]
        cos_sim = torch.mm(color_norm, color_norm.t())  # [K, K]
        
        # 转换为距离：1 - cos_sim
        color_distance = 1.0 - cos_sim  # [K, K]，范围 [0, 2]
        
        # 如果颜色距离太大，不竞争
        color_mask = color_distance < self.color_distance_threshold  # [K, K]
        
        return color_mask
    
    def _compute_scale_aware_similarity(self, scale: torch.Tensor) -> torch.Tensor:
        """
        计算基于scale的相似度修正
        
        核心：scale差异大的高斯（早期粗 vs 后期细）也可以竞争
        
        Input:  scale [K, 3]
        Output: weight [K, K]  
        """
        K = scale.shape[0]
        
        # 计算平均scale
        scale_mean = scale.mean(dim=-1)  # [K]
        
        # 计算scale比率：log(s_i / s_j)
        scale_ratio = scale_mean.unsqueeze(1) / (scale_mean.unsqueeze(0) + 1e-8)  # [K, K]
        scale_ratio = torch.abs(torch.log(scale_ratio))  # [K, K]，差异度
        
        # 转换为竞争权重：差异越大，权重越小（但仍有竞争）
        # 使用softplus确保平滑：log(1 + exp(x))
        scale_weight = torch.exp(-scale_ratio / 2.0)  # [K, K]
        
        return scale_weight
    
    def _compute_view_mask(self, source_view: torch.Tensor) -> torch.Tensor:
        """
        计算多视角竞争掩码
        
        核心：只有来自不同视角的高斯才竞争
        
        Input:  source_view [K]  视角索引
        Output: mask [K, K]  bool，True表示来自不同视角
        """
        K = source_view.shape[0]
        
        # 广播为 [K, K]
        view_i = source_view.unsqueeze(1)  # [K, 1]
        view_j = source_view.unsqueeze(0)  # [1, K]
        
        # 不同视角：True；相同视角：False
        # 我们只竞争不同视角的高斯
        different_view = view_i != view_j  # [K, K]
        
        return different_view
    
    def forward(
        self,
        gaussian_dict: dict,
        source_view: torch.Tensor,
        scene_size: torch.Tensor = None,
    ):
        """
        前向传播：应用可微竞争门控
        
        Input:
            gaussian_dict: 包含 xyz, scale, color, opacity 等的字典
            source_view: [K] 每个高斯的源视角
            scene_size: 场景大小（用于归一化）
        
        Output:
            gaussian_dict_gated: 应用竞争后的高斯（opacity被调整）
            gate: [K] 竞争门控值（用于后续损失）
        """
        
        xyz = gaussian_dict["xyz"]
        scale = gaussian_dict["scale"]
        color = gaussian_dict.get("competition_color", gaussian_dict["color"])
        opacity = gaussian_dict["opacity"]  # [K, 1]
        
        K = xyz.shape[0]
        device, dtype = xyz.device, xyz.dtype
        
        # ===== 边界情况 =====
        if K <= 1:
            return gaussian_dict, torch.ones(K, 1, device=device, dtype=dtype), {}
        
        # ===== 第1步：计算相似度矩阵 =====
        with torch.enable_grad():  # 确保这些操作有梯度
            # 位置相似度（高斯核，可微）
            pos_sim = self._compute_position_similarity(xyz)  # [K, K]
            
            # 颜色相似度掩码（二值，不可微但不影响梯度流）
            color_mask = self._compute_color_similarity(color)  # [K, K]
            
            # 多视角掩码（二值）
            view_mask = self._compute_view_mask(source_view)  # [K, K]
            
            # 联合掩码：位置近 AND 颜色近 AND 不同视角
            base_mask = color_mask & view_mask  # [K, K]
            
            # 组合相似度：位置相似度 * 掩码
            competition_similarity = pos_sim * base_mask.float()  # [K, K]
            
            # 可选：加入scale感知
            if self.use_scale_aware:
                scale_weight = self._compute_scale_aware_similarity(scale)  # [K, K]
                competition_similarity = competition_similarity * scale_weight
        
        # ===== 第2步：计算竞争得分和门控 =====
        # 方式1：基于opacity的相对排名
        opacity_flat = opacity.squeeze(-1)  # [K]
        score = torch.logit(opacity_flat.clamp(1e-4, 1.0 - 1e-4))  # [K]
        
        # 对每个高斯，计算它与竞争者的相似加权得分
        # 高相似度竞争者会压低该高斯的权重
        weighted_score = torch.matmul(competition_similarity, score.unsqueeze(-1)).squeeze(-1)  # [K]
        
        # 计算竞争强度：基于某个高斯的竞争程度
        # 竞争程度 = 有多少个类似的竞争者
        num_competitors = competition_similarity.sum(dim=1)  # [K]
        
        # 调整竞争目标
        target_strength = num_competitors.pow(self.target_power).clamp(min=1.0)  # [K]
        
        # ===== 第3步：计算软竞争门 =====
        # 使用softmax风格的门控
        # 但基于相似度加权
        exp_score = torch.exp((score - score.max()) / self.temperature)  # [K]
        exp_score = exp_score * base_mask.float().sum(dim=1)  # 按竞争数加权
        
        group_sum = torch.zeros(K, device=device, dtype=dtype)
        for i in range(K):
            # 该高斯所在的竞争组
            competitors = base_mask[i].nonzero(as_tuple=True)[0]
            if len(competitors) > 0:
                group_sum[i] = exp_score[competitors].sum()
            else:
                group_sum[i] = exp_score[i]
        
        # 防止除零
        group_sum = group_sum.clamp(min=1e-6)
        
        # 软门控：该高斯的相对强度
        gate_soft = exp_score / group_sum  # [K]
        gate_soft = gate_soft * target_strength  # 按竞争强度调整
        gate_soft = gate_soft.clamp(min=self.min_gate, max=1.0)
        
        # ===== 第4步：应用整体强度调整 =====
        if self.strength < 1.0:
            gate_soft = (1.0 - self.strength) + self.strength * gate_soft
        
        # ===== 第5步：修改opacity（梯度完全流回） =====
        gate = gate_soft.unsqueeze(-1)  # [K, 1]
        
        gaussian_dict_gated = dict(gaussian_dict)
        gaussian_dict_gated["opacity"] = gaussian_dict["opacity"] * gate  # 梯度流回！
        gaussian_dict_gated["competition_gate"] = gate.squeeze(-1)  # 保存门控用于loss
        
        # ===== 统计信息 =====
        stats = {
            "competition_gate_mean": gate.mean().detach(),
            "competition_gate_min": gate.min().detach(),
            "competition_gate_max": gate.max().detach(),
            "num_competitors_mean": num_competitors.mean().detach(),
            "num_competitors_max": num_competitors.max().detach(),
            "active_competition_count": (gate.squeeze(-1) < 1.0).sum().float(),
        }
        
        return gaussian_dict_gated, gate, stats
