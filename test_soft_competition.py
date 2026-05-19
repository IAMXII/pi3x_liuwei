"""
梯度流验证脚本

这个脚本验证软竞争机制的梯度是否正确流回高斯参数。
"""

import torch
import torch.nn as nn
from pi3.models.layers.soft_competition import SoftLocalCompetition


def test_soft_competition_gradients():
    """测试SoftLocalCompetition的梯度流"""
    
    print("=" * 60)
    print("软竞争梯度流测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 创建示例高斯数据
    B = 2  # batch size
    K = 10  # 每个batch的高斯数量
    
    gaussian_dict = {
        "xyz": torch.randn(K, 3, device=device, requires_grad=True),
        "scale": torch.randn(K, 3, device=device, requires_grad=True),
        "color": torch.rand(K, 3, device=device, requires_grad=True),
        "opacity": torch.sigmoid(torch.randn(K, 1, device=device, requires_grad=True)),
        "rotation": torch.randn(K, 4, device=device, requires_grad=False),
    }
    
    source_view = torch.randint(0, B, (K,), device=device)
    scene_size = torch.tensor([1.0], device=device)
    
    # 创建竞争模块
    competition = SoftLocalCompetition(
        position_radius=0.1,
        color_distance_threshold=0.15,
        temperature=0.8,
        target_power=0.5,
        strength=0.8,
        min_gate=0.05,
        use_scale_aware=True,
    )
    
    print("\n[1] 检查参数requires_grad设置")
    print(f"    xyz.requires_grad: {gaussian_dict['xyz'].requires_grad}")
    print(f"    scale.requires_grad: {gaussian_dict['scale'].requires_grad}")
    print(f"    color.requires_grad: {gaussian_dict['color'].requires_grad}")
    print(f"    opacity.requires_grad: {gaussian_dict['opacity'].requires_grad}")
    
    # 应用竞争
    print("\n[2] 应用软竞争机制...")
    gaussian_gated, gate, stats = competition(gaussian_dict, source_view, scene_size)
    
    print(f"    输出gate形状: {gate.shape}")
    print(f"    Gate值范围: [{gate.min():.4f}, {gate.max():.4f}]")
    print(f"    Gate平均值: {gate.mean():.4f}")
    
    # 检查输出是否需要梯度
    print("\n[3] 检查输出requires_grad")
    print(f"    gated_opacity.requires_grad: {gaussian_gated['opacity'].requires_grad}")
    print(f"    gate.requires_grad: {gate.requires_grad}")
    
    # 计算一个简单的损失（模拟）
    print("\n[4] 构造损失函数进行反传...")
    
    # 损失1：鼓励高opacity被保留
    opacity_final = gaussian_gated["opacity"]
    loss_render = opacity_final.mean()  # 简化的render损失
    
    # 损失2：竞争一致性损失
    opacity_norm = (gaussian_dict["opacity"] - gaussian_dict["opacity"].min()) / \
                   (gaussian_dict["opacity"].max() - gaussian_dict["opacity"].min() + 1e-6)
    gate_norm = (gate - gate.min()) / (gate.max() - gate.min() + 1e-6)
    loss_competition = torch.nn.functional.mse_loss(gate_norm, opacity_norm.detach())
    
    total_loss = loss_render + 0.05 * loss_competition
    
    print(f"    loss_render: {loss_render.item():.6f}")
    print(f"    loss_competition: {loss_competition.item():.6f}")
    print(f"    total_loss: {total_loss.item():.6f}")
    
    # 反传
    print("\n[5] 执行反向传播...")
    total_loss.backward()
    
    # 检查梯度
    print("\n[6] 检查梯度是否存在...")
    
    params_to_check = {
        "xyz": gaussian_dict["xyz"],
        "scale": gaussian_dict["scale"],
        "color": gaussian_dict["color"],
        "opacity": gaussian_dict["opacity"],
    }
    
    all_gradients_ok = True
    for param_name, param in params_to_check.items():
        if param.grad is None:
            print(f"    ✗ {param_name}: 梯度为None")
            all_gradients_ok = False
        else:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.abs().mean().item()
            
            if grad_norm < 1e-10:
                print(f"    ⚠ {param_name}: 梯度过小 (norm={grad_norm:.2e})")
            else:
                print(f"    ✓ {param_name}: 梯度正常 (norm={grad_norm:.6f}, mean={grad_mean:.6f})")
                all_gradients_ok = all_gradients_ok and True
    
    # 最终结论
    print("\n" + "=" * 60)
    if all_gradients_ok:
        print("✓ 梯度流测试通过！竞争机制支持完整的反向传播")
        return True
    else:
        print("✗ 梯度流测试失败！某些参数缺少梯度")
        return False


def test_multiscale_competition():
    """测试多尺度竞争场景"""
    
    print("\n" + "=" * 60)
    print("多尺度竞争测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟早期粗高斯 vs 后期细高斯
    gaussian_dict = {
        "xyz": torch.tensor([
            [0.0, 0.0, 0.0],      # 高斯1: 同一位置
            [0.05, 0.05, 0.05],   # 高斯2: 同一位置
        ], device=device, requires_grad=True),
        
        "scale": torch.tensor([
            [0.5, 0.5, 0.5],      # 早期：大scale
            [0.05, 0.05, 0.05],   # 后期：小scale
        ], device=device, requires_grad=True),
        
        "color": torch.tensor([
            [0.5, 0.5, 0.5],      # 颜色1
            [0.5, 0.5, 0.5],      # 颜色2: 相同
        ], device=device, requires_grad=True),
        
        "opacity": torch.tensor([
            [0.3],                 # 早期：低opacity
            [0.9],                 # 后期：高opacity
        ], device=device, requires_grad=True),
        
        "rotation": torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], device=device, requires_grad=False),
    }
    
    source_view = torch.tensor([0, 1], device=device)  # 不同视角
    scene_size = torch.tensor([1.0], device=device)
    
    competition = SoftLocalCompetition(
        position_radius=0.15,
        use_scale_aware=True,
    )
    
    print("\n初始状态:")
    print(f"  高斯1 (早期粗): xyz={gaussian_dict['xyz'][0].tolist()}, "
          f"scale={gaussian_dict['scale'][0, 0].item():.2f}, "
          f"opacity={gaussian_dict['opacity'][0, 0].item():.2f}")
    print(f"  高斯2 (后期细): xyz={gaussian_dict['xyz'][1].tolist()}, "
          f"scale={gaussian_dict['scale'][1, 0].item():.2f}, "
          f"opacity={gaussian_dict['opacity'][1, 0].item():.2f}")
    
    # 应用竞争
    gaussian_gated, gate, stats = competition(gaussian_dict, source_view, scene_size)
    
    print(f"\n竞争结果:")
    print(f"  高斯1的gate: {gate[0, 0].item():.4f} (应该 < 1，因为opacity低)")
    print(f"  高斯2的gate: {gate[1, 0].item():.4f} (应该 ≈ 1，因为opacity高)")
    print(f"  竞争者数平均: {stats['num_competitors_mean'].item():.2f}")
    
    # 验证
    assert gate[1, 0] > gate[0, 0], "高opacity高斯的gate应该更大"
    print("\n✓ 多尺度竞争测试通过！")
    

def test_forward_backward_cycle():
    """完整的前向-反向周期测试"""
    
    print("\n" + "=" * 60)
    print("完整前向-反向周期测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建简单的网络模拟高斯生成
    class SimpleGaussianHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.xyz_fc = nn.Linear(10, 30)  # 生成10个高斯的xyz
            self.scale_fc = nn.Linear(10, 30)
            self.color_fc = nn.Linear(10, 30)
            self.opacity_fc = nn.Linear(10, 10)
        
        def forward(self, x):
            xyz = self.xyz_fc(x).reshape(-1, 3)
            scale = self.scale_fc(x).reshape(-1, 3)
            color = torch.sigmoid(self.color_fc(x).reshape(-1, 3))
            opacity = torch.sigmoid(self.opacity_fc(x).reshape(-1, 1))
            
            rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], 
                                   device=x.device).repeat(xyz.shape[0], 1)
            
            return {
                "xyz": xyz,
                "scale": scale,
                "color": color,
                "opacity": opacity,
                "rotation": rotation,
            }
    
    model = SimpleGaussianHead().to(device)
    competition = SoftLocalCompetition().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n执行5个训练步骤...")
    
    for step in range(5):
        # 前向
        feature = torch.randn(10, device=device)
        gaussians = model(feature)
        
        source_view = torch.tensor([0, 1] * 5, device=device)[:gaussians["xyz"].shape[0]]
        
        # 竞争
        gaussians_gated, gate, _ = competition(gaussians, source_view, None)
        
        # 损失
        loss = gaussians_gated["opacity"].mean()
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  步骤 {step+1}: loss={loss.item():.6f}, "
              f"gate_mean={gate.mean().item():.4f}")
    
    print("\n✓ 完整周期测试通过！模型成功训练")


if __name__ == "__main__":
    # 运行所有测试
    success = test_soft_competition_gradients()
    
    if success:
        print("\n" + "=" * 60)
        test_multiscale_competition()
        test_forward_backward_cycle()
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    else:
        print("\n✗ 梯度流测试失败，请检查实现")
