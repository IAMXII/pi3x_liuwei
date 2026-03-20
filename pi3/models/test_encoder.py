import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import os

# ====== 导入你本地的 DINOv2 模型 ======
# 如果在项目根目录运行，请将相对导入去掉点，比如：from your_module.dinov2.hub.backbones import dinov2_vitl14_reg
from .dinov2.layers import Mlp
from .dinov2.hub.backbones import dinov2_vitl14_reg 
from safetensors.torch import load_file

def visualize_local_dinov2_features(image_path, patch_size=14, vggt_weight_path=None):
    # 1. 读取并预处理图像
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # 强制对齐到 patch_size (14) 的倍数
    new_w, new_h = (w // patch_size) * patch_size, (h // patch_size) * patch_size
    img = img.resize((new_w, new_h))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # 2. 实例化你本地的 ViT-L/14 with Registers
    print("Loading local dinov2_vitl14_reg...")
    # 默认你的实现里是 pretrained=False，所以我们需要手动把预训练权重打进去
    model = dinov2_vitl14_reg(pretrained=False)
    
    # 【非常关键】这里复用了你 Pi3_3DGS 里的加载逻辑，提取 encoder 权重
    if vggt_weight_path and os.path.exists(vggt_weight_path):
        print(f"Loading weights from {vggt_weight_path}...")
        vggt_weight = load_file(vggt_weight_path)
        vggt_enc_weight = {
            k.replace('aggregator.patch_embed.', ''): vggt_weight[k] 
            for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')
        }
        model.load_state_dict(vggt_enc_weight, strict=False)
    else:
        print("⚠️ Warning: 未提供权重文件，或者文件不存在！特征图将是随机噪声。")
        
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    
    # 3. 前向传播提取特征
    print("Extracting 1024-dim features...")
    with torch.no_grad():
        # 完全复用你前向代码里的调用方式：is_training=True
        hidden_dict = model(img_tensor, is_training=True)
        
        if isinstance(hidden_dict, dict):
            # 拿到深层 Patch Tokens，排除 CLS 和 Register Tokens
            patch_tokens = hidden_dict["x_norm_patchtokens"][0] 
        else:
            patch_tokens = hidden_dict[0] # 应对可能的不兼容返回值
            
    # patch_tokens 的形状此时应该是 (H_patch * W_patch, 1024)
    print(f"Feature tensor shape: {patch_tokens.shape}")
    
    # 4. 使用 PCA 将 1024 维降到 3 维
    print("Applying PCA...")
    features = patch_tokens.cpu().numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    # 归一化到 [0, 1] 映射成 RGB
    pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
    
    # 5. 恢复空间分辨率并插值放大
    h_patch, w_patch = new_h // patch_size, new_w // patch_size
    pca_img = pca_features.reshape(h_patch, w_patch, 3)
    pca_img_resized = cv2.resize(pca_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 6. 绘图保存
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(pca_img_resized)
    axes[1].set_title("DINOv2 ViT-L/14 Semantic Features (1024D -> 3D)", fontsize=16)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("local_dino_semantic_verification.png", dpi=150)
    print("Verification complete! Saved as 'local_dino_semantic_verification.png'.")

# === 运行示例 ===
if __name__ == '__main__':
    # 替换为你测试图像的路径，以及你本地模型权重的绝对/相对路径
    img_file = "/data/liuwei/dataset/MidAir/Kite_training/cloudy/color_left/trajectory_3000/000000.JPEG"
    ckpt_file = "ckpts/pi3/model_pi3x.safetensors" 
    
    visualize_local_dinov2_features(img_file, patch_size=14, vggt_weight_path=ckpt_file)