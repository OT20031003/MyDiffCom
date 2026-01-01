import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

# ---------------------------------------------------------
# 1. 設定とモデルのロード (DINOv2 with Registers)
# ---------------------------------------------------------
# Registers付きのモデルを指定します
model_name = "facebook/dinov2-with-registers-small" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特徴抽出用モデルとしてロード (output_attentions=True)
model = Dinov2Model.from_pretrained(model_name, output_attentions=True)
model.to(device)
model.eval()

processor = AutoImageProcessor.from_pretrained(model_name)

# ---------------------------------------------------------
# 2. Attention Rollout (レジスタ対応版)
# ---------------------------------------------------------
def get_attention_map_with_registers(outputs, img_height, img_width):
    attentions = outputs.attentions
    
    # トークン数（CLS + Registers + Patches）
    # DINOv2-smallのパッチサイズは14
    patch_size = 14
    
    num_tokens = attentions[0].shape[-1]
    result = torch.eye(num_tokens).to(device)
    
    with torch.no_grad():
        for attention in attentions:
            attn_heads_mean = attention.mean(dim=1).squeeze(0)
            
            # Rollout計算 (A = 0.5A + 0.5I)
            attn_heads_mean = 0.5 * attn_heads_mean + 0.5 * torch.eye(num_tokens).to(device)
            attn_heads_mean = attn_heads_mean / attn_heads_mean.sum(dim=-1).unsqueeze(-1)
            
            result = torch.matmul(attn_heads_mean, result)
            
    # --- ここが重要 ---
    # モデルの設定からレジスタの数を取得 (通常は4個)
    num_registers = getattr(model.config, "num_registers", 0)
    
    # [CLS]トークンが、画像パッチ [1 + num_registers :] にどれだけ注目したか
    # インデックス 0: CLS
    # インデックス 1 ～ num_registers: レジスタ (ここは可視化から除外)
    # インデックス 1 + num_registers ～ 最後: 画像パッチ
    mask = result[0, 1 + num_registers :]
    
    # グリッドサイズの計算 (正方形である前提)
    grid_size = int(np.sqrt(mask.shape[0]))
    
    mask = mask.reshape(grid_size, grid_size).cpu().numpy()
    mask = cv2.resize(mask, (img_width, img_height))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    return mask

# ---------------------------------------------------------
# 3. 実行と可視化
# ---------------------------------------------------------
def visualize_attention(image_path):
    image = Image.open(image_path).convert("RGB")
    original_img = np.array(image)
    
    # DINOv2は入力サイズを14の倍数にするのが一般的
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Attention Mapの生成
    attention_mask = get_attention_map_with_registers(outputs, original_img.shape[0], original_img.shape[1])

    # 表示用
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(original_img) / 255
    cam = cam / np.max(cam)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(attention_mask, cmap='jet')
    axs[1].set_title("Attention Map (DINOv2 w/ Registers)")
    axs[1].axis('off')
    
    axs[2].imshow(np.uint8(255 * cam))
    axs[2].set_title("Overlay")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("vit_registers_result.png")
    plt.show()
    print("Saved result to vit_registers_result.png")

# --- 実行 ---
# visualize_attention("path/to/image.jpg")
path = "testsets/ffhq_demo/69901.png"
#path = "val2017/000000000724.jpg"
#path = "results_retrans_comparison/ffhq_demo/diffcom/djscc_2/awgn_-6dB/Retrans_rate_0.1_Comparison_zeta0.3_seed22/visuals/1/2_Phase1_Recon.png"
visualize_attention(path)