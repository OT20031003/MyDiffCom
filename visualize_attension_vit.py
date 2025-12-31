import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_attention_fixed(image_path, model_name="facebook/dinov2-base"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    original_size = image.size 

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    last_attention = outputs.attentions[-1]
    last_attention = last_attention[0]
    
    # [CLS]トークンからのアテンションを取得
    cls_attention = last_attention[:, 0, 1:]
    cls_attention_mean = cls_attention.mean(dim=0)
    
    # 整形
    patch_size = 14
    input_h = inputs['pixel_values'].shape[2]
    input_w = inputs['pixel_values'].shape[3]
    feat_h = input_h // patch_size
    feat_w = input_w // patch_size
    
    attn_map = cls_attention_mean.reshape(feat_h, feat_w)
    
    # === 修正ポイント: 外れ値（アーティファクト）の除去 ===
    # アテンション値の上位1%程度の値を計算し、それ以上を丸めることで
    # 極端に高い「左端のスパイク」の影響を抑えます。
    attn_map_flat = attn_map.flatten()
    # 上位98%の値を取得（これより大きい値はクリップする）
    v_max = torch.quantile(attn_map_flat, 0.98)
    v_min = attn_map.min()
    attn_map = torch.clamp(attn_map, v_min, v_max)
    
    # 0-1に正規化（可視化を綺麗にするため）
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    # =================================================
    
    attn_map = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_map, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)
    attn_map = attn_map.squeeze().cpu().numpy()

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(image, alpha=0.5)
    # アーティファクトを除去したので、顔などの特徴が見えやすくなるはずです
    im = axes[1].imshow(attn_map, cmap='jet', alpha=0.6) 
    axes[1].set_title("DINOv2 Attention (Artifacts clipped)")
    axes[1].axis('off')
    
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target_image_path = "testsets/69900.png" # お手元の画像パス
    if os.path.exists(target_image_path):
        visualize_attention_fixed(target_image_path)
    else:
        # 画像がない場合はダミー
        dummy = Image.new('RGB', (224, 224), color='white')
        dummy.save("dummy.jpg")
        visualize_attention_fixed("dummy.jpg")