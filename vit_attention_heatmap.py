import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. モデルのロード
# ---------------------------------------------------------
print(f"Loading model {model_id} from Hugging Face Hub...")

try:
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager" # Flash Attention無効化
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print("\n【エラー】モデルのロードに失敗しました。")
    print(e)
    exit()

# ---------------------------------------------------------
# 3. 前処理
# ---------------------------------------------------------
# ViTのパッチサイズ
PATCH_SIZE = 16
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ---------------------------------------------------------
# 4. 可視化関数
# ---------------------------------------------------------
def visualize_attention_heatmap(image_path):
    # 画像読み込み
    try:
        original_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return

    original_np = np.array(original_image)
    
    # 前処理
    img_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # 推論 & アテンション取得
    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        
    # アテンション取得
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        last_layer_attn = outputs.attentions[-1]
    elif isinstance(outputs, tuple):
        last_layer_attn = outputs[-1]
    else:
        print("Attention maps not found.")
        return

    # --- 集計処理 ---
    # last_layer_attn shape: [Batch, Heads, Total_Tokens, Total_Tokens]
    
    # 1. ヘッド方向の平均 -> [1, Total_Tokens, Total_Tokens]
    attn_mat = torch.mean(last_layer_attn, dim=1)
    
    # 2. Query方向の平均 -> [1, Total_Tokens]
    patch_importance = torch.mean(attn_mat, dim=1)
    
    # 3. 画像パッチのみを抽出 (ここを修正)
    # 想定されるパッチ数
    expected_patches = (IMG_SIZE // PATCH_SIZE) ** 2  # 14*14 = 196
    
    # トークン数が多すぎる場合(Register Tokens等)、末尾からパッチ数分だけ取る
    # shape: [Batch, Tokens]
    if patch_importance.shape[1] > expected_patches:
        print(f"Token count ({patch_importance.shape[1]}) > expected ({expected_patches}). Trimming special tokens.")
        patch_importance = patch_importance[:, -expected_patches:]
    
    # --- ヒートマップ作成 ---
    num_patches = patch_importance.shape[1]
    grid_size = int(np.sqrt(num_patches)) # 14
    
    if grid_size * grid_size != num_patches:
        print(f"Error: Number of patches ({num_patches}) is not a perfect square.")
        return

    # 2次元に変形
    attn_map = patch_importance.reshape(grid_size, grid_size).cpu().numpy()
    
    # 正規化 (見やすくするため、最小値を0、最大値を1に)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # リサイズ & カラーマップ
    attn_map_resized = cv2.resize(attn_map, (original_np.shape[1], original_np.shape[0]))
    
    # ヒートマップの色付け
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # 重ね合わせ (画像の明るさを少し落としてヒートマップを目立たせる)
    cam = heatmap + np.float32(original_np) / 255 * 0.5
    cam = cam / np.max(cam)
    
    # 表示・保存
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(original_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(attn_map_resized, cmap='jet')
    axs[1].set_title(f"Mean Attention Map")
    axs[1].axis('off')
    
    axs[2].imshow(np.uint8(255 * cam))
    axs[2].set_title("Overlay")
    axs[2].axis('off')
    
    save_name = "vit_attention_hf.png"
    plt.savefig(save_name)
    print(f"Saved visualization to {save_name}")
    # plt.show()

if __name__ == "__main__":
    path = "testsets/ffhq_demo/69903.png"
    path = "testsets/imagenet/ILSVRC2012_subset_00000001.png"
    visualize_attention_heatmap(path)