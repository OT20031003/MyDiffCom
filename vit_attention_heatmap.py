import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
import os

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
    
    # 3. 画像パッチのみを抽出
    expected_patches = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
    
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
    
    # リサイズ (元画像サイズに合わせる)
    attn_map_resized = cv2.resize(attn_map, (original_np.shape[1], original_np.shape[0]))
    
    # ---------------------------------------------------------
    # 画像保存処理 (論文用に個別保存)
    # ---------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. 元画像の保存 (Original)
    # PILを使ってそのまま保存
    save_path_orig = f"{base_name}_original.png"
    Image.fromarray(original_np).save(save_path_orig)
    print(f"Saved: {save_path_orig}")

    # 2. アテンションヒートマップのみ保存 (Heatmap)
    # plt.imsaveを使うと軸なし・余白なしでカラーマップ適用して保存できる
    save_path_heat = f"{base_name}_heatmap.png"
    plt.imsave(save_path_heat, attn_map_resized, cmap='jet')
    print(f"Saved: {save_path_heat}")

    # 3. 重ね合わせ画像の保存 (Overlay)
    # カラーマップ適用 (BGRになる)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    # RGBに変換 (Pillowで扱うため)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # 重ね合わせ計算 (float計算)
    heatmap_float = np.float32(heatmap_rgb) / 255
    original_float = np.float32(original_np) / 255
    
    # ブレンド率 (元画像を少し暗くしてヒートマップを目立たせる)
    cam = heatmap_float + original_float * 0.5
    cam = cam / np.max(cam) # 正規化
    
    # uint8に戻して保存
    cam_uint8 = np.uint8(255 * cam)
    save_path_overlay = f"{base_name}_overlay.png"
    Image.fromarray(cam_uint8).save(save_path_overlay)
    print(f"Saved: {save_path_overlay}")

if __name__ == "__main__":
    path = "testsets/ffhq_demo/69906.png"
    # path = "testsets/imagenet/ILSVRC2012_subset_00000000.png"
    visualize_attention_heatmap(path)