import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

# ---------------------------------------------------------
# 1. 設定 (vit_imp_3.py より継承)
# ---------------------------------------------------------
# 有効期限内の署名付きURL (vit_imp_3.pyの内容を使用)
checkpoint_url = "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibWdldWQwMWZiMzAzZmFxYnl4cW81czBsIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3Njc4NzUwNDF9fX1dfQ__&Signature=gx2Eacr6ZFyXLP37VY0JrtpVQhoQPo3nmJ1yfOh6YjodKtvi8LJiYTP6LZx3iMXzSvp7xzQFAAIuPU5pd%7Ex6LQKKuCBoPIBiDwz97tsfu3d0vj2nIODfOPcCGnQ8s-DMsnT5gDqMdU-PVI-Pl68KFq3981iCu7jXrzGGw5PcpIwQCGIFVc%7EoIQs6g5UmHkpGwYORBTcXDLljGeGP1Eu60xYjHN688W3YsPGXl5f-fpFrmtaOytrerK0pISr2M5gD%7EGiiMxVjhxGNHBIP5DMxeSjaFHncz6Rg6NmZzkNm-fVWjHAsMuG1sC41e7PGf728aZe4HOkwJ37apuLeYXuDhQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1527870811844182" 

model_name = "dinov3_vits16"
repo = "facebookresearch/dinov3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. モデルのロード
# ---------------------------------------------------------
print(f"Loading model {model_name} from torch.hub...")
try:
    model = torch.hub.load(repo, model_name, weights=checkpoint_url, trust_repo=True)
except Exception as e:
    print(f"Error loading model: {e}")
    # ローカルキャッシュがある場合のフォールバックなどをここに記述可能
    raise e

model.to(device)
model.eval()

# ---------------------------------------------------------
# 3. 前処理
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ---------------------------------------------------------
# 4. 勾配計算・可視化ロジック
# ---------------------------------------------------------
def compute_structural_map(importance_map_2d):
    """
    Sobelフィルタを用いてImportance Mapの空間勾配(Structural Map)を計算する
    """
    # X方向、Y方向の勾配 (カーネルサイズ3)
    grad_x = cv2.Sobel(importance_map_2d, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(importance_map_2d, cv2.CV_64F, 0, 1, ksize=3)
    
    # 勾配ノルム (Gradient Magnitude)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 正規化 (0~1)
    if grad_magnitude.max() > 0:
        grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-8)
        
    return grad_magnitude

def verify_gradient_extraction(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # 画像読み込み
    original_image = Image.open(image_path).convert("RGB")
    original_np = np.array(original_image)
    H, W, _ = original_np.shape
    
    # ViT推論
    img_tensor = transform(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
        
    cls_token = features_dict["x_norm_clstoken"]       # [1, 384]
    patch_tokens = features_dict["x_norm_patchtokens"] # [1, 196, 384]
    
    # Cosine Similarity (Importance Map S)
    similarity = F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)
    
    # Reshape & Resize
    num_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches)) # 14
    
    attn_map = similarity.reshape(grid_size, grid_size).cpu().numpy()
    
    # Normalize S (0~1)
    s_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 元解像度へリサイズ (ここで滑らかになるため勾配が計算可能になる)
    s_map_resized = cv2.resize(s_map, (W, H))
    
    # --- [検証対象] 勾配計算 ---
    grad_map = compute_structural_map(s_map_resized)
    
    # --- 可視化 ---
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original
    axs[0].imshow(original_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # 2. ViT Importance (S)
    axs[1].imshow(s_map_resized, cmap='jet')
    axs[1].set_title("ViT Saliency (S)\n(Global Context)")
    axs[1].axis('off')
    
    # 3. Structural Gradient (||∇S||)
    axs[2].imshow(grad_map, cmap='magma')
    axs[2].set_title("Structural Map (||∇S||)\n(Proposed)")
    axs[2].axis('off')
    
    # 4. Overlay on Image
    # 勾配が強い部分を赤色で強調表示
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_map), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.6 * original_np/255 + 0.4 * heatmap/255
    axs[3].imshow(np.clip(overlay, 0, 1))
    axs[3].set_title("Overlay (Edges)")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig("verification_vit_gradient.png")
    plt.show()
    print("Verification image saved as 'verification_vit_gradient.png'")

if __name__ == "__main__":
    # テスト画像のパスを指定
    # path = "testsets/ffhq_demo/69901.png"
    # ない場合は適当な画像を生成するか、手持ちの画像パスを指定してください
    import urllib.request
    
    # サンプル画像がない場合のために、Webからサンプルを取得してテストする例（必要ならコメントアウト解除）
    sample_path = "sample_face.png"
    if not os.path.exists(sample_path) and not os.path.exists("testsets/ffhq_demo/69901.png"):
         try:
             url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
             print(f"Downloading sample image from {url}...")
             urllib.request.urlretrieve(url, sample_path)
             path = sample_path
         except:
             path = "testsets/ffhq_demo/69901.png" # デフォルト
    else:
         path = "testsets/ffhq_demo/69901.png" if os.path.exists("testsets/ffhq_demo/69901.png") else sample_path

    verify_gradient_extraction(path)