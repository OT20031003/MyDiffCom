import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------
# 1. 設定 (URLの設定)
# ---------------------------------------------------------
# ここに取得した "dinov3_vits16_pretrain_lvd1689m" のURLを貼り付けてください
# (有効期限があるため、最新のものを使用してください)
checkpoint_url = "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibWdldWQwMWZiMzAzZmFxYnl4cW81czBsIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3Njc4NzUwNDF9fX1dfQ__&Signature=gx2Eacr6ZFyXLP37VY0JrtpVQhoQPo3nmJ1yfOh6YjodKtvi8LJiYTP6LZx3iMXzSvp7xzQFAAIuPU5pd%7Ex6LQKKuCBoPIBiDwz97tsfu3d0vj2nIODfOPcCGnQ8s-DMsnT5gDqMdU-PVI-Pl68KFq3981iCu7jXrzGGw5PcpIwQCGIFVc%7EoIQs6g5UmHkpGwYORBTcXDLljGeGP1Eu60xYjHN688W3YsPGXl5f-fpFrmtaOytrerK0pISr2M5gD%7EGiiMxVjhxGNHBIP5DMxeSjaFHncz6Rg6NmZzkNm-fVWjHAsMuG1sC41e7PGf728aZe4HOkwJ37apuLeYXuDhQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1527870811844182" 

# モデル名 (hubconf.pyで定義されている名前)
model_name = "dinov3_vits16"
repo = "facebookresearch/dinov3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. モデルのロード (torch.hub使用)
# ---------------------------------------------------------
print(f"Loading model {model_name} from torch.hub...")
try:
    # GitHubからロード (初回はrepoのダウンロードが入ります)
    # weights引数にURLを渡すことで、そのURLから重みをロードします
    model = torch.hub.load(repo, model_name, weights=checkpoint_url, trust_repo=True)
except Exception as e:
    print("GitHubからのロードに失敗した場合、ローカルにrepoがあるなら source='local' を試してください。")
    # ローカルに 'facebookresearch/dinov3' フォルダがある場合:
    # model = torch.hub.load('facebookresearch/dinov3', model_name, source='local', weights=checkpoint_url)
    raise e

model.to(device)
model.eval()

# ---------------------------------------------------------
# 3. 前処理 (DINOv3 README推奨の設定)
# ---------------------------------------------------------
# DINOv3は 16の倍数のサイズが望ましいです
transform = transforms.Compose([
    transforms.Resize((224, 224)), # リサイズ
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ---------------------------------------------------------
# 4. 可視化関数 (Feature Similarity)
# ---------------------------------------------------------
def visualize_dinov3_similarity(image_path):
    # 画像読み込み
    original_image = Image.open(image_path).convert("RGB")
    original_np = np.array(original_image)
    
    # 前処理
    img_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # 特徴量抽出
    with torch.no_grad():
        # forward_features は辞書を返します
        # {
        #   "x_norm_clstoken": [B, D], 
        #   "x_norm_patchtokens": [B, N, D], 
        #   ...
        # }
        features_dict = model.forward_features(img_tensor)
        
    cls_token = features_dict["x_norm_clstoken"]       # [1, 384]
    patch_tokens = features_dict["x_norm_patchtokens"] # [1, 196, 384] (224x224の場合)
    
    # [CLS]トークンと各パッチトークンのコサイン類似度を計算
    # これにより「クラス分類において画像のどの部分が重要か」がわかります
    # shape: [1, 196]
    similarity = F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)
    
    # 2次元マップに変形
    num_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches)) # 例: 14 (14x14=196)
    
    attn_map = similarity.reshape(grid_size, grid_size).cpu().numpy()
    
    # 正規化 (0~1)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # 元画像サイズにリサイズ
    attn_map_resized = cv2.resize(attn_map, (original_np.shape[1], original_np.shape[0]))
    
    # ヒートマップ作成
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # 重ね合わせ
    cam = heatmap + np.float32(original_np) / 255
    cam = cam / np.max(cam)
    
    # 表示
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(attn_map_resized, cmap='jet')
    axs[1].set_title("CLS-Patch Cosine Similarity (DINOv3)")
    axs[1].axis('off')
    
    axs[2].imshow(np.uint8(255 * cam))
    axs[2].set_title("Overlay")
    axs[2].axis('off')
    plt.savefig("vit_imp_3.png")
    plt.show()

if __name__ == "__main__":
    # ここに画像パスを指定してください
    path = "testsets/ffhq_demo/69901.png"
    #path = "testsets/imagenet/ILSVRC2012_subset_00000007.png"
    visualize_dinov3_similarity(path)
    print("checkpoint_urlを設定し、visualize_dinov3_similarity(path) を呼び出してください。")