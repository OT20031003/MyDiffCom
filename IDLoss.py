import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm

# ----- 設定 -----
# 画像が保存されているディレクトリのパス (main_diffcom_comparison.pyの保存先を指定してください)
# 例: './results_retrans_comparison/ffhq_10m/diffcom/awgn_00dB/RetransComparison.../visuals'
results_dir = "./results_retrans_comparison/ffhq_demo/hifi_diffcom/djscc_2/awgn_03dB/RetransComparison_rate_0.1_zeta0.3_seed22/visuals"  # ★ここを実際の結果フォルダパスに変更してください

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- モデルの準備 (FaceNet) -----
# 顔認識用学習済みモデル (VGGFace2) をロード
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def calculate_id_dist(img_path1, img_path2):
    """2枚の画像のFace Embedding間の距離を計算"""
    try:
        # 画像読み込み & 前処理 (160x160にリサイズ, テンソル化)
        img1 = Image.open(img_path1).convert('RGB').resize((160, 160))
        img2 = Image.open(img_path2).convert('RGB').resize((160, 160))
        
        # [0, 1] -> 標準化
        batch1 = torch.tensor(np.array(img1)).permute(2, 0, 1).float().div(255).to(device)
        batch2 = torch.tensor(np.array(img2)).permute(2, 0, 1).float().div(255).to(device)
        
        # 標準化 (FaceNetの学習時設定に合わせる)
        batch1 = (batch1 - 0.5) / 0.5
        batch2 = (batch2 - 0.5) / 0.5
        
        batch1 = batch1.unsqueeze(0)
        batch2 = batch2.unsqueeze(0)

        with torch.no_grad():
            # 特徴量抽出 (512次元)
            emb1 = resnet(batch1)
            emb2 = resnet(batch2)
            
            # 距離計算 (1 - Cosine Similarity) -> 値が小さいほど「本人に似ている」
            dist = 1 - F.cosine_similarity(emb1, emb2).item()
            return dist
    except Exception as e:
        print(f"Error processing {img_path1}: {e}")
        return None

# ----- 評価実行 -----
scores_smooth = []
scores_raw = []

# ディレクトリ構造に合わせてループ (0, 1, 2... フォルダを想定)
batch_dirs = sorted(glob.glob(os.path.join(results_dir, "*"))) # 各バッチのフォルダ

print("Calculating Face Identity Distance...")
for batch_dir in tqdm(batch_dirs):
    if not os.path.isdir(batch_dir):
        continue
        
    gt_path = os.path.join(batch_dir, '0_GT.png')
    smooth_path = os.path.join(batch_dir, '3_Phase2_Refined_Smooth.png')
    raw_path = os.path.join(batch_dir, '3_Phase2_Refined_Raw.png')
    
    if os.path.exists(gt_path) and os.path.exists(smooth_path) and os.path.exists(raw_path):
        # GT vs Smooth
        d_smooth = calculate_id_dist(gt_path, smooth_path)
        if d_smooth is not None:
            scores_smooth.append(d_smooth)
            
        # GT vs Raw
        d_raw = calculate_id_dist(gt_path, raw_path)
        if d_raw is not None:
            scores_raw.append(d_raw)

# ----- 結果表示 -----
avg_smooth = np.mean(scores_smooth)
avg_raw = np.mean(scores_raw)

print("\n=== Face Identity Distance Results (Lower is Better) ===")
print(f"Phase 2 Smooth (Proposed): {avg_smooth:.4f}")
print(f"Phase 2 Raw (Comparison):  {avg_raw:.4f}")
print(f"Difference: {avg_raw - avg_smooth:.4f}")

if avg_smooth < avg_raw:
    print("\n✅ SUCCESS: The proposed method more accurately preserves the individual's identity.")
else:
    print("\n⚠️ Note: If Raw performs better in this metric as well, the Raw method may also demonstrate superior noise reduction performance for facial feature points.")