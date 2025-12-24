import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import os
import glob
import numpy as np
import shutil
from tqdm import tqdm

# ----- 設定 -----
# ★ 0dBの結果フォルダパスを指定してください
results_dir = "./results_retrans_comparison/ffhq_demo/hifi_diffcom/djscc_2/awgn_00dB/RetransComparison_rate_0.1_zeta0.3_seed22/visuals"

# ベストショットを保存するフォルダ（作成されます）
output_dir = "best_identity_examples"
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FaceNetモデル
print("Loading FaceNet...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB').resize((160, 160))
        batch = torch.tensor(np.array(img)).permute(2, 0, 1).float().div(255).to(device)
        batch = (batch - 0.5) / 0.5
        batch = batch.unsqueeze(0)
        with torch.no_grad():
            emb = resnet(batch)
        return emb
    except:
        return None

# ----- 探索実行 -----
results = []
batch_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))

print(f"Scanning {len(batch_dirs)} images for Identity Preservation wins...")

for batch_dir in tqdm(batch_dirs):
    if not os.path.isdir(batch_dir): continue
    
    p_gt = os.path.join(batch_dir, '0_GT.png')
    p_smooth = os.path.join(batch_dir, '3_Phase2_Refined_Smooth.png')
    p_raw = os.path.join(batch_dir, '3_Phase2_Refined_Raw.png')
    
    if os.path.exists(p_gt) and os.path.exists(p_smooth) and os.path.exists(p_raw):
        emb_gt = get_embedding(p_gt)
        emb_smooth = get_embedding(p_smooth)
        emb_raw = get_embedding(p_raw)
        
        if emb_gt is not None and emb_smooth is not None and emb_raw is not None:
            # ID Distance (小さいほど良い)
            dist_smooth = 1 - F.cosine_similarity(emb_gt, emb_smooth).item()
            dist_raw = 1 - F.cosine_similarity(emb_gt, emb_raw).item()
            
            # 改善量 (正の値が大きいほど、SmoothがRawより優れている)
            improvement = dist_raw - dist_smooth
            
            results.append({
                'batch_dir': batch_dir,
                'filename': os.path.basename(batch_dir), # バッチ番号など
                'dist_smooth': dist_smooth,
                'dist_raw': dist_raw,
                'improvement': improvement
            })

# ----- 結果集計 -----
# 改善量が大きい順（Smoothの圧勝順）にソート
results.sort(key=lambda x: x['improvement'], reverse=True)

print("\n" + "="*60)
print("🏆 TOP 5 IDENTITY PRESERVATION WINS (Smooth > Raw)")
print("   (Use these images for your qualitative results!)")
print("="*60)

for i, res in enumerate(results[:5]):
    print(f"Rank {i+1}: {res['filename']}")
    print(f"  ID Loss Diff: {res['improvement']:.4f} (Raw: {res['dist_raw']:.4f} -> Smooth: {res['dist_smooth']:.4f})")
    
    # 画像をコピーして保存
    src_dir = res['batch_dir']
    dst_prefix = os.path.join(output_dir, f"rank{i+1}_{res['filename']}")
    shutil.copy(os.path.join(src_dir, '0_GT.png'), f"{dst_prefix}_GT.png")
    shutil.copy(os.path.join(src_dir, '2_Phase1_Recon.png'), f"{dst_prefix}_Phase1.png")
    shutil.copy(os.path.join(src_dir, '3_Phase2_Refined_Raw.png'), f"{dst_prefix}_Raw.png")
    shutil.copy(os.path.join(src_dir, '3_Phase2_Refined_Smooth.png'), f"{dst_prefix}_Smooth.png")
    print(f"  -> Saved images to {output_dir}/")

print("\nDone. Please check the 'best_identity_examples' folder.")