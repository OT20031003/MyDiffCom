import torch
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms

# ----- 設定 -----
# ★ 分析したいディレクトリパス (0dB推奨)
results_dir = "./results_retrans_comparison/ffhq_demo/hifi_diffcom/djscc_2/awgn_00dB/RetransComparison_rate_0.1_zeta0.3_seed22/visuals"

# パッチ設定
PATCH_SIZE = 32   # パッチサイズ (32x32推奨)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- LPIPSモデルの準備 -----
try:
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    print("LPIPS model loaded successfully.")
except Exception as e:
    print(f"Failed to load LPIPS: {e}")
    exit()

# 画像前処理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_patches(img_tensor, patch_size):
    """画像をパッチに分割してバッチ化する関数"""
    kc, kh, kw = 3, patch_size, patch_size
    dc, dh, dw = 3, patch_size, patch_size
    
    patches = img_tensor.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(1, 3, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).squeeze(0) 
    return patches

# ----- 実行部分 -----
scores_smooth = []
scores_raw = []

batch_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
print(f"Found {len(batch_dirs)} directories. Starting Weighted Patch-LPIPS analysis...")

for batch_dir in tqdm(batch_dirs):
    if not os.path.isdir(batch_dir):
        continue
    
    # ファイルパス
    p_gt = os.path.join(batch_dir, '0_GT.png')
    p_p1 = os.path.join(batch_dir, '2_Phase1_Recon.png')
    p_smooth = os.path.join(batch_dir, '3_Phase2_Refined_Smooth.png')
    p_raw = os.path.join(batch_dir, '3_Phase2_Refined_Raw.png')
    
    if not (os.path.exists(p_gt) and os.path.exists(p_p1) and os.path.exists(p_smooth) and os.path.exists(p_raw)):
        continue
        
    # 画像読み込み & 前処理
    img_gt = transform(Image.open(p_gt).convert('RGB')).unsqueeze(0).to(device)
    img_p1 = transform(Image.open(p_p1).convert('RGB')).unsqueeze(0).to(device)
    img_smooth = transform(Image.open(p_smooth).convert('RGB')).unsqueeze(0).to(device)
    img_raw = transform(Image.open(p_raw).convert('RGB')).unsqueeze(0).to(device)
    
    # パッチ分割
    patches_gt = get_patches(img_gt, PATCH_SIZE)
    patches_p1 = get_patches(img_p1, PATCH_SIZE)
    patches_smooth = get_patches(img_smooth, PATCH_SIZE)
    patches_raw = get_patches(img_raw, PATCH_SIZE)
    
    # --- Weighted Analysis ---
    with torch.no_grad():
        # 1. Phase 1のエラーマップ（重み）を作成
        # 各パッチごとのGTとのLPIPS距離を計算
        dists_p1 = loss_fn_alex(patches_gt, patches_p1).view(-1)
        
        # 2. 重みの正規化 (画像内でのエラー割合にする)
        # エラーが大きいパッチほど重み(weight)が大きくなる
        total_error = dists_p1.sum()
        if total_error < 1e-8:
            weights = torch.ones_like(dists_p1) / len(dists_p1) # エラーがない場合は均等
        else:
            weights = dists_p1 / total_error
            
        # 3. SmoothとRawの各パッチのスコアを計算
        dists_smooth = loss_fn_alex(patches_gt, patches_smooth).view(-1)
        dists_raw = loss_fn_alex(patches_gt, patches_raw).view(-1)
        
        # 4. 重み付き平均スコアの算出 (Weighted Average LPIPS)
        # Σ (Score_i * Weight_i)
        weighted_score_smooth = (dists_smooth * weights).sum().item()
        weighted_score_raw = (dists_raw * weights).sum().item()
        
        scores_smooth.append(weighted_score_smooth)
        scores_raw.append(weighted_score_raw)

# ----- 結果表示 -----
print("\n" + "="*60)
print("  WEIGHTED PATCH-LPIPS ANALYSIS RESULTS")
print("="*60)

if len(scores_smooth) > 0:
    avg_smooth = np.mean(scores_smooth)
    avg_raw = np.mean(scores_raw)
    
    print(f"\n[Settings]")
    print(f"  Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Weighting Strategy: Proportional to Phase 1 LPIPS Error")
    print(f"  (Evaluated all patches, heavily penalizing errors in 'hard' regions)")

    print(f"\n[Results] Weighted LPIPS (Lower is Better)")
    print(f"  --------------------------------------------------")
    print(f"  Phase 2 Smooth (Prop): {avg_smooth:.5f}")
    print(f"  Phase 2 Raw (Comp):    {avg_raw:.5f}")
    
    diff = avg_raw - avg_smooth # 正の値ならSmoothの方が距離が小さい（勝ち）
    print(f"  Difference:            {diff:+.5f} (Positive is Good)")
    
    if diff > 0:
        print("\n  ✅ WIN: The proposed method works better on high-error regions!")
    else:
        print("\n  ⚠️ Note: Raw method still has lower weighted error.")

else:
    print("No images processed.")

print("="*60)