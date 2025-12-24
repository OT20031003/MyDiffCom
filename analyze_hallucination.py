import torch
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ----- 設定 -----
# ★ 分析したいディレクトリパス (3dB または 0dB)
results_dir = "./results_retrans_comparison/ffhq_demo/hifi_diffcom/djscc_2/awgn_00dB/RetransComparison_rate_0.1_zeta0.3_seed22/visuals"

# 評価設定
ERROR_THRESHOLD_PERCENTILE = 90
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- MTCNN Setup -----
try:
    mtcnn = MTCNN(
        keep_all=False,
        device=device,
        thresholds=[0.4, 0.5, 0.5], # 緩い閾値
        min_face_size=20
    )
    use_landmarks = True
    print("MTCNN loaded.")
except Exception as e:
    print(f"MTCNN load failed: {e}")
    use_landmarks = False

def calculate_masked_mse(img_pred, img_gt, mask):
    """マスク領域のみのMSEを計算"""
    diff = (img_pred - img_gt) ** 2
    mse = np.sum(diff * mask) / (np.sum(mask) + 1e-8)
    return mse

def calculate_landmark_distance(img_path_pred, landmarks_gt):
    """予測画像のランドマークとGTのランドマークの距離を計算"""
    try:
        img = Image.open(img_path_pred).convert('RGB')
        
        # ランドマーク検出
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        
        if landmarks is None:
            return None 
        
        # データの取り出し
        if isinstance(landmarks, list):
            lm_pred = np.array(landmarks[0])
        else:
            lm_pred = landmarks[0]
            
        # ★★★ ここで強制的にNumPy float配列に変換 (エラー回避の核心) ★★★
        lm_pred = np.array(lm_pred, dtype=np.float32)
        lm_gt_fixed = np.array(landmarks_gt, dtype=np.float32)
        
        # 距離計算 (5点の平均ユークリッド距離)
        diff = lm_pred - lm_gt_fixed
        dist = np.mean(np.linalg.norm(diff, axis=1))
        
        return dist
    except Exception as e:
        # 万が一のエラー時はNoneを返す
        return None

# ----- 実行部分 -----
res_focused_mse = {'smooth': [], 'raw': []}
res_landmark_dist = {'smooth': [], 'raw': []}

batch_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
print(f"Found {len(batch_dirs)} directories. Starting analysis...")

count_detected = 0

for batch_dir in tqdm(batch_dirs):
    if not os.path.isdir(batch_dir):
        continue
    
    # ファイルパス
    p_gt = os.path.join(batch_dir, '0_GT.png')
    p_p1 = os.path.join(batch_dir, '2_Phase1_Recon.png')
    p_smooth = os.path.join(batch_dir, '3_Phase2_Refined_Smooth.png')
    p_raw = os.path.join(batch_dir, '3_Phase2_Refined_Raw.png')
    
    # ファイル存在確認
    if not (os.path.exists(p_gt) and os.path.exists(p_p1) and os.path.exists(p_smooth) and os.path.exists(p_raw)):
        continue
        
    # 画像読み込み
    img_gt = np.array(Image.open(p_gt).convert('RGB')).astype(np.float32) / 255.0
    img_p1 = np.array(Image.open(p_p1).convert('RGB')).astype(np.float32) / 255.0
    img_smooth = np.array(Image.open(p_smooth).convert('RGB')).astype(np.float32) / 255.0
    img_raw = np.array(Image.open(p_raw).convert('RGB')).astype(np.float32) / 255.0
    
    # --- 1. Focused Repair Analysis ---
    diff_p1 = np.mean((img_p1 - img_gt) ** 2, axis=2)
    threshold = np.percentile(diff_p1, ERROR_THRESHOLD_PERCENTILE)
    error_mask = (diff_p1 >= threshold).astype(np.float32)
    error_mask_rgb = np.repeat(error_mask[:, :, np.newaxis], 3, axis=2)
    
    mse_smooth = calculate_masked_mse(img_smooth, img_gt, error_mask_rgb)
    mse_raw = calculate_masked_mse(img_raw, img_gt, error_mask_rgb)
    
    res_focused_mse['smooth'].append(mse_smooth)
    res_focused_mse['raw'].append(mse_raw)
    
    # --- 2. Landmark Analysis ---
    if use_landmarks:
        try:
            pil_gt = Image.open(p_gt).convert('RGB')
            # GT検出
            _, _, lm_gt_list = mtcnn.detect(pil_gt, landmarks=True)
            
            if lm_gt_list is not None:
                # ★ ここでも型変換しておく
                lm_gt = np.array(lm_gt_list[0], dtype=np.float32)
                
                # SmoothとRawの評価
                d_smooth = calculate_landmark_distance(p_smooth, lm_gt)
                d_raw = calculate_landmark_distance(p_raw, lm_gt)
                
                if d_smooth is not None and d_raw is not None:
                    res_landmark_dist['smooth'].append(d_smooth)
                    res_landmark_dist['raw'].append(d_raw)
                    count_detected += 1
        except Exception:
            pass

# ----- 結果表示 -----
def to_psnr(mse): return -10 * np.log10(mse)

print("\n" + "="*60)
print("  STRUCTURAL HALLUCINATION ANALYSIS RESULTS")
print("="*60)

# 1. Focused PSNR
if len(res_focused_mse['smooth']) > 0:
    mean_mse_smooth = np.mean(res_focused_mse['smooth'])
    mean_mse_raw = np.mean(res_focused_mse['raw'])
    
    print(f"\n[1] Focused PSNR (on Top {100-ERROR_THRESHOLD_PERCENTILE}% Error Regions)")
    print(f"    Phase 2 Smooth (Prop): {to_psnr(mean_mse_smooth):.4f} dB")
    print(f"    Phase 2 Raw (Comp):    {to_psnr(mean_mse_raw):.4f} dB")
    print(f"    Difference:            {to_psnr(mean_mse_smooth) - to_psnr(mean_mse_raw):+.4f} dB")

# 2. Landmark Distance
if len(res_landmark_dist['smooth']) > 0:
    lm_smooth = np.mean(res_landmark_dist['smooth'])
    lm_raw = np.mean(res_landmark_dist['raw'])
    
    print(f"\n[2] Face Landmark Error (Euclidean Distance)")
    print(f"    (Evaluated on {count_detected} images)")
    print(f"    --------------------------------------------------")
    print(f"    Phase 2 Smooth (Prop): {lm_smooth:.4f} px")
    print(f"    Phase 2 Raw (Comp):    {lm_raw:.4f} px")
    
    diff_lm = lm_raw - lm_smooth
    print(f"    Difference:            {diff_lm:+.4f} px (Positive is Good)")
    
    if diff_lm > 0:
        print("\n    ✅ WIN: The proposed method has more accurate structural alignment!")
    else:
        print("\n    ⚠️ Note: Raw method has tighter alignment.")
else:
    print("\n[2] Landmark analysis: No valid comparisons found.")

print("="*60)