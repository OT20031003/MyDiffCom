import torch
import numpy as np
from PIL import Image
import os
import glob
from facenet_pytorch import MTCNN

# ----- 設定 -----
results_dir = "./results_retrans_comparison/ffhq_demo/hifi_diffcom/djscc_2/awgn_03dB/RetransComparison_rate_0.1_zeta0.3_seed22/visuals"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

# 診断で成功した設定
mtcnn = MTCNN(
    keep_all=False,
    device=device,
    thresholds=[0.4, 0.5, 0.5],
    min_face_size=20
)

def calculate_landmark_distance_debug(img_path_pred, landmarks_gt):
    """デバッグ用: エラーを握りつぶさずに表示"""
    img = Image.open(img_path_pred).convert('RGB')
    
    # 検出
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    
    if landmarks is None:
        print(f"  [DEBUG] Detection failed for: {os.path.basename(img_path_pred)}")
        return None 
    
    # データ形状の確認
    print(f"  [DEBUG] Landmarks found. Shape: {landmarks.shape}")
    
    lm_pred = landmarks[0] # (5, 2)
    
    # 計算実行 (ここでエラーが起きている可能性大)
    diff = lm_pred - landmarks_gt
    dist = np.mean(np.linalg.norm(diff, axis=1))
    
    return dist

# ----- 最初の1つだけ実行してエラーを見る -----
batch_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
print(f"Found {len(batch_dirs)} directories.")

for i, batch_dir in enumerate(batch_dirs):
    if not os.path.isdir(batch_dir):
        continue
    
    print(f"\n--- Checking Batch {i}: {os.path.basename(batch_dir)} ---")
    
    p_gt = os.path.join(batch_dir, '0_GT.png')
    p_smooth = os.path.join(batch_dir, '3_Phase2_Refined_Smooth.png')
    
    if not os.path.exists(p_gt):
        print("GT file missing")
        continue

    # 1. GTの検出
    print("Detecting GT landmarks...")
    pil_gt = Image.open(p_gt).convert('RGB')
    _, _, lm_gt_list = mtcnn.detect(pil_gt, landmarks=True)
    
    if lm_gt_list is None:
        print("GT detection failed. Skipping this batch.")
        continue
        
    lm_gt = lm_gt_list[0]
    print(f"GT Landmarks shape: {lm_gt.shape}") # (5, 2) であるべき

    # 2. Smoothの検出 & 計算 (ここで落ちるか？)
    print("Calculating distance for Smooth...")
    try:
        d_smooth = calculate_landmark_distance_debug(p_smooth, lm_gt)
        print(f"Result: {d_smooth}")
    except Exception as e:
        print("\n" + "!"*50)
        print("CRITICAL ERROR FOUND:")
        print(e)
        print("!"*50)
        # エラーの詳細（変数の型など）を表示
        import traceback
        traceback.print_exc()
        break # 最初のエラーで止める

    # 成功したら次へ（最初の数件でエラーが出なければ、別の原因）
    if i >= 2: 
        print("--- Debug limit reached (checked 3 batches) ---")
        break