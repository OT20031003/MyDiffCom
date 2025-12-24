import torch
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

# ----- 設定 -----
# ★ 0dBの結果フォルダパスを指定
results_dir = "./results_retrans_comparison/ffhq_demo/hifi_diffcom/djscc_2/awgn_00dB/RetransComparison_rate_0.1_zeta0.3_seed22/visuals"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- モデル準備 -----
# 1. LPIPS
try:
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
except:
    print("LPIPS load failed.")
    exit()

# 2. MTCNN (Landmark)
try:
    mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.4, 0.5, 0.5], min_face_size=20)
except:
    print("MTCNN load failed.")
    exit()

# 3. FaceNet (ID Loss)
try:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
except:
    print("FaceNet load failed.")
    exit()

# 前処理
transform_lpips = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_embedding(img):
    # FaceNet用前処理
    try:
        batch = torch.tensor(np.array(img.resize((160, 160)))).permute(2, 0, 1).float().div(255).to(device)
        batch = (batch - 0.5) / 0.5
        batch = batch.unsqueeze(0)
        with torch.no_grad():
            return resnet(batch)
    except:
        return None

def calc_landmark_dist(img, lm_gt):
    try:
        boxes, _, landmarks = mtcnn.detect(img, landmarks=True)
        if landmarks is None: return None
        lm = np.array(landmarks[0], dtype=np.float32)
        return np.mean(np.linalg.norm(lm - lm_gt, axis=1))
    except:
        return None

def calc_psnr(img1, img2):
    # img1, img2: PIL Image
    arr1 = np.array(img1).astype(np.float32) / 255.0
    arr2 = np.array(img2).astype(np.float32) / 255.0
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0: return 100
    return 10 * np.log10(1.0 / mse)

# ----- 実行部分 -----
improvements = {
    'psnr': {'smooth': [], 'raw': []},
    'lpips': {'smooth': [], 'raw': []},
    'id': {'smooth': [], 'raw': []},
    'landmark': {'smooth': [], 'raw': []}
}

batch_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
print(f"Analyzing {len(batch_dirs)} samples for Relative Improvement...")

for batch_dir in tqdm(batch_dirs):
    if not os.path.isdir(batch_dir): continue
    
    p_gt = os.path.join(batch_dir, '0_GT.png')
    p_p1 = os.path.join(batch_dir, '2_Phase1_Recon.png')
    p_smooth = os.path.join(batch_dir, '3_Phase2_Refined_Smooth.png')
    p_raw = os.path.join(batch_dir, '3_Phase2_Refined_Raw.png')
    
    if not (os.path.exists(p_gt) and os.path.exists(p_p1) and os.path.exists(p_smooth) and os.path.exists(p_raw)):
        continue

    # 画像読み込み
    pil_gt = Image.open(p_gt).convert('RGB')
    pil_p1 = Image.open(p_p1).convert('RGB')
    pil_smooth = Image.open(p_smooth).convert('RGB')
    pil_raw = Image.open(p_raw).convert('RGB')
    
    # --- 1. PSNR Improvement ---
    psnr_p1 = calc_psnr(pil_gt, pil_p1)
    psnr_smooth = calc_psnr(pil_gt, pil_smooth)
    psnr_raw = calc_psnr(pil_gt, pil_raw)
    
    # PSNRは「高いほど良い」ので、(Phase2 - Phase1) / Phase1 * 100
    if psnr_p1 > 0:
        improvements['psnr']['smooth'].append((psnr_smooth - psnr_p1) / psnr_p1 * 100)
        improvements['psnr']['raw'].append((psnr_raw - psnr_p1) / psnr_p1 * 100)
    
    # --- 2. LPIPS Improvement ---
    t_gt = transform_lpips(pil_gt).unsqueeze(0).to(device)
    t_p1 = transform_lpips(pil_p1).unsqueeze(0).to(device)
    t_smooth = transform_lpips(pil_smooth).unsqueeze(0).to(device)
    t_raw = transform_lpips(pil_raw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        d_p1 = loss_fn_alex(t_gt, t_p1).item()
        d_smooth = loss_fn_alex(t_gt, t_smooth).item()
        d_raw = loss_fn_alex(t_gt, t_raw).item()
        
    # LPIPSは「低いほど良い」ので、減少率を見る (P1 - P2) / P1 * 100
    if d_p1 > 1e-6:
        improvements['lpips']['smooth'].append((d_p1 - d_smooth) / d_p1 * 100)
        improvements['lpips']['raw'].append((d_p1 - d_raw) / d_p1 * 100)

    # --- 3. ID Loss Improvement ---
    emb_gt = get_embedding(pil_gt)
    emb_p1 = get_embedding(pil_p1)
    emb_smooth = get_embedding(pil_smooth)
    emb_raw = get_embedding(pil_raw)
    
    if emb_gt is not None and emb_p1 is not None and emb_smooth is not None and emb_raw is not None:
        id_p1 = 1 - F.cosine_similarity(emb_gt, emb_p1).item()
        id_smooth = 1 - F.cosine_similarity(emb_gt, emb_smooth).item()
        id_raw = 1 - F.cosine_similarity(emb_gt, emb_raw).item()
        
        # ID Lossも「低いほど良い」ので、減少率を見る
        if id_p1 > 1e-6:
            improvements['id']['smooth'].append((id_p1 - id_smooth) / id_p1 * 100)
            improvements['id']['raw'].append((id_p1 - id_raw) / id_p1 * 100)

    # --- 4. Landmark Improvement ---
    try:
        _, _, lm_gt_list = mtcnn.detect(pil_gt, landmarks=True)
        if lm_gt_list is not None:
            lm_gt = np.array(lm_gt_list[0], dtype=np.float32)
            
            lm_dist_p1 = calc_landmark_dist(pil_p1, lm_gt)
            lm_dist_smooth = calc_landmark_dist(pil_smooth, lm_gt)
            lm_dist_raw = calc_landmark_dist(pil_raw, lm_gt)
            
            if lm_dist_p1 is not None and lm_dist_smooth is not None and lm_dist_raw is not None:
                # Landmark Errorも「低いほど良い」ので、減少率を見る
                if lm_dist_p1 > 1e-6:
                    improvements['landmark']['smooth'].append((lm_dist_p1 - lm_dist_smooth) / lm_dist_p1 * 100)
                    improvements['landmark']['raw'].append((lm_dist_p1 - lm_dist_raw) / lm_dist_p1 * 100)
    except:
        pass

# ----- 結果表示 -----
print("\n" + "="*60)
print("  RELATIVE IMPROVEMENT RATE ANALYSIS (vs Phase 1)")
print("="*60)

metrics_list = [
    ('psnr', 'Pixel Accuracy (PSNR)', 'Higher is Better'),
    ('lpips', 'Perceptual Quality (LPIPS)', 'Lower is Better'),
    ('id', 'Identity Preservation (ID Loss)', 'Lower is Better'),
    ('landmark', 'Structural Alignment (Landmark)', 'Lower is Better')
]

for metric, label, direction in metrics_list:
    if len(improvements[metric]['smooth']) > 0:
        imp_smooth = np.mean(improvements[metric]['smooth'])
        imp_raw = np.mean(improvements[metric]['raw'])
        
        print(f"\n[{label}] Improvement Rate (%)")
        print(f"  Note: Positive % means improvement over Phase 1.")
        print(f"  --------------------------------------------------")
        print(f"  Phase 2 Smooth (Prop): {imp_smooth:+.2f}%")
        print(f"  Phase 2 Raw (Comp):    {imp_raw:+.2f}%")
        
        diff = imp_smooth - imp_raw
        print(f"  Advantage (Smooth - Raw): {diff:+.2f} points")
        
        if diff > 0:
            print(f"  ✅ WIN: Proposed method improves {label} more effectively!")
        else:
            print(f"  ⚠️ Raw method shows higher relative improvement.")
    else:
        print(f"\n[{label}] No data available.")

print("="*60)