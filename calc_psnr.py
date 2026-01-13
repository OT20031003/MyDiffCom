import os
import json
import torch
import torchvision.transforms as transforms
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image
from tqdm import tqdm

def calculate_psnr_for_snr(target_path, device):
    """
    指定されたパス内の画像からPSNRを計算し保存する
    """
    if not os.path.exists(target_path):
        print(f"[Skip] Path not found: {target_path}")
        return

    # 手法定義
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        # '3_P2_temporal_raw_Unc',
        # '3_P2_temporal_raw_Sem',
        '3_P2_Random'
    ]

    # メトリクス初期化
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    visuals_dir = os.path.join(target_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"[Skip] No visuals directory in {target_path}")
        return

    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    if len(batch_dirs) == 0:
        return

    transform = transforms.Compose([transforms.ToTensor()])
    
    # 手法ごとのスコア累積用
    method_scores = {m: [] for m in methods}

    print(f"Processing {len(batch_dirs)} samples in {os.path.basename(target_path)}...")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png')
        
        if not os.path.exists(gt_path):
            continue
            
        try:
            gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
            
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    try:
                        m_img = transform(Image.open(m_path).convert('RGB')).unsqueeze(0).to(device)
                        score = psnr_metric(m_img, gt_img).item()
                        method_scores[m].append(score)
                    except Exception:
                        pass
        except Exception:
            pass

    # 平均算出
    final_results = {}
    print(f"--- PSNR Results ({os.path.basename(target_path)}) ---")
    for m in methods:
        if len(method_scores[m]) > 0:
            avg_score = sum(method_scores[m]) / len(method_scores[m])
            final_results[m] = avg_score
            print(f"{m:25s}: {avg_score:.4f}")
        else:
            print(f"{m:25s}: N/A (0 samples)")

    # 保存
    output_json = os.path.join(target_path, "post_process_psnr.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved: {output_json}\n")

if __name__ == "__main__":
    # ==========================================
    # 設定エリア
    # ==========================================
    DATASET = "ffhq_demo"
    
    # ★ 複数のSNRをリストで指定
    SNR_LABELS = ["-8","-7", "-6", "-5" ,"-4", "-3","-2"]
    
    RATE = 0.1
    EXP_FACTOR = 2.0
    GAMMA = 0.3
    
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 実行ループ
    # ==========================================
    for snr_label in SNR_LABELS:
        snr_folder = f"awgn_{snr_label}dB"
        exp_folder = f"Retrans_rate_{RATE}_Comparison_both_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        
        target_path = os.path.join(ROOT_DIR, DATASET, METHOD_PATH, snr_folder, exp_folder)
        calculate_psnr_for_snr(target_path, device)