import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# DISTSのインポート試行
try:
    from DISTS_pytorch import DISTS
except ImportError:
    print("Error: 'DISTS_pytorch' module not found. Please install via 'pip install DISTS-pytorch'.")
    exit()

def calculate_dists_from_disk(base_path, device):
    """
    保存済みの画像からDISTSを計算する (Single_Run対応版)
    """
    if not os.path.exists(base_path):
        print(f"[Skip] Path not found: {base_path}")
        return

    # 計算対象の手法
    # Single Runモードのファイル名に対応
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_Single_Run'  # 提案手法 (または比較対象)
    ]

    # DISTS 初期化
    dists_metric = DISTS().to(device)
    
    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"[Skip] No visuals directory in {base_path}")
        return

    batch_dirs = sorted([d for d in os.listdir(visuals_dir) if d.isdigit()])
    if not batch_dirs:
        print(f"No batch directories found in {visuals_dir}")
        return

    transform = transforms.Compose([transforms.ToTensor()])
    method_scores = {m: [] for m in methods}

    # パスが長いため、フォルダ名のみを表示
    folder_name = os.path.basename(base_path)
    print(f"Processing DISTS for {len(batch_dirs)} samples from: {folder_name}")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path): continue
        
        try:
            gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
            
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    try:
                        m_img = transform(Image.open(m_path).convert('RGB')).unsqueeze(0).to(device)
                        
                        # DISTS計算
                        score = dists_metric(m_img, gt_img).item()
                        method_scores[m].append(score)
                    except Exception: 
                        pass
        except Exception: 
            pass

    final_results = {}
    print(f"--- DISTS Results ({folder_name}) ---")
    for m in methods:
        if method_scores[m]:
            avg = sum(method_scores[m]) / len(method_scores[m])
            final_results[m] = avg
            print(f"{m:25s}: {avg:.4f}")
        else:
            print(f"{m:25s}: N/A (0 samples)")

    output_json = os.path.join(base_path, "post_process_dists_singlerun.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved: {output_json}\n")

    return final_results

if __name__ == "__main__":
    # ==========================================
    # 設定エリア (Configuration)
    # ==========================================
    
    # 1. データセット ("imagenet" or "ffhq_demo")
    DATASET = "ffhq_demo" 
    
    # 2. SNR リスト
    SNR_LABELS = [0]  # 例: -4, 0 など
    
    # 3. 実験パラメータ (ファイルパスの構成要素)
    RETRANS_MODE = "rate"
    RATE = 0.1
    BASIS = "semantic"
    EXP_FACTOR = 2.0
    
    # ★変更点: GAMMAをリスト形式に変更
    GAMMA_LIST = [0.0, 0.3, 0.9, 1.0]
    GAMMA_LIST = [0.6]
    # 4. 固定パラメータ
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target Gammas: {GAMMA_LIST}")
    
    # ==========================================
    # 実行ループ (SNR x Gamma)
    # ==========================================
    for snr_val in SNR_LABELS:
        # SNRの表記調整 (0 -> 00, -4 -> -4)
        snr_str = str(snr_val).zfill(2)
        snr_folder = f"awgn_{snr_str}dB"
        
        # ★変更点: GAMMAループを追加
        for gamma_val in GAMMA_LIST:
            
            # フォルダ名の構築
            # Retrans_{MODE}_{RATE}_Comparison_{BASIS}_exp{EXP}_gam{GAM}_zeta{ZETA}_seed{SEED}
            exp_folder = (
                f"Retrans_{RETRANS_MODE}_{RATE}_Comparison_{BASIS}_"
                f"exp{EXP_FACTOR}_gam{gamma_val}_zeta{ZETA}_seed{SEED}"
            )
            
            target_path = os.path.join(
                ROOT_DIR, 
                DATASET, 
                METHOD_PATH, 
                snr_folder, 
                exp_folder
            )
            
            # 存在チェックをしてから実行
            if os.path.exists(target_path):
                calculate_dists_from_disk(target_path, device)
            else:
                print(f"[Warning] Path not found: {target_path}")