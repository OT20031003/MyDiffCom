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

def calculate_dists_from_disk(base_path, device, mode):
    """
    保存済みの画像から手法ごとのDISTSを計算する
    mode: 設定されているモード (例: 'edge', 'semantic' 等)
    """
    if not os.path.exists(base_path):
        print(f"[Skip] Path not found: {base_path}")
        return

    # モード名の先頭を大文字にする
    mode_cap = mode.capitalize()

    # ★修正: Semanticの場合はファイル名の末尾が 'Sem' になるため分岐処理
    suffix = mode_cap
    if mode == "semantic":
        suffix = "Sem"

    # 計算対象の手法
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        f'3_P2_perturbation_raw_{suffix}', # 動的変更 (例: _Sem)
        '3_P2_Random'
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

    print(f"Processing DISTS for {len(batch_dirs)} samples from: {os.path.basename(base_path)} (Mode: {mode_cap} -> Suffix: {suffix})")

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
    print(f"--- DISTS Results ({os.path.basename(base_path)}) ---")
    for m in methods:
        if method_scores[m]:
            avg = sum(method_scores[m]) / len(method_scores[m])
            final_results[m] = avg
            print(f"{m:25s}: {avg:.4f}")
        else:
            print(f"{m:25s}: N/A (0 samples)")

    output_json = os.path.join(base_path, "post_process_dists.json")
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
    
    # MODE設定
    # ★修正: edge -> semantic
    MODE = "semantic"
    
    # 2. SNR リスト
    # 必要に応じてコメントアウトを解除または変更してください
    SNR_LABELS = ["-8","-7", "-6", "-5" ,"-4", "-3","-2"]
    SNR_LABELS = ["-4"]
    
    # 3. 再送率 (Retrans_rate)
    RATE = 0.1

    # 4. HPRSパラメータ
    # ★修正: 1.0 -> 2.0 (前回の成功設定に合わせる)
    EXP_FACTOR = 2.0
    GAMMA = 1.0

    # 5. その他の固定パラメータ
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================
    # 実行ループ
    # ==========================================
    for snr_label in SNR_LABELS:
        snr_folder = f"awgn_{snr_label}dB"
        exp_folder = f"Retrans_rate_{RATE}_Comparison_{MODE}_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        
        target_path = os.path.join(
            ROOT_DIR, 
            DATASET, 
            METHOD_PATH, 
            snr_folder, 
            exp_folder
        )
        
        calculate_dists_from_disk(target_path, device, MODE)