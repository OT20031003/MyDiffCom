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

def calculate_dists_from_disk(base_path, device, target_methods):
    """
    保存済みの画像から手法ごとのDISTSを計算する
    """
    if not os.path.exists(base_path):
        print(f"[Skip] Path not found: {base_path}")
        return

    # 共通の評価対象 (Phase 1)
    common_methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
    ]
    
    # 比較対象のメソッドリストを結合
    # mainスクリプトでは "3_{exp_name}.png" として保存されている
    comparison_methods = [f'3_{m}' for m in target_methods]
    
    all_methods = common_methods + comparison_methods

    # DISTS 初期化
    dists_metric = DISTS().to(device)
    
    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"[Skip] No visuals directory in {base_path}")
        return

    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    if len(batch_dirs) == 0:
        print(f"No batch directories found in {visuals_dir}")
        return

    transform = transforms.Compose([transforms.ToTensor()])
    
    # 手法ごとのスコア累積用
    method_scores = {m: [] for m in all_methods}

    print(f"Processing DISTS for {len(batch_dirs)} samples from: {os.path.basename(base_path)}")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png')
        
        if not os.path.exists(gt_path):
            continue
            
        try:
            gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
            
            for m in all_methods:
                # 画像ファイル名: "1_JSCC_Init.png", "3_6_Importance_Random.png" 等
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

    # 平均算出
    final_results = {}
    print(f"--- DISTS Results ({os.path.basename(base_path)}) ---")
    for m in all_methods:
        # 表示名をきれいにする (例: 3_6_Importance_Random -> 6_Importance_Random)
        display_name = m
        if m.startswith("3_"):
            display_name = m[2:]

        if len(method_scores[m]) > 0:
            avg_score = sum(method_scores[m]) / len(method_scores[m])
            final_results[display_name] = avg_score
            print(f"{display_name:30s}: {avg_score:.4f}")
        else:
            print(f"{display_name:30s}: N/A (0 samples)")

    # 保存
    output_json = os.path.join(base_path, "post_process_dists.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved: {output_json}\n")

    return final_results

if __name__ == "__main__":
    # ==========================================
    # 設定エリア (Configuration)
    # ==========================================
    
    # ★ バージョン選択 ('v1', 'v2', 'v3') ★
    VERSION = "v3"
    
    # 共通設定
    DATASET = "ffhq_demo"   
    SNR_LABELS = ["-8", "-7", "-6", "-5", "-4", "-3", "-2"]
    
    # 固定パラメータ
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    # --- バージョン依存パラメータの設定 ---
    if VERSION == "v3":
        # v3用の設定 (results_retrans_comparison_v3)
        ROOT_DIR = "results_retrans_comparison_v3"
        PREFIX = "Retrans_" 
        
        # フォルダ名パラメータ
        RATE = 0.1
        MODE = "rate"
        BASIS = "semantic"
        EXP_FACTOR = 2.0
        GAMMA = 0.9 
        
        # 新しい実験スイート (6, 7, 8)
        TARGET_METHODS = [
            "6_Importance_Random",
            "7_Edge_Random",
            "8_Uncertainty_Random"
        ]

    elif VERSION == "v2":
        # v2用の設定
        ROOT_DIR = "results_retrans_comparison_v2"
        PREFIX = "Retrans_v2_"
        RATE = 0.1
        MODE = "rate"
        BASIS = "semantic"
        EXP_FACTOR = 2.0
        GAMMA = 0.9
        
        TARGET_METHODS = [
            "1_Random_Baseline",
            "3_Importance_Only",
            "4_Edge_Baseline"
        ]
        
    else:
        # v1用の設定 (results_retrans_comparison_v1)
        # 注意: 古いフォルダ名が "results_retrans_comparison" (v1なし) の場合は適宜変更してください
        ROOT_DIR = "results_retrans_comparison_v1"
        PREFIX = "Retrans_"
        
        RATE = 0.1
        MODE = "rate"
        BASIS = "semantic"
        EXP_FACTOR = 2.0
        GAMMA = 0.9 
        
        TARGET_METHODS = [
            "1_Random_Baseline",
            "2_Uncertainty_Only",
            "3_Importance_Only",
            "4_Edge_Baseline",
            "5_Proposed_Method"
        ]

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target Version: {VERSION} | Root: {ROOT_DIR}")

    # ==========================================
    # 実行ループ
    # ==========================================
    for snr_label in SNR_LABELS:
        snr_folder = f"awgn_{snr_label}dB"
        
        # フォルダ名の構築ロジック
        u_mode_str = "Comparison" 
        
        exp_folder_name = (
            f"{PREFIX}{MODE}_{RATE}_{u_mode_str}_{BASIS}"
            f"_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        )
        
        target_path = os.path.join(
            ROOT_DIR, 
            DATASET, 
            METHOD_PATH, 
            snr_folder, 
            exp_folder_name
        )
        
        calculate_dists_from_disk(target_path, device, TARGET_METHODS)