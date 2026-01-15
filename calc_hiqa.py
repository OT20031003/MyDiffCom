import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import pyiqa

def calculate_hiqa_from_disk(base_path, device):
    """
    保存済みの画像から手法ごとの HyperIQA (HIQA) を計算する。
    calc_fid.py と同様の構造で実装。
    """
    if not os.path.exists(base_path):
        print(f"[Skip] Path not found: {base_path}")
        return

    # --- [モデルの準備] ---
    # PyIQAを使用してHyperIQAモデルをロード
    # 毎回ロードすると重いため、ループ外でロードして渡す設計も可能ですが、
    # calc_fidの構造に合わせるため関数内でハンドリングします(ただしcreate_metricはキャッシュされることが多いです)
    try:
        hiqa_metric = pyiqa.create_metric('hyperiqa', device=device)
    except Exception as e:
        print(f"Error loading pyiqa: {e}")
        return

    # 計算対象の手法
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        # '3_P2_temporal_raw_Unc',
        # '3_P2_temporal_raw_Sem',
        '3_P2_Random'
    ]

    # 結果格納用リスト
    hiqa_scores = {m: [] for m in methods}

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"[Skip] No visuals directory in {base_path}")
        return

    # バッチディレクトリ (0, 1, 2...) を取得
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    if len(batch_dirs) == 0:
        print(f"No batch directories found in {visuals_dir}")
        return

    # PyIQAは [0, 1] のTensor入力を期待します
    to_tensor = transforms.ToTensor()

    print(f"Processing {len(batch_dirs)} samples from: {os.path.basename(base_path)}")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        
        # HyperIQAはNo-ReferenceなのでGTは不要
        
        # 各手法の画像を処理
        for m in methods:
            m_path = os.path.join(path, f'{m}.png')
            if os.path.exists(m_path):
                try:
                    # 画像読み込み & Tensor化 [1, 3, H, W]
                    m_img = Image.open(m_path).convert('RGB')
                    m_tensor = to_tensor(m_img).unsqueeze(0).to(device)

                    # HyperIQA計算
                    with torch.no_grad():
                        score = hiqa_metric(m_tensor).item()
                        hiqa_scores[m].append(score)

                except Exception as e:
                    # エラー時はスキップ
                    pass

    # --- [集計と保存] ---
    final_results = {}
    print(f"--- HyperIQA Results ({os.path.basename(base_path)}) ---")
    
    for m in methods:
        scores = hiqa_scores[m]
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            final_results[m] = avg_score
            print(f"{m:25s}: {avg_score:.4f}")
        else:
            final_results[m] = None
            print(f"{m:25s}: N/A")

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_hiqa.json")
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
    # DATASET = "imagenet"
    
    # 比較モード (ディレクトリ名の一部)
    MODE = "both"
    
    # 2. SNR リスト
    # ここに計算したいSNRをすべて列挙します
    SNR_LABELS = ["-8", "-7", "-6", "-5", "-4", "-3", "-2"]
    
    # 3. 再送率 (Retrans_rate)
    RATE = 0.1

    # 4. HPRSパラメータ (ディレクトリ特定用)
    EXP_FACTOR = 2.0
    GAMMA = 0.3

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
        
        # calc_fid.py と同じフォルダ命名規則を使用
        exp_folder = f"Retrans_rate_{RATE}_Comparison_{MODE}_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        
        target_results_path = os.path.join(
            ROOT_DIR, 
            DATASET, 
            METHOD_PATH, 
            snr_folder, 
            exp_folder
        )
        
        calculate_hiqa_from_disk(target_results_path, device)