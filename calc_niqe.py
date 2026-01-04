import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import pyiqa

def calculate_niqe_from_disk(base_path):
    """
    保存済みの画像から手法ごとのNIQE (Naturalness Image Quality Evaluator) を計算する
    NIQEはNo-Reference指標であり、低いほど良い（自然な画像に近い）とされる。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- [モデルの準備] ---
    # PyIQAを使用してNIQEモデルをロード
    print("Loading NIQE Model...")
    try:
        niqe_metric = pyiqa.create_metric('niqe', device=device)
    except Exception as e:
        print(f"Error loading pyiqa: {e}")
        print("Please install via: pip install pyiqa")
        return

    # 計算対象の手法
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_temporal_raw_Unc',
        '3_P2_temporal_raw_Sem',
        '3_P2_Random'
    ]

    # 結果格納用リスト
    niqe_scores = {m: [] for m in methods}

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"Error: {visuals_dir} does not exist.")
        return

    # バッチディレクトリ (0, 1, 2...) を取得
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    # PyIQAは [0, 1] のTensor入力を期待します
    to_tensor = transforms.ToTensor()

    print(f"Processing {len(batch_dirs)} samples from: {visuals_dir}")

    for b_dir in tqdm(batch_dirs):
        path = os.path.join(visuals_dir, b_dir)
        
        # ※ NIQEはNo-ReferenceなのでGT画像の読み込みは必須ではありませんが、
        #    GT自体のNIQEも参考に測りたい場合はここに追加してください。
        
        # 各手法の画像を処理
        for m in methods:
            m_path = os.path.join(path, f'{m}.png')
            if os.path.exists(m_path):
                try:
                    # 画像読み込み & Tensor化 [1, 3, H, W]
                    m_img = Image.open(m_path).convert('RGB')
                    m_tensor = to_tensor(m_img).unsqueeze(0).to(device)

                    # NIQE計算
                    with torch.no_grad():
                        score = niqe_metric(m_tensor).item()
                        niqe_scores[m].append(score)

                except Exception as e:
                    print(f"Error processing {m} in batch {b_dir}: {e}")

    # --- [集計と保存] ---
    final_results = {}
    print("\n--- Final NIQE Results (Lower is Better) ---")
    print(f"{'Method':<30} | {'Average NIQE':<10}")
    print("-" * 45)

    for m in methods:
        scores = niqe_scores[m]
        
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            
            final_results[m] = {
                "niqe": avg_score,
                "num_samples": len(scores)
            }
            print(f"{m:<30} | {avg_score:.4f}")
        else:
            print(f"{m:<30} | N/A")

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_niqe.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {output_json}")

    return final_results

if __name__ == "__main__":
    # ==========================================
    # 設定エリア (Configuration)
    # ==========================================
    
    # 1. データセット ("imagenet" or "ffhq_demo")
    DATASET = "ffhq_demo"
    DATASET = "imagenet"
    # 2. SNR ("00", "-4" などフォルダ名の数値部分)
    # フォルダ名が 'awgn_-4dB' の場合は "-4" としてください
    SNR_LABEL = "-4" 
    
    # 3. 再送率 (Retrans_rate)
    RATE = 0.1

    # 4. その他の固定パラメータ (必要に応じて変更)
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    # ==========================================
    # パス構築と実行
    # ==========================================
    
    # フォルダ構成:
    # results_retrans_comparison/{DATASET}/diffcom/djscc_2/awgn_{SNR}dB/Retrans_rate_{RATE}_Comparison_both_zeta{ZETA}_seed{SEED}
    
    snr_folder = f"awgn_{SNR_LABEL}dB"
    exp_folder = f"Retrans_rate_{RATE}_Comparison_both_zeta{ZETA}_seed{SEED}"
    
    target_results_path = os.path.join(
        ROOT_DIR, 
        DATASET, 
        METHOD_PATH, 
        snr_folder, 
        exp_folder
    )
    
    print(f"Target Path: {target_results_path}")
    
    if os.path.exists(target_results_path):
        calculate_niqe_from_disk(target_results_path)
    else:
        print("\n[Error] 指定されたパスが存在しません。")
        print("設定エリアの変数 (DATASET, SNR_LABEL, RATE 等) を確認してください。")