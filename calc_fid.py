import os
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm
import json

def calculate_fids_from_disk(base_path):
    """
    保存済みの画像から手法ごとのFIDを計算する
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 計算対象の手法（ファイル名の接頭辞）を定義
    # 提供されたコードの保存形式に基づいています
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_temporal_raw_Unc',
        '3_P2_temporal_raw_Sem',
        '3_P2_Random'
    ]

    # FIDメトリクスの初期化 (手法ごと)
    fid_metrics = {
        m: FrechetInceptionDistance(feature=2048, normalize=True).to(device) 
        for m in methods
    }

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"Error: {visuals_dir} does not exist.")
        return

    # バッチディレクトリ (0, 1, 2...) を取得
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    transform = transforms.Compose([
        transforms.ToTensor(), # [0, 255] -> [0.0, 1.0]
    ])

    print(f"Processing {len(batch_dirs)} samples from: {visuals_dir}")

    for b_dir in tqdm(batch_dirs):
        path = os.path.join(visuals_dir, b_dir)
        
        # Ground Truth (Real画像) の読み込み
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path):
            continue
        
        # PILで開きRGBに変換してTensor化
        gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
        
        # 各手法の画像をFakeとして更新
        for m in methods:
            m_path = os.path.join(path, f'{m}.png')
            if os.path.exists(m_path):
                try:
                    m_img = transform(Image.open(m_path).convert('RGB')).unsqueeze(0).to(device)
                    
                    # メトリクス更新 (real=TrueはGT, real=Falseは手法の出力)
                    fid_metrics[m].update(gt_img, real=True)
                    fid_metrics[m].update(m_img, real=False)
                except Exception as e:
                    print(f"Skip {m} in batch {b_dir} due to error: {e}")

    # 最終的なスコアの集計
    final_fids = {}
    print("\n--- Final FID Results ---")
    for m in methods:
        try:
            # .compute() で最終的なFIDを算出
            score = fid_metrics[m].compute().item()
            final_fids[m] = score
            print(f"{m:25s}: {score:.4f}")
        except Exception as e:
            print(f"{m:25s}: Could not compute (maybe too few samples).")

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_fid.json")
    with open(output_json, 'w') as f:
        json.dump(final_fids, f, indent=4)
    print(f"\nResults saved to: {output_json}")

    return final_fids

if __name__ == "__main__":
    # 計算対象のディレクトリパスを指定
    target_results_path = r"results_retrans_comparison/ffhq_demo/diffcom/djscc_2/awgn_00dB/Retrans_rate_0.2_Comparison_both_zeta0.3_seed22"
    # FID計算実行
    calculate_fids_from_disk(target_results_path)