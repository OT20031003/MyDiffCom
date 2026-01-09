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
    # main_diffcom_retransmission.py の出力ファイル名に対応
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

    if len(batch_dirs) == 0:
        print(f"No batch directories found in {visuals_dir}")
        return

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
        
        try:
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
        except Exception as e:
            print(f"Error loading GT in batch {b_dir}: {e}")

    # 最終的なスコアの集計
    final_fids = {}
    print("\n--- Final FID Results ---")
    for m in methods:
        try:
            # .compute() で最終的なFIDを算出
            # サンプル数が少なすぎる場合などにエラーになる可能性があるためtry-except
            score = fid_metrics[m].compute().item()
            final_fids[m] = score
            print(f"{m:25s}: {score:.4f}")
        except Exception as e:
            print(f"{m:25s}: Could not compute (maybe too few samples or 0 samples found). Error: {e}")

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_fid.json")
    with open(output_json, 'w') as f:
        json.dump(final_fids, f, indent=4)
    print(f"\nResults saved to: {output_json}")

    return final_fids

if __name__ == "__main__":
    # ==========================================
    # 設定エリア (Configuration)
    # ==========================================
    
    # 1. データセット ("imagenet" or "ffhq_demo")
    DATASET = "ffhq_demo" 
    
    # 2. SNR 
    # 提示されたパス例 (awgn_-2dB) に合わせて設定してください
    SNR_LABEL = "-7" 
    
    # 3. 再送率 (Retrans_rate)
    RATE = 0.1

    # 4. HPRSパラメータ (ここを追加・修正しました)
    # main.py の出力フォルダ名に含まれるパラメータ
    EXP_FACTOR = 2.0  # exp2.0
    GAMMA = 0.3       # gam0.3

    # 5. その他の固定パラメータ
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    # ==========================================
    # パス構築と実行
    # ==========================================
    
    # フォルダ構成の例:
    # results_retrans_comparison/ffhq_demo/diffcom/djscc_2/awgn_-2dB/Retrans_rate_0.1_Comparison_both_exp2.0_gam0.3_zeta0.3_seed22
    
    snr_folder = f"awgn_{SNR_LABEL}dB"
    
    # main.py の config.result_name 生成ロジックに合わせたフォルダ名
    # f'Retrans_{retrans_mode}_{retrans_value}_{u_mode_str}_{retrans_basis}_exp{expansion_factor}_gam{retrans_gamma}_zeta{zeta}_seed{seed}'
    # ここでは retrans_mode='rate', basis='both', u_mode='Comparison' (複数手法実行時) を想定
    exp_folder = f"Retrans_rate_{RATE}_Comparison_both_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
    
    target_results_path = os.path.join(
        ROOT_DIR, 
        DATASET, 
        METHOD_PATH, 
        snr_folder, 
        exp_folder
    )
    
    print(f"Target Path: {target_results_path}")
    
    if os.path.exists(target_results_path):
        # FID計算実行
        calculate_fids_from_disk(target_results_path)
    else:
        print("\n[Error] 指定されたパスが存在しません。")
        print(f"Path not found: {target_results_path}")
        print("設定エリアの変数 (DATASET, SNR_LABEL, RATE, EXP_FACTOR, GAMMA 等) を確認してください。")