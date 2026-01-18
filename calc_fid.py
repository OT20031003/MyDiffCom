import os
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm
import json

def calculate_fids_from_disk(base_path, device, mode):
    """
    保存済みの画像から手法ごとのFIDを計算する
    mode: 設定されているモード (例: 'edge', 'semantic' 等)
    """
    if not os.path.exists(base_path):
        print(f"[Skip] Path not found: {base_path}")
        return

    # モード名の先頭を大文字にする (例: edge -> Edge)
    mode_cap = mode.capitalize()

    # ★修正箇所: Semanticの場合はファイル名の末尾が 'Sem' になるため分岐処理
    suffix = mode_cap
    if mode == "semantic":
        suffix = "Sem"

    # 計算対象の手法（ファイル名の接頭辞）を定義
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        f'3_P2_perturbation_raw_{suffix}', # ここが動的に変わります (例: ..._Sem, ..._Edge)
        '3_P2_Random'
    ]
    
    # FIDメトリクスの初期化 (手法ごと)
    fid_metrics = {
        m: FrechetInceptionDistance(feature=2048, normalize=True).to(device) 
        for m in methods
    }

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

    transform = transforms.Compose([
        transforms.ToTensor(), # [0, 255] -> [0.0, 1.0]
    ])

    print(f"Processing {len(batch_dirs)} samples from: {os.path.basename(base_path)} (Mode: {mode_cap} -> Suffix: {suffix})")

    for b_dir in tqdm(batch_dirs, leave=False):
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
                        # 読み込みエラー等は無視して次へ
                        pass
        except Exception as e:
            pass

    # 最終的なスコアの集計
    final_fids = {}
    print(f"--- FID Results ({os.path.basename(base_path)}) ---")
    for m in methods:
        try:
            # .compute() で最終的なFIDを算出
            score = fid_metrics[m].compute().item()
            final_fids[m] = score
            print(f"{m:25s}: {score:.4f}")
        except Exception as e:
            print(f"{m:25s}: N/A (Error or too few samples)")

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_fid.json")
    with open(output_json, 'w') as f:
        json.dump(final_fids, f, indent=4)
    print(f"Saved: {output_json}\n")

    return final_fids

if __name__ == "__main__":
    # ==========================================
    # 設定エリア (Configuration)
    # ==========================================
    
    # 1. データセット ("imagenet" or "ffhq_demo")
    DATASET = "ffhq_demo" 
    #DATASET = "imagenet"
    
    # MODE設定 ("semantic" or "edge" 等)
    MODE = "semantic"
    
    # 2. SNR リスト
    # ここに計算したいSNRをすべて列挙します
    # SNR_LABELS = ["-8","-7", "-6", "-5" ,"-4", "-3","-2"]
    SNR_LABELS = ["-4"]

    # 3. 再送率 (Retrans_rate)
    RATE = 0.08

    # 4. HPRSパラメータ
    EXP_FACTOR = 3.0
    GAMMA = 0.3  # ログを見ると0.0ではなく0.3のパスを読みたいようでしたので適宜確認してください

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
        
        # ログでPath not foundと言われたフォルダ名に合わせてパラメータを設定する必要があります
        # 例: exp2.0_gam0.3_zeta0.3_seed22
        exp_folder = f"Retrans_rate_{RATE}_Comparison_{MODE}_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        
        target_results_path = os.path.join(
            ROOT_DIR, 
            DATASET, 
            METHOD_PATH, 
            snr_folder, 
            exp_folder
        )
        
        # MODEを引数として渡す
        calculate_fids_from_disk(target_results_path, device, MODE)