import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# Facenet-PyTorch import
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("Error: 'facenet-pytorch' is required.")
    print("Please install it using: pip install facenet-pytorch")
    exit(1)

def preprocess_for_facenet(img_tensor):
    """
    Preprocess input tensor [0, 1] for Facenet (InceptionResnetV1)
    1. Resize to 160x160
    2. Whitening (Standardize): (x * 255 - 127.5) / 128.0
    """
    # Resize
    img_resized = F.interpolate(img_tensor, size=(160, 160), mode='bilinear', align_corners=False)
    
    # Normalize (Fixed image standardization)
    img_normalized = (img_resized * 255.0 - 127.5) / 128.0
    
    return img_normalized

def calculate_id_loss_for_snr(target_path, id_model, device):
    """
    指定されたパス内の画像からID Loss (1 - CosineSimilarity) を計算し保存する。
    (Single_Run対応版)
    """
    if not os.path.exists(target_path):
        print(f"[Skip] Path not found: {target_path}")
        return

    # 計算対象の手法
    # Single Runモードのファイル名に対応
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_Single_Run'  # 提案手法 (または比較対象)
    ]

    # 結果コンテナ
    id_scores = {m: [] for m in methods}     # Cosine Similarity
    id_losses = {m: [] for m in methods}     # 1 - Similarity
    
    visuals_dir = os.path.join(target_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"[Skip] No visuals directory in {target_path}")
        return

    # バッチディレクトリ取得
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    if len(batch_dirs) == 0:
        return

    to_tensor = transforms.ToTensor()

    # パスが長いため、フォルダ名のみを表示
    folder_name = os.path.basename(target_path)
    print(f"Processing ID Loss for {len(batch_dirs)} samples in {folder_name}")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        
        # Load Ground Truth
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path):
            continue
        
        try:
            gt_img = Image.open(gt_path).convert('RGB')
            gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device) # [1, 3, H, W]

            with torch.no_grad():
                gt_input = preprocess_for_facenet(gt_tensor)
                gt_emb = id_model(gt_input) # [1, 512]

            # Process each method
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    try:
                        m_img = Image.open(m_path).convert('RGB')
                        m_tensor = to_tensor(m_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            m_input = preprocess_for_facenet(m_tensor)
                            m_emb = id_model(m_input)

                            # Calculate Similarity & Loss
                            similarity = F.cosine_similarity(gt_emb, m_emb).item()
                            loss = 1.0 - similarity

                            id_scores[m].append(similarity)
                            id_losses[m].append(loss)
                            
                    except Exception:
                        pass
                else:
                    pass

        except Exception as e:
            pass

    # --- [Summarize & Save] ---
    final_results = {}
    print(f"--- ID Loss Results ({folder_name}) ---")
    print(f"{'Method':<30} | {'ID Loss':<10} | {'Similarity':<10}")
    print("-" * 60)

    for m in methods:
        scores = id_scores[m]
        losses = id_losses[m]
        
        if len(losses) > 0:
            avg_loss = sum(losses) / len(losses)
            avg_score = sum(scores) / len(scores)
            
            final_results[m] = {
                "id_loss": avg_loss,
                "id_similarity": avg_score,
                "num_samples": len(losses)
            }
            print(f"{m:<30} | {avg_loss:.4f}     | {avg_score:.4f}")
        else:
            print(f"{m:<30} | N/A        | N/A")

    # Save results to JSON
    output_json = os.path.join(target_path, "post_process_id_loss_singlerun.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved: {output_json}\n")

if __name__ == "__main__":
    # ==========================================
    # 設定エリア
    # ==========================================
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target Gammas: {GAMMA_LIST}")

    # --- [Load Model Once] ---
    print("Loading Face Recognition Model (InceptionResnetV1)...")
    try:
        id_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

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
                calculate_id_loss_for_snr(target_path, id_model, device)
            else:
                print(f"[Warning] Path not found: {target_path}")