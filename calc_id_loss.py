import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# Facenet-PyTorchのインポート
# インストールされていない場合はエラーメッセージを表示
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("Error: 'facenet-pytorch' is required.")
    print("Please install it using: pip install facenet-pytorch")
    exit(1)

def calculate_id_loss_from_disk(base_path):
    """
    保存済みの画像から手法ごとのID Loss (Identity Loss) を計算する
    ID Loss = 1 - CosineSimilarity(GT_embedding, Fake_embedding)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- [モデルの準備] ---
    # VGGFace2で事前学習されたInceptionResnetV1を使用
    print("Loading Face Recognition Model (InceptionResnetV1)...")
    id_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 計算対象の手法（calc_fid.pyと同じ構成）
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
    # { 'MethodName': [loss_sample1, loss_sample2, ...] }
    id_scores = {m: [] for m in methods}     # Cosine Similarity (高いほど似ている)
    id_losses = {m: [] for m in methods}     # 1 - Similarity (低いほど良い)

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"Error: {visuals_dir} does not exist.")
        return

    # バッチディレクトリ (0, 1, 2...) を取得
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    # --- [前処理] ---
    # Facenet-PyTorchは通常、固定の正規化を期待しますが、
    # ここでは一般的なToTensor([0,1])を行い、embedding取得時に関数内で調整します。
    to_tensor = transforms.ToTensor()

    print(f"Processing {len(batch_dirs)} samples from: {visuals_dir}")

    for b_dir in tqdm(batch_dirs):
        path = os.path.join(visuals_dir, b_dir)
        
        # Ground Truth (Real画像) の読み込み
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path):
            continue
        
        try:
            # 画像読み込み & Tensor化
            gt_img = Image.open(gt_path).convert('RGB')
            gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device) # [1, 3, H, W]

            # GTの特徴量を抽出
            with torch.no_grad():
                # Facenet用にリサイズ (160x160) と正規化
                gt_input = preprocess_for_facenet(gt_tensor)
                gt_emb = id_model(gt_input) # [1, 512]

            # 各手法の画像を処理
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    m_img = Image.open(m_path).convert('RGB')
                    m_tensor = to_tensor(m_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        m_input = preprocess_for_facenet(m_tensor)
                        m_emb = id_model(m_input)

                        # Cosine Similarity計算
                        similarity = F.cosine_similarity(gt_emb, m_emb).item()
                        
                        # ID Loss = 1 - Similarity
                        loss = 1.0 - similarity

                        id_scores[m].append(similarity)
                        id_losses[m].append(loss)

        except Exception as e:
            print(f"Error processing batch {b_dir}: {e}")
            continue

    # --- [集計と保存] ---
    final_results = {}
    print("\n--- Final ID Loss Results (Lower is Better) ---")
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

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_id_loss.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {output_json}")

    return final_results

def preprocess_for_facenet(img_tensor):
    """
    入力画像 ([0, 1] range tensor) を Facenet (InceptionResnetV1) 用に前処理する
    1. 160x160 にリサイズ
    2. Whitening (標準化): (x - 127.5) / 128.0 相当の処理
       ※ img_tensorは [0, 1] なので、 (x * 255 - 127.5) / 128.0 を行う
    """
    # Resize
    img_resized = F.interpolate(img_tensor, size=(160, 160), mode='bilinear', align_corners=False)
    
    # Normalize (Fixed image standardization)
    # [0, 1] -> [0, 255] -> Standardize
    img_normalized = (img_resized * 255.0 - 127.5) / 128.0
    
    return img_normalized

if __name__ == "__main__":
    # 計算対象のディレクトリパスを指定（環境に合わせて変更してください）
    target_results_path = r"results_retrans_comparison/ffhq_demo/diffcom/djscc_2/awgn_-4dB/Retrans_rate_0.1_Comparison_both_zeta0.3_seed22"
    
    if os.path.exists(target_results_path):
        calculate_id_loss_from_disk(target_results_path)
    else:
        print(f"Path not found: {target_results_path}")