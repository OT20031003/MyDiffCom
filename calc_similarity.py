import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# Transformersライブラリのインポート
try:
    from transformers import AutoModel, AutoImageProcessor
except ImportError:
    print("Error: 'transformers' is required.")
    print("Please install it using: pip install transformers")
    exit(1)

def calculate_dino_similarity_from_disk(base_path, model_name="facebook/dinov2-base"):
    """
    保存済みの画像から手法ごとの DINO Embedding Similarity を計算する
    Similarity = CosineSimilarity(GT_embedding, Fake_embedding)
    
    Args:
        base_path (str): 'visuals' ディレクトリを含むルートパス
        model_name (str): HuggingFaceのモデルID (例: 'facebook/dinov2-base', 'facebook/dinov3-...')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- [モデルの準備] ---
    print(f"Loading DINO Model ({model_name})...")
    try:
        # DINOモデルのロード (DINOv2, v3対応)
        # trust_remote_code=TrueはDINOv3等の新しいモデルで必要な場合があります
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(device)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return

    # ImageNet Normalization Parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 計算対象の手法リスト (calc_id_loss.pyと同様)
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_temporal_raw_Unc',
        '3_P2_temporal_raw_Sem',
        '3_P2_Random'
    ]

    # 結果格納用辞書
    # { 'MethodName': [sim_sample1, sim_sample2, ...] }
    dino_scores = {m: [] for m in methods}

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"Error: {visuals_dir} does not exist.")
        return

    # バッチディレクトリ (数字のみ) を取得
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    # 基本的なToTensor変換
    to_tensor = transforms.ToTensor()

    print(f"Processing {len(batch_dirs)} samples from: {visuals_dir}")

    # --- [メインループ] ---
    for b_dir in tqdm(batch_dirs):
        path = os.path.join(visuals_dir, b_dir)
        
        # Ground Truth (Real画像) の読み込み
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path):
            continue
        
        try:
            # 画像読み込み & Tensor化 [1, 3, H, W]
            gt_img = Image.open(gt_path).convert('RGB')
            gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)

            # GTの特徴量を抽出
            with torch.no_grad():
                gt_input = preprocess_for_dino(gt_tensor, mean, std)
                gt_emb = extract_dino_feature(model, gt_input)

            # 各手法の画像を処理
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    m_img = Image.open(m_path).convert('RGB')
                    m_tensor = to_tensor(m_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        m_input = preprocess_for_dino(m_tensor, mean, std)
                        m_emb = extract_dino_feature(model, m_input)

                        # Cosine Similarity計算 (1.0に近いほど意味的に類似)
                        similarity = F.cosine_similarity(gt_emb, m_emb).item()
                        dino_scores[m].append(similarity)

        except Exception as e:
            print(f"Error processing batch {b_dir}: {e}")
            continue

    # --- [集計と保存] ---
    final_results = {}
    print("\n--- Final DINO Semantic Similarity (Higher is Better) ---")
    print(f"{'Method':<30} | {'DINO Sim':<10}")
    print("-" * 45)

    for m in methods:
        scores = dino_scores[m]
        
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            
            final_results[m] = {
                "dino_similarity": avg_score,
                "num_samples": len(scores)
            }
            print(f"{m:<30} | {avg_score:.4f}")
        else:
            print(f"{m:<30} | N/A")

    # 結果をJSONとして保存
    output_json = os.path.join(base_path, "post_process_dino_sim.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {output_json}")

    return final_results

def preprocess_for_dino(img_tensor, mean, std, target_size=(224, 224)):
    """
    DINO入力用の前処理 (Resize -> Normalize)
    img_tensor: [B, 3, H, W] (0~1)
    """
    # 1. Resize to 224x224 (DINO standard)
    img_resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    # 2. Normalize with ImageNet mean/std
    img_normalized = (img_resized - mean) / std
    
    return img_normalized

def extract_dino_feature(model, inputs):
    """
    モデルからCLSトークンの特徴量を取得し、L2正規化して返す
    """
    outputs = model(inputs)
    
    # DINOv2 / ViT 系の出力処理
    # last_hidden_state: [B, Seq_Len, Dim]
    # class token (CLS) は通常 index 0
    last_hidden_state = outputs.last_hidden_state
    cls_token = last_hidden_state[:, 0, :]  # [B, Dim]
    
    # Cosine Similarity計算用に正規化しておく
    features = F.normalize(cls_token, dim=1, p=2)
    
    return features

if __name__ == "__main__":
    # 計算対象のディレクトリパスを指定
    # 必要に応じてパスを書き換えてください
    target_results_path = r"results_retrans_comparison/imagenet/diffcom/djscc_2/awgn_-3dB/Retrans_rate_0.1_Comparison_both_exp5.0_gam0.7_zeta0.3_seed22"
    
    # 使用するモデル:
    # 一般的な比較には "facebook/dinov2-base" が推奨されます。
    # 前回のコードで使用していた "facebook/dinov3-vitb16-pretrain-lvd1689m" も指定可能です。
    target_model = "facebook/dinov2-base"
    target_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    
    if os.path.exists(target_results_path):
        calculate_dino_similarity_from_disk(target_results_path, model_name=target_model)
    else:
        print(f"Path not found: {target_results_path}")