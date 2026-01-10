import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# 必要なライブラリ
try:
    from transformers import CLIPProcessor, CLIPModel
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    print("Please install: pip install transformers torchvision")
    exit(1)

# ImageNetのクラスIDとラベル名のマッピング用 (簡易版)
# 本来は ImageNet 1k のラベルファイルが必要ですが、
# ここではCLIP用にラベル名を推論または外部ファイルから取得する想定です。
# 今回は簡略化のため、CLIP Score計算時に「クラスラベルが不明」な場合の処理を含めます。

def calculate_semantic_metrics(base_path, gt_labels_json=None):
    """
    Args:
        base_path (str): visualsディレクトリを含むルートパス
        gt_labels_json (str): 画像ファイル名とImageNetクラス名の対応JSON (Optional)
                              形式: {"filename.png": "goldfish, Carassius auratus"}
                              これがない場合、CLIPスコアは計算できません(分類精度のみ計算)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Models ---
    print("Loading ResNet50 (Classifier)...")
    # ImageNetで学習済みのResNet50
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    resnet_transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    print("Loading CLIP (Text-Image Metric)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 手法リスト
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_Random'
    ]

    # 結果格納
    # accuracy: ResNetがGTと同じクラスを予測できた確率 (今回はGTラベルがないため、GT画像の予測クラスを正解とみなす 'Consistency' を測る)
    metrics = {m: {'clip_score': [], 'classifier_conf': [], 'consistency': []} for m in methods}

    visuals_dir = os.path.join(base_path, 'visuals')
    batch_dirs = sorted([d for d in os.listdir(visuals_dir) if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()])

    print(f"Processing {len(batch_dirs)} samples...")

    for b_dir in tqdm(batch_dirs):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png') # Ground Truth
        
        if not os.path.exists(gt_path): continue

        try:
            # GT画像の読み込み
            gt_img_pil = Image.open(gt_path).convert('RGB')
            
            # --- ResNetによる「正解クラス」の取得 ---
            # ImageNetのGTラベルファイルがない場合、GT画像の予測トップ1を「正解」と仮定します
            gt_tensor = resnet_transform(gt_img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                gt_logits = resnet(gt_tensor)
                gt_probs = F.softmax(gt_logits, dim=1)
                pred_class_idx = torch.argmax(gt_probs, dim=1).item()
                gt_confidence = gt_probs[0, pred_class_idx].item()
                
                # クラス名取得 (CLIP用)
                class_name = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][pred_class_idx]

            # --- 各手法の評価 ---
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    m_img_pil = Image.open(m_path).convert('RGB')

                    # 1. Classification Consistency & Confidence
                    m_tensor = resnet_transform(m_img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = resnet(m_tensor)
                        probs = F.softmax(logits, dim=1)
                        
                        # GTと同じクラスに対する確率 (Confidence recovery)
                        confidence_on_gt_class = probs[0, pred_class_idx].item()
                        metrics[m]['classifier_conf'].append(confidence_on_gt_class)

                        # Top-1予測がGTと一致したか (Accuracy)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        is_correct = 1.0 if pred_idx == pred_class_idx else 0.0
                        metrics[m]['consistency'].append(is_correct)

                    # 2. CLIP Score (Image vs Class Name)
                    # "A photo of a {class_name}" との類似度
                    text_input = [f"a photo of a {class_name}"]
                    inputs = clip_processor(text=text_input, images=m_img_pil, return_tensors="pt", padding=True).to(device)
                    
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        # image_embedsとtext_embedsのCosine Similarity * logit_scale
                        logits_per_image = outputs.logits_per_image  # shape [1, 1]
                        score = logits_per_image.item()
                        metrics[m]['clip_score'].append(score)

        except Exception as e:
            print(f"Error in batch {b_dir}: {e}")
            continue

    # --- 集計と表示 ---
    print("\n" + "="*80)
    print(f"{'Method':<30} | {'Acc (Consistency)':<18} | {'ResNet Conf':<12} | {'CLIP Score':<10}")
    print("-" * 80)
    
    final_results = {}
    for m in methods:
        res = metrics[m]
        if len(res['consistency']) == 0:
            print(f"{m:<30} | N/A")
            continue
            
        avg_acc = sum(res['consistency']) / len(res['consistency'])
        avg_conf = sum(res['classifier_conf']) / len(res['classifier_conf'])
        avg_clip = sum(res['clip_score']) / len(res['clip_score'])
        
        final_results[m] = {
            "accuracy": avg_acc,
            "resnet_confidence": avg_conf,
            "clip_score": avg_clip
        }
        
        print(f"{m:<30} | {avg_acc:.4f}             | {avg_conf:.4f}       | {avg_clip:.4f}")

    # 保存
    out_path = os.path.join(base_path, "semantic_metrics_results.json")
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    # ここに解析したい結果フォルダのパスを入れてください
    target_path = r"results_retrans_comparison/imagenet/diffcom/djscc_2/awgn_-6dB/Retrans_rate_0.1_Comparison_both_exp5.0_gam0.7_zeta0.3_seed22"
    
    if os.path.exists(target_path):
        calculate_semantic_metrics(target_path)
    else:
        print("Path not found.")