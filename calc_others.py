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

def calculate_semantic_metrics(base_path, gt_labels_json=None):
    """
    Args:
        base_path (str): visualsディレクトリを含むルートパス
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Models ---
    print("Loading ResNet50 (Classifier)...")
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
    
    # 比較対象の手法キー（コード内の文字列と一致させる）
    KEY_UNC = '3_P2_perturbation_raw_Unc'
    KEY_SEM = '3_P2_perturbation_raw_Sem'

    # 結果格納
    metrics = {m: {'clip_score': [], 'classifier_conf': [], 'consistency': []} for m in methods}
    failed_indices = {m: [] for m in methods}
    
    # ★追加: UncとSemの比較用リスト
    # Uncが正解(OK)で、Semが不正解(NG)のインデックス
    diff_unc_ok_sem_ng = []
    # Semが正解(OK)で、Uncが不正解(NG)のインデックス
    diff_sem_ok_unc_ng = []

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
            gt_tensor = resnet_transform(gt_img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                gt_logits = resnet(gt_tensor)
                gt_probs = F.softmax(gt_logits, dim=1)
                pred_class_idx = torch.argmax(gt_probs, dim=1).item()
                
                # クラス名取得 (CLIP用)
                class_name = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][pred_class_idx]

            # 現在の画像の各手法の正解可否を保存する辞書
            current_sample_correctness = {}

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
                        
                        confidence_on_gt_class = probs[0, pred_class_idx].item()
                        metrics[m]['classifier_conf'].append(confidence_on_gt_class)

                        pred_idx = torch.argmax(probs, dim=1).item()
                        
                        if pred_idx == pred_class_idx:
                            is_correct = 1.0
                            current_sample_correctness[m] = True # 正解
                        else:
                            is_correct = 0.0
                            current_sample_correctness[m] = False # 不正解
                            failed_indices[m].append(b_dir)
                            
                        metrics[m]['consistency'].append(is_correct)

                    # 2. CLIP Score
                    text_input = [f"a photo of a {class_name}"]
                    inputs = clip_processor(text=text_input, images=m_img_pil, return_tensors="pt", padding=True).to(device)
                    
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        score = outputs.logits_per_image.item()
                        metrics[m]['clip_score'].append(score)
                else:
                    # 画像がない場合は評価不能（False扱いにしておく）
                    current_sample_correctness[m] = False

            # --- ★追加: Unc vs Sem の比較判定 ---
            # 両方のキーが存在する場合のみ比較
            if KEY_UNC in current_sample_correctness and KEY_SEM in current_sample_correctness:
                is_unc_ok = current_sample_correctness[KEY_UNC]
                is_sem_ok = current_sample_correctness[KEY_SEM]

                if is_unc_ok and not is_sem_ok:
                    diff_unc_ok_sem_ng.append(b_dir)
                elif is_sem_ok and not is_unc_ok:
                    diff_sem_ok_unc_ng.append(b_dir)

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
            "clip_score": avg_clip,
            "failed_indices": failed_indices[m]
        }
        
        print(f"{m:<30} | {avg_acc:.4f}             | {avg_conf:.4f}       | {avg_clip:.4f}")

    # --- ★追加: Unc vs Sem 差異の詳細ログ ---
    print("\n" + "="*80)
    print("Unc vs Sem: Discrepancy Analysis")
    print("-" * 80)
    
    print(f"Case 1: Unc Correct (O) / Sem Wrong (X) -> Count: {len(diff_unc_ok_sem_ng)}")
    if diff_unc_ok_sem_ng:
        print(f"Indices: {', '.join(diff_unc_ok_sem_ng)}")
    
    print("-" * 40)
    
    print(f"Case 2: Sem Correct (O) / Unc Wrong (X) -> Count: {len(diff_sem_ok_unc_ng)}")
    if diff_sem_ok_unc_ng:
        print(f"Indices: {', '.join(diff_sem_ok_unc_ng)}")
        
    print("="*80 + "\n")

    # --- 失敗リストのログ表示（既存機能） ---
    print("Misclassified Sample Indices (Failed to match GT prediction):")
    print("-" * 80)
    for m in methods:
        failures = failed_indices[m]
        if len(failures) > 0:
            failures_str = ", ".join(failures)
            print(f"[{m}] Count: {len(failures)}")
            # 長すぎる場合は省略表示（必要に応じてコメントアウト解除）
            # if len(failures_str) > 100: failures_str = failures_str[:100] + "..."
            print(f"Indices: {failures_str}\n")
        else:
            print(f"[{m}] No failures (Perfect Consistency)\n")

    # 結果をJSONに保存（差異分析も追加）
    final_results["discrepancy_analysis"] = {
        "unc_ok_sem_ng": diff_unc_ok_sem_ng,
        "sem_ok_unc_ng": diff_sem_ok_unc_ng
    }

    out_path = os.path.join(base_path, "semantic_metrics_results.json")
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    target_path = r"results_retrans_comparison/imagenet/diffcom/djscc_2/awgn_-4dB/Retrans_rate_0.1_Comparison_both_exp2.0_gam0.7_zeta0.3_seed22"
    
    if os.path.exists(target_path):
        calculate_semantic_metrics(target_path)
    else:
        print("Path not found.")