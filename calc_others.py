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
    # 強力な分類モデル ConvNeXt Large を使用
    from torchvision.models import convnext_large, ConvNeXt_Large_Weights
except ImportError:
    print("Please install: pip install transformers torchvision")
    exit(1)

def load_models(device):
    """
    モデルを一度だけロードして返す関数
    """
    print(f"Loading Models on {device}...")
    
    # 1. Classifier: ConvNeXt Large (Acc ~87.5%)
    print(" - Loading Classifier (ConvNeXt Large)...")
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    classifier = convnext_large(weights=weights).eval().to(device)
    classifier_transform = weights.transforms()
    
    # 2. CLIP
    print(" - Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return classifier, classifier_transform, clip_model, clip_processor, weights

def calculate_semantic_metrics(base_path, models, device):
    """
    Args:
        base_path (str): 処理対象のディレクトリパス
        models (tuple): ロード済みのモデル群
        device (torch.device): デバイス
    """
    classifier, classifier_transform, clip_model, clip_processor, cls_weights = models
    
    # 手法リスト
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_Random'
    ]
    
    KEY_UNC = '3_P2_perturbation_raw_Unc'
    KEY_SEM = '3_P2_perturbation_raw_Sem'

    # 結果格納
    metrics = {m: {'clip_score': [], 'classifier_conf': [], 'consistency': []} for m in methods}
    failed_indices = {m: [] for m in methods}
    
    diff_unc_ok_sem_ng = []
    diff_sem_ok_unc_ng = []

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"Skipping: 'visuals' directory not found in {base_path}")
        return

    batch_dirs = sorted([d for d in os.listdir(visuals_dir) if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()])

    print(f"Processing {len(batch_dirs)} samples in: {os.path.basename(base_path)}")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png') 
        
        if not os.path.exists(gt_path): continue

        try:
            # GT画像の読み込み
            gt_img_pil = Image.open(gt_path).convert('RGB')
            
            # --- Classifierによる「正解クラス」の取得 ---
            gt_tensor = classifier_transform(gt_img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                gt_logits = classifier(gt_tensor)
                gt_probs = F.softmax(gt_logits, dim=1)
                pred_class_idx = torch.argmax(gt_probs, dim=1).item()
                
                # クラス名取得
                class_name = cls_weights.meta["categories"][pred_class_idx]

            current_sample_correctness = {}

            # --- 各手法の評価 ---
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    m_img_pil = Image.open(m_path).convert('RGB')

                    # 1. Classification Consistency & Confidence
                    m_tensor = classifier_transform(m_img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = classifier(m_tensor)
                        probs = F.softmax(logits, dim=1)
                        
                        confidence_on_gt_class = probs[0, pred_class_idx].item()
                        metrics[m]['classifier_conf'].append(confidence_on_gt_class)

                        pred_idx = torch.argmax(probs, dim=1).item()
                        
                        if pred_idx == pred_class_idx:
                            is_correct = 1.0
                            current_sample_correctness[m] = True
                        else:
                            is_correct = 0.0
                            current_sample_correctness[m] = False
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
                    current_sample_correctness[m] = False

            # --- Unc vs Sem の比較判定 ---
            if KEY_UNC in current_sample_correctness and KEY_SEM in current_sample_correctness:
                is_unc_ok = current_sample_correctness[KEY_UNC]
                is_sem_ok = current_sample_correctness[KEY_SEM]

                if is_unc_ok and not is_sem_ok:
                    diff_unc_ok_sem_ng.append(b_dir)
                elif is_sem_ok and not is_unc_ok:
                    diff_sem_ok_unc_ng.append(b_dir)

        except Exception as e:
            # print(f"Error in batch {b_dir}: {e}") # エラーログが多い場合はコメントアウト
            continue

    # --- 集計結果の保存 ---
    final_results = {}
    
    # 簡易表示用ヘッダー
    print(f"{'Method':<30} | {'Acc':<10} | {'Conf':<10} | {'CLIP':<10}")
    
    for m in methods:
        res = metrics[m]
        if len(res['consistency']) == 0:
            continue
            
        avg_acc = sum(res['consistency']) / len(res['consistency'])
        avg_conf = sum(res['classifier_conf']) / len(res['classifier_conf'])
        avg_clip = sum(res['clip_score']) / len(res['clip_score'])
        
        final_results[m] = {
            "accuracy": avg_acc,
            "classifier_confidence": avg_conf,
            "clip_score": avg_clip,
            "failed_indices": failed_indices[m]
        }
        print(f"{m:<30} | {avg_acc:.4f}     | {avg_conf:.4f}     | {avg_clip:.4f}")

    final_results["discrepancy_analysis"] = {
        "unc_ok_sem_ng": diff_unc_ok_sem_ng,
        "sem_ok_unc_ng": diff_sem_ok_unc_ng
    }

    # ファイル名は以前と同じものを使用
    out_path = os.path.join(base_path, "semantic_metrics_results.json")
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved: {out_path}\n")

if __name__ == "__main__":
    
    # ================= SETTINGS =================
    
    # 1. 処理したいSNRのリスト
    SNR_LIST = [-8, -7, -6, -5, -4, -3] 
    #SNR_LIST = [-8, -6, -4]
    # 2. パスのテンプレート ({snr} の部分がリストの値に置換されます)
    # パスがSNR以外共通の場合はこちらを使用してください
    PATH_TEMPLATE = r"results_retrans_comparison/imagenet/diffcom/djscc_2/awgn_{snr}dB/Retrans_rate_0.1_Comparison_both_exp2.0_gam0.3_zeta0.3_seed22"
    
    # ============================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Device: {device}")
    
    # モデルのロード（ループの外で1回だけ実行）
    models = load_models(device)
    
    print("="*60)
    print(f"Starting batch processing for SNRs: {SNR_LIST}")
    print("="*60)

    for snr in SNR_LIST:
        # パスの生成
        target_path = PATH_TEMPLATE.format(snr=snr)
        
        if os.path.exists(target_path):
            print(f"Processing SNR: {snr}dB")
            calculate_semantic_metrics(target_path, models, device)
        else:
            print(f"[Warning] Path not found for SNR {snr}dB:\n{target_path}\n")
    
    print("All Finished.")