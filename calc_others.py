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
    import timm
    from transformers import CLIPProcessor, CLIPModel
    from torchvision.models import swin_v2_b, Swin_V2_B_Weights
except ImportError:
    print("Please install required libraries: pip install timm transformers torchvision")
    exit(1)

def load_models(device, model_name='convnext_v2'):
    """
    モデルをロードして返す関数
    """
    print(f"Loading Models on {device}...")
    print(f" - Selected Classifier: {model_name}")

    meta_weights = Swin_V2_B_Weights.IMAGENET1K_V1
    
    classifier = None
    classifier_transform = None

    if model_name == 'swin_v2':
        print(" - Loading Swin Transformer V2 Base...")
        classifier = swin_v2_b(weights=meta_weights).eval().to(device)
        classifier_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif model_name == 'convnext_v2':
        print(" - Loading ConvNeXt V2 Base (timm)...")
        classifier = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained=True)
        classifier.eval().to(device)
        classifier_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    # 3. CLIP (共通)
    print(" - Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return classifier, classifier_transform, clip_model, clip_processor, meta_weights

def calculate_semantic_metrics(base_path, models, device):
    """
    Args:
        base_path (str): 処理対象のディレクトリパス
        models (tuple): ロード済みのモデル群
        device (torch.device): デバイス
    """
    classifier, classifier_transform, clip_model, clip_processor, cls_weights = models
    
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
                
            # --- 【変更点】GT画像のCLIP特徴量を計算 (Image-to-Image用) ---
            # テキストではなく、GT画像自体をエンコードします
            gt_clip_inputs = clip_processor(images=gt_img_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                gt_clip_features = clip_model.get_image_features(**gt_clip_inputs)
                # 正規化 (L2 Norm)
                gt_clip_features = gt_clip_features / gt_clip_features.norm(p=2, dim=-1, keepdim=True)

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
                            current_sample_correctness[m] = True
                            metrics[m]['consistency'].append(1.0)
                        else:
                            current_sample_correctness[m] = False
                            failed_indices[m].append(b_dir)
                            metrics[m]['consistency'].append(0.0)

                    # 2. CLIP Score (Image-to-Image Cosine Similarity)
                    # 復元画像のCLIP特徴量を取得
                    m_clip_inputs = clip_processor(images=m_img_pil, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        m_clip_features = clip_model.get_image_features(**m_clip_inputs)
                        
                        # 正規化 (L2 Norm)
                        m_clip_features = m_clip_features / m_clip_features.norm(p=2, dim=-1, keepdim=True)
                        
                        # Cosine Similarity を計算 (GT特徴量 と 復元画像特徴量 の内積)
                        # CLIPの埋め込み空間におけるGTと復元画像の類似度(最大1.0)
                        score = torch.matmul(gt_clip_features, m_clip_features.t()).item()
                        
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
            # print(f"Error in batch {b_dir}: {e}") 
            continue

    # --- 集計結果の保存 ---
    final_results = {}
    
    print(f"{'Method':<30} | {'Acc':<10} | {'Conf':<10} | {'CLIP(I2I)':<10}")
    
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
            "clip_score": avg_clip, # plot_others.py と互換性を保つためキー名は 'clip_score' のまま
            "failed_indices": failed_indices[m]
        }
        print(f"{m:<30} | {avg_acc:.4f}     | {avg_conf:.4f}     | {avg_clip:.4f}")

    final_results["discrepancy_analysis"] = {
        "unc_ok_sem_ng": diff_unc_ok_sem_ng,
        "sem_ok_unc_ng": diff_sem_ok_unc_ng
    }

    model_suffix = "convnext" if "convnext" in str(classifier.__class__).lower() else "swin"
    # plot_others.py が読み込めるファイル名に戻す (i2i_results ではなく results)
    out_path = os.path.join(base_path, f"semantic_metrics_results_{model_suffix}.json")
    
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved: {out_path}\n")

if __name__ == "__main__":
    
    # ================= SETTINGS =================
    MODEL_SELECTION = 'convnext_v2'
    SNR_LIST = [-8, -7, -6, -5, -4, -3, -2] 
    PATH_TEMPLATE = r"results_retrans_comparison/imagenet/diffcom/djscc_2/awgn_{snr}dB/Retrans_rate_0.1_Comparison_both_exp2.0_gam0.7_zeta0.3_seed22"
    # ============================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Device: {device}")
    
    models = load_models(device, model_name=MODEL_SELECTION)
    
    print("="*60)
    print(f"Starting batch processing (Image-to-Image CLIP) for SNRs: {SNR_LIST}")
    print(f"Model: {MODEL_SELECTION}")
    print("="*60)

    for snr in SNR_LIST:
        target_path = PATH_TEMPLATE.format(snr=snr)
        
        if os.path.exists(target_path):
            print(f"Processing SNR: {snr}dB")
            calculate_semantic_metrics(target_path, models, device)
        else:
            print(f"[Warning] Path not found for SNR {snr}dB:\n{target_path}\n")
    
    print("All Finished.")