import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import lpips

# Saliency検出用 (簡易的に torchvisionのセグメンテーションモデルを使用)
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def get_saliency_mask(model, img_tensor, device):
    """
    DeepLabV3を使って「背景以外」の領域をマスクとして取得する
    Return: [1, 1, H, W] (0.0 ~ 1.0)
    """
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
        output_predictions = output.argmax(0) # [H, W]
        
        # COCO/VOCのラベルで 0=background なので、0以外を1とする
        mask = (output_predictions > 0).float().unsqueeze(0).unsqueeze(0)
        
        # マスクが真っ黒(何も検出されない)場合は、全体を1にする(安全策)
        if mask.sum() < 10:
            mask = torch.ones_like(mask)
            
    return mask.to(device)

def calculate_masked_metrics(base_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Models ---
    print("Loading LPIPS...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    print("Loading Segmentation Model (for Masking)...")
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    seg_model = deeplabv3_resnet101(weights=weights).eval().to(device)
    seg_transform = weights.transforms()

    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_Random'
    ]
    
    # metrics: {method: {'masked_psnr': [], 'masked_lpips': []}}
    results = {m: {'masked_psnr': [], 'masked_lpips': []} for m in methods}

    visuals_dir = os.path.join(base_path, 'visuals')
    batch_dirs = sorted([d for d in os.listdir(visuals_dir) if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()])

    print(f"Processing {len(batch_dirs)} samples with Saliency Masking...")

    to_tensor = transforms.ToTensor()

    for b_dir in tqdm(batch_dirs):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png')
        
        if not os.path.exists(gt_path): continue
        
        try:
            # Load GT
            gt_img = Image.open(gt_path).convert('RGB')
            gt_t = to_tensor(gt_img).unsqueeze(0).to(device) # [1, 3, H, W]
            
            # Generate Mask from GT (GTの物体領域を特定)
            # Segモデル用に入力を正規化
            gt_seg_in = seg_transform(gt_img).unsqueeze(0).to(device)
            mask = get_saliency_mask(seg_model, gt_seg_in, device) # [1, 1, H, W]
            
            # Maskを画像サイズにリサイズ (念のため)
            if mask.shape[-2:] != gt_t.shape[-2:]:
                mask = F.interpolate(mask, size=gt_t.shape[-2:], mode='nearest')

            # Maskの面積 (正規化用)
            mask_sum = mask.sum() + 1e-8

            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    m_img = Image.open(m_path).convert('RGB')
                    m_t = to_tensor(m_img).unsqueeze(0).to(device)

                    # --- 1. Masked MSE / PSNR ---
                    diff = (gt_t - m_t) ** 2
                    # マスク領域のみのMSE
                    masked_mse = (diff * mask).sum() / (mask_sum * 3) # 3 channels
                    masked_psnr = -10 * torch.log10(masked_mse + 1e-8).item()
                    
                    results[m]['masked_psnr'].append(masked_psnr)

                    # --- 2. Masked LPIPS ---
                    # LPIPSは [B, C, H, W] のマップを返す設定にするか、
                    # あるいは 画像自体をマスクして背景を黒/白にしてから計算する
                    # ここでは「背景をグレー(0.5)で塗りつぶした画像」同士でLPIPSを測る
                    
                    # 背景を隠す (Mask=1なら画像、0なら灰色)
                    gt_masked = gt_t * mask + 0.5 * (1 - mask)
                    m_masked = m_t * mask + 0.5 * (1 - mask)
                    
                    # LPIPS計算 (Normalize to -1~1 for LPIPS model is usually handled inside or assume 0-1 depending on version, 
                    # but lpips lib expects normalized tensors roughly. Standard usage:
                    # input should be 0-1, convert to -1 to 1 inside? 
                    # lpipsライブラリは通常 -1~1 を期待しますが、ここでは簡易化のため
                    # そのまま渡して相対比較します(内部で正規化される場合多し))
                    
                    # LPIPS入力用に -1 ~ 1 に変換
                    gt_m_norm = gt_masked * 2 - 1
                    m_m_norm = m_masked * 2 - 1
                    
                    dist = loss_fn_alex(gt_m_norm, m_m_norm).item()
                    results[m]['masked_lpips'].append(dist)

        except Exception as e:
            # print(e)
            continue

    # --- 集計 ---
    print("\n" + "="*60)
    print(f"{'Method':<30} | {'Masked PSNR':<12} | {'Masked LPIPS':<12}")
    print("-" * 60)
    
    final_data = {}
    for m in methods:
        dat = results[m]
        if not dat['masked_psnr']: continue
        
        avg_psnr = sum(dat['masked_psnr']) / len(dat['masked_psnr'])
        avg_lpips = sum(dat['masked_lpips']) / len(dat['masked_lpips'])
        
        final_data[m] = {'masked_psnr': avg_psnr, 'masked_lpips': avg_lpips}
        print(f"{m:<30} | {avg_psnr:.4f}       | {avg_lpips:.4f}")
        
    # JSON保存
    with open(os.path.join(base_path, 'masked_metrics.json'), 'w') as f:
        json.dump(final_data, f, indent=4)
        
if __name__ == "__main__":
    target_path = r"results_retrans_comparison/imagenet/diffcom/djscc_2/awgn_-5dB/Retrans_rate_0.1_Comparison_both_exp2.0_gam0.3_zeta0.3_seed22"
    if os.path.exists(target_path):
        calculate_masked_metrics(target_path)