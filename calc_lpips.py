import os
import json
import torch
import torchvision.transforms as transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
from tqdm import tqdm

def calculate_lpips_for_snr(target_path, device):
    if not os.path.exists(target_path):
        print(f"[Skip] Path not found: {target_path}")
        return

    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_Random'
    ]

    # LPIPS 初期化 (alexnet推奨)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    
    visuals_dir = os.path.join(target_path, 'visuals')
    if not os.path.exists(visuals_dir): return

    batch_dirs = sorted([d for d in os.listdir(visuals_dir) if d.isdigit()])
    if not batch_dirs: return

    # LPIPSは [0,1] 入力を想定 (normalize=Trueで内部正規化も可能だがToTensorで[0,1]にする)
    transform = transforms.Compose([transforms.ToTensor()])
    method_scores = {m: [] for m in methods}

    print(f"Processing LPIPS for {os.path.basename(target_path)}...")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path): continue
        
        try:
            gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
            # LPIPSは入力が[-1, 1]を期待する場合もあるが、torchmetricsの実装はnormalize=True引数で調整可能
            # ここではToTensorで[0,1]にし、metric初期化時にnormalize=Trueとしているため内部で[-1,1]にスケーリングされる
            
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    try:
                        m_img = transform(Image.open(m_path).convert('RGB')).unsqueeze(0).to(device)
                        score = lpips_metric(m_img, gt_img).item()
                        method_scores[m].append(score)
                    except: pass
        except: pass

    final_results = {}
    print(f"--- LPIPS Results ---")
    for m in methods:
        if method_scores[m]:
            avg = sum(method_scores[m]) / len(method_scores[m])
            final_results[m] = avg
            print(f"{m:25s}: {avg:.4f}")
            
    with open(os.path.join(target_path, "post_process_lpips.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    DATASET = "ffhq_demo"
    SNR_LABELS = ["-8","-7", "-6", "-5" ,"-4", "-3","-2"]
    RATE = 0.1
    EXP_FACTOR = 2.0
    GAMMA = 0.3
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    SEED = 22
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for snr_label in SNR_LABELS:
        folder = f"awgn_{snr_label}dB"
        exp = f"Retrans_rate_{RATE}_Comparison_both_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        calculate_lpips_for_snr(os.path.join(ROOT_DIR, DATASET, METHOD_PATH, folder, exp), device)