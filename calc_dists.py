import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# DISTSのインポート試行
try:
    from DISTS_pytorch import DISTS
except ImportError:
    print("Error: 'DISTS_pytorch' module not found. Please install via 'pip install DISTS-pytorch'.")
    exit()

def calculate_dists_for_snr(target_path, device):
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

    # DISTS 初期化
    dists_metric = DISTS().to(device)
    
    visuals_dir = os.path.join(target_path, 'visuals')
    if not os.path.exists(visuals_dir): return

    batch_dirs = sorted([d for d in os.listdir(visuals_dir) if d.isdigit()])
    if not batch_dirs: return

    transform = transforms.Compose([transforms.ToTensor()])
    method_scores = {m: [] for m in methods}

    print(f"Processing DISTS for {os.path.basename(target_path)}...")

    for b_dir in tqdm(batch_dirs, leave=False):
        path = os.path.join(visuals_dir, b_dir)
        gt_path = os.path.join(path, '0_GT.png')
        if not os.path.exists(gt_path): continue
        
        try:
            gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
            
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
                    try:
                        m_img = transform(Image.open(m_path).convert('RGB')).unsqueeze(0).to(device)
                        score = dists_metric(m_img, gt_img).item()
                        method_scores[m].append(score)
                    except: pass
        except: pass

    final_results = {}
    print(f"--- DISTS Results ---")
    for m in methods:
        if method_scores[m]:
            avg = sum(method_scores[m]) / len(method_scores[m])
            final_results[m] = avg
            print(f"{m:25s}: {avg:.4f}")

    with open(os.path.join(target_path, "post_process_dists.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    DATASET = "ffhq_demo"
    #DATASET = "imagenet"
    SNR_LABELS = ["-8","-7", "-6", "-5" ,"-4", "-3","-2"]
    RATE = 0.1
    EXP_FACTOR = 10.0
    GAMMA = 1.0
    ROOT_DIR = "results_retrans_comparison"
    METHOD_PATH = "diffcom/djscc_2"
    ZETA = 0.3
    MODE = "semantic"
    SEED = 22
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for snr_label in SNR_LABELS:
        folder = f"awgn_{snr_label}dB"
        exp = f"Retrans_rate_{RATE}_Comparison_{MODE}_exp{EXP_FACTOR}_gam{GAMMA}_zeta{ZETA}_seed{SEED}"
        calculate_dists_for_snr(os.path.join(ROOT_DIR, DATASET, METHOD_PATH, folder, exp), device)