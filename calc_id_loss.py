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

def calculate_id_loss_from_disk(base_path):
    """
    Calculate ID Loss (1 - CosineSimilarity) for saved images.
    Also logs the top 3 images with the largest difference between Unc and Sem methods.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- [Load Model] ---
    print("Loading Face Recognition Model (InceptionResnetV1)...")
    id_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Methods to evaluate
    methods = [
        '1_JSCC_Init',
        '2_Phase1_Recon',
        '3_P2_perturbation_raw_Unc',
        '3_P2_perturbation_raw_Sem',
        '3_P2_temporal_raw_Unc',
        '3_P2_temporal_raw_Sem',
        '3_P2_Random'
    ]

    # Result containers
    id_scores = {m: [] for m in methods}     # Cosine Similarity
    id_losses = {m: [] for m in methods}     # 1 - Similarity

    # Containers for tracking differences [ (batch_id, diff, loss_unc, loss_sem), ... ]
    diff_records_perturbation = []
    diff_records_temporal = []

    visuals_dir = os.path.join(base_path, 'visuals')
    if not os.path.exists(visuals_dir):
        print(f"Error: {visuals_dir} does not exist.")
        return

    # Get batch directories
    batch_dirs = sorted([
        d for d in os.listdir(visuals_dir) 
        if os.path.isdir(os.path.join(visuals_dir, d)) and d.isdigit()
    ])

    to_tensor = transforms.ToTensor()

    print(f"Processing {len(batch_dirs)} samples from: {visuals_dir}")

    for b_dir in tqdm(batch_dirs):
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

            # Store current batch losses temporarily to compute differences
            current_batch_losses = {}

            # Process each method
            for m in methods:
                m_path = os.path.join(path, f'{m}.png')
                if os.path.exists(m_path):
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
                        
                        # Store for difference calculation
                        current_batch_losses[m] = loss
                else:
                    # If image missing, cannot compute difference
                    current_batch_losses[m] = None

            # --- Calculate Differences for Top 3 Tracking ---
            
            # 1. Perturbation: Unc vs Sem
            unc_key_p = '3_P2_perturbation_raw_Unc'
            sem_key_p = '3_P2_perturbation_raw_Sem'
            if current_batch_losses.get(unc_key_p) is not None and current_batch_losses.get(sem_key_p) is not None:
                u_val = current_batch_losses[unc_key_p]
                s_val = current_batch_losses[sem_key_p]
                diff = abs(u_val - s_val)
                diff_records_perturbation.append( (b_dir, diff, u_val, s_val) )

            # 2. Temporal: Unc vs Sem
            unc_key_t = '3_P2_temporal_raw_Unc'
            sem_key_t = '3_P2_temporal_raw_Sem'
            if current_batch_losses.get(unc_key_t) is not None and current_batch_losses.get(sem_key_t) is not None:
                u_val = current_batch_losses[unc_key_t]
                s_val = current_batch_losses[sem_key_t]
                diff = abs(u_val - s_val)
                diff_records_temporal.append( (b_dir, diff, u_val, s_val) )

        except Exception as e:
            print(f"Error processing batch {b_dir}: {e}")
            continue

    # --- [Summarize & Save] ---
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

    # --- [Print Top 3 Differences] ---
    print("\n" + "="*60)
    print(" TOP 3 SAMPLES WITH LARGEST DIFFERENCE (Unc vs Sem)")
    print("="*60)

    def print_top3(records, category_name):
        # Sort by difference (descending)
        # record structure: (batch_id, diff, loss_unc, loss_sem)
        sorted_recs = sorted(records, key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\n[{category_name}] Largest Abs Differences:")
        print(f"{'Batch ID':<10} | {'Diff':<10} | {'Unc Loss':<10} | {'Sem Loss':<10}")
        print("-" * 50)
        for r in sorted_recs:
            b_id, diff, u_l, s_l = r
            print(f"{b_id:<10} | {diff:.4f}     | {u_l:.4f}     | {s_l:.4f}")

    print_top3(diff_records_perturbation, "Perturbation (Raw)")
    print_top3(diff_records_temporal, "Temporal (Raw)")

    # Save results to JSON
    output_json = os.path.join(base_path, "post_process_id_loss.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {output_json}")

    return final_results

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

if __name__ == "__main__":
    # Adjust the target path as needed
    target_results_path = r"results_retrans_comparison/ffhq_demo/diffcom/djscc_2/awgn_-5dB/Retrans_rate_0.1_Comparison_both_exp2.0_gam0.3_zeta0.3_seed22"
    
    if os.path.exists(target_results_path):
        calculate_id_loss_from_disk(target_results_path)
    else:
        print(f"Path not found: {target_results_path}")