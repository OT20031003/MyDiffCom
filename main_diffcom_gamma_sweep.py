import argparse
import logging
import os
import os.path
import random
import shutil
import json
import copy
import sys
import signal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from transformers import AutoModel

# --- [FID calculation library] ---
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    IS_TORCHMETRICS_AVAILABLE = True
except ImportError:
    IS_TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. FID calculation will be skipped.")
# ----------------------------------

# Custom modules
import conditioning_method.diffcom as diffcom_module
from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
from data.datasets import get_test_loader
from guided_diffusion.measurement import get_operator
from guided_diffusion.noise_schedule import NoiseSchedule
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from utils.util import Config, MetricWrapper, DictAverageMeter
from utils import util, utils_logger, utils_model

# 提案手法(Proposed Method)において、Gamma(意味的重要度の重視割合)を変化させて比較する
EXPERIMENT_SUITE = [
    # 1. Gamma = 0.0
    #   (候補領域内で不確実性のみを基準に選択)
    {
        "name": "Proposed_Gamma_0.0",
        "mode": "rate", "value": 0.1, "expansion": 2.0, "gamma": 0.0, "basis": "semantic"
    },
    # 2. Gamma = 0.2
    {
        "name": "Proposed_Gamma_0.2",
        "mode": "rate", "value": 0.1, "expansion": 2.0, "gamma": 0.2, "basis": "semantic"
    },
    # 3. Gamma = 0.4
    {
        "name": "Proposed_Gamma_0.4",
        "mode": "rate", "value": 0.1, "expansion": 2.0, "gamma": 0.4, "basis": "semantic"
    },
    # 4. Gamma = 0.6
    {
        "name": "Proposed_Gamma_0.6",
        "mode": "rate", "value": 0.1, "expansion": 2.0, "gamma": 0.6, "basis": "semantic"
    },
    # 5. Gamma = 0.8
    {
        "name": "Proposed_Gamma_0.8",
        "mode": "rate", "value": 0.1, "expansion": 2.0, "gamma": 0.8, "basis": "semantic"
    },
    # 6. Gamma = 1.0
    #   (候補領域内で意味的重要性のみを基準に選択)
    {
        "name": "Proposed_Gamma_1.0",
        "mode": "rate", "value": 0.1, "expansion": 2.0, "gamma": 1.0, "basis": "semantic"
    }
]

# --- JSON Custom Encoder ---
class NumpyEncoder(json.JSONEncoder):
    """
    Encoder to convert NumPy data types (float32, float64, ndarray, etc.)
    to standard Python types for JSON saving.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# ---------------------------------------

# --- ViT Saliency Extractor (DINOv3 based) ---
class ViTSaliencyExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        
        # Hugging Face Model ID
        self.model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"

        print(f"Loading model {self.model_id} from Hugging Face Hub...")
        try:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                attn_implementation="eager" # Disable Flash Attention
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"\n[Error] Failed to load model: {e}")
            raise e

        # ImageNet Normalization params
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # ViT Settings
        self.patch_size = 16
        self.img_size = 224

    @torch.no_grad()
    def get_importance_map(self, images):
        """
        DINOv3 Attention Heatmap Logic
        images: Tensor [B, 3, H, W] (0~1)
        returns: Tensor [B, 1, H, W] (normalized 0~1)
        """
        B, C, H, W = images.shape
        
        # 1. Resize to 224x224 (Model Input Size)
        images_resized = F.interpolate(images, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # 2. Normalize (ImageNet mean/std)
        inputs = (images_resized - self.mean) / self.std
        
        # 3. Forward Pass & Get Attentions
        outputs = self.model(inputs, output_attentions=True)
        
        # Get attention (last layer)
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            last_layer_attn = outputs.attentions[-1]
        elif isinstance(outputs, tuple):
            last_layer_attn = outputs[-1]
        else:
            raise ValueError("Attention maps not found in model outputs.")

        # --- Aggregation (Batch support) ---
        # 1. Mean over heads -> [Batch, Total_Tokens, Total_Tokens]
        attn_mat = torch.mean(last_layer_attn, dim=1)
        
        # 2. Mean over query dim(dim=1) -> [Batch, Total_Tokens]
        patch_importance = torch.mean(attn_mat, dim=1)
        
        # 3. Extract image patches
        expected_patches = (self.img_size // self.patch_size) ** 2  # 196
        
        if patch_importance.shape[1] > expected_patches:
            patch_importance = patch_importance[:, -expected_patches:]
        
        # 4. Reshape to heatmap
        grid_size = int(np.sqrt(expected_patches)) # 14
        
        # [Batch, N] -> [Batch, 1, Grid, Grid]
        similarity_map = patch_importance.reshape(B, 1, grid_size, grid_size)
        
        # 5. Resize back to original resolution (H, W)
        importance_resized = F.interpolate(similarity_map, size=(H, W), mode='bilinear', align_corners=False)
        
        # 6. Min-Max Normalize per image in batch
        flat = importance_resized.flatten(2) # [B, 1, H*W]
        i_min = flat.min(2, keepdim=True)[0].unsqueeze(-1)
        i_max = flat.max(2, keepdim=True)[0].unsqueeze(-1)
        
        importance_normalized = (importance_resized - i_min) / (i_max - i_min + 1e-8)
        
        return importance_normalized

def compute_heuristic_importance_map(images, method='edge'):
    """
    Helper function to compute simple image processing based importance maps.
    Args:
        images: Tensor [B, 3, H, W] (0~1)
        method: 'edge' (Sobel) or 'variance' (Local Variance)
    Returns:
        Tensor [B, 1, H, W] (normalized 0~1)
    """
    B, C, H, W = images.shape
    device = images.device

    # Grayscale conversion (Rec. 601)
    # [B, 3, H, W] -> [B, 1, H, W]
    gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]

    if method == 'edge':
        # Sobel filter edge detection
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

        # Padding and convolution
        gx = F.conv2d(gray, kernel_x, padding=1)
        gy = F.conv2d(gray, kernel_y, padding=1)

        # Gradient magnitude
        feat_map = torch.sqrt(gx**2 + gy**2 + 1e-8)
    
    elif method == 'variance':
        # Local Variance (3x3 neighborhood)
        # E[x^2] - (E[x])^2
        kernel_size = 3
        padding = kernel_size // 2
        
        # Average Pooling to compute local mean
        mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding)
        mean_sq = F.avg_pool2d(gray**2, kernel_size=kernel_size, stride=1, padding=padding)
        
        feat_map = mean_sq - mean**2
        # Prevent negative values due to numerical errors
        feat_map = torch.clamp(feat_map, min=0.0)

    else:
        raise ValueError(f"Unknown heuristic method: {method}")

    # Min-Max Normalize per image
    flat = feat_map.flatten(2) # [B, 1, H*W]
    v_min = flat.min(2, keepdim=True)[0].unsqueeze(-1)
    v_max = flat.max(2, keepdim=True)[0].unsqueeze(-1)
    
    normalized_map = (feat_map - v_min) / (v_max - v_min + 1e-8)
    return normalized_map

def reconstruct_full_summary(history):
    """
    Recalculate average (summary) of all data from full history (list of dict).
    Aggregates according to structure changes in History (inside comparison_results).
    """
    if not history:
        return {}

    # Accumulator dict: { "metric_key": { "psnr": [val...], "lpips": [val...] } }
    accumulator = {}

    def add_values(meter_key, metrics_dict):
        if meter_key not in accumulator:
            accumulator[meter_key] = {}
        for m_name, m_val in metrics_dict.items():
            if isinstance(m_val, (int, float)):
                if m_name not in accumulator[meter_key]:
                    accumulator[meter_key][m_name] = []
                accumulator[meter_key][m_name].append(m_val)

    for record in history:
        # 1. jscc_init
        if 'jscc_init' in record:
            add_values('jscc_init', record['jscc_init'])
        
        # 2. phase1_recon
        if 'phase1' in record:
            add_values('phase1_recon', record['phase1'])
            
        # 3. comparison_results (Suite execution results)
        if 'comparison_results' in record:
            for method_name, content in record['comparison_results'].items():
                if 'metrics' in content:
                    add_values(method_name, content['metrics'])
        
        # 4. (Legacy support) modes / random
        if 'modes' in record:
            for u_mode, u_content in record['modes'].items():
                if 'results' in u_content:
                    for sub_key, strat_dict in u_content['results'].items():
                        for strat_name, metrics in strat_dict.items():
                            meter_key = f"{u_mode}_{sub_key}_{strat_name}"
                            add_values(meter_key, metrics)
        if 'random' in record:
            add_values('random', record['random'])

    # Calculate averages
    final_summary = {}
    for meter_key, metrics_list in accumulator.items():
        final_summary[meter_key] = {}
        for m_name, values in metrics_list.items():
            if values:
                final_summary[meter_key][m_name] = sum(values) / len(values)
                
    return final_summary

def simulate_semantic_retransmission(operator, input_image, measurement, uncertainty_map, 
                                     mode='rate', value=0.1, logger=None, vit_importance_map=None,
                                     expansion_factor=2.0, gamma=0.6, basis='uncertainty'):
    """
    Hybrid-Priority Retransmission Simulation (HPRS)
    """
    device = input_image.device
    channel_wrapper = operator.channel
    cand_mask_vis = None # Init visualization variable
    
    if not hasattr(channel_wrapper, 'shuffled_indices') or channel_wrapper.shuffled_indices is None:
        if logger: logger.warning("Channel indices not found. Is this run after observe? Skipping.")
        return measurement, 0.0, None, None, None

    saved_indices = channel_wrapper.shuffled_indices.to(device)
    saved_avg_pwr = channel_wrapper.avg_pwr
    
    with torch.no_grad():
        s_raw = operator.encode(input_image) 
        B, N_s = s_raw.shape
        
        if saved_indices.dim() == 1:
            indices_expanded = saved_indices.unsqueeze(0).expand(B, -1)
        else:
            indices_expanded = saved_indices
            
        s_shuffled = torch.gather(s_raw, 1, indices_expanded)
        
        pwr_tensor = torch.as_tensor(saved_avg_pwr, device=device).float()
        if pwr_tensor.numel() == 1 and pwr_tensor.item() == 0:
            pwr_tensor = torch.tensor(1.0, device=device)
            
        y_clean = s_shuffled / torch.sqrt(pwr_tensor)

    y_dirty = measurement['ofdm_sig']
    mask_vis = None
    mask_lat_spatial = None
    
    # Calculate spatial size of latent representation
    if hasattr(operator, 's_shape'):
        latent_H, latent_W = operator.s_shape[2], operator.s_shape[3]
        C_feat = operator.s_shape[1]
    else:
        latent_H, latent_W = input_image.shape[2] // 16, input_image.shape[3] // 16
        C_feat = s_raw.shape[1] // (latent_H * latent_W)

    # ---------------------------------------------------------------------
    # Mode 1: Oracle
    # ---------------------------------------------------------------------
    if mode == 'oracle':
        if y_dirty.shape != y_clean.shape:
             y_clean = y_clean.view(y_dirty.shape)
        diff = torch.abs(y_dirty - y_clean)
        diff_flat = diff.view(B, -1)
        k = int(diff_flat.shape[1] * value)
        if k < 1: k = 1
        top_val, _ = torch.topk(diff_flat, k, dim=1)
        thresh = top_val[:, -1].view(B, *([1]*(len(diff.shape)-1)))
        mask_for_y = (diff >= thresh).float()
        mask_vis = torch.zeros(B, 1, input_image.shape[2], input_image.shape[3]).to(device)

    # ---------------------------------------------------------------------
    # Mode 2: Random
    # ---------------------------------------------------------------------
    elif mode == 'random':
        u_map_lat = torch.rand(B, 1, latent_H, latent_W, device=device)
        u_flat = u_map_lat.view(B, -1)
        k = int(u_flat.shape[1] * value)
        if k < 1: k = 1
        top_val, _ = torch.topk(u_flat, k, dim=1)
        thresh = top_val[:, -1].view(B, 1, 1, 1)
        mask_lat_spatial = (u_map_lat >= thresh).float()
        
        mask_vis = F.interpolate(mask_lat_spatial, size=input_image.shape[-2:], mode='nearest')
        mask_expanded = mask_lat_spatial.repeat(1, C_feat, 1, 1)
        mask_flat = mask_expanded.view(B, -1)
        
        target_len = indices_expanded.shape[1]
        current_len = mask_flat.shape[1]
        
        if current_len != target_len:
            if current_len < target_len:
                padding = torch.zeros(B, target_len - current_len, device=device)
                mask_flat = torch.cat([mask_flat, padding], dim=1)
            else:
                mask_flat = mask_flat[:, :target_len]

        mask_shuffled = torch.gather(mask_flat, 1, indices_expanded)
        mask_for_y = mask_shuffled.view(y_dirty.shape)

    # ---------------------------------------------------------------------
    # Mode 3: Hybrid-Priority Retransmission (HPRS)
    # ---------------------------------------------------------------------
    else:
        # Switch map based on basis
        if basis in ['edge', 'variance']:
            u_map = compute_heuristic_importance_map(input_image, method=basis)
            u_map = u_map.to(device)
        else:
            if uncertainty_map is None:
                if logger: logger.warning("Uncertainty map is None in non-random mode!")
                return measurement, 0.0, None, None, None
            u_map = uncertainty_map.to(device)

        u_map_lat = F.adaptive_avg_pool2d(u_map, output_size=(latent_H, latent_W))
        
        if mode == 'rate':
            # === Step 1: Candidate Mask Generation ===
            u_flat = u_map_lat.view(B, -1)
            total_pixels = u_flat.shape[1]
            
            k_total = int(total_pixels * value)
            if k_total < 1: k_total = 1
            
            k_cand = int(total_pixels * value * expansion_factor)
            k_cand = min(k_cand, total_pixels)
            if k_cand < k_total: k_cand = k_total

            # Get candidate indices (high uncertainty)
            _, cand_indices = torch.topk(u_flat, k_cand, dim=1)

            # Candidate region mask (for visualization)
            cand_mask_flat = torch.zeros_like(u_flat)
            cand_mask_flat.scatter_(1, cand_indices, 1.0)
            cand_mask_lat = cand_mask_flat.view(B, 1, latent_H, latent_W)
            cand_mask_vis = F.interpolate(cand_mask_lat, size=input_image.shape[-2:], mode='nearest')

            # === Step 2: Budget Division ===
            k_sem = int(k_total * gamma)
            k_struct = k_total - k_sem
            
            # Prepare ViT map
            if vit_importance_map is not None:
                vit_lat = F.adaptive_avg_pool2d(vit_importance_map.to(device), output_size=(latent_H, latent_W))
                vit_flat = vit_lat.view(B, -1)
            else:
                # Fallback to u_flat if ViT missing or gamma=0
                vit_flat = u_flat

            # === Step 3: Selection within Candidates ===
            gathered_vit = torch.gather(vit_flat, 1, cand_indices)

            # --- Step 3-A: Semantic Slot (Top k_sem) ---
            _, sort_idx_local = torch.sort(gathered_vit, descending=True, dim=1)
            idx_sem_local = sort_idx_local[:, :k_sem]

            # --- Step 3-B: Structural Slot (Random k_struct from remaining) ---
            if k_struct > 0:
                idx_remain_local = sort_idx_local[:, k_sem:]
                n_remain = idx_remain_local.shape[1]
                
                if n_remain > 0:
                    rand_perm = torch.rand(B, n_remain, device=device).argsort(dim=1)
                    idx_rand_local = torch.gather(idx_remain_local, 1, rand_perm[:, :k_struct])
                    final_local_indices = torch.cat([idx_sem_local, idx_rand_local], dim=1)
                else:
                    final_local_indices = idx_sem_local
            else:
                final_local_indices = idx_sem_local

            # === Step 4: Mapping to Global Indices ===
            final_global_indices = torch.gather(cand_indices, 1, final_local_indices)

            # Create Mask
            mask_flat_spatial = torch.zeros_like(u_flat)
            mask_flat_spatial.scatter_(1, final_global_indices, 1.0)
            mask_lat_spatial = mask_flat_spatial.view(B, 1, latent_H, latent_W)

        else:
            # Conventional Threshold mode
            mask_lat_spatial = (u_map_lat > value).float()

        # Reshape and Apply Mask
        mask_vis = F.interpolate(mask_lat_spatial, size=input_image.shape[-2:], mode='nearest')
        mask_expanded = mask_lat_spatial.repeat(1, C_feat, 1, 1)
        mask_flat = mask_expanded.view(B, -1)
        
        target_len = indices_expanded.shape[1]
        current_len = mask_flat.shape[1]
        
        if current_len != target_len:
            if current_len < target_len:
                padding = torch.zeros(B, target_len - current_len, device=device)
                mask_flat = torch.cat([mask_flat, padding], dim=1)
            else:
                mask_flat = mask_flat[:, :target_len]

        mask_shuffled = torch.gather(mask_flat, 1, indices_expanded)
        mask_for_y = mask_shuffled.view(y_dirty.shape)

    retransmission_ratio = mask_for_y.float().mean().item()
    
    high_snr_value = 20.0
    with torch.no_grad():
        s_high = operator.encode(input_image, snr_override=high_snr_value)
        cof_for_forward = measurement.get('cof_est', None)
        y_high = operator.forward(s_high, cof=cof_for_forward)
    
    if y_high.shape != y_dirty.shape:
        y_high = y_high.view(y_dirty.shape)

    new_measurement = copy.deepcopy(measurement)
    new_measurement['retrans_sig'] = y_high
    new_measurement['retrans_mask'] = mask_for_y
    
    return new_measurement, retransmission_ratio, mask_vis, mask_lat_spatial, cand_mask_vis

def simulate_hybrid_global_random(operator, input_image, measurement, mode='hybrid_global', 
                                  value=0.1, logger=None, vit_importance_map=None,
                                  gamma=0.5, basis='semantic'):
    """
    New Baseline: Hybrid Global Random.
    Mixes Top-K (Semantic or Edge) with Random selection from the ENTIRE latent space.
    """
    device = input_image.device
    channel_wrapper = operator.channel
    
    # 1. Channel Indices Setup
    if not hasattr(channel_wrapper, 'shuffled_indices') or channel_wrapper.shuffled_indices is None:
        if logger: logger.warning("Channel indices not found. Is this run after observe? Skipping.")
        return measurement, 0.0, None, None, None

    saved_indices = channel_wrapper.shuffled_indices.to(device)
    saved_avg_pwr = channel_wrapper.avg_pwr
    
    with torch.no_grad():
        s_raw = operator.encode(input_image) 
        B, N_s = s_raw.shape
        
        if saved_indices.dim() == 1:
            indices_expanded = saved_indices.unsqueeze(0).expand(B, -1)
        else:
            indices_expanded = saved_indices
            
        s_shuffled = torch.gather(s_raw, 1, indices_expanded)
        
        pwr_tensor = torch.as_tensor(saved_avg_pwr, device=device).float()
        if pwr_tensor.numel() == 1 and pwr_tensor.item() == 0:
            pwr_tensor = torch.tensor(1.0, device=device)
            
        y_clean = s_shuffled / torch.sqrt(pwr_tensor)

    y_dirty = measurement['ofdm_sig']
    
    # Latent spatial size calculation
    if hasattr(operator, 's_shape'):
        latent_H, latent_W = operator.s_shape[2], operator.s_shape[3]
        C_feat = operator.s_shape[1]
    else:
        latent_H, latent_W = input_image.shape[2] // 16, input_image.shape[3] // 16
        C_feat = s_raw.shape[1] // (latent_H * latent_W)

    # -----------------------------------------------------------
    # Logic: Global Hybrid (Priority + Random from Remainder)
    # -----------------------------------------------------------
    
    # 1. Get Base Importance Map
    if basis in ['edge', 'variance']:
        map_spatial = compute_heuristic_importance_map(input_image, method=basis)
        map_spatial = map_spatial.to(device)
    elif basis == 'uncertainty':
        # [UPDATED] Handling uncertainty basis for hybrid random
        if vit_importance_map is None:
             if logger: logger.warning("Uncertainty map is None for hybrid_global uncertainty mode. Using random.")
             map_spatial = torch.rand(B, 1, input_image.shape[2], input_image.shape[3], device=device)
        else:
             map_spatial = vit_importance_map.to(device)
    else:
        # basis='semantic'
        if vit_importance_map is None:
             if logger: logger.warning("ViT map is None for hybrid_global semantic mode. Using random.")
             map_spatial = torch.rand(B, 1, input_image.shape[2], input_image.shape[3], device=device)
        else:
             map_spatial = vit_importance_map.to(device)

    # 2. Resize to Latent & Flatten
    map_lat = F.adaptive_avg_pool2d(map_spatial, output_size=(latent_H, latent_W))
    map_flat = map_lat.view(B, -1) # [B, N_lat]
    total_pixels = map_flat.shape[1]
    
    # 3. Budget Calculation
    k_total = int(total_pixels * value)
    if k_total < 1: k_total = 1
    
    k_priority = int(k_total * gamma)
    k_random = k_total - k_priority
    
    # 4. Selection Process
    # Sort entire latent space by importance
    _, sorted_indices = torch.sort(map_flat, descending=True, dim=1)
    
    # 4-A: Priority Part (Top-k_priority)
    indices_priority = sorted_indices[:, :k_priority]
    
    # 4-B: Random Part (From the rest)
    if k_random > 0:
        indices_remainder = sorted_indices[:, k_priority:]
        n_remain = indices_remainder.shape[1]
        
        if n_remain > 0:
            # Randomly permute the remainder and pick k_random
            rand_perm = torch.rand(B, n_remain, device=device).argsort(dim=1)
            indices_random_selected = torch.gather(indices_remainder, 1, rand_perm[:, :k_random])
            
            final_indices = torch.cat([indices_priority, indices_random_selected], dim=1)
        else:
            final_indices = indices_priority
    else:
        final_indices = indices_priority

    # 5. Mask Construction
    mask_flat_spatial = torch.zeros_like(map_flat)
    mask_flat_spatial.scatter_(1, final_indices, 1.0)
    mask_lat_spatial = mask_flat_spatial.view(B, 1, latent_H, latent_W)
    
    # Visualization
    mask_vis = F.interpolate(mask_lat_spatial, size=input_image.shape[-2:], mode='nearest')
    
    # 6. Apply to Channels (Expand -> Flatten -> Shuffle)
    mask_expanded = mask_lat_spatial.repeat(1, C_feat, 1, 1)
    mask_flat = mask_expanded.view(B, -1)
    
    target_len = indices_expanded.shape[1]
    current_len = mask_flat.shape[1]
    
    if current_len != target_len:
        if current_len < target_len:
            padding = torch.zeros(B, target_len - current_len, device=device)
            mask_flat = torch.cat([mask_flat, padding], dim=1)
        else:
            mask_flat = mask_flat[:, :target_len]

    mask_shuffled = torch.gather(mask_flat, 1, indices_expanded)
    mask_for_y = mask_shuffled.view(y_dirty.shape)
    
    retransmission_ratio = mask_for_y.float().mean().item()
    
    # 7. Encode High SNR Signal (Simulation)
    high_snr_value = 20.0
    with torch.no_grad():
        s_high = operator.encode(input_image, snr_override=high_snr_value)
        cof_for_forward = measurement.get('cof_est', None)
        y_high = operator.forward(s_high, cof=cof_for_forward)
    
    if y_high.shape != y_dirty.shape:
        y_high = y_high.view(y_dirty.shape)

    new_measurement = copy.deepcopy(measurement)
    new_measurement['retrans_sig'] = y_high
    new_measurement['retrans_mask'] = mask_for_y
    
    return new_measurement, retransmission_ratio, mask_vis, mask_lat_spatial, None


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom_-4.yaml', help="Path to option YMAL file.")
    parser.add_argument("--retrans_mode", type=str, default='rate', choices=['rate', 'threshold', 'oracle', 'hybrid_global'])
    parser.add_argument("--retrans_value", type=float, default=0.1)
    parser.add_argument("--expansion_factor", type=float, default=2.0)
    parser.add_argument("--retrans_gamma", type=float, default=0.9)
    parser.add_argument("--retrans_basis", type=str, default='semantic', choices=['uncertainty', 'semantic', 'both', 'edge', 'variance'])
    parser.add_argument("--resume_index", type=int, default=0, help="Index to resume processing from (0-based).")
    parser.add_argument("--enable_random", action='store_true', help="Enable random retransmission baseline.")
    
    # --- Comparison Experiment Suite Flag ---
    parser.add_argument("--run_suite", action='store_true', help="Run the full comparison suite (Phase 1 once, multiple Phase 2).")
    # ---------------------------------------------

    args = parser.parse_args()
    
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    
    config.retrans_mode = args.retrans_mode
    config.retrans_value = args.retrans_value
    config.expansion_factor = args.expansion_factor
    config.retrans_gamma = args.retrans_gamma
    config.retrans_basis = args.retrans_basis
    config.resume_index = args.resume_index
    config.enable_random = args.enable_random
    config.run_suite = args.run_suite # Added to Config
    
    cond_config = Config(config.getattr('diffcom_series'))
    conditioning_method = Config(cond_config.getattr(config.conditioning_method))
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt
    config.skip = cond_config.num_train_timesteps // cond_config.iter_num
    config.sigma = np.sqrt(1.0 / (2 * 10 ** (config.CSNR / 10)))

    config.model_zoo = os.path.join(config.cwd, 'model_zoo')
    config.testsets = os.path.join(config.cwd, 'testsets')
    config.results = os.path.join(config.cwd, 'results_retrans_comparison')
    config.results = os.path.join(config.results, config.testset_name)
    config.results = os.path.join(config.results, config.conditioning_method)

    if config.operator_name == 'djscc':
        config.results = os.path.join(config.results, config.operator_name + '_{}'.format(config.djscc['channel_num']))
    
    config.results = os.path.join(config.results, f'{config.channel_type}_{config.CSNR.__str__().zfill(2)}dB')
    
    u_mode = cond_config.uncertainty_mode
    u_mode_str = "Comparison" if isinstance(u_mode, list) else str(u_mode)
    
    config.result_name = f'Retrans_{config.retrans_mode}_{config.retrans_value}_{u_mode_str}_{config.retrans_basis}'
    # Include gamma in filename
    config.result_name += f'_exp{config.expansion_factor}_gam{config.retrans_gamma}_zeta{conditioning_method.zeta}_seed{config.seed}'
    
    config.model_path = os.path.join(config.model_zoo, config.model_name + '.pt')
    config.testsets_path = os.path.join(config.testsets, config.testset_name)
    config.save_path = os.path.join(config.results, config.result_name)
    
    
    util.mkdir(config.save_path)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    return config

def run_diffusion_process(config, noise_schedule, unet, diffusion, operator, cond_method, 
                          measurement, input_image, device, phase_name="Phase1"):
    
    ofdm_config = Config(config.ofdm_tdl)
    
    x_ref = measurement['x_mse'] 
    
    x_init = noise_schedule.sqrt_alphas_cumprod[noise_schedule.t_start] * (2 * x_ref - 1) + \
             noise_schedule.sqrt_1m_alphas_cumprod[noise_schedule.t_start] * torch.randn_like(input_image)

    if config.conditioning_method == 'blind_diffcom':
        power = torch.exp(-torch.arange(ofdm_config.L).float() / ofdm_config.decay).view(1, 1, ofdm_config.L).to(device)
        power = power / sum(power)
        cof_init_real = torch.randn_like(measurement['cof_gt'][..., :ofdm_config.L]) * power
        cof_init_imag = torch.randn_like(measurement['cof_gt'][..., :ofdm_config.L]) * power
        cof_init = cof_init_real + 1j * cof_init_imag
        cof_init = noise_schedule.sqrt_alphas_cumprod[noise_schedule.t_start] * cof_init + \
                   noise_schedule.sqrt_1m_alphas_cumprod[noise_schedule.t_start] * torch.randn_like(cof_init)
    else:
        cof_gt = 0 + 0j
        cof_init = measurement['cof_est']
        power = None

    seq = noise_schedule.seq
    x_t = x_init
    h_t = cof_init
    
    pbar = tqdm(range(len(seq)), ncols=120, desc=f"{phase_name}", leave=False)
    
    for i in pbar:
        t_step = seq[i]
        
        x_0_hat, h_0_hat, x_t_prev, h_t_prev, norm = cond_method(
            config, i, noise_schedule,
            x_init if i == 0 else x_t,
            cof_init if i == 0 else h_t,
            power if config.conditioning_method == 'blind_diffcom' else None,
            measurement, unet, diffusion, operator, 
            loss_wrapper=None,
            last_timestep=(seq[i] == seq[-1])
        )
        
        x_t = x_t_prev
        h_t = h_t_prev
        
    x_recon = (x_t / 2 + 0.5)

    raw_maps = diffcom_module.latest_uncertainty_map
    final_uncertainty_maps = {}

    if isinstance(raw_maps, dict):
        for key, val in raw_maps.items():
            if isinstance(val, dict):
                final_uncertainty_maps[key] = {}
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, torch.Tensor):
                        final_uncertainty_maps[key][sub_k] = sub_v.detach().clone()
                    else:
                        final_uncertainty_maps[key][sub_k] = sub_v
            elif isinstance(val, torch.Tensor):
                final_uncertainty_maps[key] = val.detach().clone()
            else:
                final_uncertainty_maps[key] = val

    diffcom_module.latest_uncertainty_map = {}
    return x_recon.detach(), final_uncertainty_maps

def p_sample_loop(config, noise_schedule, unet, diffusion, operator, cond_method, dataloader, device, logger):
    # --- [Updated] Experiment List Selection ---
    if config.run_suite:
        target_experiments = EXPERIMENT_SUITE
        logger.info(f"Running Experiment Suite with {len(target_experiments)} methods.")
    else:
        # Single Run (Legacy mode)
        target_experiments = [{
            "name": "Single_Run",
            "mode": config.retrans_mode,
            "value": config.retrans_value,
            "expansion": config.expansion_factor,
            "gamma": config.retrans_gamma,
            "basis": config.retrans_basis
        }]
    # -----------------------------------------
    
    metric_wrapper = MetricWrapper().to(device)
    loss_wrapper = ConsistencyLoss(config, device)
    
    def format_metrics(m):
        s = f"PSNR: {m.get('psnr', 0):.2f}dB"
        if 'lpips' in m: s += f" | LPIPS: {m['lpips']:.4f}"
        if 'fid' in m:   s += f" | FID: {m['fid']}"
        return s

    # --- [ViT Initialization Check] ---
    # Load ViT if any suite experiment uses semantic/both/hybrid with semantic
    use_vit = False
    for exp in target_experiments:
        if exp['basis'] in ['semantic', 'both'] or (exp.get('mode') == 'hybrid_global' and exp['basis'] == 'semantic'):
            use_vit = True
            break
    
    vit_extractor = None
    if use_vit:
        try:
            vit_extractor = ViTSaliencyExtractor(device=device)
            logger.info("[ViT] Saliency Extractor Initialized Successfully.")
        except Exception as e:
            logger.warning(f"[ViT] Init Failed: {e}. Semantic modes may fail.")
    else:
        logger.info("[ViT] Saliency Extractor Skipped (No semantic methods in suite).")
    # -----------------------------

    results_meters = {}
    fid_meters = {}
    
    def get_meter(key):
        if key not in results_meters:
            results_meters[key] = DictAverageMeter()
        return results_meters[key]

    def update_fid(key, real_img, fake_img):
        global IS_TORCHMETRICS_AVAILABLE
        if not IS_TORCHMETRICS_AVAILABLE: return
        if key not in fid_meters:
            fid_meters[key] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        real_norm = torch.clamp(real_img, 0, 1)
        fake_norm = torch.clamp(fake_img, 0, 1)
        fid_meters[key].update(real_norm, real=True)
        fid_meters[key].update(fake_norm, real=False)

    def wrapped_cond_method(*args, **kwargs):
        kwargs['loss_wrapper'] = loss_wrapper
        return cond_method(*args, **kwargs)

    json_filename = f"{config.result_name}.json"
    json_path = os.path.join(config.save_path, json_filename)
    import gc

    all_results_history = []
    
    # Load history
    if config.resume_index > 0 and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
                all_results_history = existing_data.get('history', [])
            logger.info(f"Loaded {len(all_results_history)} previous records.")
        except Exception as e:
            logger.warning(f"Failed to load existing JSON: {e}.")

    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        for idx, batch in enumerate(dataloader):
            if idx < config.resume_index:
                continue
            
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]

            # --- [ViT Batch Calculation] ---
            vit_map = None
            if vit_extractor is not None:
                try:
                    vit_map = vit_extractor.get_importance_map(input_image).detach()
                except Exception as e:
                    logger.warning(f"Batch {idx}: ViT calculation failed ({e}).")
            # -----------------------------------------------------
            
            # --- Phase 1 (Common): Run Once ---
            torch.manual_seed(config.seed + idx)
            measurement_phase1 = operator.observe_and_transpose(input_image)
            
            metrics_jscc_p1 = metric_wrapper(measurement_phase1['x_mse'].detach(), input_image)
            get_meter('jscc_init').update(metrics_jscc_p1)
            
            log_msg_jscc = f"Batch {idx+1}/{len(dataloader)} | [Phase 1 Init] | {format_metrics(metrics_jscc_p1)}"
            logger.info(log_msg_jscc)

            save_dir = os.path.join(config.save_path, 'visuals', str(idx))
            util.mkdir(save_dir)
            torchvision.utils.save_image(input_image[0].cpu(), os.path.join(save_dir, '0_GT.png'))
            torchvision.utils.save_image(measurement_phase1['x_mse'][0].cpu(), os.path.join(save_dir, '1_JSCC_Init.png'))
            
            if vit_map is not None:
                v_vis = vit_map[0, 0].cpu().numpy()
                plt.imsave(os.path.join(save_dir, 'ViT_Importance.png'), v_vis, cmap='jet')

            batch_record = {
                "batch_idx": idx + 1,
                "filename": names[0],
                "jscc_init": {k: float(v) for k, v in metrics_jscc_p1.items()},
                "comparison_results": {} # Container for results
            }

            torch.manual_seed(config.seed + idx)
            diffcom_module.latest_uncertainty_map = {} 

            x_recon_p1, uncertainty_container_p1 = run_diffusion_process(
                config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                measurement_phase1, input_image, device, phase_name="Phase1"
            )
            
            metrics_p1 = metric_wrapper(x_recon_p1.detach(), input_image)
            get_meter('phase1_recon').update(metrics_p1)
            batch_record['phase1'] = {k: float(v) for k, v in metrics_p1.items()}
            update_fid('phase1', input_image, x_recon_p1.detach())

            logger.info(f"  -> Phase 1 Done | {format_metrics(metrics_p1)}")
            torchvision.utils.save_image(x_recon_p1[0].cpu(), os.path.join(save_dir, f'2_Phase1_Recon.png'))
            
            # Extract Phase 1 uncertainty (use first mode)
            raw_uncertainty_map = None
            if uncertainty_container_p1:
                first_key = list(uncertainty_container_p1.keys())[0]
                raw_uncertainty_map = uncertainty_container_p1[first_key].get('raw')
            
            # --- [Updated] Phase 2 Loop (In-Memory) ---
            for exp in target_experiments:
                exp_name = exp["name"]
                
                # Parameters
                c_mode = exp["mode"]
                c_val = exp["value"]
                c_exp = exp["expansion"]
                c_gam = exp["gamma"]
                c_basis = exp["basis"]
                
                # Determine current vit usage
                current_vit = vit_map if c_basis in ['semantic', 'both'] else None

                # Simulation Execution
                if c_mode == 'hybrid_global':
                    # Determine input map for hybrid_global function based on basis
                    hybrid_input_map = None
                    if c_basis == 'uncertainty':
                        hybrid_input_map = raw_uncertainty_map
                    elif c_basis in ['semantic', 'both']:
                        hybrid_input_map = vit_map

                    # Call New Function
                    meas_p2, ratio, mask_vis, _, cand_vis = simulate_hybrid_global_random(
                        operator, input_image, measurement_phase1,
                        mode=c_mode,
                        value=c_val,
                        vit_importance_map=hybrid_input_map, # Pass selected map
                        gamma=c_gam,
                        basis=c_basis,
                        logger=logger
                    )
                else:
                    # Call Existing Function
                    meas_p2, ratio, mask_vis, _, cand_vis = simulate_semantic_retransmission(
                        operator, input_image, measurement_phase1, 
                        raw_uncertainty_map, 
                        mode=c_mode, 
                        value=c_val,
                        vit_importance_map=current_vit,
                        expansion_factor=c_exp,
                        basis=c_basis,
                        gamma=c_gam,
                        logger=logger
                    )

                # Phase 2 Diffusion
                torch.manual_seed(config.seed + idx)
                p2_phase_name = f"P2_{exp_name}"
                
                x_recon_p2, _ = run_diffusion_process(
                    config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                    meas_p2, input_image, device, phase_name=p2_phase_name
                )
                
                # Eval & Save
                metrics_p2 = metric_wrapper(x_recon_p2.detach(), input_image)
                get_meter(exp_name).update(metrics_p2)
                update_fid(exp_name, input_image, x_recon_p2.detach())
                
                # Record
                batch_record["comparison_results"][exp_name] = {
                    "metrics": {k: float(v) for k, v in metrics_p2.items()},
                    "ratio": ratio,
                    "params": exp
                }

                logger.info(f"    [{exp_name}] Ratio: {ratio:.2%} | {format_metrics(metrics_p2)}")
                
                # Image Save
                torchvision.utils.save_image(x_recon_p2[0].cpu(), os.path.join(save_dir, f'3_{exp_name}.png'))
                if mask_vis is not None:
                    plt.imsave(os.path.join(save_dir, f'Mask_{exp_name}.png'), mask_vis[0, 0].cpu().numpy(), cmap='gray')
                if cand_vis is not None:
                    plt.imsave(os.path.join(save_dir, f'Cand_{exp_name}.png'), cand_vis[0, 0].cpu().numpy(), cmap='gray')

                # Free Memory
                del meas_p2, x_recon_p2
                torch.cuda.empty_cache()
            
            # -----------------------------------------------------

            all_results_history.append(batch_record)
            logger.info('-' * 80)
            
            del input_image, measurement_phase1, x_recon_p1, uncertainty_container_p1
            gc.collect() 
            torch.cuda.empty_cache() 

    except KeyboardInterrupt:
        logger.info("\n[!] Process Interrupted by User. Saving current results...")
    except Exception as e:
        logger.error(f"\n[!] Unexpected Error: {e}. Saving current results...")
        import traceback
        traceback.print_exc()
    finally:
        # Create & Save Summary
        current_session_summary = {}
        for k, meter in results_meters.items():
            current_session_summary[k] = meter.avg
        
        # Calculate FID
        if IS_TORCHMETRICS_AVAILABLE and len(fid_meters) > 0:
            logger.info("Calculating FID scores...")
            try:
                for k, fid_obj in fid_meters.items():
                    if k not in current_session_summary: current_session_summary[k] = {}
                    try:
                        score = fid_obj.compute().item()
                        current_session_summary[k]['fid'] = score
                        logger.info(f"  -> FID [{k}]: {score:.4f}")
                    except Exception as e:
                        logger.warning(f"FID Error {k}: {e}")
            except KeyboardInterrupt:
                logger.warning("FID calc interrupted.")
        
        # Merge with History
        if len(all_results_history) > 0:
            logger.info("Recalculating global summary...")
            final_summary = reconstruct_full_summary(all_results_history)
            
            # Merge Session FIDs
            for k, v in current_session_summary.items():
                if 'fid' in v and k in final_summary:
                    final_summary[k]['fid'] = v['fid']
        else:
            final_summary = current_session_summary

        output_data = {"summary": final_summary, "history": all_results_history}

        if len(all_results_history) > 0:
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Saved results to {json_path}")
        
        # Final Log
        logger.info("=== Final Comparison Summary ===")
        for k in sorted(final_summary.keys()):
            logger.info(f"{k:30s} | {format_metrics(final_summary[k])}")

    return results_meters

def main():
    config = parse_args_and_config()
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    config.device = device

    logger_name = config.result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(config.save_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    
    dataloader = get_test_loader(config.testsets_path, batch_size=config.batch_size, shuffle=False)
    
    if config.model_name == 'ffhq_10m':
        model_config = dict(
            model_path=config.model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        )
    elif config.model_name == 'lsun_uncond_100M_1200K_bs128': 
        model_config = dict(
            model_path=config.model_path,
            image_size=256,
            num_channels=128,
            num_res_blocks=2,
            num_heads=1,
            learn_sigma=True,
            use_scale_shift_norm=False,
            attention_resolutions="16",
            diffusion_steps=1000,
            noise_schedule="linear",
            rescale_learned_sigmas=False,
            rescale_timesteps=False,
        )
    else:
        model_config = dict(
            model_path=config.model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    
    args_unet = utils_model.create_argparser(model_config).parse_args([])
    unet, diffusion = create_model_and_diffusion(
        **args_to_dict(args_unet, model_and_diffusion_defaults().keys()))
    unet.load_state_dict(torch.load(args_unet.model_path, map_location="cpu"))
    unet.eval()
    unet = unet.to(device)

    shutil.copyfile(config.opt, os.path.join(config.save_path, os.path.basename('config.yaml')))

    operator = get_operator(config.operator_name, config=config, logger=logger, device=device)
    operator.model = operator.model.to(device)
    ns = NoiseSchedule(config, logger, device)

    cond_method = get_conditioning_method(name=config.conditioning_method)
    cond_method = cond_method.conditioning
    
    p_sample_loop(config, ns, unet, diffusion, operator, cond_method, dataloader, device, logger)


if __name__ == '__main__':
    main()