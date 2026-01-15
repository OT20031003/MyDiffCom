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

# --- [FID計算用のライブラリ] ---
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    IS_TORCHMETRICS_AVAILABLE = True
except ImportError:
    IS_TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. FID calculation will be skipped.")
# ----------------------------------

# カスタムモジュール
import conditioning_method.diffcom as diffcom_module
from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
from data.datasets import get_test_loader
from guided_diffusion.measurement import get_operator
from guided_diffusion.noise_schedule import NoiseSchedule
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from utils.util import Config, MetricWrapper, DictAverageMeter
from utils import util, utils_logger, utils_model

# --- JSON保存用カスタムエンコーダー ---
class NumpyEncoder(json.JSONEncoder):
    """
    NumPyのデータ型(float32, float64, ndarray等)を
    標準のPython型に変換してJSON保存するためのエンコーダー
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

# --- ViT重要度抽出クラス (DINOv3 based) ---
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
                attn_implementation="eager" # Flash Attention無効化
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"\n【エラー】モデルのロードに失敗しました: {e}")
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
        
        # アテンション取得 (最終層)
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            last_layer_attn = outputs.attentions[-1]
        elif isinstance(outputs, tuple):
            last_layer_attn = outputs[-1]
        else:
            raise ValueError("Attention maps not found in model outputs.")

        # --- 集計処理 (Batch対応) ---
        # 1. ヘッド方向の平均 -> [Batch, Total_Tokens, Total_Tokens]
        attn_mat = torch.mean(last_layer_attn, dim=1)
        
        # 2. Query方向(dim=1)の平均 -> [Batch, Total_Tokens]
        patch_importance = torch.mean(attn_mat, dim=1)
        
        # 3. 画像パッチのみを抽出
        expected_patches = (self.img_size // self.patch_size) ** 2  # 196
        
        if patch_importance.shape[1] > expected_patches:
            patch_importance = patch_importance[:, -expected_patches:]
        
        # 4. ヒートマップ整形
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

def reconstruct_full_summary(history):
    """
    全履歴データ(list of dict)から、全データの平均値(summary)を再計算する。
    Historyのネスト構造を、Summaryのフラットなキー構造(phase1_recon, perturbation_raw_Uncなど)に変換して集計する。
    """
    if not history:
        return {}

    # 集計用辞書: { "metric_key": { "psnr": [val, val...], "lpips": [val...] } }
    accumulator = {}

    def add_values(meter_key, metrics_dict):
        if meter_key not in accumulator:
            accumulator[meter_key] = {}
        for m_name, m_val in metrics_dict.items():
            # 数値型のみ集計対象にする
            if isinstance(m_val, (int, float)):
                if m_name not in accumulator[meter_key]:
                    accumulator[meter_key][m_name] = []
                accumulator[meter_key][m_name].append(m_val)

    for record in history:
        # 1. jscc_init
        if 'jscc_init' in record:
            add_values('jscc_init', record['jscc_init'])
        
        # 2. phase1 -> Summaryキーは 'phase1_recon'
        if 'phase1' in record:
            add_values('phase1_recon', record['phase1'])
            
        # 3. random
        if 'random' in record:
            add_values('random', record['random'])
            
        # 4. modes (再送モード) -> Summaryキーは 'perturbation_raw_Unc' 等の形式
        # record['modes'] = { "perturbation": { "results": { "raw": { "Unc": {...}, "Sem": {...} } } } }
        if 'modes' in record:
            for u_mode, u_content in record['modes'].items():
                if 'results' in u_content:
                    for sub_key, strat_dict in u_content['results'].items():
                        for strat_name, metrics in strat_dict.items():
                            # キーの構築: 例 "perturbation_raw_Unc"
                            meter_key = f"{u_mode}_{sub_key}_{strat_name}"
                            add_values(meter_key, metrics)

    # 平均値の計算
    final_summary = {}
    for meter_key, metrics_list in accumulator.items():
        final_summary[meter_key] = {}
        for m_name, values in metrics_list.items():
            if values:
                final_summary[meter_key][m_name] = sum(values) / len(values)
                
    return final_summary

def simulate_semantic_retransmission(operator, input_image, measurement, uncertainty_map, 
                                     mode='rate', value=0.1, logger=None, vit_importance_map=None,
                                     expansion_factor=2.0, gamma=0.6):
    """
    Hybrid-Priority Retransmission Simulation (HPRS)
    
    Args:
        gamma (float): Ratio of budget for Semantic Priority (0.0 ~ 1.0). 
                       The rest (1.0 - gamma) is used for Random Structural Sampling.
    """
    device = input_image.device
    channel_wrapper = operator.channel
    
    if not hasattr(channel_wrapper, 'shuffled_indices') or channel_wrapper.shuffled_indices is None:
        if logger: logger.warning("Channel indices not found. Is this run after observe? Skipping.")
        return measurement, 0.0, None, None

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
    
    # 潜在表現の空間サイズ計算
    if hasattr(operator, 's_shape'):
        latent_H, latent_W = operator.s_shape[2], operator.s_shape[3]
        C_feat = operator.s_shape[1]
    else:
        latent_H, latent_W = input_image.shape[2] // 16, input_image.shape[3] // 16
        C_feat = s_raw.shape[1] // (latent_H * latent_W)

    # ---------------------------------------------------------------------
    # Mode 1: Oracle (正解との差分に基づく理想的な再送)
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
    # Mode 2: Random (ランダム再送 - ベースライン)
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
    # Mode 3: Hybrid-Priority Retransmission (HPRS) - 提案法
    # ---------------------------------------------------------------------
    else:
        if uncertainty_map is None:
            return measurement, 0.0, None, None

        u_map = uncertainty_map.to(device)
        u_map_lat = F.adaptive_avg_pool2d(u_map, output_size=(latent_H, latent_W))
        
        if mode == 'rate':
            # === Step 1 (Rx): 候補マスク生成 (Candidate Generation) ===
            u_flat = u_map_lat.view(B, -1)
            total_pixels = u_flat.shape[1]
            
            # 再送総予算 (Total Budget)
            k_total = int(total_pixels * value)
            if k_total < 1: k_total = 1
            
            # フィードバック候補数 (expansion_factor倍)
            k_cand = int(total_pixels * value * expansion_factor)
            k_cand = min(k_cand, total_pixels)
            if k_cand < k_total: k_cand = k_total  # 予算より候補が少ない場合は合わせる

            # 不確実性が高い順に候補インデックスを取得
            # cand_indices: [B, k_cand]
            _, cand_indices = torch.topk(u_flat, k_cand, dim=1)

            # === Step 2 (Tx): 予算分割 (Budget Split) ===
            k_sem = int(k_total * gamma)
            k_struct = k_total - k_sem
            
            # ViTマップの準備 (Semantic Guide)
            if vit_importance_map is not None:
                vit_lat = F.adaptive_avg_pool2d(vit_importance_map.to(device), output_size=(latent_H, latent_W))
                vit_flat = vit_lat.view(B, -1)
            else:
                # ViTがない場合は、不確実性マップそのものを重要度として代用
                vit_flat = u_flat

            # === Step 3 (Selection): 候補内での選別 ===
            # 候補領域内の「意味的重要性 (ViT)」を取得
            # gathered_vit: [B, k_cand]
            gathered_vit = torch.gather(vit_flat, 1, cand_indices)

            # --- Step 3-A: Semantic枠 (上位 k_sem) ---
            # 候補の中でViT値が高い順にソート
            # sort_idx_local: [B, k_cand] (0 ~ k_cand-1 の範囲)
            _, sort_idx_local = torch.sort(gathered_vit, descending=True, dim=1)
            
            # Semantic枠のローカルインデックス
            idx_sem_local = sort_idx_local[:, :k_sem]

            # --- Step 3-B: Structural枠 (残りからランダム k_struct) ---
            if k_struct > 0:
                # Semantic枠で選ばれなかった残りのインデックス群
                idx_remain_local = sort_idx_local[:, k_sem:]
                
                # 残り候補数
                n_remain = idx_remain_local.shape[1]
                
                if n_remain > 0:
                    # ランダム順列を生成して先頭 k_struct 個を取得
                    rand_perm = torch.rand(B, n_remain, device=device).argsort(dim=1)
                    idx_rand_local = torch.gather(idx_remain_local, 1, rand_perm[:, :k_struct])
                    
                    # 結合: Semantic枠 + Random枠
                    final_local_indices = torch.cat([idx_sem_local, idx_rand_local], dim=1)
                else:
                    # まれなケース: 候補数 == 予算数 の場合など
                    final_local_indices = idx_sem_local
            else:
                final_local_indices = idx_sem_local

            # === Step 4: グローバルインデックスへのマッピング ===
            # local (0~k_cand) -> global (0~TotalPixels)
            final_global_indices = torch.gather(cand_indices, 1, final_local_indices)

            # マスクの作成
            mask_flat_spatial = torch.zeros_like(u_flat)
            # scatterで1を立てる
            mask_flat_spatial.scatter_(1, final_global_indices, 1.0)
            
            mask_lat_spatial = mask_flat_spatial.view(B, 1, latent_H, latent_W)

        else:
            # 従来のThresholdモード (フォールバック)
            mask_lat_spatial = (u_map_lat > value).float()

        # マスクの整形と適用
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
    
    return new_measurement, retransmission_ratio, mask_vis, mask_lat_spatial

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml', help="Path to option YMAL file.")
    parser.add_argument("--retrans_mode", type=str, default='rate', choices=['rate', 'threshold', 'oracle'])
    parser.add_argument("--retrans_value", type=float, default=0.1)
    parser.add_argument("--expansion_factor", type=float, default=2.0, help="Expansion factor for candidate mask generation.")
    # --- [HPRS用に追加] ---
    parser.add_argument("--retrans_gamma", type=float, default=0.3, 
                        help="Ratio of budget allocated to semantic priority. Remaining is used for random structural sampling.")
    # ----------------------
    parser.add_argument("--retrans_basis", type=str, default='both', choices=['uncertainty', 'semantic', 'both'],
                        help="Basis for retransmission: 'uncertainty' (U only), 'semantic' (U * ViT), or 'both'.")
    parser.add_argument("--resume_index", type=int, default=50, help="Index to resume processing from (0-based).")
    parser.add_argument("--enable_random", action='store_true', help="Enable random retransmission baseline.")

    args = parser.parse_args()
    
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    
    config.retrans_mode = args.retrans_mode
    config.retrans_value = args.retrans_value
    config.expansion_factor = args.expansion_factor
    config.retrans_gamma = args.retrans_gamma  # Configに追加
    config.retrans_basis = args.retrans_basis
    config.resume_index = args.resume_index
    config.enable_random = args.enable_random
    
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
    # ファイル名にgammaも含めて実験条件を明記
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
    logger.info(f'【Config】: Retransmission Mode: {config.retrans_mode}, Value: {config.retrans_value}, ExpFactor: {config.expansion_factor}')
    logger.info(f'【Config】: Retransmission Basis: {config.retrans_basis} (Using ViT: {config.retrans_basis in ["semantic", "both"]})')
    logger.info(f'【Config】: Random Baseline Enabled: {config.enable_random}')
    
    config_modes = config.diffcom_series['uncertainty_mode']
    logger.info(f"Target Uncertainty Modes: {config_modes}")

    metric_wrapper = MetricWrapper().to(device)
    loss_wrapper = ConsistencyLoss(config, device)
    
    # --- [修正: 文字列と数値を安全にフォーマットするヘルパー] ---
    def format_metrics(m):
        s = f"PSNR: {m.get('psnr', 0):.2f}dB"
        if 'lpips' in m: s += f" | LPIPS: {m['lpips']:.4f}"
        if 'dists' in m: s += f" | DISTS: {m['dists']:.4f}"
        if 'fid' in m:   
            val = m['fid']
            # 数値型ならフォーマット、文字列(エラーメッセージ)ならそのまま表示
            if isinstance(val, (int, float)):
                s += f" | FID: {val:.4f}"
            else:
                s += f" | FID: {val}"
        if 'corr' in m:  s += f" | Corr: {m['corr']:.3f}"
        return s
    # --------------------------------------------------------

    vit_extractor = None
    if config.retrans_basis in ['semantic', 'both']:
        try:
            vit_extractor = ViTSaliencyExtractor(device=device)
            logger.info("[ViT] Saliency Extractor Initialized Successfully (DINOv3).")
        except Exception as e:
            logger.warning(f"[ViT] Initialization Failed: {e}. Falling back to standard uncertainty if possible.")
    else:
        logger.info("[ViT] Saliency Extractor Skipped (Basis is 'uncertainty' only).")

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
            try:
                fid_meters[key] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            except (ModuleNotFoundError, ImportError, RuntimeError) as e:
                logger.warning(f"\n[Warning] FID init failed: {e}. FID disabled.")
                IS_TORCHMETRICS_AVAILABLE = False
                return
        real_norm = torch.clamp(real_img, 0, 1)
        fake_norm = torch.clamp(fake_img, 0, 1)
        fid_meters[key].update(real_norm, real=True)
        fid_meters[key].update(fake_norm, real=False)

    def wrapped_cond_method(*args, **kwargs):
        kwargs['loss_wrapper'] = loss_wrapper
        return cond_method(*args, **kwargs)

    json_filename = f"SNR{config.CSNR}_{config.result_name}.json"
    json_path = os.path.join(config.save_path, json_filename)
    import gc

    all_results_history = []
    
    # indexが指定されており、かつファイルが存在する場合は読み込む
    if config.resume_index > 0 and os.path.exists(json_path):
        logger.info(f"Found existing JSON at {json_path}. Loading history to resume...")
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
                all_results_history = existing_data.get('history', [])
            logger.info(f"Loaded {len(all_results_history)} previous records from history.")
        except Exception as e:
            logger.warning(f"Failed to load existing JSON: {e}. Starting with empty history.")

    # --- [シグナルハンドリングの定義] ---
    def handle_sigterm(signum, frame):
        """killコマンド(SIGTERM)を受け取った時に例外を投げてfinallyブロックへ誘導する"""
        logger.info(f"Received Signal {signum}. Raising KeyboardInterrupt to save results...")
        raise KeyboardInterrupt

    # SIGTERM (killデフォルト) をキャッチ
    signal.signal(signal.SIGTERM, handle_sigterm)
    # ------------------------------------

    try:
        for idx, batch in enumerate(dataloader):
            if idx < config.resume_index:
                continue
            
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]

            vit_map = None
            if vit_extractor is not None:
                try:
                    vit_map = vit_extractor.get_importance_map(input_image).detach()
                except Exception as e:
                    logger.warning(f"Batch {idx}: ViT map calculation failed ({e}).")
            
            torch.manual_seed(config.seed + idx)
            measurement_phase1 = operator.observe_and_transpose(input_image)
            
            metrics_jscc_p1 = metric_wrapper(measurement_phase1['x_mse'].detach(), input_image)
            get_meter('jscc_init').update(metrics_jscc_p1)
            
            log_msg_jscc = f"Batch {idx+1}/{len(dataloader)} | [Base JSCC] Init | {format_metrics(metrics_jscc_p1)}"
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
                "modes": {}
            }

            # Phase 1
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

            log_msg_p1 = f"  -> Phase 1        | {format_metrics(metrics_p1)}"
            logger.info(log_msg_p1)

            torchvision.utils.save_image(x_recon_p1[0].cpu(), os.path.join(save_dir, f'2_Phase1_Recon.png'))
            
            error_map = torch.abs(x_recon_p1 - input_image).mean(dim=1, keepdim=True)
            e_flat = error_map.detach().cpu().flatten().numpy()

            # Phase 2 Loop
            available_modes = list(uncertainty_container_p1.keys()) if uncertainty_container_p1 else []
            if not available_modes and config.retrans_mode != 'oracle':
                 logger.warning("No uncertainty maps found!")

            for u_mode in available_modes:
                mode_maps = uncertainty_container_p1[u_mode]
                mode_result = {"correlation": {}, "results": {}}
                
                u_map_tensor = mode_maps.get('raw')
                if u_map_tensor is None: continue
                
                sub_key = 'raw'

                if u_map_tensor.shape[-2:] != error_map.shape[-2:]:
                        u_raw_resized = F.interpolate(u_map_tensor, size=error_map.shape[-2:], mode='bilinear')
                        u_flat_val = u_raw_resized.flatten().cpu().numpy()
                else:
                        u_flat_val = u_map_tensor.flatten().cpu().numpy()
                
                if np.isnan(u_flat_val).any() or np.isnan(e_flat).any():
                    corr = 0.0
                else:
                    corr, _ = pearsonr(u_flat_val, e_flat)
                
                mode_result["correlation"][sub_key] = corr

                if config.retrans_mode != 'oracle':
                    strategies = []
                    if config.retrans_basis in ['uncertainty', 'both']:
                        strategies.append((None, "Unc"))
                    if config.retrans_basis in ['semantic', 'both']:
                        if vit_map is not None:
                            strategies.append((vit_map, "Sem"))
                        else:
                            logger.warning("Skipping Semantic mode because ViT map is missing.")

                    for v_map_arg, strategy_name in strategies:
                        # 候補提示型再送シミュレーション (expansion_factorを渡す)
                        meas_p2, ratio, mask_vis, _ = simulate_semantic_retransmission(
                            operator, input_image, measurement_phase1, 
                            u_map_tensor, 
                            mode=config.retrans_mode, value=config.retrans_value,
                            vit_importance_map=v_map_arg,
                            expansion_factor=config.expansion_factor
                        )
                        
                        base_key = f"{u_mode}_{sub_key}_{strategy_name}"

                        torch.manual_seed(config.seed + idx)
                        x_recon_p2, _ = run_diffusion_process(
                            config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                            meas_p2, input_image, device, phase_name=f"P2_{base_key}"
                        )
                        
                        metrics_p2 = metric_wrapper(x_recon_p2.detach(), input_image)
                        metrics_p2['corr'] = float(corr)
                        get_meter(base_key).update(metrics_p2)
                        update_fid(base_key, input_image, x_recon_p2.detach())
                        
                        if sub_key not in mode_result["results"]:
                            mode_result["results"][sub_key] = {}
                        
                        mode_result["results"][sub_key][strategy_name] = {k: float(v) for k, v in metrics_p2.items()}
                        mode_result["results"][sub_key][strategy_name]['ratio'] = ratio
                        
                        log_msg_p2 = f"    [{u_mode[:4]}-{sub_key:6s}-{strategy_name:3s}] Ratio: {ratio:.2%} | {format_metrics(metrics_p2)}"
                        logger.info(log_msg_p2)

                        torchvision.utils.save_image(x_recon_p2[0].cpu(), os.path.join(save_dir, f'3_P2_{base_key}.png'))
                        if mask_vis is not None:
                            plt.imsave(os.path.join(save_dir, f'Mask_{base_key}.png'), mask_vis[0, 0].cpu().numpy(), cmap='gray')
                        
                        if strategy_name == "Sem" and v_map_arg is not None:
                            p_map = u_map_tensor.to(device) * v_map_arg.to(device)
                            p_vis = p_map[0, 0].cpu().numpy()
                            p_vis = (p_vis - p_vis.min()) / (p_vis.max() - p_vis.min() + 1e-8)
                            plt.imsave(os.path.join(save_dir, f'Priority_{base_key}.png'), p_vis, cmap='jet')
                        else:
                            u_vis = u_map_tensor[0, 0].cpu().numpy()
                            u_vis = (u_vis - u_vis.min()) / (u_vis.max() - u_vis.min() + 1e-8)
                            plt.imsave(os.path.join(save_dir, f'Uncertainty_{base_key}.png'), u_vis, cmap='jet')
                        
                        del meas_p2, x_recon_p2
                        torch.cuda.empty_cache()

                batch_record["modes"][u_mode] = mode_result

            # Random Baseline
            if config.retrans_mode != 'oracle' and config.enable_random:
                 meas_rnd, ratio_rnd, mask_vis_rnd, _ = simulate_semantic_retransmission(
                     operator, input_image, measurement_phase1, None, mode='random', value=config.retrans_value,
                     expansion_factor=config.expansion_factor
                 )
                 
                 torch.manual_seed(config.seed + idx)
                 x_recon_rnd, _ = run_diffusion_process(
                     config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                     meas_rnd, input_image, device, phase_name="P2_Random"
                 )
                 metrics_rnd = metric_wrapper(x_recon_rnd.detach(), input_image)
                 get_meter('random').update(metrics_rnd)
                 update_fid('random', input_image, x_recon_rnd.detach())

                 batch_record['random'] = {k: float(v) for k, v in metrics_rnd.items()}
                 batch_record['random']['ratio'] = ratio_rnd
                 
                 log_msg_rnd = f"    [Random             ] Ratio: {ratio_rnd:.2%} | {format_metrics(metrics_rnd)}"
                 logger.info(log_msg_rnd)

                 torchvision.utils.save_image(x_recon_rnd[0].cpu(), os.path.join(save_dir, '3_P2_Random.png'))
                 if mask_vis_rnd is not None:
                     plt.imsave(os.path.join(save_dir, 'Mask_Random.png'), mask_vis_rnd[0, 0].cpu().numpy(), cmap='gray')
                
                 del meas_rnd, x_recon_rnd
                 torch.cuda.empty_cache()

            all_results_history.append(batch_record)
            logger.info('-' * 80)
            
            del input_image, measurement_phase1
            if 'x_recon_p1' in locals(): del x_recon_p1
            if 'uncertainty_container_p1' in locals(): del uncertainty_container_p1
            if 'vit_map' in locals(): del vit_map
            
            gc.collect() 
            torch.cuda.empty_cache() 

    except KeyboardInterrupt:
        logger.info("\n[!] Process Interrupted by User (Ctrl+C or kill). Saving current results...")
    except Exception as e:
        logger.error(f"\n[!] Unexpected Error Occurred: {e}. Saving current results...")
        import traceback
        traceback.print_exc()
    finally:
        # サマリー作成 (全履歴から再計算)
        current_session_summary = {}
        for k, meter in results_meters.items():
            current_session_summary[k] = meter.avg
        
        # --- [修正: FID計算中の二重割り込み(KeyboardInterrupt)をガード] ---
        if IS_TORCHMETRICS_AVAILABLE and len(fid_meters) > 0:
            logger.info("Calculating FID scores... (might take a while)")
            
            # FID計算全体を try ブロックで囲む
            try:
                for k, fid_obj in fid_meters.items():
                    if k not in current_session_summary: current_session_summary[k] = {}
                    try:
                        # 計算試行
                        score = fid_obj.compute().item()
                        current_session_summary[k]['fid'] = score
                        logger.info(f"  -> FID [{k}]: {score:.4f}")
                    except KeyboardInterrupt:
                         # ここで再度のCtrl+Cをキャッチしてループを抜ける
                        logger.warning(f"FID calculation for {k} interrupted by user!")
                        current_session_summary[k]['fid'] = "Interrupted_User"
                        raise KeyboardInterrupt # 外側のexceptへ飛ばす
                    except Exception as e:
                        # 計算エラーなどは個別に記録
                        logger.warning(f"Failed to compute FID for {k} (Error): {e}")
                        current_session_summary[k]['fid'] = f"Error: {str(e)}"
            
            except KeyboardInterrupt:
                logger.warning("\n[!] FID calculation interrupted. Skipping remaining FIDs to save data immediately.")
        # ------------------------------------
        
        # 履歴がある場合は、全履歴からSummaryを再構築する
        if len(all_results_history) > 0:
            logger.info(f"Recalculating global summary from {len(all_results_history)} total records...")
            final_summary = reconstruct_full_summary(all_results_history)
            
            # FIDなど履歴に含まれない情報をマージ
            for k, v in current_session_summary.items():
                if 'fid' in v and k in final_summary:
                    final_summary[k]['fid'] = v['fid']
                elif k not in final_summary: # 万が一履歴にないが今回計測されたもの
                    final_summary[k] = v
        else:
            final_summary = current_session_summary

        output_data = {"summary": final_summary, "history": all_results_history}

        if len(all_results_history) > 0:
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Saved {len(all_results_history)} results to {json_path}")
        else:
            logger.warning("No complete results to save.")
        
        # 最終ログ表示
        logger.info("=== Final Comparison Summary ===")
        
        if 'jscc_init' in final_summary:
            logger.info(f"Init (Base)  | {format_metrics(final_summary['jscc_init'])}")
        if 'phase1_recon' in final_summary:
            logger.info(f"Phase 1      | {format_metrics(final_summary['phase1_recon'])}")
        if 'random' in final_summary:
            logger.info(f"Random       | {format_metrics(final_summary['random'])}")

        for k in sorted(final_summary.keys()):
            if k in ['jscc_init', 'random', 'phase1_recon', 'random_jscc']: continue
            if 'jscc' in k: continue 
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
    
    
    # モデル名に応じた設定の分岐
    if config.model_name == 'ffhq_10m':
        model_config = dict(
            model_path=config.model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        )
    # ▼▼▼ LSUN Bedroom用設定を追加 (ファイル名は適宜合わせてください) ▼▼▼
    elif config.model_name == 'lsun_uncond_100M_1200K_bs128': 
        model_config = dict(
            model_path=config.model_path,
            image_size=256,
            num_channels=128,           # LSUNモデルの仕様
            num_res_blocks=2,           # LSUNモデルの仕様
            num_heads=1,                # LSUNモデルの仕様
            learn_sigma=True,           # LSUNモデルの仕様
            use_scale_shift_norm=False, # LSUNモデルの仕様
            attention_resolutions="16", # LSUNモデルの仕様
            diffusion_steps=1000,       # DiffComの標準に合わせるかモデルに合わせる
            noise_schedule="linear",    # LSUNモデルはlinear
            rescale_learned_sigmas=False,
            rescale_timesteps=False,
        )
    # ▲▲▲ 追加ここまで ▲▲▲
    else:
        # ImageNet等のデフォルト設定
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