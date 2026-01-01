import argparse
import logging
import os
import os.path
import random
import shutil
import json
import copy
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from transformers import AutoImageProcessor, Dinov2Model

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

# --- ViT重要度抽出クラス ---
class ViTSaliencyExtractor:
    def __init__(self, model_name="facebook/dinov2-with-registers-small", device="cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name, output_attentions=True).to(device)
        self.model.eval()

    @torch.no_grad()
    def get_importance_map(self, images):
        """
        images: Tensor [B, 3, H, W] (0~1)
        returns: Tensor [B, 1, H, W] (normalized 0~1)
        """
        B, C, H, W = images.shape
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        attentions = outputs.attentions 
        last_layer_attn = attentions[-1] 
        
        num_registers = getattr(self.model.config, "num_registers", 0)
        cls_attn = last_layer_attn[:, :, 0, 1 + num_registers:]
        importance = cls_attn.mean(dim=1) 
        
        grid_size = int(importance.shape[-1]**0.5)
        importance = importance.reshape(B, 1, grid_size, grid_size)
        
        importance_resized = F.interpolate(importance, size=(H, W), mode='bilinear', align_corners=False)
        
        i_min = importance_resized.flatten(2).min(2, keepdim=True)[0].unsqueeze(-1)
        i_max = importance_resized.flatten(2).max(2, keepdim=True)[0].unsqueeze(-1)
        importance_resized = (importance_resized - i_min) / (i_max - i_min + 1e-8)
        
        return importance_resized

# =========================================================================
# 提案法: 意味的再送シミュレーション関数 (Signal Replacement)
# =========================================================================
def simulate_semantic_retransmission(operator, input_image, measurement, uncertainty_map, 
                                     mode='rate', value=0.1, logger=None, vit_importance_map=None):
    """
    不確かさマップに基づく信号置換 (Signal Replacement)
    vit_importance_map が指定された場合、P = U * A (Uncertainty * ViT Attention) に基づく再送を行う。
    Noneの場合は従来のUncertaintyのみ(U)に基づく再送を行う。
    """
    device = input_image.device
    
    # 1. 送信時の状態(State)の復元
    channel_wrapper = operator.channel
    
    if not hasattr(channel_wrapper, 'shuffled_indices') or channel_wrapper.shuffled_indices is None:
        if logger: logger.warning("Channel indices not found. Is this run after observe? Skipping.")
        return measurement, 0.0, None, None

    saved_indices = channel_wrapper.shuffled_indices.to(device)
    saved_avg_pwr = channel_wrapper.avg_pwr
    
    # 2. 理想的な受信信号 (y_clean) の生成
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

    # 実際の受信信号
    y_dirty = measurement['ofdm_sig']

    # 3. 再送マスクの生成 (Pixel -> Latent)
    mask_vis = None
    
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

    elif mode == 'random':
        if hasattr(operator, 's_shape'):
            latent_H, latent_W = operator.s_shape[2], operator.s_shape[3]
            C_feat = operator.s_shape[1]
        else:
            latent_H, latent_W = input_image.shape[2] // 16, input_image.shape[3] // 16
            C_feat = s_raw.shape[1] // (latent_H * latent_W)
        
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
        
    else:
        # 通常モード (不確実性 + [Optional] ViT重要度)
        if uncertainty_map is None:
            return measurement, 0.0, None, None

        u_map = uncertainty_map.to(device) 

        # --- [ViT Integration Logic] ---
        if vit_importance_map is not None:
            # P = U * A (Uncertainty * ViT Attention)
            priority_map = u_map * vit_importance_map.to(device)
            u_map = priority_map
        # -------------------------------
        
        if hasattr(operator, 's_shape'):
            latent_H, latent_W = operator.s_shape[2], operator.s_shape[3]
            C_feat = operator.s_shape[1]
        else:
            latent_H, latent_W = input_image.shape[2] // 16, input_image.shape[3] // 16
            C_feat = s_raw.shape[1] // (latent_H * latent_W)

        u_map_lat = F.adaptive_avg_pool2d(u_map, output_size=(latent_H, latent_W))
        
        if mode == 'rate':
            u_flat = u_map_lat.view(B, -1)
            k = int(u_flat.shape[1] * value)
            if k < 1: k = 1
            top_val, _ = torch.topk(u_flat, k, dim=1)
            thresh = top_val[:, -1].view(B, 1, 1, 1)
            mask_lat_spatial = (u_map_lat >= thresh).float()
        else: 
            mask_lat_spatial = (u_map_lat > value).float()

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

    # 4. 観測値の更新
    new_measurement = copy.deepcopy(measurement)
    
    if y_clean.shape != y_dirty.shape:
        y_clean = y_clean.view(y_dirty.shape)
        
    y_new = (1 - mask_for_y) * y_dirty + mask_for_y * y_clean
    new_measurement['ofdm_sig'] = y_new
    
    h_dirty = measurement.get('cof_est', None)
    if h_dirty is not None:
         if h_dirty.shape == y_dirty.shape:
             h_new = (1 - mask_for_y) * h_dirty + mask_for_y * torch.ones_like(h_dirty)
             new_measurement['cof_est'] = h_new

    with torch.no_grad():
        cof_for_transpose = new_measurement.get('cof_est', None)
        s_hat_new = operator.transpose(y_new, cof_for_transpose)
        x_mse_new = operator.decode(s_hat_new)
        new_measurement['x_mse'] = x_mse_new

    return new_measurement, retransmission_ratio, mask_vis, mask_lat_spatial


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml', help="Path to option YMAL file.")
    parser.add_argument("--retrans_mode", type=str, default='rate', choices=['rate', 'threshold', 'oracle'])
    parser.add_argument("--retrans_value", type=float, default=0.1)
    parser.add_argument("--retrans_basis", type=str, default='both', choices=['uncertainty', 'semantic', 'both'],
                        help="Basis for retransmission: 'uncertainty' (U only), 'semantic' (U * ViT), or 'both'.")

    args = parser.parse_args()
    
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    
    config.retrans_mode = args.retrans_mode
    config.retrans_value = args.retrans_value
    config.retrans_basis = args.retrans_basis
    
    cond_config = Config(config.getattr('diffcom_series'))
    conditioning_method = Config(cond_config.getattr(config.conditioning_method))
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt
    config.skip = cond_config.num_train_timesteps // cond_config.iter_num
    config.sigma = np.sqrt(1.0 / (2 * 10 ** (config.CSNR / 10)))

    config.model_zoo = os.path.join(config.cwd, 'model_zoo')
    
    # ------------------ [FIXED HERE] ------------------
    # 'config.testsets' がYAMLに存在しないため、cwdから生成する
    config.testsets = os.path.join(config.cwd, 'testsets')
    # --------------------------------------------------

    config.results = os.path.join(config.cwd, 'results_retrans_comparison')
    config.results = os.path.join(config.results, config.testset_name)
    config.results = os.path.join(config.results, config.conditioning_method)

    if config.operator_name == 'djscc':
        config.results = os.path.join(config.results, config.operator_name + '_{}'.format(config.djscc['channel_num']))
    
    config.results = os.path.join(config.results, f'{config.channel_type}_{config.CSNR.__str__().zfill(2)}dB')
    
    u_mode = cond_config.uncertainty_mode
    u_mode_str = "Comparison" if isinstance(u_mode, list) else str(u_mode)
    
    config.result_name = f'Retrans_{config.retrans_mode}_{config.retrans_value}_{u_mode_str}_{config.retrans_basis}'
    config.result_name += f'_zeta{conditioning_method.zeta}_seed{config.seed}'
    
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
    final_uncertainty_maps = diffcom_module.latest_uncertainty_map
    
    return x_recon, final_uncertainty_maps

def p_sample_loop(config, noise_schedule, unet, diffusion, operator, cond_method, dataloader, device, logger):
    logger.info(f'【Config】: Retransmission Mode: {config.retrans_mode}, Value: {config.retrans_value}')
    logger.info(f'【Config】: Retransmission Basis: {config.retrans_basis} (Using ViT: {config.retrans_basis in ["semantic", "both"]})')
    
    config_modes = config.diffcom_series['uncertainty_mode']
    logger.info(f"Target Uncertainty Modes: {config_modes}")

    metric_wrapper = MetricWrapper().to(device)
    loss_wrapper = ConsistencyLoss(config, device)
    
    # --- [METRICS FORMATTER] ---
    def format_metrics(m):
        """メトリクス辞書を文字列にフォーマットするヘルパー"""
        s = f"PSNR: {m.get('psnr', 0):.2f}dB"
        if 'lpips' in m: s += f" | LPIPS: {m['lpips']:.4f}"
        if 'dists' in m: s += f" | DISTS: {m['dists']:.4f}"
        if 'fid' in m:   s += f" | FID: {m['fid']:.4f}"
        return s
    # ---------------------------

    # --- [ViT Extractor Init (Conditional)] ---
    vit_extractor = None
    if config.retrans_basis in ['semantic', 'both']:
        try:
            vit_extractor = ViTSaliencyExtractor(device=device)
            logger.info("[ViT] Saliency Extractor Initialized Successfully.")
        except Exception as e:
            logger.warning(f"[ViT] Initialization Failed: {e}. Falling back to standard uncertainty if possible.")
    else:
        logger.info("[ViT] Saliency Extractor Skipped (Basis is 'uncertainty' only).")
    # ------------------------------------------

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

    all_results_history = []
    json_filename = f"SNR{config.CSNR}_{config.result_name}.json"
    json_path = os.path.join(config.save_path, json_filename)

    try:
        for idx, batch in enumerate(dataloader):
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]

            # --- [ViT Importance Map Calculation] ---
            vit_map = None
            if vit_extractor is not None:
                try:
                    vit_map = vit_extractor.get_importance_map(input_image)
                except Exception as e:
                    logger.warning(f"Batch {idx}: ViT map calculation failed ({e}).")
            # ----------------------------------------
            
            # Step 0: Initial Transmission
            torch.manual_seed(config.seed + idx)
            measurement_phase1 = operator.observe_and_transpose(input_image)
            
            metrics_jscc_p1 = metric_wrapper(measurement_phase1['x_mse'].detach(), input_image)
            get_meter('jscc_init').update(metrics_jscc_p1)
            
            # [LOG UPDATED]
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

            # -----------------------------------------------------
            # Step 1: Phase 1 (Diffusion)
            # -----------------------------------------------------
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

            # [LOG UPDATED]
            log_msg_p1 = f"  -> Phase 1        | {format_metrics(metrics_p1)}"
            logger.info(log_msg_p1)

            torchvision.utils.save_image(x_recon_p1[0].cpu(), os.path.join(save_dir, f'2_Phase1_Recon.png'))
            
            error_map = torch.abs(x_recon_p1 - input_image).mean(dim=1, keepdim=True)
            e_flat = error_map.detach().cpu().flatten().numpy()

            # -----------------------------------------------------
            # Step 2: Phase 2 (Retransmission Loop)
            # -----------------------------------------------------
            
            available_modes = list(uncertainty_container_p1.keys()) if uncertainty_container_p1 else []
            if not available_modes and config.retrans_mode != 'oracle':
                 logger.warning("No uncertainty maps found!")

            for u_mode in available_modes:
                mode_maps = uncertainty_container_p1[u_mode]
                mode_result = {"correlation": {}, "results": {}}
                
                # Smoothを削除し、常にRawを使用
                u_map_tensor = mode_maps.get('raw')
                if u_map_tensor is None: continue
                
                # キーは 'raw' で統一
                sub_key = 'raw'

                # 相関計算
                # Rawの場合はサイズが異なる可能性があるためリサイズして計算
                if u_map_tensor.shape[-2:] != error_map.shape[-2:]:
                        u_raw_resized = F.interpolate(u_map_tensor, size=error_map.shape[-2:], mode='bilinear')
                        u_flat_val = u_raw_resized.flatten().numpy()
                else:
                        u_flat_val = u_map_tensor.flatten().numpy()
                corr, _ = pearsonr(u_flat_val, e_flat)
                mode_result["correlation"][sub_key] = corr

                if config.retrans_mode != 'oracle':
                    # 実行する再送方法のリストを作成
                    strategies = []
                    if config.retrans_basis in ['uncertainty', 'both']:
                        strategies.append((None, "Unc")) # 元の手法 (ViTなし)
                    if config.retrans_basis in ['semantic', 'both']:
                        # ViTが取得できていれば実行、なければスキップ
                        if vit_map is not None:
                            strategies.append((vit_map, "Sem"))
                        else:
                            logger.warning("Skipping Semantic mode because ViT map is missing.")

                    for v_map_arg, strategy_name in strategies:
                        meas_p2, ratio, mask_vis, _ = simulate_semantic_retransmission(
                            operator, input_image, measurement_phase1, 
                            u_map_tensor, 
                            mode=config.retrans_mode, value=config.retrans_value,
                            vit_importance_map=v_map_arg
                        )
                        
                        metrics_jscc_p2 = metric_wrapper(meas_p2['x_mse'].detach(), input_image)
                        
                        # キーの生成 (例: perturbation_raw_Unc_jscc)
                        base_key = f"{u_mode}_{sub_key}_{strategy_name}"
                        get_meter(f"{base_key}_jscc").update(metrics_jscc_p2)
                        
                        # [LOG UPDATED]
                        log_msg_jscc = f"    [{u_mode[:4]}-{sub_key:6s}-{strategy_name:3s} JSCC] {format_metrics(metrics_jscc_p2)}"
                        logger.info(log_msg_jscc)

                        # Diffusion
                        torch.manual_seed(config.seed + idx)
                        x_recon_p2, _ = run_diffusion_process(
                            config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                            meas_p2, input_image, device, phase_name=f"P2_{base_key}"
                        )
                        
                        metrics_p2 = metric_wrapper(x_recon_p2.detach(), input_image)
                        get_meter(base_key).update(metrics_p2)
                        update_fid(base_key, input_image, x_recon_p2.detach())
                        
                        if sub_key not in mode_result["results"]:
                            mode_result["results"][sub_key] = {}
                        
                        # 結果辞書の構造: results[raw][Unc] = {...}
                        mode_result["results"][sub_key][strategy_name] = {k: float(v) for k, v in metrics_p2.items()}
                        mode_result["results"][sub_key][strategy_name]['ratio'] = ratio
                        mode_result["results"][sub_key][strategy_name]['jscc'] = {k: float(v) for k, v in metrics_jscc_p2.items()}
                        
                        # [LOG UPDATED]
                        log_msg_p2 = f"    [{u_mode[:4]}-{sub_key:6s}-{strategy_name:3s}] Ratio: {ratio:.2%} | {format_metrics(metrics_p2)} | Corr: {corr:.3f}"
                        logger.info(log_msg_p2)

                        torchvision.utils.save_image(x_recon_p2[0].cpu(), os.path.join(save_dir, f'3_P2_{base_key}.png'))
                        plt.imsave(os.path.join(save_dir, f'Mask_{base_key}.png'), mask_vis[0, 0].cpu().numpy(), cmap='gray')
                        
                        # Priority mapの保存 (Semの場合)
                        if strategy_name == "Sem":
                            p_map = u_map_tensor.to(device) * v_map_arg.to(device)
                            p_vis = p_map[0, 0].cpu().numpy()
                            p_vis = (p_vis - p_vis.min()) / (p_vis.max() - p_vis.min() + 1e-8)
                            plt.imsave(os.path.join(save_dir, f'Priority_{base_key}.png'), p_vis, cmap='jet')
                        else:
                            # Uncの場合は生のUncertaintyを保存
                            u_vis = u_map_tensor[0, 0].numpy()
                            u_vis = (u_vis - u_vis.min()) / (u_vis.max() - u_vis.min() + 1e-8)
                            plt.imsave(os.path.join(save_dir, f'Uncertainty_{base_key}.png'), u_vis, cmap='jet')

                batch_record["modes"][u_mode] = mode_result

            # -----------------------------------------------------------------
            # Random Baseline
            # -----------------------------------------------------------------
            if config.retrans_mode != 'oracle':
                 meas_rnd, ratio_rnd, mask_vis_rnd, _ = simulate_semantic_retransmission(
                     operator, input_image, measurement_phase1, None, mode='random', value=config.retrans_value
                 )
                 
                 metrics_jscc_rnd = metric_wrapper(meas_rnd['x_mse'].detach(), input_image)
                 get_meter('random_jscc').update(metrics_jscc_rnd)
                 
                 # [LOG UPDATED]
                 log_msg_jscc_rnd = f"    [Random          JSCC] {format_metrics(metrics_jscc_rnd)}"
                 logger.info(log_msg_jscc_rnd)

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
                 
                 # [LOG UPDATED]
                 log_msg_rnd = f"    [Random             ] Ratio: {ratio_rnd:.2%} | {format_metrics(metrics_rnd)}"
                 logger.info(log_msg_rnd)

                 torchvision.utils.save_image(x_recon_rnd[0].cpu(), os.path.join(save_dir, '3_P2_Random.png'))
                 plt.imsave(os.path.join(save_dir, 'Mask_Random.png'), mask_vis_rnd[0, 0].cpu().numpy(), cmap='gray')

            all_results_history.append(batch_record)
            logger.info('-' * 80)

    except KeyboardInterrupt:
        logger.info("\n[!] Process Interrupted by User. Saving current results...")
    except Exception as e:
        logger.error(f"\n[!] Unexpected Error Occurred: {e}. Saving current results...")
        import traceback
        traceback.print_exc()
    finally:
        # サマリー作成
        final_summary = {}
        for k, meter in results_meters.items():
            final_summary[k] = meter.avg
        
        if IS_TORCHMETRICS_AVAILABLE and len(fid_meters) > 0:
            logger.info("Calculating FID scores...")
            for k, fid_obj in fid_meters.items():
                try:
                    score = fid_obj.compute().item()
                    if k not in final_summary: final_summary[k] = {}
                    final_summary[k]['fid'] = score
                    logger.info(f"  -> FID [{k}]: {score:.4f}")
                except Exception as e:
                    logger.error(f"Failed to compute FID for {k}: {e}")
        
        output_data = {"summary": final_summary, "history": all_results_history}

        if len(all_results_history) > 0:
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Saved {len(all_results_history)} results to {json_path}")
        
        # 最終ログ表示
        logger.info("=== Final Comparison Summary ===")
        # format_metrics は関数の先頭ですでに定義されているものを利用

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
    
    model_config = dict(
        model_path=config.model_path,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
    ) if config.model_name == 'ffhq_10m' \
        else dict(
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