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

# カスタムモジュール
import conditioning_method.diffcom as diffcom_module
from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
from data.datasets import get_test_loader
from guided_diffusion.measurement import get_operator
from guided_diffusion.noise_schedule import NoiseSchedule
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from utils.util import Config, MetricWrapper, DictAverageMeter
from utils import util, utils_logger, utils_model

# =========================================================================
# 提案法: 意味的再送シミュレーション関数 (Signal Replacement)
# =========================================================================
def simulate_semantic_retransmission(operator, input_image, measurement, uncertainty_map, 
                                     mode='rate', value=0.1, logger=None):
    """
    不確かさマップに基づく信号置換 (Signal Replacement)
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
        # 通常モード
        if uncertainty_map is None:
            return measurement, 0.0, None, None

        u_map = uncertainty_map.to(device) 
        
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

    args = parser.parse_args()
    
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    
    config.retrans_mode = args.retrans_mode
    config.retrans_value = args.retrans_value
    
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
    
    # uncertainty_modeがリストか文字列かでファイル名を変える
    u_mode = cond_config.uncertainty_mode
    u_mode_str = "Comparison" if isinstance(u_mode, list) else str(u_mode)
    
    config.result_name = f'Retrans_{config.retrans_mode}_{config.retrans_value}_{u_mode_str}'
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
    
    # プログレスバーの表示
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
    
    # 実行する不確かさモード（リスト or 文字列）
    config_modes = config.diffcom_series['uncertainty_mode']
    logger.info(f"Target Uncertainty Modes: {config_modes}")

    metric_wrapper = MetricWrapper().to(device)
    loss_wrapper = ConsistencyLoss(config, device)
    
    # 結果保存用の動的なDictionary Meterを作成
    # results_meters は、"temporal_smooth" や "temporal_smooth_jscc" などのキーを自動生成します
    results_meters = {}

    def get_meter(key):
        if key not in results_meters:
            results_meters[key] = DictAverageMeter()
        return results_meters[key]

    def wrapped_cond_method(*args, **kwargs):
        kwargs['loss_wrapper'] = loss_wrapper
        return cond_method(*args, **kwargs)

    all_results_history = []
    json_filename = f"{config.result_name}.json"
    json_path = os.path.join(config.save_path, json_filename)

    try:
        for idx, batch in enumerate(dataloader):
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]
            
            # Step 0: Initial Transmission (共通)
            torch.manual_seed(config.seed + idx)
            measurement_phase1 = operator.observe_and_transpose(input_image)
            
            # Phase 1 JSCC (Baseline) Init
            metrics_jscc_p1 = metric_wrapper(measurement_phase1['x_mse'].detach(), input_image)
            get_meter('jscc_init').update(metrics_jscc_p1)
            
            # ログ: Phase 1 Init JSCC
            log_msg_jscc = f"Batch {idx+1}/{len(dataloader)} | [Base JSCC] Init | PSNR: {metrics_jscc_p1['psnr']:.2f}dB"
            if 'lpips' in metrics_jscc_p1: log_msg_jscc += f" | LPIPS: {metrics_jscc_p1['lpips']:.4f}"
            if 'dists' in metrics_jscc_p1: log_msg_jscc += f" | DISTS: {metrics_jscc_p1['dists']:.4f}"
            logger.info(log_msg_jscc)

            # 保存用ディレクトリ作成
            save_dir = os.path.join(config.save_path, 'visuals', str(idx))
            util.mkdir(save_dir)
            torchvision.utils.save_image(input_image[0].cpu(), os.path.join(save_dir, '0_GT.png'))
            torchvision.utils.save_image(measurement_phase1['x_mse'][0].cpu(), os.path.join(save_dir, '1_JSCC_Init.png'))

            # バッチ結果レコード初期化
            batch_record = {
                "batch_idx": idx + 1,
                "filename": names[0],
                "jscc_init": {k: float(v) for k, v in metrics_jscc_p1.items()},
                "modes": {}
            }

            # -----------------------------------------------------
            # Step 1: Phase 1 (Diffusion & Calc Uncertainty)
            # -----------------------------------------------------
            # シードをリセットしてPhase 1実行
            torch.manual_seed(config.seed + idx)
            diffcom_module.latest_uncertainty_map = {} # グローバル変数リセット

            x_recon_p1, uncertainty_container_p1 = run_diffusion_process(
                config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                measurement_phase1, input_image, device, phase_name="Phase1"
            )
            
            metrics_p1 = metric_wrapper(x_recon_p1.detach(), input_image)
            get_meter('phase1_recon').update(metrics_p1)
            batch_record['phase1'] = {k: float(v) for k, v in metrics_p1.items()}

            # ログ: Phase 1 Diffusion Result
            log_msg_p1 = f"  -> Phase 1        | PSNR: {metrics_p1['psnr']:.2f}dB"
            if 'lpips' in metrics_p1: log_msg_p1 += f" | LPIPS: {metrics_p1['lpips']:.4f}"
            if 'dists' in metrics_p1: log_msg_p1 += f" | DISTS: {metrics_p1['dists']:.4f}"
            logger.info(log_msg_p1)

            torchvision.utils.save_image(x_recon_p1[0].cpu(), os.path.join(save_dir, f'2_Phase1_Recon.png'))
            
            # エラーマップ（相関計算用）
            error_map = torch.abs(x_recon_p1 - input_image).mean(dim=1, keepdim=True)
            e_flat = error_map.detach().cpu().flatten().numpy()

            # -----------------------------------------------------
            # Step 2: Phase 2 (Retransmission Comparison Loop)
            # -----------------------------------------------------
            
            available_modes = list(uncertainty_container_p1.keys()) if uncertainty_container_p1 else []
            if not available_modes and config.retrans_mode != 'oracle':
                 logger.warning("No uncertainty maps found!")

            # モードごとループ (Temporal / Perturbation)
            for u_mode in available_modes:
                mode_maps = uncertainty_container_p1[u_mode]
                mode_result = {"correlation": {}, "results": {}}
                
                # サブタイプごとループ (Smoothed / Raw)
                sub_types = [('smooth', mode_maps.get('smoothed')), ('raw', mode_maps.get('raw'))]
                
                for sub_key, u_map_tensor in sub_types:
                    if u_map_tensor is None: continue

                    # 相関計算
                    corr = 0.0
                    if sub_key == 'smooth':
                        corr, _ = pearsonr(u_map_tensor.flatten().numpy(), e_flat)
                    else:
                        if u_map_tensor.shape[-2:] != error_map.shape[-2:]:
                             u_raw_resized = F.interpolate(u_map_tensor, size=error_map.shape[-2:], mode='bilinear')
                             u_flat_val = u_raw_resized.flatten().numpy()
                        else:
                             u_flat_val = u_map_tensor.flatten().numpy()
                        corr, _ = pearsonr(u_flat_val, e_flat)
                    
                    mode_result["correlation"][sub_key] = corr

                    # Oracle以外なら再送シミュレーション実行
                    if config.retrans_mode != 'oracle':
                        meas_p2, ratio, mask_vis, _ = simulate_semantic_retransmission(
                            operator, input_image, measurement_phase1, 
                            u_map_tensor, 
                            mode=config.retrans_mode, value=config.retrans_value
                        )
                        
                        # Phase 2 JSCC の品質評価
                        metrics_jscc_p2 = metric_wrapper(meas_p2['x_mse'].detach(), input_image)
                        
                        # [重要] JSCCの平均値を記録
                        meter_key_jscc = f"{u_mode}_{sub_key}_jscc"
                        get_meter(meter_key_jscc).update(metrics_jscc_p2)
                        
                        # ログ: JSCC品質の詳細表示
                        log_msg_jscc = f"    [{u_mode[:4]}-{sub_key:6s} JSCC] PSNR: {metrics_jscc_p2['psnr']:.2f}dB"
                        if 'lpips' in metrics_jscc_p2: log_msg_jscc += f" | LPIPS: {metrics_jscc_p2['lpips']:.4f}"
                        if 'dists' in metrics_jscc_p2: log_msg_jscc += f" | DISTS: {metrics_jscc_p2['dists']:.4f}"
                        logger.info(log_msg_jscc)

                        # Phase 2 Diffusion
                        torch.manual_seed(config.seed + idx)
                        x_recon_p2, _ = run_diffusion_process(
                            config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                            meas_p2, input_image, device, phase_name=f"P2_{u_mode}_{sub_key}"
                        )
                        
                        metrics_p2 = metric_wrapper(x_recon_p2.detach(), input_image)
                        
                        # Phase 2 Diffusionの平均値を記録
                        meter_key = f"{u_mode}_{sub_key}"
                        get_meter(meter_key).update(metrics_p2)
                        
                        mode_result["results"][sub_key] = {k: float(v) for k, v in metrics_p2.items()}
                        mode_result["results"][sub_key]['ratio'] = ratio
                        mode_result["results"][sub_key]['jscc'] = {k: float(v) for k, v in metrics_jscc_p2.items()}
                        
                        # ログ: Diffusion後の詳細表示
                        log_msg_p2 = f"    [{u_mode[:4]}-{sub_key:6s}] Ratio: {ratio:.2%} | PSNR: {metrics_p2['psnr']:.2f}dB"
                        if 'lpips' in metrics_p2: log_msg_p2 += f" | LPIPS: {metrics_p2['lpips']:.4f}"
                        if 'dists' in metrics_p2: log_msg_p2 += f" | DISTS: {metrics_p2['dists']:.4f}"
                        log_msg_p2 += f" | Corr: {corr:.3f}"
                        logger.info(log_msg_p2)

                        # 画像保存
                        torchvision.utils.save_image(x_recon_p2[0].cpu(), os.path.join(save_dir, f'3_P2_{u_mode}_{sub_key}.png'))
                        plt.imsave(os.path.join(save_dir, f'Mask_{u_mode}_{sub_key}.png'), mask_vis[0, 0].cpu().numpy(), cmap='gray')
                        
                        u_vis = u_map_tensor[0, 0].numpy()
                        u_vis = (u_vis - u_vis.min()) / (u_vis.max() - u_vis.min() + 1e-8)
                        plt.imsave(os.path.join(save_dir, f'Uncertainty_{u_mode}_{sub_key}.png'), u_vis, cmap='jet')
                
                batch_record["modes"][u_mode] = mode_result

            # -----------------------------------------------------------------
            # Random Baseline (Once per batch)
            # -----------------------------------------------------------------
            if config.retrans_mode != 'oracle':
                 meas_rnd, ratio_rnd, mask_vis_rnd, _ = simulate_semantic_retransmission(
                     operator, input_image, measurement_phase1, None, mode='random', value=config.retrans_value
                 )
                 
                 # Random JSCC品質
                 metrics_jscc_rnd = metric_wrapper(meas_rnd['x_mse'].detach(), input_image)
                 
                 # [重要] Random JSCCの平均値を記録
                 get_meter('random_jscc').update(metrics_jscc_rnd)
                 
                 # ログ: Random JSCC詳細
                 log_msg_jscc_rnd = f"    [Random      JSCC] PSNR: {metrics_jscc_rnd['psnr']:.2f}dB"
                 if 'lpips' in metrics_jscc_rnd: log_msg_jscc_rnd += f" | LPIPS: {metrics_jscc_rnd['lpips']:.4f}"
                 if 'dists' in metrics_jscc_rnd: log_msg_jscc_rnd += f" | DISTS: {metrics_jscc_rnd['dists']:.4f}"
                 logger.info(log_msg_jscc_rnd)

                 torch.manual_seed(config.seed + idx)
                 x_recon_rnd, _ = run_diffusion_process(
                     config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                     meas_rnd, input_image, device, phase_name="P2_Random"
                 )
                 metrics_rnd = metric_wrapper(x_recon_rnd.detach(), input_image)
                 
                 # Random Diffusionの平均値を記録
                 get_meter('random').update(metrics_rnd)
                 
                 batch_record['random'] = {k: float(v) for k, v in metrics_rnd.items()}
                 batch_record['random']['ratio'] = ratio_rnd
                 batch_record['random']['jscc'] = {k: float(v) for k, v in metrics_jscc_rnd.items()}
                 
                 # ログ: Random Diffusion詳細
                 log_msg_rnd = f"    [Random         ] Ratio: {ratio_rnd:.2%} | PSNR: {metrics_rnd['psnr']:.2f}dB"
                 if 'lpips' in metrics_rnd: log_msg_rnd += f" | LPIPS: {metrics_rnd['lpips']:.4f}"
                 if 'dists' in metrics_rnd: log_msg_rnd += f" | DISTS: {metrics_rnd['dists']:.4f}"
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
        
        output_data = {
            "summary": final_summary,
            "history": all_results_history
        }

        if len(all_results_history) > 0:
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Saved {len(all_results_history)} results to {json_path}")
        
        # 最終ログ表示
        logger.info("=== Final Comparison Summary ===")
        
        # 1. Base JSCC
        if 'jscc_init' in final_summary:
            m = final_summary['jscc_init']
            logger.info(f"Init (Base)  | PSNR: {m['psnr']:.2f}dB | LPIPS: {m.get('lpips',0):.4f} | DISTS: {m.get('dists',0):.4f}")
        
        # 2. Random
        if 'random' in final_summary:
            m = final_summary['random']
            logger.info(f"Random       | PSNR: {m['psnr']:.2f}dB | LPIPS: {m.get('lpips',0):.4f} | DISTS: {m.get('dists',0):.4f}")

        # 3. Dynamic Modes (JSCC results will appear here automatically because they are in keys)
        # ソートして出力することで、同じモードのJSCCとReconが近くに並びます
        for k in sorted(final_summary.keys()):
            if k in ['jscc_init', 'random', 'phase1_recon']: continue
            m = final_summary[k]
            # キー名が長くなる可能性があるため、フォーマット調整
            logger.info(f"{k:20s} | PSNR: {m['psnr']:.2f}dB | LPIPS: {m.get('lpips',0):.4f} | DISTS: {m.get('dists',0):.4f}")

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