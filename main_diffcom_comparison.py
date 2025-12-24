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
    
    # -------------------------------------------------------------------------
    # 1. 送信時の状態(State)の復元
    # -------------------------------------------------------------------------
    channel_wrapper = operator.channel
    
    if not hasattr(channel_wrapper, 'shuffled_indices') or channel_wrapper.shuffled_indices is None:
        if logger: logger.warning("Channel indices not found. Is this run after observe? Skipping.")
        return measurement, 0.0, None, None

    # インデックスをGPUへ
    saved_indices = channel_wrapper.shuffled_indices.to(device)
    saved_avg_pwr = channel_wrapper.avg_pwr
    
    # -------------------------------------------------------------------------
    # 2. 理想的な受信信号 (y_clean) の生成
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 3. 再送マスクの生成 (Pixel -> Latent)
    # -------------------------------------------------------------------------
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
        
    else:
        # 通常モード: 不確かさマップに基づく
        if uncertainty_map is None:
            return measurement, 0.0, None, None

        u_map = uncertainty_map.to(device) # [B, 1, H, W]
        
        # 3-1. Latentサイズの取得
        if hasattr(operator, 's_shape'):
            latent_H, latent_W = operator.s_shape[2], operator.s_shape[3]
            C_feat = operator.s_shape[1]
        else:
            latent_H, latent_W = input_image.shape[2] // 16, input_image.shape[3] // 16
            C_feat = s_raw.shape[1] // (latent_H * latent_W)

        # 3-2. ダウンサンプリング
        # Rawマップの場合、ここでLatentサイズへのグリッド化（平均）が行われる
        u_map_lat = F.adaptive_avg_pool2d(u_map, output_size=(latent_H, latent_W))
        
        # 3-3. マスク生成 (空間方向 1xHxW)
        if mode == 'rate':
            u_flat = u_map_lat.view(B, -1)
            k = int(u_flat.shape[1] * value)
            if k < 1: k = 1
            top_val, _ = torch.topk(u_flat, k, dim=1)
            thresh = top_val[:, -1].view(B, 1, 1, 1)
            mask_lat_spatial = (u_map_lat >= thresh).float()
        else: 
            mask_lat_spatial = (u_map_lat > value).float()

        # 可視化用
        mask_vis = F.interpolate(mask_lat_spatial, size=input_image.shape[-2:], mode='nearest')
        
        # 3-4. チャネル方向への拡張
        mask_expanded = mask_lat_spatial.repeat(1, C_feat, 1, 1)
        
        # 3-5. フラット化とシャッフル
        mask_flat = mask_expanded.view(B, -1)
        
        target_len = indices_expanded.shape[1]
        current_len = mask_flat.shape[1]
        
        if current_len != target_len:
            if logger: logger.warning(f"Resizing mask from {current_len} to {target_len}")
            if current_len < target_len:
                padding = torch.zeros(B, target_len - current_len, device=device)
                mask_flat = torch.cat([mask_flat, padding], dim=1)
            else:
                mask_flat = mask_flat[:, :target_len]

        mask_shuffled = torch.gather(mask_flat, 1, indices_expanded)
        mask_for_y = mask_shuffled.view(y_dirty.shape)

    retransmission_ratio = mask_for_y.float().mean().item()

    # -------------------------------------------------------------------------
    # 4. 観測値の更新 (Signal Replacement)
    # -------------------------------------------------------------------------
    new_measurement = copy.deepcopy(measurement)
    
    # (A) 受信信号 y の更新
    if y_clean.shape != y_dirty.shape:
        y_clean = y_clean.view(y_dirty.shape)
        
    y_new = (1 - mask_for_y) * y_dirty + mask_for_y * y_clean
    new_measurement['ofdm_sig'] = y_new
    
    # (B) チャネル推定値 H の更新
    h_dirty = measurement.get('cof_est', None)
    if h_dirty is not None:
         if h_dirty.shape == y_dirty.shape:
             h_new = (1 - mask_for_y) * h_dirty + mask_for_y * torch.ones_like(h_dirty)
             new_measurement['cof_est'] = h_new

    # (C) ガイド画像 x_mse の更新
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
    
    config.result_name = f'RetransComparison_{config.retrans_mode}_{config.retrans_value}'
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
    metric_wrapper = MetricWrapper().to(device)
    
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
    
    pbar = tqdm(range(len(seq)), ncols=165, desc=f"{phase_name}", leave=False)
    
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
        
        with torch.no_grad():
            metrics_inter = metric_wrapper((x_0_hat / 2 + 0.5).detach(), input_image)
        
        l_m_val = norm['ofdm_sig'].item() if 'ofdm_sig' in norm.keys() else 0.0
        
        message = {
            't': t_step,
            'L_m': f"{l_m_val:.3f}",
            'PSNR': f"{metrics_inter['psnr']:.2f}",
            'LPIPS': f"{metrics_inter['lpips']:.3f}", 
        }
        pbar.set_postfix(message, refresh=True)

    x_recon = (x_t / 2 + 0.5)
    final_uncertainty = diffcom_module.latest_uncertainty_map
    
    return x_recon, final_uncertainty


def p_sample_loop(config, noise_schedule, unet, diffusion, operator, cond_method, dataloader, device, logger):
    logger.info(f'【Config】: Retransmission Mode: {config.retrans_mode}, Value: {config.retrans_value}')
    
    metric_wrapper = MetricWrapper().to(device)
    
    # メトリクス記録用
    results_smooth = DictAverageMeter()
    results_raw = DictAverageMeter()
    
    loss_wrapper = ConsistencyLoss(config, device)
    
    def wrapped_cond_method(*args, **kwargs):
        kwargs['loss_wrapper'] = loss_wrapper
        return cond_method(*args, **kwargs)

    all_results_history = []
    json_path = os.path.join(config.save_path, 'retransmission_comparison.json')

    try:
        for idx, batch in enumerate(dataloader):
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]
            
            # Step 0: Initial Transmission
            measurement_phase1 = operator.observe_and_transpose(input_image)
            
            torch.manual_seed(config.seed + idx)
            
            # ---------------------------------------------------------------------
            # Step 1: Phase 1 (Initial Reconstruction)
            # ---------------------------------------------------------------------
            logger.info(f"Batch {idx+1}/{len(dataloader)}: Phase 1 (Initial)...")
            diffcom_module.latest_uncertainty_map = None
            
            x_recon_p1, uncertainty_container_p1 = run_diffusion_process(
                config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                measurement_phase1, input_image, device, phase_name="Phase1"
            )
            
            metrics_p1 = metric_wrapper(x_recon_p1.detach(), input_image)
            logger.info(f"  -> Phase 1 | PSNR: {metrics_p1['psnr']:.2f}dB | LPIPS: {metrics_p1['lpips']:.4f} | DISTS: {metrics_p1['dists']:.4f}")

            # 【追加】Phase 1 JSCC (Baseline) の診断
            metrics_jscc_p1 = metric_wrapper(measurement_phase1['x_mse'].detach(), input_image)
            log_msg_jscc = f"  -> [Base JSCC] Init | PSNR: {metrics_jscc_p1['psnr']:.2f}dB"
            if 'lpips' in metrics_jscc_p1: log_msg_jscc += f" | LPIPS: {metrics_jscc_p1['lpips']:.4f}"
            if 'dists' in metrics_jscc_p1: log_msg_jscc += f" | DISTS: {metrics_jscc_p1['dists']:.4f}"
            logger.info(log_msg_jscc)

            # 相関分析 & マップ準備
            u_smooth, u_raw = None, None
            corr_smooth, corr_raw = 0.0, 0.0

            if uncertainty_container_p1 is not None:
                # 誤差マップ計算
                error_map = torch.abs(x_recon_p1 - input_image).mean(dim=1, keepdim=True)
                e_flat = error_map.detach().cpu().flatten().numpy()
                
                u_smooth = uncertainty_container_p1.get('smoothed', None)
                u_raw = uncertainty_container_p1.get('raw', None)
                
                if u_smooth is not None:
                    corr_smooth, _ = pearsonr(u_smooth.detach().cpu().flatten().numpy(), e_flat)
                
                if u_raw is not None:
                    if u_raw.shape[-2:] != error_map.shape[-2:]:
                        u_raw_resized = F.interpolate(u_raw, size=error_map.shape[-2:], mode='bilinear')
                        u_raw_flat = u_raw_resized.detach().cpu().flatten().numpy()
                    else:
                        u_raw_flat = u_raw.detach().cpu().flatten().numpy()
                    corr_raw, _ = pearsonr(u_raw_flat, e_flat)
                
                logger.info(f"  -> [Corr] Smoothed: {corr_smooth:.4f} | Raw: {corr_raw:.4f}")
            
            # ---------------------------------------------------------------------
            # Step 2 & 3: Retransmission Simulation & Phase 2 Refinement
            # ---------------------------------------------------------------------
            
            # 基本情報を記録 (Phase2の結果は後で追加)
            batch_record = {
                "batch_idx": idx + 1,
                "filename": names[0],
                "phase1": {k: float(v) for k, v in metrics_p1.items()},
                "jscc_init": {k: float(v) for k, v in metrics_jscc_p1.items()},
                "correlation": {"smooth": corr_smooth, "raw": corr_raw},
                "phase2_smooth": None,
                "phase2_raw": None
            }
            
            save_dir = os.path.join(config.save_path, 'visuals', str(idx))
            util.mkdir(save_dir)
            torchvision.utils.save_image(input_image[0].cpu(), os.path.join(save_dir, '0_GT.png'))
            torchvision.utils.save_image(measurement_phase1['x_mse'][0].cpu(), os.path.join(save_dir, '1_JSCC_Init.png'))
            torchvision.utils.save_image(x_recon_p1[0].cpu(), os.path.join(save_dir, '2_Phase1_Recon.png'))
            
            # ====== Branch A: Smoothed Map ======
            if config.retrans_mode == 'oracle' or u_smooth is not None:
                logger.info(f"  -> Processing Phase 2 (Smoothed)...")
                
                meas_p2_s, ratio_s, mask_vis_s, _ = simulate_semantic_retransmission(
                    operator, input_image, measurement_phase1, 
                    u_smooth if config.retrans_mode != 'oracle' else None,
                    mode=config.retrans_mode, value=config.retrans_value, logger=None
                )

                # 【追加】再送後のJSCC品質評価
                metrics_jscc_p2_s = metric_wrapper(meas_p2_s['x_mse'].detach(), input_image)
                log_msg_jscc_s = f"     [Smooth JSCC] PSNR: {metrics_jscc_p2_s['psnr']:.2f}dB"
                if 'lpips' in metrics_jscc_p2_s: log_msg_jscc_s += f" | LPIPS: {metrics_jscc_p2_s['lpips']:.4f}"
                if 'dists' in metrics_jscc_p2_s: log_msg_jscc_s += f" | DISTS: {metrics_jscc_p2_s['dists']:.4f}"
                logger.info(log_msg_jscc_s)
                
                x_recon_p2_s, _ = run_diffusion_process(
                    config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                    meas_p2_s, input_image, device, phase_name="P2_Smooth"
                )
                
                metrics_p2_s = metric_wrapper(x_recon_p2_s.detach(), input_image)
                results_smooth.update(metrics_p2_s)
                
                # 詳細なログ出力を作成
                log_msg = f"     [Smooth] Ratio: {ratio_s:.2%} | PSNR: {metrics_p2_s['psnr']:.2f}dB"
                if 'lpips' in metrics_p2_s: log_msg += f" | LPIPS: {metrics_p2_s['lpips']:.4f}"
                if 'dists' in metrics_p2_s: log_msg += f" | DISTS: {metrics_p2_s['dists']:.4f}"
                logger.info(log_msg)
                
                # JSON用データの更新
                batch_record['phase2_smooth'] = {k: float(v) for k, v in metrics_p2_s.items()}
                batch_record['phase2_smooth']['ratio'] = ratio_s
                batch_record['phase2_smooth']['jscc'] = {k: float(v) for k, v in metrics_jscc_p2_s.items()}
                
                torchvision.utils.save_image(x_recon_p2_s[0].cpu(), os.path.join(save_dir, '3_Phase2_Refined_Smooth.png'))
                plt.imsave(os.path.join(save_dir, 'Mask_Smooth.png'), mask_vis_s[0, 0].cpu().numpy(), cmap='gray')
                
                if u_smooth is not None:
                    u_vis = u_smooth[0, 0].cpu().numpy()
                    u_vis = (u_vis - u_vis.min()) / (u_vis.max() - u_vis.min() + 1e-8)
                    plt.imsave(os.path.join(save_dir, 'Uncertainty_Smooth.png'), u_vis, cmap='jet')

            # ====== Branch B: Raw Map ======
            if config.retrans_mode != 'oracle' and u_raw is not None:
                logger.info(f"  -> Processing Phase 2 (Raw)...")
                
                meas_p2_r, ratio_r, mask_vis_r, _ = simulate_semantic_retransmission(
                    operator, input_image, measurement_phase1, 
                    u_raw, 
                    mode=config.retrans_mode, value=config.retrans_value, logger=None
                )
                
                # 【追加】再送後のJSCC品質評価
                metrics_jscc_p2_r = metric_wrapper(meas_p2_r['x_mse'].detach(), input_image)
                log_msg_jscc_r = f"     [Raw JSCC   ] PSNR: {metrics_jscc_p2_r['psnr']:.2f}dB"
                if 'lpips' in metrics_jscc_p2_r: log_msg_jscc_r += f" | LPIPS: {metrics_jscc_p2_r['lpips']:.4f}"
                if 'dists' in metrics_jscc_p2_r: log_msg_jscc_r += f" | DISTS: {metrics_jscc_p2_r['dists']:.4f}"
                logger.info(log_msg_jscc_r)
                
                x_recon_p2_r, _ = run_diffusion_process(
                    config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
                    meas_p2_r, input_image, device, phase_name="P2_Raw"
                )
                
                metrics_p2_r = metric_wrapper(x_recon_p2_r.detach(), input_image)
                results_raw.update(metrics_p2_r)
                
                # 詳細なログ出力を作成
                log_msg = f"     [Raw   ] Ratio: {ratio_r:.2%} | PSNR: {metrics_p2_r['psnr']:.2f}dB"
                if 'lpips' in metrics_p2_r: log_msg += f" | LPIPS: {metrics_p2_r['lpips']:.4f}"
                if 'dists' in metrics_p2_r: log_msg += f" | DISTS: {metrics_p2_r['dists']:.4f}"
                logger.info(log_msg)
                
                # JSON用データの更新
                batch_record['phase2_raw'] = {k: float(v) for k, v in metrics_p2_r.items()}
                batch_record['phase2_raw']['ratio'] = ratio_r
                batch_record['phase2_raw']['jscc'] = {k: float(v) for k, v in metrics_jscc_p2_r.items()}
                
                torchvision.utils.save_image(x_recon_p2_r[0].cpu(), os.path.join(save_dir, '3_Phase2_Refined_Raw.png'))
                plt.imsave(os.path.join(save_dir, 'Mask_Raw.png'), mask_vis_r[0, 0].cpu().numpy(), cmap='gray')
                
                u_raw_vis = u_raw[0, 0].cpu().numpy()
                u_raw_vis = (u_raw_vis - u_raw_vis.min()) / (u_raw_vis.max() - u_raw_vis.min() + 1e-8)
                plt.imsave(os.path.join(save_dir, 'Uncertainty_Raw.png'), u_raw_vis, cmap='jet')
                
            elif config.retrans_mode == 'oracle':
                pass

            all_results_history.append(batch_record)
            logger.info('--------------------------------------------')

    except KeyboardInterrupt:
        logger.info("\n[!] Process Interrupted by User. Saving current results...")
    except Exception as e:
        logger.error(f"\n[!] Unexpected Error Occurred: {e}. Saving current results...")
        import traceback
        traceback.print_exc()
    finally:
        # 最後に必ず保存 (中断・エラー・正常終了すべてに対応)
        if len(all_results_history) > 0:
            with open(json_path, 'w') as f:
                json.dump(all_results_history, f, indent=4)
            logger.info(f"Saved {len(all_results_history)} results to {json_path}")
        
        logger.info(f'Final Avg | P2(Smooth) PSNR: {results_smooth.avg.get("psnr", 0):.2f}dB | P2(Raw) PSNR: {results_raw.avg.get("psnr", 0):.2f}dB')

    return results_smooth


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