import argparse
import logging
import os
import os.path
import random
import shutil
import json
import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
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
# 提案法: ピクセルベース再送シミュレーション関数 (Patch & Re-encode)
# =========================================================================
def simulate_pixel_retransmission(operator, input_image, measurement, uncertainty_map, 
                                  mode='rate', value=0.1, logger=None):
    """
    ピクセルベースの再送シミュレーション (Patch & Re-encode)
    """
    device = input_image.device
    
    # 1. 現在の復元画像
    x_current = measurement['x_mse']
    
    # 2. マスクの生成 (Pixel空間)
    if mode == 'oracle':
        # Oracle: 画素ごとの誤差が大きい場所
        diff = torch.abs(x_current - input_image).mean(dim=1, keepdim=True)
        B = diff.shape[0]
        diff_flat = diff.view(B, -1)
        k = int(diff_flat.shape[1] * value)
        if k < 1: k = 1
        
        top_val, _ = torch.topk(diff_flat, k, dim=1)
        thresh = top_val[:, -1].view(B, 1, 1, 1)
        mask = (diff >= thresh).float()
        
    else:
        # Uncertainty Map利用
        if uncertainty_map is None:
             return measurement, 0.0, torch.zeros_like(x_current[:, :1])
             
        u_map = uncertainty_map.to(device)
        
        if mode == 'rate':
            B = u_map.shape[0]
            u_flat = u_map.view(B, -1)
            k = int(u_flat.shape[1] * value)
            if k < 1: k = 1
            top_val, _ = torch.topk(u_flat, k, dim=1)
            thresh = top_val[:, -1].view(B, 1, 1, 1)
            mask = (u_map >= thresh).float()
        else: # threshold
            mask = (u_map > value).float()

    mask_vis = mask
    retransmission_ratio = mask.mean().item()

    # 3. 画像のパッチング
    x_patched = x_current * (1 - mask) + input_image * mask

    # 4. 観測値の更新
    new_measurement = copy.deepcopy(measurement)
    new_measurement['x_mse'] = x_patched
    
    with torch.no_grad():
        s_new = operator.encode(x_patched)
        
        if hasattr(operator, 'channel'):
            avg_pwr = operator.channel.avg_pwr if hasattr(operator.channel, 'avg_pwr') else 1.0
            channel_type = operator.channel.channel_type if hasattr(operator.channel, 'channel_type') else 'awgn'
            
            pwr_tensor = torch.as_tensor(avg_pwr * 2.0, device=device).float()
            
            if channel_type == 'awgn':
                 y_clean = s_new / torch.sqrt(pwr_tensor)
                 new_measurement['ofdm_sig'] = y_clean
            else:
                 new_measurement['ofdm_sig'] = s_new
        else:
            new_measurement['ofdm_sig'] = s_new

        if new_measurement.get('cof_est') is not None:
             new_measurement['cof_est'] = torch.ones_like(new_measurement['cof_est'])

    return new_measurement, retransmission_ratio, mask_vis


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
    # フォルダ名を区別
    config.results = os.path.join(config.cwd, 'results_retrans_hard') 
    config.results = os.path.join(config.results, config.testset_name)
    config.results = os.path.join(config.results, config.conditioning_method)

    if config.operator_name == 'djscc':
        config.results = os.path.join(config.results, config.operator_name + '_{}'.format(config.djscc['channel_num']))
    
    config.results = os.path.join(config.results, f'{config.channel_type}_{config.CSNR.__str__().zfill(2)}dB')
    
    config.result_name = f'RetransPixel_{config.retrans_mode}_{config.retrans_value}'
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
                          measurement, input_image, device, phase_name="Phase1", 
                          hard_mask=None):
    ofdm_config = Config(config.ofdm_tdl)
    metric_wrapper = MetricWrapper().to(device)
    
    x_ref = measurement['x_mse'] 
    
    # 初期値の設定
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
    
    # tqdmの設定 (LPIPS表示用に幅を確保)
    pbar = tqdm(range(len(seq)), ncols=165, desc=f"{phase_name}", leave=False)
    
    for i in pbar:
        t_step = seq[i]
        
        # 1. DiffCom Step
        x_0_hat, h_0_hat, x_t_prev, h_t_prev, norm = cond_method(
            config, i, noise_schedule,
            x_init if i == 0 else x_t,
            cof_init if i == 0 else h_t,
            power if config.conditioning_method == 'blind_diffcom' else None,
            measurement, unet, diffusion, operator, 
            loss_wrapper=None,
            last_timestep=(seq[i] == seq[-1])
        )
        
        # 2. Hard Inpainting (Replacement) Step
        if hard_mask is not None:
            x_ref_norm = 2 * x_ref - 1
            noise = torch.randn_like(x_ref_norm)
            
            if i < len(seq) - 1:
                t_next = seq[i+1]
                sqrt_alpha_next = noise_schedule.sqrt_alphas_cumprod[t_next]
                sqrt_1m_alpha_next = noise_schedule.sqrt_1m_alphas_cumprod[t_next]
                x_known_next = sqrt_alpha_next * x_ref_norm + sqrt_1m_alpha_next * noise
                
                # 上書き
                x_t_prev = x_t_prev * (1 - hard_mask) + x_known_next * hard_mask
            else:
                x_t_prev = x_t_prev * (1 - hard_mask) + x_ref_norm * hard_mask

        # 変数更新
        x_t = x_t_prev
        h_t = h_t_prev
        
        # メトリクス計算と表示 (LPIPS追加)
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
    results = DictAverageMeter()
    
    loss_wrapper = ConsistencyLoss(config, device)
    
    def wrapped_cond_method(*args, **kwargs):
        kwargs['loss_wrapper'] = loss_wrapper
        return cond_method(*args, **kwargs)

    all_results_history = []

    for idx, batch in enumerate(dataloader):
        input_image, names = batch
        input_image = input_image.to(device)
        config.batch_size = input_image.shape[0]
        
        # Step 0: Initial Transmission
        measurement_phase1 = operator.observe_and_transpose(input_image)
        
        torch.manual_seed(config.seed + idx)
        
        # Step 1: Phase 1
        logger.info(f"Batch {idx+1}/{len(dataloader)}: Phase 1 (Initial Reconstruction)...")
        diffcom_module.latest_uncertainty_map = None
        
        x_recon_p1, uncertainty_map_p1 = run_diffusion_process(
            config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
            measurement_phase1, input_image, device, phase_name="Phase1"
        )
        
        metrics_p1 = metric_wrapper(x_recon_p1.detach(), input_image)
        # main_diffcom風のログ出力
        logger.info(f"  -> Phase 1 Results | PSNR: {metrics_p1['psnr']:.2f}dB | LPIPS: {metrics_p1['lpips']:.4f} | DISTS: {metrics_p1['dists']:.4f}")

        # Phase 1 JSCC精度
        metrics_jscc_p1 = metric_wrapper(measurement_phase1['x_mse'].detach(), input_image)
        logger.info(f"  -> [Diagnosis] Init JSCC | PSNR: {metrics_jscc_p1['psnr']:.2f}dB | LPIPS: {metrics_jscc_p1['lpips']:.4f}")

        # Step 2: Retransmission (Pixel-based)
        logger.info(f"Batch {idx+1}/{len(dataloader)}: Processing Pixel-based Retransmission...")
        
        if config.retrans_mode == 'oracle' or uncertainty_map_p1 is not None:
            measurement_phase2, ret_ratio, mask_vis = simulate_pixel_retransmission(
                operator, input_image, measurement_phase1, 
                uncertainty_map_p1, 
                mode=config.retrans_mode,
                value=config.retrans_value,
                logger=logger
            )
            logger.info(f"  -> Retransmission Ratio: {ret_ratio:.2%} (Target={config.retrans_value})")
            
            metrics_jscc_p2 = metric_wrapper(measurement_phase2['x_mse'].detach(), input_image)
            jscc_gain = metrics_jscc_p2['psnr'] - metrics_jscc_p1['psnr']
            logger.info(f"  -> [Diagnosis] After Patching | PSNR: {metrics_jscc_p2['psnr']:.2f}dB (+{jscc_gain:.2f}dB)")
            
            hard_mask_for_phase2 = mask_vis 

        else:
            logger.warning("  -> No uncertainty map found! Skipping retransmission.")
            measurement_phase2 = measurement_phase1
            ret_ratio = 0.0
            mask_vis = torch.zeros_like(input_image[:, :1, :, :])
            hard_mask_for_phase2 = None

        # Step 3: Phase 2 (Refinement with Hard Constraint)
        logger.info(f"Batch {idx+1}/{len(dataloader)}: Phase 2 (Refinement)...")
        
        x_recon_p2, _ = run_diffusion_process(
            config, noise_schedule, unet, diffusion, operator, wrapped_cond_method,
            measurement_phase2, input_image, device, phase_name="Phase2",
            hard_mask=hard_mask_for_phase2
        )
        
        metrics_p2 = metric_wrapper(x_recon_p2.detach(), input_image)
        # main_diffcom風のログ出力
        logger.info(f"  -> Phase 2 Results | PSNR: {metrics_p2['psnr']:.2f}dB | LPIPS: {metrics_p2['lpips']:.4f} | DISTS: {metrics_p2['dists']:.4f}")
        logger.info(f"  -> Gain vs Phase 1 | PSNR: +{metrics_p2['psnr'] - metrics_p1['psnr']:.2f}dB | LPIPS: {metrics_p2['lpips'] - metrics_p1['lpips']:.4f}")
        
        results.update(metrics_p2)
        batch_record = {
            "batch_idx": idx + 1,
            "filename": names[0],
            "retransmission_ratio": ret_ratio,
            "phase1": {k: float(v) for k, v in metrics_p1.items()},
            "phase2": {k: float(v) for k, v in metrics_p2.items()}
        }
        all_results_history.append(batch_record)

        # 画像保存
        save_dir = os.path.join(config.save_path, 'visuals', str(idx))
        util.mkdir(save_dir)
        torchvision.utils.save_image(input_image[0].cpu(), os.path.join(save_dir, '0_GT.png'))
        torchvision.utils.save_image(measurement_phase1['x_mse'][0].cpu(), os.path.join(save_dir, '1_JSCC_Init.png'))
        torchvision.utils.save_image(measurement_phase2['x_mse'][0].cpu(), os.path.join(save_dir, '1_JSCC_Patched.png'))
        torchvision.utils.save_image(x_recon_p1[0].cpu(), os.path.join(save_dir, '2_Phase1_Recon.png'))
        torchvision.utils.save_image(x_recon_p2[0].cpu(), os.path.join(save_dir, '3_Phase2_Refined.png'))
        
        if uncertainty_map_p1 is not None:
            u_vis = uncertainty_map_p1[0, 0].cpu().numpy()
            u_vis = (u_vis - u_vis.min()) / (u_vis.max() - u_vis.min() + 1e-8)
            plt.imsave(os.path.join(save_dir, 'Phase1_Uncertainty.png'), u_vis, cmap='jet')
        
        m_vis = mask_vis[0, 0].cpu().numpy()
        plt.imsave(os.path.join(save_dir, 'Retransmission_Mask.png'), m_vis, cmap='gray')

        logger.info('--------------------------------------------')

    json_path = os.path.join(config.save_path, 'retransmission_summary.json')
    with open(json_path, 'w') as f:
        json.dump(all_results_history, f, indent=4)
        
    logger.info(f'Average Phase 2 Results | PSNR: {results.avg["psnr"]:.2f}dB | LPIPS: {results.avg["lpips"]:.4f}')
    return results


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