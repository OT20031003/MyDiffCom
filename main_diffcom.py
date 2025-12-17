import argparse
import logging
import os
import os.path
import random
import shutil
import json  # ★追加: JSON保存用

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from tqdm.auto import tqdm
from scipy.stats import pearsonr

# diffcomモジュールをインポート
import conditioning_method.diffcom as diffcom_module
from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
from data.datasets import get_test_loader
from guided_diffusion.measurement import get_operator
from guided_diffusion.noise_schedule import NoiseSchedule
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from utils.util import Config, MetricWrapper, DictAverageMeter
from utils import util, utils_logger, utils_model


def evaluate_latent_correlation(uncertainty_map, x_recon, input_image, operator):
    """
    不確かさマップ(Pixel由来)と、潜在空間での誤差(Latent Error)の相関を計算する関数
    """
    if uncertainty_map is None:
        return 0.0, None, None

    # 1. 画像を潜在空間(z)へエンコード (GPUで実行)
    with torch.no_grad():
        z_gt_flat = operator.encode(input_image)
        z_recon_flat = operator.encode(x_recon)
        
        z_gt = None
        z_recon = None
        
        if hasattr(operator, 's_shape'): # DeepJSCC
            z_gt = z_gt_flat.reshape(operator.s_shape)
            z_recon = z_recon_flat.reshape(operator.s_shape)
        elif hasattr(operator, 's_masked_shape'): # NTSCC
            z_gt = z_gt_flat.reshape(operator.s_masked_shape)
            z_recon = z_recon_flat.reshape(operator.s_masked_shape)
        else:
            return 0.0, None, None

    # 2. 潜在空間での誤差マップを計算
    latent_error_map = torch.mean((z_gt - z_recon) ** 2, dim=1).detach().cpu()
    
    # 3. 不確かさマップのリサイズ
    u_map = uncertainty_map.detach().cpu()
    target_size = latent_error_map.shape[-2:]
    
    if u_map.shape[-2:] != target_size:
        try:
            u_map_resized = torch.nn.functional.interpolate(
                u_map, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        except Exception as e:
            return 0.0, None, None
    else:
        u_map_resized = u_map

    # 4. 相関の計算
    u_flat = u_map_resized.flatten().numpy()
    e_flat = latent_error_map.flatten().numpy()
    
    if np.std(u_flat) == 0 or np.std(e_flat) == 0:
        return 0.0, latent_error_map, u_map_resized

    corr, _ = pearsonr(u_flat, e_flat)
    
    return corr, latent_error_map, u_map_resized


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml', help="Path to option YMAL file.")
    args = parser.parse_args()
    # Load the YAML file
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    if config.conditioning_method == 'blind_diffcom':
        assert config.channel_type == 'ofdm_tdl'
        assert not config.CSNR_adapt_t_start

    cond_config = Config(config.getattr('diffcom_series'))
    conditioning_method = Config(cond_config.getattr(config.conditioning_method))
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt
    config.skip = cond_config.num_train_timesteps // cond_config.iter_num
    config.sigma = np.sqrt(1.0 / (2 * 10 ** (config.CSNR / 10)))

    # paths
    config.model_zoo = os.path.join(config.cwd, 'model_zoo')
    config.testsets = os.path.join(config.cwd, 'testsets')
    config.results = os.path.join(config.cwd, 'results')
    config.results = os.path.join(config.results, config.testset_name)
    config.results = os.path.join(config.results, config.conditioning_method)

    if config.operator_name == 'djscc':
        config.results = os.path.join(config.results, config.operator_name + '_{}'.format(config.djscc['channel_num']))
    elif config.operator_name == 'ntscc':
        if config.ntscc['compatible']:
            config.results = os.path.join(config.results, config.operator_name + '_{}_{}'.format(config.ntscc['eta'], config.ntscc['qp_level']))
        else:
            config.results = os.path.join(config.results, config.operator_name + '_plus_{}'.format(config.ntscc['qp_level']))

    config.results = os.path.join(config.results, f'{config.channel_type}_{config.CSNR.__str__().zfill(2)}dB')

    config.result_name = f'zeta{conditioning_method.zeta}'
    config.result_name += f'_seed{config.seed}'
    config.result_name += f'_gamma{conditioning_method.gamma}'
    config.result_name += f'_faststart_N{config.N}' if config.CSNR_adapt_t_start else ''
    if config.channel_type == 'ofdm_tdl':
        ofdm_config = Config(config.ofdm_tdl)
        config.result_name += '_BLIND_h_lr{}_'.format(
            conditioning_method.h_lr) if config.conditioning_method == 'blind_diffcom' else f'_{ofdm_config.channel_est}_{ofdm_config.equalization}'
        if ofdm_config.is_clip:
            config.result_name += '_CLIP{}'.format(ofdm_config.clip_ratio)
        if ofdm_config.K < ofdm_config.L:
            config.result_name += f'_ISI'

    config.result_name += f'_NFE{cond_config.iter_num}_{config.model_name}'
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


def p_sample_loop(config, noise_schedule, unet, diffusion, operator, cond_method, dataloader, device, logger):
    logger.info('【Config】: model_name: {}'.format(config.model_name))
    logger.info('【Config】: testset_name: {}'.format(config.testset_name))
    logger.info('【Config】: conditioning_method: {}'.format(config.conditioning_method))
    for key, value in config.diffcom_series[config.conditioning_method].items():
        logger.info('【Config】: {}: {}'.format(key, value))
    logger.info('【Config】: channel_type: {}'.format(config.channel_type))
    logger.info('【Config】: CSNR: {}'.format(config.CSNR))
    
    ofdm_config = Config(config.ofdm_tdl)
    logger.info('【Config】: {} channel estimation'.format(ofdm_config.channel_est))
    logger.info('【Config】: {} equalization'.format(ofdm_config.equalization))
    logger.info('【Config】: 【BLIND MODE】') if config.conditioning_method == 'blind_diffcom' else None

    metric_wrapper = MetricWrapper().to(device)
    results = DictAverageMeter()
    loss_wrapper = ConsistencyLoss(config, device)

    # ★追加: 全結果を保存するためのリスト
    all_results_history = []

    try: # ★追加: try-finallyブロックで囲むことで、中断時も保存されるようにする
        for idx, batch in enumerate(dataloader):
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]
            measurement = operator.observe_and_transpose(input_image)
            
            torch.manual_seed(config.seed + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed + 1)
            np.random.seed(config.seed + 1)
            random.seed(config.seed + 1)

            if config.channel_type == 'ofdm_tdl' and not (config.conditioning_method == 'blind_diffcom'):
                H_loss_gt = torch.linalg.norm(measurement['cof_est'] - measurement["cof_gt"])
                logger.info(f"batch{idx + 1:->4d}--> 【Init】 H_Loss cof_gt: {H_loss_gt:.4f}")

            util.mkdir(config.save_path + '/measurement')
            util.imsave_batch(util.tensor2uint_batch(measurement['x_mse']), names, config.save_path + '/measurement',
                              f"measurement_")
            
            # Baseline Metrics Calculation
            baseline_metric = metric_wrapper(measurement['x_mse'], input_image)
            cbr_val = measurement['channel_usage'] / measurement['x_mse'].numel()
            
            logger.info(
                f"batch{idx + 1:->4d}--> 【Baseline】"
                f"CBR: {cbr_val:.4f},"
                f"PSNR: {baseline_metric['psnr']:.2f}dB, "
                f"LPIPS: {baseline_metric['lpips']:.4f}, "
                f"DISTS: {baseline_metric['dists']:.4f}, "
                f"MSSSIM: {baseline_metric['msssim']:.4f}")

            # ★追加: Baselineデータを辞書に記録
            batch_record = {
                "batch_idx": idx + 1,
                "filename": names[0],
                "baseline": {
                    "CBR": float(cbr_val),
                    "PSNR": float(baseline_metric['psnr']),
                    "LPIPS": float(baseline_metric['lpips']),
                    "DISTS": float(baseline_metric['dists']),
                    "MSSSIM": float(baseline_metric['msssim'])
                },
                "reconstruction": {},
                "uncertainty": {}
            }

            x_init = noise_schedule.sqrt_alphas_cumprod[noise_schedule.t_start] * (2 * measurement['x_mse'] - 1) + \
                     noise_schedule.sqrt_1m_alphas_cumprod[
                         noise_schedule.t_start] * torch.randn_like(input_image)

            if config.conditioning_method == 'blind_diffcom':
                # plot measurement['cof_gt']... (omitted plotting code for brevity, logic remains)
                plt.clf()
                # ... (plotting logic) ...
                plt.close()

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
            psnr_list = []
            lpips_list = []
            dists_list = []
            L_m_list = []
            H_loss_list = []
            
            pbar = tqdm(range(len(seq)), ncols=140)
            for i in pbar:
                x_0_hat, h_0_hat, x_t, h_t, norm = cond_method(config, i, noise_schedule,
                                                               x_init if i == 0 else x_t,
                                                               cof_init if i == 0 else h_t,
                                                               power if config.conditioning_method == 'blind_diffcom' else None,
                                                               measurement, unet, diffusion, operator, loss_wrapper,
                                                               last_timestep=(seq[i] == seq[-1]))

                # ... (Intermediate Saving Logic) ...
                
                metrics_inter = metric_wrapper((x_0_hat / 2 + 0.5).detach(), input_image)
                
                l_m_val = norm['ofdm_sig'].item() if 'ofdm_sig' in norm.keys() else 0.0
                l_c_val = norm['x_mse'].item() if 'x_mse' in norm.keys() else 0.0
                
                message = {'t_step': seq[i],
                           'L_m': l_m_val,
                           'L_c': l_c_val,
                           'PSNR': metrics_inter['psnr']}
                           
                L_m_list.append(l_m_val)
                # L_c_list.append(l_c_val)

                if config.channel_type == 'ofdm_tdl':
                    h_dist_val = torch.linalg.norm(
                        h_t[..., :ofdm_config.L] - measurement["cof_gt"][..., :ofdm_config.L]).item()
                    H_loss_list.append(h_dist_val)
                    message['H_dist'] = h_dist_val
                else:
                    H_loss_list.append(0.0)

                pbar.set_postfix(message, refresh=True)
                psnr_list.append(metrics_inter['psnr'])
                lpips_list.append(metrics_inter['lpips'])
                dists_list.append(metrics_inter['dists'])

            # ... (Plotting Code) ...

            # --------------------------------
            # Final Metrics & Logging
            # --------------------------------
            x_recon = (x_t / 2 + 0.5)
            metrics = metric_wrapper(x_recon.detach(), input_image)
            metrics['L_m'] = L_m_list[-1]
            # metrics['L_c'] = L_c_list[-1]
            metrics['H_Loss'] = H_loss_list[-1]
            results.update(metrics)
            
            # ログ出力
            logger.info(
                f"batch{idx + 1:->4d}--> 【Recon】"
                f'H_Loss: {metrics["H_Loss"]:.4f},'
                f'L_m: {metrics["L_m"]:.4f},'
                f'L_c: 0.0000,' # L_c is currently 0 in this logic
                f"PSNR: {metrics['psnr']:.2f}dB, LPIPS: {metrics['lpips']:.4f}, "
                f"DISTS: {metrics['dists']:.4f}, MSSSIM: {metrics['msssim']:.4f}")

            # ★追加: Reconデータを辞書に記録
            batch_record["reconstruction"] = {
                "H_Loss": float(metrics["H_Loss"]),
                "L_m": float(metrics["L_m"]),
                "L_c": 0.0,
                "PSNR": float(metrics['psnr']),
                "LPIPS": float(metrics['lpips']),
                "DISTS": float(metrics['dists']),
                "MSSSIM": float(metrics['msssim'])
            }

            # =========================================================
            # 不確かさの相関評価 (Latent Space)
            # =========================================================
            u_map = diffcom_module.latest_uncertainty_map
            
            if u_map is not None:
                # Latent空間での相関を計算 (x_recon, input_image はGPUのまま)
                corr, lat_err, u_map_resized = evaluate_latent_correlation(
                    u_map, x_recon, input_image, operator
                )
                
                if lat_err is not None:
                    logger.info(f"Batch {idx}: Uncertainty-LatentError Correlation = {corr:.4f}")
                    
                    # ★追加: 相関データを記録
                    batch_record["uncertainty"]["latent_correlation"] = float(corr)
                    
                    # 可視化保存
                    save_debug_path = os.path.join(config.save_path, 'debug_maps', str(idx))
                    util.mkdir(save_debug_path)
                    
                    e_vis = lat_err[0].numpy()
                    e_vis = (e_vis - e_vis.min()) / (e_vis.max() - e_vis.min() + 1e-8)
                    plt.imsave(os.path.join(save_debug_path, 'latent_error.png'), e_vis, cmap='jet')
                    
                    if u_map_resized is not None:
                        u_vis = u_map_resized[0, 0].numpy()
                        u_vis = (u_vis - u_vis.min()) / (u_vis.max() - u_vis.min() + 1e-8)
                        plt.imsave(os.path.join(save_debug_path, 'resized_uncertainty.png'), u_vis, cmap='jet')
                    
                    torchvision.utils.save_image(x_recon[0].cpu(), os.path.join(save_debug_path, 'recon.png'))
                    torchvision.utils.save_image(input_image[0].cpu(), os.path.join(save_debug_path, 'gt.png'))
                
                diffcom_module.latest_uncertainty_map = None
            # =========================================================

            # ★追加: リストに追加
            all_results_history.append(batch_record)

            logger.info('--------------------------------------------')
            recon_image = util.tensor2uint_batch(x_recon)
            util.imsave_batch(recon_image, names, config.save_path + '/recon',
                              f"{config.model_name}_")

    except KeyboardInterrupt:
        logger.info("Execution Interrupted by User (Ctrl+C). Saving current results...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
    finally:
        # ★追加: 最後に必ずJSONファイルを保存する
        json_path = os.path.join(config.save_path, 'metrics_summary.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(all_results_history, f, indent=4)
            logger.info(f"-----------> Metrics saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics JSON: {e}")

    # --------------------------------
    # Average PSNR and LPIPS for all images
    # --------------------------------

    logger.info('-----------> Method: {}'.format(config.conditioning_method))
    logger.info('-----------> Average PSNR (RGB) of ({}), SNR: ({}): {} -> {}'.format(config.testset_name, config.CSNR,
                                                                                  baseline_metric['psnr'], results.avg['psnr']))
    logger.info('-----------> Average LPIPS of ({}), SNR: ({}): {} -> {}'.format(config.testset_name, config.CSNR,
                                                                           baseline_metric['lpips'], results.avg['lpips']))
    logger.info('-----------> Average DISTS of ({}), SNR: ({}): {} -> {}'.format(config.testset_name, config.CSNR,
                                                                           baseline_metric['dists'], results.avg['dists']))
    logger.info('-----------> Average MSSSIM of ({}), SNR: ({}): {} -> {}'.format(config.testset_name, config.CSNR,
                                                                            baseline_metric['msssim'], results.avg['msssim']))

    if config.conditioning_method == 'blind_diffcom':
        logger.info('-----------> Average H_Loss of ({}), SNR: ({}): {}'.format(config.testset_name, config.CSNR,
                                                                                results.avg['H_Loss']))

    logger.info(
        '-----------> Average Measurement Loss L_m of {}, SNR: {}dB: {}'.format(config.testset_name, config.CSNR,
                                                                                results.avg['L_m']))
    logger.info('-----------> Results Save to {}'.format(config.save_path))
    return results


def main():
    config = parse_args_and_config()
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    config.device = device

    # set up logger
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
    args = utils_model.create_argparser(model_config).parse_args([])
    unet, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    unet.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    unet.eval()

    unet = unet.to(device)

    # save config
    shutil.copyfile(config.opt, os.path.join(config.save_path, os.path.basename('config.yaml')))

    # get operator
    operator = get_operator(config.operator_name, config=config, logger=logger, device=device)
    operator.model = operator.model.to(device)
    ns = NoiseSchedule(config, logger, device)

    cond_method = get_conditioning_method(name=config.conditioning_method)

    cond_method = cond_method.conditioning
    p_sample_loop(config, ns, unet, diffusion, operator, cond_method, dataloader, device, logger)


if __name__ == '__main__':
    main()