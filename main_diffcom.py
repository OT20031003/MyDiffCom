import argparse
import logging
import os
import os.path
import random
import shutil
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from tqdm.auto import tqdm
from scipy.stats import pearsonr

# ==========================================
# カスタムモジュールのインポート
# ==========================================
# conditioning_method: 拡散モデルの逆拡散過程で条件付けを行うためのモジュール
import conditioning_method.diffcom as diffcom_module
from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
# data: データセットの読み込み機能
from data.datasets import get_test_loader
# guided_diffusion: 拡散モデル(Guided Diffusion)のコア機能（演算子、ノイズスケジュール、モデル定義など）
from guided_diffusion.measurement import get_operator
from guided_diffusion.noise_schedule import NoiseSchedule
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
# utils: 設定管理、メトリクス計算、ログ記録などのユーティリティ
from utils.util import Config, MetricWrapper, DictAverageMeter
from utils import util, utils_logger, utils_model


def evaluate_latent_correlation(uncertainty_map, x_recon, input_image, operator):
    """
    不確かさマップ(Pixel由来)と、潜在空間での誤差(Latent Error)の相関を計算する関数
    
    役割:
    モデルが「ここは自信がない（不確かさが高い）」と判断した場所が、
    実際に「復元に失敗している（誤差が大きい）」場所と一致しているかを確認します。
    """
    if uncertainty_map is None:
        return 0.0, None, None

    # 1. 画像を潜在空間(z)へエンコード (GPUで実行)
    # ここでは、画素データ(Pixel)を、モデル内部の表現(Latent/特徴量)に変換しています。
    with torch.no_grad():
        print(f"input_image.shape = {input_image.shape}")
        
        z_gt_flat = operator.encode(input_image)  # 正解画像の潜在表現
        z_recon_flat = operator.encode(x_recon)   # 復元画像の潜在表現
        
        z_gt = None
        z_recon = None
        print(f"z.shape = {z_gt_flat.shape}")
        # モデルの種類(DeepJSCCかNTSCCか)によって形状を整えます
        if hasattr(operator, 's_shape'): # DeepJSCCの場合
            z_gt = z_gt_flat.reshape(operator.s_shape)
            z_recon = z_recon_flat.reshape(operator.s_shape)
        elif hasattr(operator, 's_masked_shape'): # NTSCCの場合
            z_gt = z_gt_flat.reshape(operator.s_masked_shape)
            z_recon = z_recon_flat.reshape(operator.s_masked_shape)
        else:
            return 0.0, None, None

    # 2. 潜在空間での誤差マップを計算
    # 特徴量レベルでどれくらいズレているかを計算します (平均二乗誤差)
    latent_error_map = torch.mean((z_gt - z_recon) ** 2, dim=1).detach().cpu()
    
    # 3. 不確かさマップのリサイズ
    # 不確かさマップと誤差マップのサイズが違う場合、比較できるようにサイズを合わせます
    u_map = uncertainty_map.detach().cpu()
    print(f"u_map shape = {u_map.shape}")
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
    # マップを1次元配列にならして、ピアソンの相関係数を計算します
    u_flat = u_map_resized.flatten().numpy()
    e_flat = latent_error_map.flatten().numpy()
    
    # 標準偏差が0（値が全部同じ）だと相関が計算できないためチェック
    if np.std(u_flat) == 0 or np.std(e_flat) == 0:
        return 0.0, latent_error_map, u_map_resized

    corr, _ = pearsonr(u_flat, e_flat)
    
    return corr, latent_error_map, u_map_resized


def parse_args_and_config():
    """
    コマンドライン引数の解析と、設定ファイル(YAML)の読み込みを行う関数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml', help="Path to option YMAL file.")
    args = parser.parse_args()
    
    # YAMLファイルをロード
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    
    # 特殊なモード(blind_diffcom)の場合の整合性チェック
    if config.conditioning_method == 'blind_diffcom':
        assert config.channel_type == 'ofdm_tdl'
        assert not config.CSNR_adapt_t_start

    # 設定値の整理と計算
    cond_config = Config(config.getattr('diffcom_series'))
    conditioning_method = Config(cond_config.getattr(config.conditioning_method))
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt
    config.skip = cond_config.num_train_timesteps // cond_config.iter_num # ステップのスキップ幅
    config.sigma = np.sqrt(1.0 / (2 * 10 ** (config.CSNR / 10))) # ノイズレベル(Sigma)の計算

    # パス(保存先など)の設定
    config.model_zoo = os.path.join(config.cwd, 'model_zoo') # 学習済みモデル置き場
    config.testsets = os.path.join(config.cwd, 'testsets')   # テスト用画像置き場
    config.results = os.path.join(config.cwd, 'results')     # 結果出力先
    config.results = os.path.join(config.results, config.testset_name)
    config.results = os.path.join(config.results, config.conditioning_method)

    # オペレータ（通信路モデル）ごとの結果保存フォルダ分け
    if config.operator_name == 'djscc':
        config.results = os.path.join(config.results, config.operator_name + '_{}'.format(config.djscc['channel_num']))
    elif config.operator_name == 'ntscc':
        if config.ntscc['compatible']:
            config.results = os.path.join(config.results, config.operator_name + '_{}_{}'.format(config.ntscc['eta'], config.ntscc['qp_level']))
        else:
            config.results = os.path.join(config.results, config.operator_name + '_plus_{}'.format(config.ntscc['qp_level']))

    config.results = os.path.join(config.results, f'{config.channel_type}_{config.CSNR.__str__().zfill(2)}dB')

    # 結果フォルダ名の生成（パラメータ情報を含める）
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
    util.mkdir(config.save_path) # フォルダ作成

    # 再現性確保のために乱数シードを固定
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    return config


def p_sample_loop(config, noise_schedule, unet, diffusion, operator, cond_method, dataloader, device, logger):
    """
    推論（サンプリング）を行うメインループ
    データセットの各バッチに対して、通信路シミュレーションを行い、拡散モデルで復元を試みます。
    """
    # ログへの設定情報出力
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

    # メトリクス（評価指標）計算用ツールの準備
    metric_wrapper = MetricWrapper().to(device)
    results = DictAverageMeter() # 結果の平均値を管理するクラス
    loss_wrapper = ConsistencyLoss(config, device)

    all_results_history = [] # 全バッチの結果を保存するリスト

    try:
        # データローダーからバッチごとに画像を取得して処理ループ開始
        for idx, batch in enumerate(dataloader):
            input_image, names = batch
            input_image = input_image.to(device)
            config.batch_size = input_image.shape[0]
            
            # 1. 観測プロセスの実行 (画像を送信し、ノイズが乗った状態などをシミュレート)
            measurement = operator.observe_and_transpose(input_image)
            
            # 各バッチでシードを調整（再現性のため）
            torch.manual_seed(config.seed + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed + 1)
            np.random.seed(config.seed + 1)
            random.seed(config.seed + 1)

            # チャネル推定誤差の計算（OFDMかつBlindでない場合）
            if config.channel_type == 'ofdm_tdl' and not (config.conditioning_method == 'blind_diffcom'):
                H_loss_gt = torch.linalg.norm(measurement['cof_est'] - measurement["cof_gt"])
                logger.info(f"batch{idx + 1:->4d}--> 【Init】 H_Loss cof_gt: {H_loss_gt:.4f}")

            # 観測画像(MSEベースの単純復元画像など)の保存
            util.mkdir(config.save_path + '/measurement')
            util.imsave_batch(util.tensor2uint_batch(measurement['x_mse']), names, config.save_path + '/measurement',
                              f"measurement_")
            
            # 2. ベースライン指標の計算（拡散モデルによる復元前の状態の評価）
            baseline_metric = metric_wrapper(measurement['x_mse'], input_image)
            cbr_val = measurement['channel_usage'] / measurement['x_mse'].numel() # Channel Bandwidth Ratio
            
            logger.info(
                f"batch{idx + 1:->4d}--> 【Baseline】"
                f"CBR: {cbr_val:.4f},"
                f"PSNR: {baseline_metric['psnr']:.2f}dB, "
                f"LPIPS: {baseline_metric['lpips']:.4f}, "
                f"DISTS: {baseline_metric['dists']:.4f}, "
                f"MSSSIM: {baseline_metric['msssim']:.4f}")

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

            # 3. 拡散プロセスの初期化
            # ノイズスケジュールに基づいて初期ノイズ画像(x_init)を作成
            x_init = noise_schedule.sqrt_alphas_cumprod[noise_schedule.t_start] * (2 * measurement['x_mse'] - 1) + \
                     noise_schedule.sqrt_1m_alphas_cumprod[
                         noise_schedule.t_start] * torch.randn_like(input_image)

            # Blind設定（チャネル情報が未知）の場合の初期化処理
            if config.conditioning_method == 'blind_diffcom':
                # (Plotting logic omitted for brevity)
                plt.clf()
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

            seq = noise_schedule.seq # タイムステップのシーケンス
            psnr_list = []
            lpips_list = []
            dists_list = []
            L_m_list = []
            H_loss_list = []
            
            # 4. 逆拡散過程（サンプリングループ）
            # tqdmを使ってプログレスバーを表示しながらループ
            pbar = tqdm(range(len(seq)), ncols=140)
            for i in pbar:
                # cond_method: 現在の状態からノイズを除去し、次のステップの画像を推定する
                x_0_hat, h_0_hat, x_t, h_t, norm = cond_method(config, i, noise_schedule,
                                                               x_init if i == 0 else x_t,
                                                               cof_init if i == 0 else h_t,
                                                               power if config.conditioning_method == 'blind_diffcom' else None,
                                                               measurement, unet, diffusion, operator, loss_wrapper,
                                                               last_timestep=(seq[i] == seq[-1]))

                # 中間結果のプロット（設定されていれば）
                if (seq[i]) % config.diffcom_series['save_recon_every'] == 0:
                    pass

                # 現在のステップでの推定画像(x_0_hat)の評価
                metrics_inter = metric_wrapper((x_0_hat / 2 + 0.5).detach(), input_image)
                l_m_val = norm['ofdm_sig'].item() if 'ofdm_sig' in norm.keys() else 0.0
                
                # ★修正: 変数名を metrics -> metrics_inter に修正
                message = {'t_step': seq[i],
                           'L_m': l_m_val,
                           'PSNR': metrics_inter['psnr'],
                           'LPIPS': metrics_inter['lpips'],
                           'DISTS': metrics_inter['dists']}
                           
                L_m_list.append(l_m_val)

                # チャネル推定誤差の記録
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

            # --------------------------------
            # 5. 最終結果の評価とログ記録
            # --------------------------------
            x_recon = (x_t / 2 + 0.5) # 値の範囲を[0, 1]に戻す
            metrics = metric_wrapper(x_recon.detach(), input_image)
            metrics['L_m'] = L_m_list[-1]
            metrics['H_Loss'] = H_loss_list[-1]
            results.update(metrics) # 平均値の更新
            
            logger.info(
                f"batch{idx + 1:->4d}--> 【Recon】"
                f'H_Loss: {metrics["H_Loss"]:.4f},'
                f'L_m: {metrics["L_m"]:.4f},'
                f'L_c: 0.0000,'
                f"PSNR: {metrics['psnr']:.2f}dB, LPIPS: {metrics['lpips']:.4f}, "
                f"DISTS: {metrics['dists']:.4f}, MSSSIM: {metrics['msssim']:.4f}")

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
            # 6. 不確かさの相関評価 (Latent Space)
            # =========================================================
            # 直前のステップで計算された不確かさマップを取得
            u_map = diffcom_module.latest_uncertainty_map
            
            if u_map is not None:
                # 不確かさと、実際の復元誤差（潜在空間）との相関を計算
                corr, lat_err, u_map_resized = evaluate_latent_correlation(
                    u_map, x_recon, input_image, operator
                )
                
                if lat_err is not None:
                    logger.info(f"Batch {idx}: Uncertainty-LatentError Correlation = {corr:.4f}")
                    batch_record["uncertainty"]["latent_correlation"] = float(corr)
                    
                    # デバッグ用にマップや画像を保存
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
                
                # 次のバッチのためにリセット
                diffcom_module.latest_uncertainty_map = None
            # =========================================================

            all_results_history.append(batch_record)

            logger.info('--------------------------------------------')
            # 復元画像を保存
            recon_image = util.tensor2uint_batch(x_recon)
            util.imsave_batch(recon_image, names, config.save_path + '/recon',
                              f"{config.model_name}_")

    except KeyboardInterrupt:
        logger.info("Execution Interrupted by User (Ctrl+C). Saving current results...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
    finally:
        # エラーで止まっても、終了時でも、JSONに結果サマリを書き出す
        json_path = os.path.join(config.save_path, 'metrics_summary.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(all_results_history, f, indent=4)
            logger.info(f"-----------> Metrics saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics JSON: {e}")

    # 最終的な平均スコアのログ出力
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
    """
    プログラムのエントリーポイント
    環境設定、モデルのロードを行い、メインループを実行します。
    """
    config = parse_args_and_config()
    
    # GPUが使えるならGPUを、そうでなければCPUを使用
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    config.device = device

    # ロガー（記録係）のセットアップ
    logger_name = config.result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(config.save_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    
    # テストデータの読み込み準備
    dataloader = get_test_loader(config.testsets_path, batch_size=config.batch_size, shuffle=False)
    
    # モデルの構成パラメータ設定（モデル名に応じて切り替え）
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
    
    # 引数の解析とモデル作成
    args = utils_model.create_argparser(model_config).parse_args([])
    unet, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    
    # 学習済みモデルの重みをロード
    unet.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    unet.eval() # 推論モードに設定

    unet = unet.to(device) # モデルをデバイス(GPUなど)へ転送

    # 設定ファイルを保存先にコピー（後で再現確認できるように）
    shutil.copyfile(config.opt, os.path.join(config.save_path, os.path.basename('config.yaml')))

    # オペレータ（通信路や劣化モデル）の取得
    operator = get_operator(config.operator_name, config=config, logger=logger, device=device)
    operator.model = operator.model.to(device)
    
    # ノイズスケジュールの設定
    ns = NoiseSchedule(config, logger, device)

    # 条件付け手法(DiffComなど)の取得
    cond_method = get_conditioning_method(name=config.conditioning_method)

    cond_method = cond_method.conditioning
    
    # メインループの実行
    p_sample_loop(config, ns, unet, diffusion, operator, cond_method, dataloader, device, logger)


if __name__ == '__main__':
    main()