import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import utils_model

__CONDITIONING_METHOD__ = {}

# 評価・可視化用に不確実性マップを一時保存するグローバル変数
# 変更後構造: {'temporal': {'raw': ...}, 'perturbation': {'raw': ...}}
# smoothキーは削除され、常にrawのみが格納されます
latest_uncertainty_map = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](**kwargs)


def calculate_temporal_uncertainty(x0_preds_list):
    """
    時間的分散 (Temporal Variance) を計算するヘルパー関数
    Smoothing処理は削除されました。
    """
    if len(x0_preds_list) < 2:
        return None

    # Stack: (T, B, C, H, W)
    preds_stack = torch.stack(x0_preds_list)
    
    # Variance over time dimension (dim=0) -> (B, C, H, W)
    # Mean over channel dimension -> (B, 1, H, W)
    V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True)
    
    return V_t

def calculate_perturbation_uncertainty(unet, diffusion, x_0_hat, sigma_t, ns, t_step, M, config):
    """
    摂動分散 (Perturbation Variance) を計算するヘルパー関数
    Smoothing処理は削除されました。
    """
    with torch.no_grad():
        alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
        sqrt_alpha = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

        preds = []
        for _ in range(M):
            eps_k = torch.randn_like(x_0_hat)
            x_t_k = sqrt_alpha * x_0_hat + sqrt_one_minus_alpha * eps_k
            x_0_hat_k = utils_model.model_fn(
                x_t_k,
                noise_level=sigma_t * 255,
                model_out_type='pred_xstart',
                model_diffusion=unet,
                diffusion=diffusion,
                ddim_sample=config.ddim_sample
            )
            preds.append(x_0_hat_k)

        preds_stack = torch.stack(preds) 
        V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True)
            
    return V_t


def average_uncertainty_buffer(buffer_list):
    """
    バッファ内の不確実性マップを平均化するヘルパー関数
    buffer_list: list of Tensor (Raw maps)
    return: Tensor (Averaged Raw map)
    """
    if not buffer_list:
        return None
    
    # バッファ内の各要素を取り出してスタック
    raw_stack = torch.stack(buffer_list)
    
    # 平均を計算 (dim=0 はバッファの要素方向)
    return raw_stack.mean(dim=0)


class ConsistencyLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        
        # 設定の読み込み
        method_config = config.diffcom_series.get(config.conditioning_method, {})
        zeta = method_config.get('zeta', 1.0)
        gamma = method_config.get('gamma', 0.0)
        
        self.weight = {
            'x_mse': gamma,
            'ofdm_sig': zeta,
        }

    def forward(self, measurement, x_0_hat, cof, operator, operation_mode):
        x_0_hat = (x_0_hat / 2 + 0.5)  # [-1, 1] -> [0, 1]
        loss = {}

        # Hybrid-Quality Guidance (HQG) Logic for Phase 2
        # measurement に retrans_mask がある場合、異なるSNR設定でガイダンスを分離する
        if 'retrans_mask' in measurement:
            mask = measurement['retrans_mask']
            y_low = measurement['ofdm_sig']
            y_high = measurement['retrans_sig']
            
            # 1. Background (Low Quality) Guidance
            # Phase 1 と同様に現在のCSNRでエンコード
            s_low = operator.encode(x_0_hat, snr_override=self.config.CSNR)
            
            # チャネルシミュレーションを通して比較
            if operation_mode == 'latent':
                y_hat_low = operator.forward(s_low, cof=cof)
                loss_bg = torch.linalg.norm((1 - mask) * (y_low - y_hat_low))
            elif operation_mode == 'pixel':
                # Pixel mode の場合、DeepJSCC等ではないため通常の比較
                loss_bg = torch.linalg.norm((1 - mask) * (measurement['x_mse'] - x_0_hat))
            else:
                 y_hat_low = operator.forward(s_low, cof=cof)
                 loss_bg = torch.linalg.norm((1 - mask) * (y_low - y_hat_low))

            # 2. Retransmission (High Quality) Guidance
            # 高SNR (20dB) でエンコード
            high_snr_value = 20.0
            s_high = operator.encode(x_0_hat, snr_override=high_snr_value)
            
            # 高品質比較 (channel simulationを通すが、noiseは付与されない想定)
            if operation_mode == 'latent':
                y_hat_high = operator.forward(s_high, cof=cof)
                loss_fg = torch.linalg.norm(mask * (y_high - y_hat_high))
            elif operation_mode == 'pixel':
                loss_fg = 0.0 # Pixel mode does not support dual encoding
            else:
                y_hat_high = operator.forward(s_high, cof=cof)
                loss_fg = torch.linalg.norm(mask * (y_high - y_hat_high))

            # 合算して ofdm_sig の損失とする
            loss['ofdm_sig'] = self.weight['ofdm_sig'] * (loss_bg + loss_fg)

            if operation_mode == 'joint':
                 # joint の場合の x_mse ロス等は今回は省略または低SNR側で代表
                 # 厳密にはここも分離すべきだが、主眼は Latent Guidance
                 pass

        else:
            # Phase 1 (Standard DiffCom)
            s = operator.encode(x_0_hat)
            
            if operation_mode == 'latent':
                recon_measurement = {
                    'ofdm_sig': operator.forward(s, cof)
                }
            elif operation_mode == 'pixel':
                recon_measurement = {
                    'x_mse': x_0_hat
                }
            elif operation_mode == 'joint':
                ofdm_sig = operator.forward(s, cof)
                s_hat = operator.transpose(ofdm_sig, cof)
                x_confirming = operator.decode(s_hat)
                recon_measurement = {
                    'ofdm_sig': ofdm_sig,
                    'x_mse': x_confirming
                }
            
            for key in recon_measurement.keys():
                loss[key] = self.weight[key] * torch.linalg.norm(measurement[key] - recon_measurement[key])
                
        return loss


def get_lr(config, t, T):
    lr_base = config['learning_rate']
    if config['lr_schedule'] == 'exp':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * np.exp(-t / T)
    elif config['lr_schedule'] == 'linear':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * (t / T)
    else:
        lr = lr_base
    return lr


@register_conditioning_method(name='diffcom')
class DiffCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'latent'
        
        self.M = 5            
        
        self.x0_history = []
        self.uncertainty_modes = ['perturbation'] # デフォルトはリスト
        self.uncertainty_buffer = {} # 累積平均用のバッファ

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        global latest_uncertainty_map
        
        # 初回ステップで設定を読み込み、リスト化して保持、バッファを初期化
        if i == 0:
            u_mode = config.diffcom_series.get('uncertainty_mode', 'perturbation')
            if isinstance(u_mode, str):
                self.uncertainty_modes = [u_mode]
            else:
                self.uncertainty_modes = u_mode
                
            self.x0_history = []
            self.uncertainty_buffer = {mode: [] for mode in self.uncertainty_modes}

        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        
        # 1. Prediction Step
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        # 2. Uncertainty Estimation & Accumulation
        if 'temporal' in self.uncertainty_modes:
            self.x0_history.append(x_0_hat.detach())

        should_calc_uncertainty = (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            
            # 各モードについて計算し、バッファに追加
            for mode in self.uncertainty_modes:
                V_t = None
                
                if mode == 'temporal':
                    V_t = calculate_temporal_uncertainty(self.x0_history)
                
                elif mode == 'perturbation':
                    V_t = calculate_perturbation_uncertainty(
                        unet, diffusion, x_0_hat, sigma_t, ns, t_step, 
                        self.M, config
                    )
                
                if V_t is not None:
                    # バッファに追加 (CPUへ退避してメモリ節約)
                    self.uncertainty_buffer[mode].append(V_t.detach().cpu())

            # 現在のバッファを用いて累積平均を計算し、グローバル変数を更新
            # これにより、プロセスの途中でも最新の「アンサンブル平均」が参照可能になる
            averaged_maps = {}
            for mode in self.uncertainty_modes:
                avg_tensor = average_uncertainty_buffer(self.uncertainty_buffer[mode])
                if avg_tensor is not None:
                    # mainスクリプトとの互換性のため 'raw' キーに格納
                    averaged_maps[mode] = {'raw': avg_tensor}
            
            if averaged_maps:
                latest_uncertainty_map = averaged_maps

        # 3. Gradient Update
        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_t, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            
            x_grad = torch.autograd.grad(outputs=total_loss, inputs=x_t)[0]
            
            learning_rate = get_lr(config.diffcom_series[config.conditioning_method], t_step, ns.t_start - 1)
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss


@register_conditioning_method(name='hifi_diffcom')
class HiFiDiffCom(DiffCom):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'joint'
        
        self.M = 5            
        self.kappa = 1.0      
        
        self.mask_boost = None
        self.mask_suppress = None
        
        self.x0_history = []
        self.uncertainty_modes = ['perturbation']
        self.uncertainty_buffer = {}

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        global latest_uncertainty_map

        if i == 0:
            u_mode = config.diffcom_series.get('uncertainty_mode', 'perturbation')
            if isinstance(u_mode, str):
                self.uncertainty_modes = [u_mode]
            else:
                self.uncertainty_modes = u_mode
            
            self.x0_history = [] 
            self.uncertainty_buffer = {mode: [] for mode in self.uncertainty_modes}

        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        
        # 1. Prediction Step
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        # 2. Uncertainty Estimation & Accumulation
        if 'temporal' in self.uncertainty_modes:
            self.x0_history.append(x_0_hat.detach())

        should_calc_uncertainty = (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            
            # 各モードについて計算し、バッファに追加
            for mode in self.uncertainty_modes:
                V_t = None
                
                if mode == 'temporal':
                    V_t = calculate_temporal_uncertainty(self.x0_history)
                elif mode == 'perturbation':
                    V_t = calculate_perturbation_uncertainty(
                        unet, diffusion, x_0_hat, sigma_t, ns, t_step, 
                        self.M, config
                    )
                
                if V_t is not None:
                    self.uncertainty_buffer[mode].append(V_t.detach().cpu())

            # バッファから累積平均を計算
            averaged_maps = {}
            for mode in self.uncertainty_modes:
                avg_tensor = average_uncertainty_buffer(self.uncertainty_buffer[mode])
                if avg_tensor is not None:
                    averaged_maps[mode] = {'raw': avg_tensor}

            if averaged_maps:
                latest_uncertainty_map = averaged_maps

                # HiFi Guidance Mask Update (累積平均マップを使用)
                # 使用するモードの優先順位: リストの順序通り
                primary_U_t = None
                for mode in self.uncertainty_modes:
                    if mode in averaged_maps:
                        # Smoothingが削除されたため 'raw' を使用します
                        primary_U_t = averaged_maps[mode]['raw']
                        break
                
                if primary_U_t is not None:
                    # デバイスを戻す (計算はCPUで行われている可能性があるため)
                    primary_U_t = primary_U_t.to(x_t.device)
                    
                    u_min = primary_U_t.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                    u_max = primary_U_t.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                    U_norm = (primary_U_t - u_min) / (u_max - u_min + 1e-8)
                    
                    self.mask_boost = 1.0 + self.kappa * U_norm
                    self.mask_suppress = 1.0 / (1.0 + self.kappa * U_norm)

        # 3. Guidance & Update
        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss_dict = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            
            grad_m = torch.zeros_like(x_t)
            grad_c = torch.zeros_like(x_t)

            if 'ofdm_sig' in loss_dict:
                grad_m = torch.autograd.grad(outputs=loss_dict['ofdm_sig'], inputs=x_t, retain_graph=True)[0]
                if self.mask_suppress is not None:
                    grad_m = grad_m * self.mask_suppress

            if 'x_mse' in loss_dict:
                grad_c = torch.autograd.grad(outputs=loss_dict['x_mse'], inputs=x_t)[0]
                if self.mask_boost is not None:
                    grad_c = grad_c * self.mask_boost

            total_grad = grad_m + grad_c
            learning_rate = config.diffcom_series['hifi_diffcom'].get('learning_rate', 1.0)
            
            x_t_minus_1 = x_t_minus_1_prime - total_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss_dict


@register_conditioning_method(name='blind_diffcom')
class BlindDiffCom(DiffCom):
    """
    Original Blind-DiffCom Implementation
    """
    def __init__(self):
        super().__init__()

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        assert (config.conditioning_method == 'blind_diffcom')

        h_t = h_t.requires_grad_()
        h_score = - h_t / (power ** 2)
        h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
                h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
        h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
                            ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad, h_t_grad = torch.autograd.grad(outputs=total_loss, inputs=[x_t, h_t])
            learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
            learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            lr_h = config.diffcom_series['blind_diffcom']['h_lr']
            lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
            h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
            h_t_minus_1 = h_t_minus_1.detach_()
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss