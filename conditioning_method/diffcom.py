import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils_model

__CONDITIONING_METHOD__ = {}

# 評価用に不確実性マップを一時保存するグローバル変数
latest_uncertainty_map = None


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


class ConsistencyLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        zeta = config.diffcom_series[config.conditioning_method]['zeta']
        gamma = config.diffcom_series[config.conditioning_method]['gamma']
        self.weight = {
            'x_mse': gamma,
            'ofdm_sig': zeta,
        }

    def forward(self, measurement, x_0_hat, cof, operator, operation_mode):
        x_0_hat = (x_0_hat / 2 + 0.5)  # .clip(0, 1)
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
        
        loss = {}
        for key in recon_measurement.keys():
            loss[key] = self.weight[key] * torch.linalg.norm(measurement[key] - recon_measurement[key])
        return loss


def get_lr(config, t, T):
    lr_base = config['learning_rate']
    # exponential decay to 0
    if config['lr_schedule'] == 'exp':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * np.exp(-t / T)
    # linear decay
    elif config['lr_schedule'] == 'linear':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * (t / T)
    # constant
    else:
        lr = lr_base
    return lr


@register_conditioning_method(name='diffcom')
class DiffCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'latent'

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)
        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_t, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad = torch.autograd.grad(outputs=total_loss, inputs=x_t)[0]
            learning_rate = get_lr(config.diffcom_series[config.conditioning_method], t_step,
                                   ns.t_start - 1)
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss


@register_conditioning_method(name='hifi_diffcom')
class HiFiDiffCom(DiffCom):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'joint'


@register_conditioning_method(name='blind_diffcom')
class BlindDiffCom(DiffCom):
    def __init__(self):
        super().__init__()
        # Algorithm 3 のパラメータ設定
        self.M = 3            # Perturbation samples
        self.kappa = 5.0      # Sensitivity parameter (scale)
        self.k_s = 16         # Kernel size for Spatial Smoothing (AvgPool)
        
        # ★追加: 計算したマスクを保持する変数を初期化
        self.cached_mask = None 
        
    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        # グローバル変数を参照
        global latest_uncertainty_map

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        
        # 1. Denoising Prediction (Step 1)
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        assert (config.conditioning_method == 'blind_diffcom')

        # チャネル推定の更新 (Update Channel)
        h_t = h_t.requires_grad_()
        h_score = - h_t / (power ** 2)
        h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
                h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
        h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
                            ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))


        # =================================================================
        # 2. Spatially Smoothed Uncertainty Estimation (Step 2)
        # =================================================================
        
        # 計算条件: t <= 400 (中盤以降) かつ 20ステップ毎
        should_calc_uncertainty = (t_step <= 400) and (t_step > 0) and (i % 20 == 0)

        # マスクの計算と更新（キャッシュ）
        if not last_timestep and should_calc_uncertainty:
            with torch.no_grad():
                alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                preds = []
                # (a) Perturbation
                for _ in range(self.M):
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

                # (a) Variance Calculation [B, 1, H, W]
                preds_stack = torch.stack(preds) 
                V_t = torch.var(preds_stack, dim=0) # meanを削除してRGBを保つ
                
                # (b) Spatial Smoothing (AvgPool)
                U_t = F.avg_pool2d(V_t, kernel_size=self.k_s, stride=self.k_s)
                
                # グローバル変数に保存 (評価用)
                latest_uncertainty_map = U_t.detach().cpu()

                # (c) Mask Generation & Normalization
                u_min = U_t.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                u_max = U_t.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                U_norm = (U_t - u_min) / (u_max - u_min + 1e-8)
                
                W_small = 1.0 + self.kappa * U_norm

                # (d) Upsampling & Caching
                # ★ここでクラス変数 self.cached_mask に保存
                self.cached_mask = F.interpolate(
                    W_small, 
                    size=x_t.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        # =================================================================

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            # 3. Measurement Consistency Gradient (Step 3)
            loss_dict = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            
            if 'ofdm_sig' in loss_dict:
                target_loss = loss_dict['ofdm_sig']
            else:
                target_loss = sum(loss_dict.values())

            # 勾配計算 (g_meas)
            x_grad, h_t_grad = torch.autograd.grad(outputs=target_loss, inputs=[x_t, h_t])

            # =================================================================
            # 4. State Update with Uncertainty Mask (Step 4)
            # =================================================================
            
            # ★修正: キャッシュされたマスクがあれば、毎ステップ適用する
            if self.cached_mask is not None:
                x_grad = x_grad * self.cached_mask
            
            # =================================================================

            learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
            learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
            
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            lr_h = config.diffcom_series['blind_diffcom']['h_lr']
            lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
            h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
            h_t_minus_1 = h_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss_dict