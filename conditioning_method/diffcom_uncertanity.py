import numpy as np
import torch
import torch.nn as nn

from utils import utils_model

__CONDITIONING_METHOD__ = {}

# ★追加: 評価用に不確実性マップを一時保存するグローバル変数
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
        # パラメータ設定
        self.uncertainty_samples = 3  # 不確実性推定のためのサンプリング数
        self.guidance_scale = 5.0     # ガイダンスの強さ（診断時はマスク適用するかどうかで使い分け）

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        # グローバル変数を参照
        global latest_uncertainty_map

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        
        # 1. 通常のデノイズステップ
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        assert (config.conditioning_method == 'blind_diffcom')

        # 2. チャネル推定の更新
        h_t = h_t.requires_grad_()
        h_score = - h_t / (power ** 2)
        h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
                h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
        h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
                            ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))


        # =================================================================
        # 3. 不確実性 (Uncertainty) の計算と保存
        # =================================================================
        uncertainty_mask = None

        # ★修正箇所: 計算タイミングの限定
        # (1) t_step <= 400: 生成の後半（形が見えてきた段階）
        # (2) t_step > 0: 最後のステップは除く
        # (3) i % 20 == 0: 20ステップごとに計算（頻度を少し上げる）
        should_calc_uncertainty = (t_step <= 1000) and (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            with torch.no_grad():
                alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                preds = []
                for _ in range(self.uncertainty_samples):
                    # Perturbation: 現在の予測値にノイズを加えて再入力
                    eps_k = torch.randn_like(x_0_hat)
                    x_t_k = sqrt_alpha * x_0_hat + sqrt_one_minus_alpha * eps_k
                    
                    # Re-prediction: x_0の再予測
                    x_0_hat_k = utils_model.model_fn(
                        x_t_k,
                        noise_level=sigma_t * 255,
                        model_out_type='pred_xstart',
                        model_diffusion=unet,
                        diffusion=diffusion,
                        ddim_sample=config.ddim_sample
                    )
                    preds.append(x_0_hat_k)

                # 分散計算 (Pixel-wise Uncertainty)
                preds_stack = torch.stack(preds) # [M, B, C, H, W]
                # チャンネル平均をとって [B, 1, H, W] または [B, H, W] にする
                variance = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True)
                
                # ★グローバル変数に保存 (CPUへ退避)
                latest_uncertainty_map = variance.detach().cpu()
                
                # マスク作成 (診断のため、ここでのマスク適用は一旦コメントアウト)
                # v_min = variance.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                # v_max = variance.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                # variance_norm = (variance - v_min) / (v_max - v_min + 1e-8)
                # uncertainty_mask = 1.0 + self.guidance_scale * variance_norm
        # =================================================================

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad, h_t_grad = torch.autograd.grad(outputs=total_loss, inputs=[x_t, h_t])

            # ガイダンス適用 (診断のためコメントアウト中)
            # if uncertainty_mask is not None:
            #     x_grad = x_grad * uncertainty_mask

            learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
            learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            lr_h = config.diffcom_series['blind_diffcom']['h_lr']
            lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
            h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
            h_t_minus_1 = h_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss