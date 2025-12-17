import numpy as np
import torch
import torch.nn as nn

from utils import utils_model

__CONDITIONING_METHOD__ = {}


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
        # 必要に応じてconfigから読み込むように変更可能です
        self.uncertainty_samples = 3  # 不確実性推定のためのサンプリング回数 (M)
        self.guidance_scale = 0.5     # 不確実な領域をどれだけ強く補正するか (kappa)

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        
        # 1. 通常のデノイズステップ (x_t -> x_0_hat)
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        assert (config.conditioning_method == 'blind_diffcom')

        # 2. チャネル推定の更新 (DiffCom標準の処理)
        h_t = h_t.requires_grad_()
        h_score = - h_t / (power ** 2)
        h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
                h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
        h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
                            ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))

        # =================================================================
        # 3. Pixel-wise Aleatoric Uncertainty Estimation & Mask Generation
        # =================================================================
        uncertainty_mask = None
        
        # 計算コスト削減のため、勾配計算が必要なステップでのみ実行
        # また、diffusionの極初期や最後期では効果が薄いため、中間ステップ(例: t=100~900)のみ適用するなどの工夫も有効
        if not last_timestep: 
            with torch.no_grad():
                # ノイズスケジュールの係数取得
                alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                preds = []
                for _ in range(self.uncertainty_samples):
                    # A. Perturbation: 推定したx_0_hatに対して再度ノイズを加えて時刻tの状態に戻す
                    eps_k = torch.randn_like(x_0_hat)
                    x_t_k = sqrt_alpha * x_0_hat + sqrt_one_minus_alpha * eps_k
                    
                    # B. Re-prediction: 摂動を加えた画像から再度推論を行う
                    # model_out_type='pred_xstart' を指定して x_0 の予測値のみを取得
                    x_0_hat_k = utils_model.model_fn(
                        x_t_k,
                        noise_level=sigma_t * 255,
                        model_out_type='pred_xstart',
                        model_diffusion=unet,
                        diffusion=diffusion,
                        ddim_sample=config.ddim_sample
                    )
                    preds.append(x_0_hat_k)

                # C. Variance Calculation: 予測のばらつきを不確実性とする
                preds_stack = torch.stack(preds) # shape: [M, B, C, H, W]
                # チャンネル方向の平均を取り、空間的な不確実性マップを作成
                variance = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True) 
                
                # D. Mask Normalization & Scaling
                # 画像ごとに[0, 1]に正規化
                v_min = variance.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                v_max = variance.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                variance_norm = (variance - v_min) / (v_max - v_min + 1e-8)
                
                # 重みマスク: 通常時は1.0、不確実性が高い場所は (1.0 + scale) 倍の勾配を適用
                uncertainty_mask = 1.0 + self.guidance_scale * variance_norm
        # =================================================================

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            # 4. 損失計算と勾配の適用
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            
            # x_t と h_t の勾配を計算
            x_grad, h_t_grad = torch.autograd.grad(outputs=total_loss, inputs=[x_t, h_t])

            # =================================================================
            # 5. Apply Uncertainty Mask to Gradients
            # =================================================================
            if uncertainty_mask is not None:
                # 不確実な領域ほど、受信信号との整合性を取るための修正（勾配）を強くする
                x_grad = x_grad * uncertainty_mask
            # =================================================================

            # x の更新
            learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
            learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            # h (チャネル推定) の更新
            lr_h = config.diffcom_series['blind_diffcom']['h_lr']
            lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
            h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
            h_t_minus_1 = h_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss
# @register_conditioning_method(name='blind_diffcom')
# class BlindDiffCom(DiffCom):
#     def __init__(self):
#         super().__init__()

#     def conditioning(self, config, i, ns, x_t, h_t, power,
#                      measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
#         t_step = ns.seq[i]
#         sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
#         x_t = x_t.requires_grad_()
#         x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
#                                                              noise_level=sigma_t * 255,
#                                                              model_out_type='pred_x_prev_and_start', \
#                                                              model_diffusion=unet,
#                                                              diffusion=diffusion,
#                                                              ddim_sample=config.ddim_sample)

#         assert (config.conditioning_method == 'blind_diffcom')

#         h_t = h_t.requires_grad_()
#         h_score = - h_t / (power ** 2)
#         h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
#                 h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
#         h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
#                             ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))

#         # x_0_hatここからUncertanityの測定
        
        

#         if last_timestep:
#             loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
#             return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
#         else:
#             loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
#             total_loss = sum(loss.values())
#             x_grad, h_t_grad = torch.autograd.grad(outputs=total_loss, inputs=[x_t, h_t])
#             learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
#             learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
#             x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
#             x_t_minus_1 = x_t_minus_1.detach_()
#             lr_h = config.diffcom_series['blind_diffcom']['h_lr']
#             lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
#             h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
#             h_t_minus_1 = h_t_minus_1.detach_()
#             return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss