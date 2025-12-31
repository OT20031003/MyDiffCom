import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils_model

__CONDITIONING_METHOD__ = {}

# 評価・可視化用に不確実性マップを一時保存するグローバル変数
# {'raw': ..., 'smoothed': ...} の辞書を保持する
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


def calculate_temporal_uncertainty(x0_preds_list, k_s=16):
    """
    時間的分散 (Temporal Variance) を計算するヘルパー関数
    x0_preds_list: List of tensor (B, C, H, W) - 各ステップの予測結果
    k_s: 平滑化カーネルサイズ
    """
    if len(x0_preds_list) < 2:
        return None, None

    # Stack: (T, B, C, H, W)
    preds_stack = torch.stack(x0_preds_list)
    
    # Variance over time dimension (dim=0) -> (B, C, H, W)
    # Mean over channel dimension -> (B, 1, H, W)
    V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True)
    
    # Spatial Smoothing
    if V_t.shape[-1] >= k_s:
        U_t = F.avg_pool2d(V_t, kernel_size=k_s, stride=1, padding=k_s//2)
        U_t = F.interpolate(U_t, size=V_t.shape[-2:], mode='bilinear', align_corners=False)
    else:
        U_t = V_t
        
    return V_t, U_t


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
            # HiFi-DiffCom用: 潜在空間とピクセル空間の両方で計算
            ofdm_sig = operator.forward(s, cof)
            s_hat = operator.transpose(ofdm_sig, cof)
            x_confirming = operator.decode(s_hat)
            recon_measurement = {
                'ofdm_sig': ofdm_sig,
                'x_mse': x_confirming
            }
        
        loss = {}
        for key in recon_measurement.keys():
            # ノルム計算により損失を算出
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
        
        # --- 不確かさ推定パラメータ ---
        self.M = 5            # Perturbation samples (摂動サンプルの数)
        self.k_s = 16         # 平滑化カーネルサイズ
        
        # 履歴保存用 (Temporal Variance用)
        self.x0_history = []
        self.uncertainty_mode = 'perturbation' # デフォルト

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        # 可視化用にグローバル変数を参照
        global latest_uncertainty_map
        
        # 初回ステップで設定を読み込む
        if i == 0:
            self.uncertainty_mode = config.diffcom_series.get('uncertainty_mode', 'perturbation')
            self.x0_history = [] # 履歴リセット

        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        
        # -----------------------------------------------------------
        # 1. Prediction Step (通常のリバース拡散ステップ)
        # -----------------------------------------------------------
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        # -----------------------------------------------------------
        # 2. Uncertainty Estimation (不確かさの計算)
        # -----------------------------------------------------------
        
        # Temporal Modeの場合、常に履歴を保存
        if self.uncertainty_mode == 'temporal':
            # メモリ節約のため detach して保存
            self.x0_history.append(x_0_hat.detach())

        # 計算を実行するかどうかの判定
        # (a) t > 0: 最後のステップは除く
        # (b) i % 20 == 0: 計算コスト削減のため20ステップに1回だけ更新
        should_calc_uncertainty = (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            V_t, U_t = None, None

            # === Branch A: Temporal Variance (新手法) ===
            if self.uncertainty_mode == 'temporal':
                # 蓄積した履歴から分散を計算
                V_t, U_t = calculate_temporal_uncertainty(self.x0_history, k_s=self.k_s)

            # === Branch B: Perturbation Variance (既存手法) ===
            elif self.uncertainty_mode == 'perturbation':
                with torch.no_grad():
                    alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                    sqrt_alpha = torch.sqrt(alpha_bar)
                    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                    preds = []
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

                    preds_stack = torch.stack(preds) 
                    V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True)
                    
                    if V_t.shape[-1] >= self.k_s: 
                        U_t = F.avg_pool2d(V_t, kernel_size=self.k_s, stride=1, padding=self.k_s//2)
                        U_t = F.interpolate(U_t, size=V_t.shape[-2:], mode='bilinear', align_corners=False)
                    else:
                        U_t = V_t

            # グローバル変数への保存とマスク生成
            if V_t is not None and U_t is not None:
                latest_uncertainty_map = {
                    'raw': V_t.detach().cpu(),
                    'smoothed': U_t.detach().cpu()
                }
                
                u_min = U_t.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                u_max = U_t.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                self.uncertainty_mask = (U_t - u_min) / (u_max - u_min + 1e-8)

        # -----------------------------------------------------------
        # 3. Gradient Update
        # -----------------------------------------------------------
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
    """
    Uncertainty-Aware HiFi-DiffCom Implementation with selectable Uncertainty Mode
    """
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'joint'
        
        # --- 不確かさ推定のためのパラメータ ---
        self.M = 5            
        self.kappa = 1.0      
        self.k_s = 16         
        
        self.mask_boost = None
        self.mask_suppress = None
        
        self.x0_history = []
        self.uncertainty_mode = 'perturbation'

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        global latest_uncertainty_map

        # 初回ステップで設定を読み込む
        if i == 0:
            self.uncertainty_mode = config.diffcom_series.get('uncertainty_mode', 'perturbation')
            self.x0_history = [] 

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

        # 2. Uncertainty Estimation
        if self.uncertainty_mode == 'temporal':
            self.x0_history.append(x_0_hat.detach())

        should_calc_uncertainty = (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            V_t, U_t = None, None

            # === Branch A: Temporal Variance ===
            if self.uncertainty_mode == 'temporal':
                V_t, U_t = calculate_temporal_uncertainty(self.x0_history, k_s=self.k_s)

            # === Branch B: Perturbation Variance ===
            elif self.uncertainty_mode == 'perturbation':
                with torch.no_grad():
                    alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                    sqrt_alpha = torch.sqrt(alpha_bar)
                    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                    preds = []
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

                    preds_stack = torch.stack(preds) 
                    V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True)
                    
                    if V_t.shape[-1] >= self.k_s: 
                        U_t = F.avg_pool2d(V_t, kernel_size=self.k_s, stride=1, padding=self.k_s//2)
                        U_t = F.interpolate(U_t, size=V_t.shape[-2:], mode='bilinear', align_corners=False)
                    else:
                        U_t = V_t
            
            # Mask Generation Logic (共通)
            if V_t is not None and U_t is not None:
                latest_uncertainty_map = {
                    'raw': V_t.detach().cpu(),
                    'smoothed': U_t.detach().cpu()
                }

                u_min = U_t.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                u_max = U_t.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                U_norm = (U_t - u_min) / (u_max - u_min + 1e-8)
                
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