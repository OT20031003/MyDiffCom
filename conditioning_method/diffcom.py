import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils_model

__CONDITIONING_METHOD__ = {}

# 評価・可視化用に不確実性マップを一時保存するグローバル変数
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
        
        # --- 不確かさ推定パラメータ (DiffCom用) ---
        self.M = 5            # サンプリング数 (Perturbation samples)
        self.k_s = 16         # 平滑化カーネルサイズ (Spatial kernel size)
        
        # マスク用 (可視化や将来的な再送判定用)
        self.uncertainty_mask = None

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        # 可視化用にグローバル変数を参照
        global latest_uncertainty_map

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
        # 2. Uncertainty Estimation & Mask Generation (不確かさの計算)
        # -----------------------------------------------------------
        
        # 計算条件: 
        # (a) t > 0: 最後のステップは除く
        # (b) i % 20 == 0: 計算コスト削減のため20ステップに1回だけ更新
        should_calc_uncertainty = (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            with torch.no_grad():
                alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                preds = []
                # Perturbation (摂動を与えて再予測)
                for _ in range(self.M):
                    eps_k = torch.randn_like(x_0_hat)
                    # 現在の推定x_0にノイズを加えてx_t相当に戻す
                    x_t_k = sqrt_alpha * x_0_hat + sqrt_one_minus_alpha * eps_k
                    
                    # 再度x_0を予測
                    x_0_hat_k = utils_model.model_fn(
                        x_t_k,
                        noise_level=sigma_t * 255,
                        model_out_type='pred_xstart',
                        model_diffusion=unet,
                        diffusion=diffusion,
                        ddim_sample=config.ddim_sample
                    )
                    preds.append(x_0_hat_k)

                # Variance Calculation (分散計算) [B, 1, H, W]
                preds_stack = torch.stack(preds) 
                V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True) # チャンネル平均
                
                # Spatial Smoothing (空間平滑化)
                if V_t.shape[-1] >= self.k_s: 
                    U_t = F.avg_pool2d(V_t, kernel_size=self.k_s, stride=1, padding=self.k_s//2)
                    U_t = F.interpolate(U_t, size=V_t.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    U_t = V_t
                
                # グローバル変数に保存 (評価・可視化用)
                latest_uncertainty_map = U_t.detach().cpu()

                # Mask Generation (マスク生成) - 再送判定用
                # 正規化: [0, 1]
                u_min = U_t.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                u_max = U_t.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                
                # この mask は再送要求の閾値判定などに使用可能
                self.uncertainty_mask = (U_t - u_min) / (u_max - u_min + 1e-8)

        # -----------------------------------------------------------
        # 3. Gradient Update (通常の更新)
        # -----------------------------------------------------------

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_t, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            
            # Gradient Calculation
            x_grad = torch.autograd.grad(outputs=total_loss, inputs=x_t)[0]
            
            # Update x
            learning_rate = get_lr(config.diffcom_series[config.conditioning_method], t_step,
                                   ns.t_start - 1)
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss


@register_conditioning_method(name='hifi_diffcom')
class HiFiDiffCom(DiffCom):
    """
    Uncertainty-Aware HiFi-DiffCom Implementation
    Hybrid Strategy:
    1. Boost 'grad_c' (x_mse) in uncertain regions to enforce structure.
    2. Suppress 'grad_m' (ofdm_sig) in uncertain regions to avoid noise amplification.
    """
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'joint'
        
        # --- 不確かさ推定のためのパラメータ ---
        self.M = 5            # Perturbation samples (摂動サンプルの数)
        self.kappa = 1.0      # Sensitivity parameter (不確かさをマスク強度に変換する係数)
        self.k_s = 16         # Spatial kernel size (平滑化カーネルサイズ)
        
        # 計算したマスクをキャッシュする変数 (Hybrid)
        self.mask_boost = None
        self.mask_suppress = None

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        # 可視化用にグローバル変数を参照
        global latest_uncertainty_map

        # HiFiではチャネル推定の更新は行わないため、h_tはそのまま保持
        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        
        # -----------------------------------------------------------
        # 1. Prediction Step (通常のリバース拡散ステップ)
        # -----------------------------------------------------------
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        # -----------------------------------------------------------
        # 2. Uncertainty Estimation & Mask Generation (不確かさの計算)
        # -----------------------------------------------------------
        
        # 計算条件: 
        # (a) t > 0: 最後のステップは除く
        # (b) i % 20 == 0: 計算コスト削減のため20ステップに1回だけ更新
        should_calc_uncertainty = (t_step > 0) and (i % 20 == 0)

        if not last_timestep and should_calc_uncertainty:
            with torch.no_grad():
                alpha_bar = ns.alphas_cumprod[t_step].to(x_0_hat.device)
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)

                preds = []
                # Perturbation (摂動を与えて再予測)
                for _ in range(self.M):
                    eps_k = torch.randn_like(x_0_hat)
                    # 現在の推定x_0にノイズを加えてx_t相当に戻す
                    x_t_k = sqrt_alpha * x_0_hat + sqrt_one_minus_alpha * eps_k
                    
                    # 再度x_0を予測
                    x_0_hat_k = utils_model.model_fn(
                        x_t_k,
                        noise_level=sigma_t * 255,
                        model_out_type='pred_xstart',
                        model_diffusion=unet,
                        diffusion=diffusion,
                        ddim_sample=config.ddim_sample
                    )
                    preds.append(x_0_hat_k)

                # Variance Calculation (分散計算) [B, 1, H, W]
                preds_stack = torch.stack(preds) 
                V_t = torch.var(preds_stack, dim=0).mean(dim=1, keepdim=True) # チャンネル平均
                
                # Spatial Smoothing (空間平滑化)
                if V_t.shape[-1] >= self.k_s: 
                    U_t = F.avg_pool2d(V_t, kernel_size=self.k_s, stride=1, padding=self.k_s//2)
                    U_t = F.interpolate(U_t, size=V_t.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    U_t = V_t
                
                # グローバル変数に保存 (評価・可視化用)
                latest_uncertainty_map = U_t.detach().cpu()

                # Mask Generation (マスク生成)
                # 正規化: [0, 1]
                u_min = U_t.flatten(2).min(2, keepdim=True)[0].unsqueeze(2)
                u_max = U_t.flatten(2).max(2, keepdim=True)[0].unsqueeze(2)
                U_norm = (U_t - u_min) / (u_max - u_min + 1e-8)
                
                # ==== Hybrid Mask Strategy ====
                
                # (1) Boost Mask: 不確かな場所はJSCC出力 (x_mse) を強く信頼する
                # Range: [1.0, 1.0 + kappa]
                self.mask_boost = 1.0 + self.kappa * U_norm

                # (2) Suppress Mask: 不確かな場所は受信信号 (ofdm_sig) を無視する
                # Range: [1.0, 1.0 / (1.0 + kappa)] -> 減衰させる
                self.mask_suppress = 1.0 / (1.0 + self.kappa * U_norm)

        # -----------------------------------------------------------
        # 3. Guidance Gradient Calculation & Update (ガイダンス適用と更新)
        # -----------------------------------------------------------
        
        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            # 損失計算 (L_m: ofdm_sig, L_c: x_mse)
            loss_dict = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            
            # 勾配変数の初期化
            grad_m = torch.zeros_like(x_t)
            grad_c = torch.zeros_like(x_t)

            # (A) 受信信号ガイダンス (L_m) の勾配計算
            if 'ofdm_sig' in loss_dict:
                # retain_graph=True で計算グラフを保持
                grad_m = torch.autograd.grad(outputs=loss_dict['ofdm_sig'], inputs=x_t, retain_graph=True)[0]
                
                # ★ Suppress: ノイズが多い不確か領域は、受信信号を「無視」する
                if self.mask_suppress is not None:
                    grad_m = grad_m * self.mask_suppress

            # (B) 確実性制約 (L_c) の勾配計算
            if 'x_mse' in loss_dict:
                grad_c = torch.autograd.grad(outputs=loss_dict['x_mse'], inputs=x_t)[0]

                # ★ Boost: 不確か領域は、構造が正しいJSCC出力を「信頼」する
                if self.mask_boost is not None:
                    grad_c = grad_c * self.mask_boost

            # 合計勾配
            total_grad = grad_m + grad_c

            # 状態更新
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