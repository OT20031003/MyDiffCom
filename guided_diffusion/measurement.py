"""
Simulate the measurement y = f(x) + n.
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torchvision import torch

from _djscc.network import ADJSCC
#from _ntsccp.net.ntscc import CompatibleNTSCC_plus, NTSCC_plus
from channel.channel import Channel
from channel.ofdm_channel import LMMSE_channel_est, LS_channel_est, MMSE_equalization, OFDM, ZF_equalization
from utils.util import Config

# OPERATOR CLASSES -> f(·)
# 演算子クラスを登録するための辞書（名前 -> クラス）
__OPERATOR__ = {}


def register_operator(name: str):
    """
    オペレーターを名前で登録するデコレータ。
    使い方:
        @register_operator(name='djscc')
        class DeepJSCC(NonlinearOperator): ...
    同じ名前が既に登録されていると例外になる。
    """
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    """
    登録済みオペレーターを名前で取得しインスタンス化するヘルパー。
    未定義の名前が来たら例外を投げる。
    """
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class NonlinearOperator(ABC):
    """
    非線形演算子の抽象基底クラス。
    - forward: 観測モデル f(x) の順方向（例: チャネルを通す）
    - transpose: 転置（または逆変換）操作（例: H^T * x）
    - ortho_project / project: 投影に関するユーティリティ
    実装クラスは forward と transpose を必ず実装すること。
    """
    @abstractmethod
    def forward(self, data, **kwargs):
        # H * x のような順方向作用
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # H^T * x のような転置／逆作用
        pass

    def ortho_project(self, data, **kwargs):
        """
        直交射影の補助:
        (I - H^T * H) * x を計算するユーティリティ。
        """
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        """
        プロジェクションのヘルパー:
        (I - H^T * H) * y - H * x を計算する。
        """
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


def shuffle(x, shuffled_indices=None):
    """
    シンボル列をシャッフル（インターリーブ）する関数。
    - x の形状は [B, N_s] を想定。
    - shuffled_indices が与えられればその順序でシャッフルし、与えられなければランダム生成。
    - (シャッフル後テンソル, 使用したインデックス) を返す。
    """
    B, N_s = x.shape
    if shuffled_indices is None:
        shuffled_indices = torch.randperm(N_s)
    x = x.reshape(B, -1)[..., shuffled_indices].reshape(B, N_s)
    return x, shuffled_indices


def de_shuffle(x, shuffled_indices):
    """
    シャッフルを元に戻す関数。
    - shuffled_indices を用いて元の位置に復元する。
    """
    B, N_s = x.shape
    x = x.reshape(B, -1)
    x_rx = torch.zeros_like(x)
    x_rx[..., shuffled_indices] = x
    x_rx = x_rx.reshape(B, N_s)
    return x_rx


class ChannelWrapper(nn.Module):
    """
    チャネル操作をまとめて扱うラッパークラス。
    目的:
      - 設定(config)に応じて AWGN / Rayleigh / OFDM(TDL) を扱う
      - 送信側の observe (チャネル通過) と受信側の transpose (等化・デシャッフル等) を提供
      - ofdm 用のチャネル推定や等化のラッパー関数を内包する
    注意:
      - config.ofdm_tdl があるときは OFDM クラスを用いる
      - rescale フラグは送受信時の再スケーリングの振る舞いを制御
    """
    def __init__(self, config, logger, device, rescale):
        super(ChannelWrapper, self).__init__()
        self.channel_type = config.channel_type
        self.CSNR = config.CSNR
        self.shuffled_indices = None
        self.rescale = rescale
        if config.channel_type == 'awgn' or config.channel_type == 'rayleigh':
            # AWGN / Rayleigh 専用の Channel オブジェクトを使用
            self.channel = Channel(config.channel_type, config.CSNR, logger, device, rescale=False)
        elif config.channel_type == 'ofdm_tdl':
            # OFDM 用の設定を読み込み OFDM オブジェクトを生成
            self.opt = Config(config.ofdm_tdl)
            self.channel = OFDM(self.opt, device)
        else:
            raise NotImplementedError(f"Channel type {config.channel_type} is not supported.")

    def channel_estimation_wrapper(self, H_t, info_pilot, noise_pwr, M):
        """
        OFDM 用チャネル推定のラッパー。
        - self.opt.channel_est に応じて 'perfect' / 'LS' / 'LMMSE' を切り替える。
        - 引数:
            H_t: 真のチャネルインパルス応答 (time domain)
            info_pilot: パイロット受信信号
            noise_pwr: ノイズパワー
            M: OFDM フレーム長（サブキャリア数等）
        """
        if self.opt.channel_est == 'perfect':
            H_est = H_t.unsqueeze(1)
        elif self.opt.channel_est == 'LS':
            H_est = LS_channel_est(self.channel.pilot, info_pilot)
        elif self.opt.channel_est == 'LMMSE':
            H_est = LMMSE_channel_est(self.channel.pilot, info_pilot, M * noise_pwr)
        return H_est

    def channel_equalization_wrapper(self, H_est, info_sig, M, noise_pwr):
        """
        OFDM 用等化のラッパー。
        - self.opt.equalization に応じて 'ZF' / 'MMSE' を選択する。
        """
        if self.opt.equalization == 'ZF':
            s_equal = ZF_equalization(H_est, info_sig)
        elif self.opt.equalization == 'MMSE':
            s_equal = MMSE_equalization(H_est, info_sig, M * noise_pwr)
        return s_equal

    def observe(self, s, mask):
        '''
        送信処理: シンボルをチャネルに通す。
        - 入力:
            s: 送信シンボル、shape: [B, N_s]
            mask: シンボル利用マスク（可変レート時に未使用部分を示す等）
        - 出力:
            info_sig: チャネル出力（OFDM 信号等）
            cof_est: 推定された周波数領域係数（等化用、存在しない場合は None）
            cof_gt: 真の周波数係数（評価用）
            channel_usage: 使用したチャネルシンボル数
        '''
        self.s_shape = s.shape
        B, N_s = self.s_shape
        # 送信前にインターリーブ (シャッフル)
        s, shuffled_indices = shuffle(s)
        mask_sig, _ = shuffle(mask, shuffled_indices)
        self.shuffled_indices = shuffled_indices
        avg_pwr = torch.sum(s ** 2) / mask.sum()
        self.avg_pwr = avg_pwr
        cof_est = None
        cof_gt = None
        if self.channel_type == 'awgn':
            # AWGN の場合は単純にチャネルに通す
            info_sig, channel_usage = self.channel.forward(s, avg_pwr)
            info_sig = info_sig * mask_sig
        # elif self.channel_type == 'rayleigh':
        #     info_sig, H_est, channel_usage = self.channel.forward(s)
        elif self.channel_type == 'ofdm_tdl':
            # 可変長シンボルを OFDM ブロック長に揃えるためにゼロパディング
            ofdm_size = 2 * self.opt.P * self.opt.S
            if N_s % ofdm_size != 0:
                s = torch.cat([s, torch.zeros(B, ofdm_size - N_s % ofdm_size, device=s.get_device())], dim=-1)
            # OFDM 形状にリシェイプ (実部/虚部 を別チャネルに並べた形)
            s_ofdm = s.reshape(B, self.opt.P * 2, self.opt.S, -1)
            M = s_ofdm.shape[-1]
            self.ofdm_shape = s_ofdm.shape
            # 実部・虚部を複素表現に組み合わせる
            s_ofdm = s_ofdm[:, :self.opt.P] + 1j * s_ofdm[:, self.opt.P:]
            channel_usage = s_ofdm.numel()
            self.channel.set_pilot(M)
            # 正規化：平均パワーで割る
            s_ofdm = s_ofdm / torch.sqrt(avg_pwr)
            # チャネル呼び出し：ノイズ付加あり
            info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = self.channel(s_ofdm,
                                                                               self.CSNR,
                                                                               cof=None,
                                                                               add_noise=True)
            cof_gt = self.channel.get_cof_from_H(H_t)
            self.noise_pwr = noise_pwr
            if self.opt.blind:
                cof_est = None
            else:
                # チャネル推定を行い、周波数領域係数に変換
                H_est = self.channel_estimation_wrapper(H_t, info_pilot, noise_pwr, M)
                cof_est = self.channel.get_cof_from_H(H_est)

        return info_sig, cof_est, cof_gt, channel_usage

    def transpose(self, data, cof):
        """
        受信側の逆変換処理。
        - OFDM の場合: 等化 → 実部/虚部分離 → 元サイズにトリム → デシャッフル
        - AWGN 等: 必要であればスケール戻し → デシャッフル
        """
        if self.channel_type == 'ofdm_tdl':
            B, N_s = self.s_shape
            M = self.ofdm_shape[-1]
            s_ofdm_hat = torch.zeros(self.ofdm_shape, device=data.get_device())

            if self.opt.blind:
                # blind モードでは等化を行わない（data が既に等化済みと仮定）
                s_equal = data
            else:
                # cof は周波数係数、逆 FFT を使って等化に用いる
                H_est = torch.fft.fft(cof, dim=-1)
                s_equal = self.channel_equalization_wrapper(H_est, data, M, self.noise_pwr)

            if self.rescale:
                # 送信時に使った平均パワーで戻す（スケーリング）
                s_equal = s_equal * torch.sqrt(self.avg_pwr)

            # 実部・虚部を分離して格納
            s_ofdm_hat[:, :self.opt.P] = torch.real(s_equal)
            s_ofdm_hat[:, self.opt.P:] = torch.imag(s_equal)
            # 1次元シーケンスに戻し、パディング分を切り捨てて元の長さにする
            s_hat = s_ofdm_hat.reshape(B, -1)[:, :N_s]
        else:
            # AWGN 等の単純チャネルではスケール復元のみ
            if self.rescale:
                data = data * torch.sqrt(self.avg_pwr * 2)
            s_hat = data
        # 最後にデシャッフルして元の順序に戻す
        s_hat = de_shuffle(s_hat, self.shuffled_indices)
        return s_hat

    def forward(self, s, mask, cof=None):
        """
        テスト時などに直接チャネルへ s を送り OFDM 信号等を得るための forward。
        - s はシャッフル済みのものを受け取る想定（ここで再度シャッフルを行う）。
        - チャネルの種類に応じて処理を分岐する。
        """
        B, N_s = s.shape
        s, _ = shuffle(s, self.shuffled_indices)

        # 受信側が平均パワーを知っている前提で正規化に使用
        avg_pwr = self.avg_pwr

        if self.channel_type == 'awgn':
            # AWGN の場合はスケーリングして返す
            ofdm_sig = s / torch.sqrt(avg_pwr * 2)

        elif self.channel_type == 'ofdm_tdl':
            # OFDM 形状に戻して複素表現にし、チャネルへ投入（add_noise=False）
            s = s.reshape(B, self.opt.P * 2, self.opt.S, -1)
            M = s.shape[-1]
            s = s[:, :self.opt.P] + 1j * s[:, self.opt.P:]
            self.channel.set_pilot(M)
            # 正規化
            s = s / torch.sqrt(avg_pwr)
            # チャネル呼び出し（ノイズは付加しない）
            info_pilot, ofdm_sig, H_t, noise_pwr, papr, papr_cp = self.channel(s,
                                                                               self.CSNR,
                                                                               cof=cof,
                                                                               add_noise=False)
        return ofdm_sig


@register_operator(name='djscc')
class DeepJSCC(NonlinearOperator):
    """
    DeepJSCC を演算子として扱うラッパー。
    - ChannelWrapper を内部に持ち、ADJSCC モデルでエンコード/デコードする。
    - observe_and_transpose: エンコード→チャネル→等化→デコード の一連処理を提供。
    """
    def __init__(self, config, logger, device):
        self.device = device
        self.config = config
        # ChannelWrapper を生成（rescale=False）
        self.channel = ChannelWrapper(config, logger, device, rescale=False)
        # ADJSCC モデルを生成（チャネル情報を渡す）
        self.model = ADJSCC(config.djscc['channel_num'], self.channel, device)
        state_dict = torch.load(config.djscc['jscc_model_path'], map_location=device)
        # 保存された state_dict のキー名が古い命名の場合に合わせる（Encoder -> jscc_encoder 等）
        for key in list(state_dict.keys()):
            if key.startswith('distortion_loss'):
                state_dict.pop(key)
                continue
            if key.startswith('Encoder'):
                state_dict[key.replace('Encoder', 'jscc_encoder')] = state_dict.pop(key)
            elif key.startswith('Decoder'):
                state_dict[key.replace('Decoder', 'jscc_decoder')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def observe_and_transpose(self, x):
        """
        入力画像 x を:
          1) エンコードしてシンボル s を得る
          2) チャネルに通し ofdm_sig, cof_est, cof_gt, channel_usage を得る
          3) 等化して s_hat を得る
          4) デコードして復元画像 x_mse を得る
        戻り値は辞書形式で結果を返す。
        """
        s = self.encode(x)
        ofdm_sig, cof_est, cof_gt, channel_usage = self.channel.observe(s, torch.ones_like(s))
        s_hat = self.channel.transpose(ofdm_sig, cof_est)
        x_mse = self.decode(s_hat)
        return {"x_mse": x_mse,
                "ofdm_sig": ofdm_sig,
                "s_hat": s_hat,
                "cof_est": cof_est,
                "cof_gt": cof_gt,
                "channel_usage": channel_usage}

    def encode(self, data, snr_override=None):
        """
        ADJSCC モデルで入力画像を符号化してシンボル列 s を返す。
        - 出力 s は flatten して形状 [B, -1] にして返す。
        - self.s_shape に元の符号化出力形状を保持し、デコード時に使用する。
        - snr_override: 指定された場合、設定値(self.config.CSNR)の代わりに使用する
        """
        B, C, H, W = data.shape
        target_snr = snr_override if snr_override is not None else self.config.CSNR
        s = self.model.encode(data, given_SNR=target_snr)
        self.s_shape = s.shape
        # 平坦化して返す
        s = s.reshape(B, -1)
        return s

    def forward(self, s, cof=None):
        # ChannelWrapper を介して OFDM 信号を生成
        ofdm_sig = self.channel.forward(s, cof=cof, mask=torch.ones_like(s))
        return ofdm_sig

    def transpose(self, ofdm_sig, cof=None):
        # 受信した ofdm_sig を s_hat に変換して返す
        s_hat = self.channel.transpose(ofdm_sig, cof)
        return s_hat

    def decode(self, s_hat):
        """
        s_hat を元の形状に戻して ADJSCC のデコーダで復元を行う。
        - self.s_shape を用いてリシェイプし、モデル.decode を呼ぶ。
        """
        s_hat = s_hat.reshape(self.s_shape)
        x_mse = self.model.decode(s_hat, given_SNR=self.config.CSNR)
        return x_mse


@register_operator(name='ntscc')
class NTSCC(NonlinearOperator):
    """
    NTSCC ラッパー:
    - Compatible モードと通常モードの両方に対応
    - 事前学習済みモデルをロードしてエンコード/デコードを行う
    - ChannelWrapper を介してチャネルを通す
    - 可変レート用の mask/index 処理を扱う
    """
    def __init__(self, config, logger, device):
        self.device = device
        self.config = config
        self.compatible = config.ntscc['compatible']

        if config.ntscc['compatible']:
            # 互換モード用設定: multiple_rate リストや pretrained パスを指定
            self.ntscc_config = Config({
                'multiple_rate': [1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
                                  240, 256, 272, 288, 304, 320],
                'pretrained': '/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/compatible_NTSCC.pth.tar',
                'eta': config.ntscc['eta'],
                'qp_level': config.ntscc['qp_level']
            })
            self.model = CompatibleNTSCC_plus(self.ntscc_config,
                                              register_channel=False,
                                              qr_anchor_num=6)
        else:
            # 非互換モードでは複数の pretrained パスから qp_level に応じたものを選択
            pretrained_list = ['/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/ckbd2_lmbd_0.013.pth.tar',
                               '/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/ckbd2_lmbd_0.0483.pth.tar',
                               '/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/ckbd2_lmbd_0.18.pth.tar',
                               '/media/D/wangsixian/NTSCC_plus/checkpoint/ckbd2_lmbd_0.36.pth.tar',
                               '/media/D/wangsixian/NTSCC_plus/checkpoint/ckbd2_lmbd_0.72.pth.tar']
            self.ntscc_config = Config({
                'multiple_rate': [1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
                                  240, 256, 272, 288, 304, 320],
                'pretrained': pretrained_list[config.ntscc['qp_level']],
                'eta': config.ntscc['eta'],
                'qp_level': config.ntscc['qp_level']
            })
            self.model = NTSCC_plus(self.ntscc_config, register_channel=False)

        pretrained = torch.load(self.ntscc_config.pretrained, map_location='cpu')
        if 'state_dict' in pretrained:
            pretrained = pretrained['state_dict']
        result_dict = {}
        # 一部のキー（attn_mask 等）は読み込まないようにフィルタリング
        for key, weight in pretrained.items():
            result_key = key
            if 'attn_mask' not in key and 'rate_adaption_enc.mask' not in key:
                result_dict[result_key] = weight
        print(self.model.load_state_dict(result_dict, strict=False))
        self.model.to(device)
        self.model.eval()

        # ChannelWrapper を生成（rescale=True）
        self.channel = ChannelWrapper(config, logger, device, rescale=True)
        self.indexes = None

    @torch.no_grad()
    def observe_and_transpose(self, x):
        """
        NTSCC のパイプライン:
        - encode で得た channel_input, mask, indexes をチャネルへ流す
        - 等化してデコードし復元画像を返す
        - channel_usage は使用したチャネルシンボル数を返す
        """
        self.indexes = None
        channel_input, mask, indexes = self.encode(x)
        channel_usage = mask.sum().item() / 2
        ofdm_sig, cof_est, cof_gt, _ = self.channel.observe(channel_input, mask)
        s_hat = self.channel.transpose(ofdm_sig, cof_est)
        x_mse = self.decode(s_hat)
        return {"x_mse": x_mse,
                "ofdm_sig": ofdm_sig,
                # "s_hat": s_hat,
                "cof_est": cof_est,
                "cof_gt": cof_gt,
                "channel_usage": channel_usage}

    def encode(self, data, snr_override=None):
        """
        NTSCC のエンコード:
        - self.compatible によって model.encode の呼び出し方が変わる
        - 戻り値は channel_input（flatten されたシンボル列）, mask, indexes
        - mask は boolean に変換して扱う
        - 初回呼び出し時のみ self.indexes を保存して次回以降に再利用する
        - snr_override: 指定された場合、設定値(self.config.CSNR)の代わりに使用する
        """
        B, _, _, _ = data.shape
        target_snr = snr_override if snr_override is not None else self.config.CSNR

        # モデルの encode を呼び出す（compatible モードは追加引数あり）
        if self.compatible:
            s_masked, mask, indexes = self.model.encode(data,
                                                        self.indexes,
                                                        eta=self.ntscc_config.eta,
                                                        qp_level=self.ntscc_config.qp_level,
                                                        snr=target_snr)
        else:
            s_masked, mask, indexes = self.model.encode(data,
                                                        self.indexes)
        mask = mask.bool()

        # torch.masked_select の勾配周りの問題を避けるため reshape を使用
        channel_input = s_masked.reshape(B, -1)
        mask = mask.reshape(B, -1)

        if self.indexes is not None:
            # すでに indexes があれば channel_input のみ返す（推論の2回目以降を想定）
            return channel_input
        else:
            # 初回呼び出し時は index 情報等を保存して返す
            self.indexes = indexes
            self.mask = mask
            self.s_masked_shape = s_masked.shape
            return channel_input, mask, indexes

    def forward(self, channel_input, cof=None):
        # ChannelWrapper を介して OFDM 信号を生成して返す
        ofdm_sig = self.channel.forward(channel_input, self.mask, cof=cof)
        return ofdm_sig

    def decode(self, s_masked_hat):
        """
        受信した s_masked_hat を元のマスク付き形状に戻してモデル.decode を呼ぶ。
        - compatible モードかどうかで decode のシグネチャが変わる。
        """
        s_masked_hat = s_masked_hat.reshape(self.s_masked_shape)

        if self.compatible:
            x_mse = self.model.decode(s_masked_hat, self.indexes, qp_level=self.ntscc_config.qp_level,
                                      snr=self.config.CSNR)
        else:
            x_mse = self.model.decode(s_masked_hat, self.indexes)
        return x_mse

    def transpose(self, ofdm_sig, cof=None):
        # ChannelWrapper の transpose を呼んで s_hat を得る
        s_hat = self.channel.transpose(ofdm_sig, cof)
        return s_hat


if __name__ == "__main__":
    # スクリプトとして単体実行したときのテストコード
    import argparse
    import logging
    from torchvision import transforms
    from PIL import Image
    import torchvision
    import numpy as np

    logger = logging.getLogger('test')

    # 引数パーサを生成（デフォルト値をここで定義）
    config = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 引数追加
    config.add_argument('--channel_num', type=int, default=2, help='Number of channels')
    config.add_argument('--channel_type', type=str, default='ofdm_tdl', help='Type of channel')

    # OFDM パラメータ
    config.add_argument('--P', type=int, default=1, help='OFDM parameter P')
    config.add_argument('--S', type=int, default=8, help='OFDM parameter S')
    config.add_argument('--K', type=int, default=16, help='OFDM parameter K')
    config.add_argument('--L', type=int, default=8, help='OFDM parameter L')
    config.add_argument('--decay', type=int, default=4, help='OFDM parameter decay')
    config.add_argument('--N_pilot', type=int, default=1, help='OFDM parameter N_pilot')
    config.add_argument('--is_clip', type=bool, default=False, help='OFDM parameter is_clip')
    config.add_argument('--channel_est', type=str, default='perfect', help='OFDM parameter channel_est')
    config.add_argument('--equalization', type=str, default='MMSE', help='OFDM parameter equalization')

    # テスト用の ADJSCC と TDL チャネルの呼び出し例（コメントアウト）
    # 実行時には必要なパスと GPU を適切に設定して使用すること

    # Test compatible NTSCC+
    config = config.parse_args()
    config.blind = False
    config.SNR = 10
    config.ntscc = {
        'eta': 0.15,
        'qp_level': 15
    }  # 0 to 100
    config.channel_type = 'awgn'
    operator = NTSCC(config, logger, 'cuda')
    operator.model = operator.model.to('cuda')
    image_path = '/media/D/wangsixian/DiffComm/testsets/demo_test/69037.png'
    x = Image.open(image_path).convert('RGB')
    x = transforms.Resize((256, 256))(x)
    x = transforms.ToTensor()(x).unsqueeze(0).to(0)
    x = x.cuda()
    batch_size = 8
    x = x.repeat(batch_size, 1, 1, 1)
    results = operator.observe_and_transpose(x)
    print(results['ofdm_sig'].flatten()[:40])
    x_mse = results["x_mse"]
    mse = torch.mean((x - x_mse) ** 2, dim=(1, 2, 3)).cpu().numpy()
    psnr = 10 * np.log10(1 / mse)
    print(psnr, psnr.mean())
    channel_usage = results["channel_usage"]
    print(channel_usage / 256 / 256 / 3 / batch_size)
    torchvision.utils.save_image(x_mse, 'x_mse.png')

    s = operator.encode(x)
    ofdm_sig = operator.forward(s)
    print(ofdm_sig.flatten()[:40])
    s_hat = operator.transpose(ofdm_sig)
    x_gluing = operator.decode(s_hat)
    torchvision.utils.save_image(x_gluing, 'x_gluing.png')