import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm import tqdm

# 既存モジュールのインポート
# 実行ディレクトリによってはパスの調整が必要な場合があります
from utils.util import Config
from guided_diffusion.measurement import get_operator
# プロジェクト内のLPIPSモジュールを使用
from _pdjscc.loss_utils.perceptual_similarity.perceptual_loss import PerceptualLoss

# ==========================================
# 1. データセット定義 (COCO val2017用)
# ==========================================
class CocoSimpleDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        # jpg, pngなどを収集
        self.image_paths = glob(os.path.join(root_dir, '*.jpg')) + \
                           glob(os.path.join(root_dir, '*.png'))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # エラー時はランダムなノイズを返す（学習を止めないため）
            return torch.rand(3, 256, 256)

# ==========================================
# 2. 重要度予測モデル (Importance Predictor)
# ==========================================
class ImportancePredictor(nn.Module):
    """
    エンコーダの特徴量 z を入力とし、その重要度マップ(0~1)を出力するモデル。
    """
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1), # 入力と同じチャンネル数を出力
            nn.Sigmoid() # 0~1に正規化
        )

    def forward(self, z):
        return self.net(z)

# ==========================================
# 3. 学習用スクリプト
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Importance Predictor")
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml', help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COCO val2017 directory")
    parser.add_argument("--save_dir", type=str, default='./results/importance_model_elementwise', help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configの読み込み
    with open(args.opt, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Config(config_dict)
    
    # GPU設定
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device 

    print(f"Loading Operator: {config.operator_name}...")
    # ロガーのダミー（DeepJSCC等の初期化に必要）
    import logging
    logger = logging.getLogger("dummy")
    logger.setLevel(logging.INFO)
    
    # エンコーダ/デコーダ (Operator) のロード
    operator = get_operator(config.operator_name, config=config, logger=logger, device=device)
    
    # 【重要】モデルをGPUへ転送
    operator.model = operator.model.to(device)
    
    # 学習させないように固定
    for param in operator.model.parameters():
        param.requires_grad = False
    operator.model.eval()

    # データセット
    print(f"Loading Dataset from {args.data_dir}...")
    dataset = CocoSimpleDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # LPIPS計算用モデル (Spatial=Trueで空間マップを取得)
    print("Loading LPIPS Metric...")
    lpips_loss_fn = PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=True)
    lpips_loss_fn.to(device)
    lpips_loss_fn.eval()

    # 特徴量のチャンネル数を取得するためのダミーフォワード
    dummy_input = torch.zeros(1, 3, 256, 256).to(device)
    with torch.no_grad():
        if config.operator_name == 'djscc':
            dummy_z = operator.model.encode(dummy_input, given_SNR=config.CSNR)
        else:
            raise NotImplementedError("This script supports 'djscc' operator currently.")
    
    feat_channels = dummy_z.shape[1]
    print(f"Feature shape: {dummy_z.shape}, Channels: {feat_channels} (Element-wise Prediction)")

    # 重要度予測モデル
    imp_model = ImportancePredictor(in_channels=feat_channels).to(device)
    optimizer = optim.Adam(imp_model.parameters(), lr=args.lr)
    
    # 保存ディレクトリ
    os.makedirs(args.save_dir, exist_ok=True)

    print("Start Training...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0.0 

        for i, images in enumerate(pbar):
            images = images.to(device)
            B = images.shape[0]

            # 1. 伝送シミュレーション & 勾配ベースの重要度算出 (教師データ作成)
            
            # (A) 特徴量 z を取得 (勾配計算の始点)
            with torch.no_grad():
                z_raw = operator.model.encode(images, given_SNR=config.CSNR)
            
            # z を計算グラフの葉として扱う
            z = z_raw.detach()
            z.requires_grad = True
            
            # (B) デコードして画像を再構成 (チャネル通過を含む)
            if config.operator_name == 'djscc':
                # --- エラー修正箇所 ---
                # operator.model.feature_pass_channel の代わりに
                # operator.channel (ChannelWrapper) を直接使用して微分可能なパスを通す
                
                # 1. Flatten
                z_flat = z.reshape(B, -1)
                
                # 2. Channel Observe (ノイズ付加)
                # maskは全て1 (DeepJSCCは全シンボル送信)
                mask = torch.ones_like(z_flat)
                # observeメソッドは微分可能（内部でtorch.normalを使用）
                # 戻り値: received_signal, cof_est, cof_gt, usage
                ofdm_sig, cof_est, _, _ = operator.channel.observe(z_flat, mask)
                
                # 3. Channel Transpose (デシャッフル・等化)
                s_hat = operator.channel.transpose(ofdm_sig, cof_est)
                
                # 4. Reshape back to (B, C, H, W)
                noisy_z = s_hat.reshape(z.shape)
                
                # 5. Decode
                recon_imgs = operator.model.jscc_decoder(noisy_z, torch.ones([B, 1]).to(device) * config.CSNR)
                # ---------------------
            else:
                raise NotImplementedError("Currently only djscc is supported.")

            # (C) LPIPSロスの計算
            loss_lpips = lpips_loss_fn(images, recon_imgs, normalize=True).mean()
            
            # (D) z についてバックプロパゲーション (勾配計算)
            optimizer.zero_grad() # モデルの勾配は不要
            if z.grad is not None:
                z.grad.zero_()
            
            loss_lpips.backward()
            
            # (E) 勾配の絶対値を取得して教師データ化
            with torch.no_grad():
                grads = z.grad.abs() # (B, C, H, W)
                target_map = grads
                
                # 正規化 (特徴量全体でのMin-Max)
                min_v = target_map.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
                max_v = target_map.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
                target_map = (target_map - min_v) / (max_v - min_v + 1e-8)
                
                z.requires_grad = False
                z = z.detach()

            # 2. 重要度予測 (モデルの学習)
            pred_map = imp_model(z)

            # 3. ロス計算 & 更新
            loss = nn.MSELoss()(pred_map, target_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- プログレスバー表示更新 ---
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (i + 1)
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # エポックごとにモデル保存
        torch.save(imp_model.state_dict(), os.path.join(args.save_dir, f"imp_model_epoch_{epoch}.pth"))
        
        # 可視化画像の保存 
        if epoch % 1 == 0:
            from torchvision.utils import save_image
            with torch.no_grad():
                # 平均化して可視化
                vis_pred_src = pred_map.mean(dim=1, keepdim=True)
                vis_target_src = target_map.mean(dim=1, keepdim=True)
                
                vis_pred = torch.nn.functional.interpolate(vis_pred_src, size=(256, 256), mode='nearest')
                vis_target = torch.nn.functional.interpolate(vis_target_src, size=(256, 256), mode='nearest')
                
                # 可視化用正規化
                vis_pred = (vis_pred - vis_pred.min()) / (vis_pred.max() - vis_pred.min() + 1e-8)
                vis_target = (vis_target - vis_target.min()) / (vis_target.max() - vis_target.min() + 1e-8)
            
            comparison = torch.cat([images[:4], vis_target[:4].repeat(1,3,1,1), vis_pred[:4].repeat(1,3,1,1)], dim=0)
            save_image(comparison, os.path.join(args.save_dir, f"vis_epoch_{epoch}.png"), nrow=4)

    print("Training Finished!")

if __name__ == "__main__":
    main()