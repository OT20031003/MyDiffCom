import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image
from scipy import linalg
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID from mixed directory structure on CPU")
    parser.add_argument("--path", type=str, default="results_retrans_comparison/ffhq_demo_100/diffcom/djscc_2/awgn_-2dB/Retrans_rate_0.1_Comparison_zeta0.3_seed22/visuals", help="Path to the 'visuals' directory (containing 0, 1, 2... folders)")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for InceptionV3 inference")
    return parser.parse_args()

class InceptionV3FeatureExtractor(nn.Module):
    """
    FID計算用のInceptionV3ラッパー。
    最終的な分類層の手前のPooling層(2048次元)の特徴量を取得します。
    """
    def __init__(self):
        super().__init__()
        # Pretrainedモデルのロード (CPU)
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception

    def forward(self, x):
        # Inception v3 expects (299, 299)
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalization (Inception specific: range -1 to 1)
        # Input x is assumed to be [0, 1]
        x = (x - 0.5) / 0.5
        
        # Forward pass steps to get features before FC
        # torchvisionのInceptionV3はforwardが複雑なため、必要な部分だけ通します
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)

        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)

        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

def get_statistics(image_paths, model, batch_size=10, device='cpu'):
    """
    画像パスのリストから特徴量を抽出し、平均(mu)と共分散(sigma)を計算する
    """
    model.eval()
    
    # 画像前処理
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    activations = []
    
    # バッチごとに処理
    for i in tqdm(range(0, len(image_paths), batch_size), leave=False):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                img_t = preprocess(img)
                batch_imgs.append(img_t)
            except Exception as e:
                print(f"Warning: Failed to load {p}. Skipping.")

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feat = model(batch_tensor)
        
        activations.append(feat.cpu().numpy())

    if not activations:
        return None, None

    activations = np.concatenate(activations, axis=0)

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    2つのガウス分布間のFrechet Distanceを計算
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight complex component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def main():
    args = parse_args()
    device = torch.device('cpu') # 強制的にCPU
    
    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist.")
        return

    print(f"Loading InceptionV3 model on {device}...")
    model = InceptionV3FeatureExtractor().to(device)

    # 1. ファイルの収集とグループ化
    # visuals/0/0_GT.png, visuals/1/0_GT.png ... のように散らばっているのをまとめる
    
    gt_files = []
    methods_files = {} # {'3_P2_hoge.png': [path1, path2, ...]}

    print("Scanning directories...")
    # visuals以下のサブディレクトリ（0, 1, 2...）を走査
    subdirs = [d for d in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, d))]
    
    count_imgs = 0
    for d in subdirs:
        dir_path = os.path.join(args.path, d)
        files = os.listdir(dir_path)
        
        for f in files:
            # フィルタリング条件
            if not f.endswith('.png'): continue
            if f.startswith('Mask_'): continue
            if f.startswith('Uncertainty_'): continue
            
            full_path = os.path.join(dir_path, f)
            
            if f == '0_GT.png':
                gt_files.append(full_path)
            else:
                if f not in methods_files:
                    methods_files[f] = []
                methods_files[f].append(full_path)
            
            count_imgs += 1

    if not gt_files:
        print("Error: No '0_GT.png' files found.")
        return
    
    print(f"Found {len(gt_files)} GT images.")
    print(f"Found {len(methods_files)} methods to compare.")
    
    # 2. GTの統計量を計算 (キャッシュとして一度だけ計算)
    print("\nCalculating statistics for Ground Truth (GT)...")
    mu_gt, sigma_gt = get_statistics(gt_files, model, batch_size=args.batch_size, device=device)
    
    if mu_gt is None:
        print("Error calculating GT statistics.")
        return

    # 3. 各手法のFID計算
    print("\nCalculating FID for each method...")
    results = []

    # ファイル名でソートして実行
    for method_name in sorted(methods_files.keys()):
        file_list = methods_files[method_name]
        
        # 枚数がGTと大きく異なるとFIDの信頼性が下がるため警告
        if len(file_list) != len(gt_files):
            print(f"Warning: {method_name} has {len(file_list)} images (GT has {len(gt_files)}).")

        print(f"Processing {method_name} ...")
        mu_method, sigma_method = get_statistics(file_list, model, batch_size=args.batch_size, device=device)
        
        if mu_method is not None:
            fid_value = calculate_frechet_distance(mu_gt, sigma_gt, mu_method, sigma_method)
            results.append((method_name, fid_value))
            print(f"  -> FID: {fid_value:.4f}")
        else:
            print(f"  -> Error: Could not calculate statistics.")

    # 4. 結果のサマリー表示
    print("\n" + "="*50)
    print("FID RESULTS SUMMARY (Lower is better)")
    print("="*50)
    # FIDが良い順（小さい順）にソート
    results.sort(key=lambda x: x[1])
    
    print(f"{'Method / Filename':<40} | {'FID':<10}")
    print("-" * 52)
    for name, score in results:
        print(f"{name:<40} | {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()