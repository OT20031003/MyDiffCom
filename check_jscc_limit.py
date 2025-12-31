import argparse
import os
import torch
import torchvision
import yaml
import numpy as np
from utils.util import Config, MetricWrapper, DictAverageMeter
from guided_diffusion.measurement import get_operator
from data.datasets import get_test_loader
from utils import util
import logging
from tqdm import tqdm  # 進捗表示用

def check_noiseless_jscc_all():
    # 1. 設定読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml')
    args = parser.parse_args()
    
    with open(args.opt, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Config(config_dict)
    
    config.world_size = 1
    config.model_zoo = os.path.join(config.cwd, 'model_zoo')
    config.testsets = os.path.join(config.cwd, 'testsets')
    config.results = os.path.join(config.cwd, 'results_noiseless_check_all') # 保存先フォルダ名を変更
    util.mkdir(config.results)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger("NoiselessCheck")
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading Operator: {config.operator_name}...")
    operator = get_operator(config.operator_name, config=config, logger=logger, device=device)
    operator.model = operator.model.to(device)
    operator.model.eval()

    config.testsets_path = os.path.join(config.testsets, config.testset_name)
    dataloader = get_test_loader(config.testsets_path, batch_size=1, shuffle=False)
    metric_wrapper = MetricWrapper().to(device)

    # 平均スコア記録用
    avg_meters = DictAverageMeter()

    logger.info(f"Starting Fixed Noiseless Check for {len(dataloader)} images...")
    
    # tqdmを使って進捗バーを表示
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_image, names = batch
        input_image = input_image.to(device)
        
        # ファイル名を取得 (例: "00001.png")
        fname = names[0]
        fname_no_ext = os.path.splitext(fname)[0]
        
        with torch.no_grad():
            # ==========================================
            # 修正ポイント: パワー正規化
            # ==========================================
            
            # 1. Encode
            s_raw = operator.encode(input_image)
            
            # 2. Power Normalization
            # 平均パワー(分散)を計算し、正規化
            current_power = torch.mean(s_raw ** 2)
            s_normalized = s_raw / torch.sqrt(current_power)
            
            # 3. Decode
            x_recon = operator.decode(s_normalized)
            
            # 4. Clamp
            x_recon = torch.clamp(x_recon, 0.0, 1.0)
            
            metrics = metric_wrapper(x_recon, input_image)
            avg_meters.update(metrics)

        # 保存 (ファイル名を使ってユニークにする)
        save_path = config.results
        
        # GTは毎回保存しなくても良いかもしれませんが、念のため対応するペアとして保存します
        torchvision.utils.save_image(input_image, os.path.join(save_path, f'{fname_no_ext}_GT.png'))
        torchvision.utils.save_image(x_recon, os.path.join(save_path, f'{fname_no_ext}_Recon.png'))
        
        # ログが多すぎると見づらいので、10枚おき、または重要なときだけ詳細表示するなど調整可能
        # ここでは tqdm を使っているので、print は控えめにします

    logger.info("========================================")
    logger.info(f"Finished processing {len(dataloader)} images.")
    logger.info(f"Average PSNR: {avg_meters.avg['psnr']:.2f} dB")
    logger.info(f"Average LPIPS: {avg_meters.avg['lpips']:.4f}")
    logger.info(f"Results saved to: {config.results}")
    logger.info("========================================")

if __name__ == '__main__':
    check_noiseless_jscc_all()