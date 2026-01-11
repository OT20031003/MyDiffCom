import os
import requests
import tarfile
from PIL import Image
from tqdm import tqdm
import io

def download_and_process_imagenet(output_dir, num_images=100, image_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    
    # Imagenette (320px version) の直接ダウンロードURL
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    
    print(f"Streaming Imagenette from source fetching {num_images} images...")
    
    # ストリーミングでダウンロード開始
    response = requests.get(url, stream=True)
    
    count = 0
    # tarファイルをストリームとして開く
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        # メンバーを順次取得（全データをメモリに乗せない）
        for member in tar:
            if count >= num_images:
                break
            
            # ディレクトリはスキップ
            if not member.isfile():
                continue
            
            # 検証用データ(valフォルダ)のみを対象にする
            # 構造例: imagenette2-320/val/n01440764/ILSVRC2012_val_00000293.JPEG
            if "/val/" not in member.name:
                continue
                
            # 画像ファイル拡張子のチェック
            if not member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                # ファイルオブジェクトを抽出
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                # PILで読み込み
                image = Image.open(f)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # リサイズ (256x256)
                image = image.resize(image_size, Image.BICUBIC)

                # 保存
                save_path = os.path.join(output_dir, f"ILSVRC2012_subset_{count:08d}.png")
                image.save(save_path)
                
                count += 1
                
                # 進捗表示（簡易的）
                if count % 10 == 0:
                    print(f"Saved {count}/{num_images}...", end="\r")
                    
            except Exception as e:
                print(f"Skipped {member.name}: {e}")
                continue

    print(f"\nSuccessfully saved {count} images to {output_dir}")

if __name__ == "__main__":
    # 保存先
    OUTPUT_DIR = "testsets/imagenetadd" 
    download_and_process_imagenet(OUTPUT_DIR, num_images=100)