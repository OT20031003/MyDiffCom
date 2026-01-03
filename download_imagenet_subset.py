import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_and_process_imagenet(output_dir, num_images=100, image_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Streaming Imagenette (subset of ImageNet) fetching {num_images} images...")
    
    # 変更点: "imagenet-1k" -> "frgfm/imagenette"
    # "320px" は解像度設定（256pxを作るのに丁度よいサイズ）
    dataset = load_dataset("frgfm/imagenette", "320px", split="validation", streaming=True)
    
    count = 0
    for item in tqdm(dataset, total=num_images):
        if count >= num_images:
            break
            
        try:
            image = item['image']
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # リサイズ (256x256)
            image = image.resize(image_size, Image.BICUBIC)

            # 保存
            save_path = os.path.join(output_dir, f"ILSVRC2012_subset_{count:08d}.png")
            image.save(save_path)
            
            count += 1
            
        except Exception as e:
            print(f"Skipped: {e}")
            continue

    print(f"Successfully saved {count} images to {output_dir}")

if __name__ == "__main__":
    # 保存先
    OUTPUT_DIR = "testsets/imagenet" 
    download_and_process_imagenet(OUTPUT_DIR, num_images=50)