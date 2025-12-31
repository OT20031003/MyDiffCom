import easyocr
import os
import shutil
import glob
from tqdm import tqdm

# --- 設定項目 ---
INPUT_DIR = 'testsets/ffhq_demo/'  # FFHQ画像のパス
OUTPUT_DIR = './ffhq-text-images_256'             # 抽出先のフォルダ
CONFIDENCE_THRESHOLD = 0.4                    # 確信度（0.0〜1.0）。誤検出を減らすには高めに設定
USE_GPU = True                                # GPUがある場合はTrue（推奨）

def extract_text_images():
    # OCRリーダーの初期化（英語を対象）
    # FFHQは多様なデータですが、主に英数字の検出が目的であれば 'en' で十分です
    reader = easyocr.Reader(['en'], gpu=USE_GPU)

    # 出力ディレクトリの作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 画像リストの取得（pngとjpgに対応）
    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

    print(f"Total images found: {len(image_paths)}")
    print("Processing start...")

    copied_count = 0

    # 画像を1枚ずつ処理
    for img_path in tqdm(image_paths):
        try:
            # テキスト検出実行
            # detail=1 は座標や確信度などの詳細情報を取得する設定
            result = reader.readtext(img_path, detail=1)

            # 結果の検証
            has_text = False
            for (bbox, text, prob) in result:
                # 確信度が閾値を超え、かつ1文字以上のテキストがある場合
                if prob > CONFIDENCE_THRESHOLD and len(text.strip()) > 0:
                    has_text = True
                    break
            
            # 文字が含まれていると判定された場合、コピーする
            if has_text:
                filename = os.path.basename(img_path)
                destination = os.path.join(OUTPUT_DIR, filename)
                shutil.copy(img_path, destination)
                copied_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Processing complete. {copied_count} images copied to {OUTPUT_DIR}")

if __name__ == '__main__':
    extract_text_images()