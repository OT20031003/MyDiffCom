from pathlib import Path
from PIL import Image

def resize_images(input_dir, output_dir, target_size=(256, 256)):
    """
    指定ディレクトリの画像をリサイズして別ディレクトリに保存します。
    
    Args:
        input_dir (str): 元画像があるフォルダのパス
        output_dir (str): リサイズ後の画像を保存するフォルダのパス
        target_size (tuple): 目標の解像度 (幅, 高さ)
    """
    # パスオブジェクトの作成
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # 出力フォルダが存在しない場合は作成（親フォルダも含めて作成）
    out_path.mkdir(parents=True, exist_ok=True)

    # 対象とする拡張子（必要に応じて追加してください）
    target_extensions = {'.png', '.jpeg', '.jpg', '.bmp', '.webp'}

    count = 0
    print(f"処理を開始します: {input_dir} -> {output_dir}")

    # ディレクトリ内のファイルを走査
    for file_path in in_path.iterdir():
        # ファイルかつ、拡張子が対象リストに含まれているかチェック
        if file_path.is_file() and file_path.suffix.lower() in target_extensions:
            try:
                with Image.open(file_path) as img:
                    # 画像をRGBモードに変換（PNGの透過情報などでエラーが出るのを防ぐため）
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # リサイズ実行 (LANCZOSは高品質なフィルタです)
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 保存先のパスを作成
                    save_path = out_path / file_path.name
                    
                    # 保存（画質を少し調整したい場合は quality=95 などを引数に追加）
                    resized_img.save(save_path)
                    print(f"成功: {file_path.name}")
                    count += 1
            except Exception as e:
                print(f"エラー: {file_path.name} の処理中に問題が発生しました。理由: {e}")

    print(f"--- 完了: 合計 {count} 枚の画像を処理しました ---")

# --- 実行設定 ---
if __name__ == "__main__":
    # ここに自分のフォルダパスを指定してください
    # Windowsの例: r"C:\Users\Name\Pictures\Input"
    # Mac/Linuxの例: "/Users/Name/Pictures/Input"
    
    INPUT_FOLDER = "testsets/ffhq_demo/"  # 元画像のフォルダ
    OUTPUT_FOLDER = "testsets/resized"  # 保存先のフォルダ

    resize_images(INPUT_FOLDER, OUTPUT_FOLDER)