from datasets import load_dataset
import os

# 保存先フォルダ
output_dir = "ffhq_samples"
os.makedirs(output_dir, exist_ok=True)

print("データセットへの接続を試みています...")

# 修正点: 削除されたリポジトリの代わりに 'marcosv/ffhq-dataset' を使用します
# このデータセットはFFHQのミラーです
try:
    dataset = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True)
except Exception as e:
    print(f"エラーが発生しました: {e}")
    exit()

print("ダウンロードを開始します...")

count = 0
for i, item in enumerate(dataset):
    if count >= 10:
        break
    
    # 画像データを取り出して保存
    # 'image'キーにPIL形式の画像が入っています
    if 'image' in item:
        image = item['image']
        image.save(f"{output_dir}/image_{i:03d}.png")
        count += 1
        if count % 10 == 0:
            print(f"{count}枚 保存完了...")

print(f"完了: {output_dir} に{count}枚の画像を保存しました。")