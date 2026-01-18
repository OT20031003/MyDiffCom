# DiffCom: Semantic Retransmission with Uncertainty Guidance

このリポジトリは、通信路のノイズで劣化した画像を、**「不確実性（Uncertainty）」**と**「意味的重要性（Semantic Saliency）」**を頼りに賢く修復・再送するシステムの実装です。

論文 **"DiffCom"** をベースに、さらに **HPRS (Hybrid-Priority Retransmission Simulation)** という拡張機能を導入しています。

---

## 📂 ファイル構成と使い分け

このプロジェクトには、実験の目的（一括実行か、詳細分析か、事後評価か）に応じて複数のスクリプトが用意されています。

### 1. メイン実験スクリプト (Simulation)

| ファイル名 | 目的 | 説明 |
| :--- | :--- | :--- |
| **`main_diffcom_retransmission.py`** | 📊 **一括比較** | 「提案手法 vs ランダム vs エッジ検出」など、複数の手法を一度に実行して性能差を比較したい時に使います。 |
| **`main_diffcom_hqg.py`** | 🔬 **詳細分析** | 「特定のパラメータ($\gamma$など)の効果を見たい」「不確実性マップを詳しく見たい」時に使います。 |
| **`run_comparison.sh`** | 🤖 **自動化** | 複数のSNR（例: -8dB, -6dB...）に対して、`retransmission.py` の一括比較実験を連続で実行します。 |

### 2. 評価・可視化ツール (Evaluation & Visualization)
実験が終わった後に、保存された画像データから数値を計算したり、解析するためのツールです。

| ファイル名 | 目的 | 説明 |
| :--- | :--- | :--- |
| **`calc_fid.py`** | 🖼️ **画質評価** | 保存済み画像から **FID** (Frechet Inception Distance) を再計算します。 |
| **`calc_id_loss.py`** | 👤 **顔認証評価** | FFHQなどの顔画像に対して、**ID Loss** (本人同一性損失) を計算します。 |
| **`calc_dists.py`** | 📏 **テクスチャ評価** | **DISTS** (Deep Image Structure and Texture Similarity) 指標を計算します。 |
| **`vit_attention_heatmap.py`** | 🔥 **可視化** | 指定した画像に対して、DINOv3 (ViT) がどこに注目しているかをヒートマップ画像として出力します。 |

---

## 🛠️ インストール

必要なライブラリをインストールします。評価ツール用にいくつか追加パッケージが必要です。

```bash
# 基本ライブラリ
pip install torch torchvision transformers pyyaml scipy tqdm matplotlib numpy

# 評価指標用 (FID, DISTS, FaceNet)
pip install torchmetrics facenet-pytorch DISTS-pytorch

# ViT (DINOv3) 用
pip install timm opencv-python
```

---

## 🚀 実行方法 1: 手法間の性能を一括比較する

**「とりあえず、提案手法が他の手法より優れているか確認したい」** という場合はこちら。

`main_diffcom_retransmission.py` を `--run_suite` フラグ付きで実行します。これだけで、以下の5つの実験が自動的に走ります。

1.  **Random Baseline**: ランダムに再送 (比較用)
2.  **Uncertainty Only**: 不確実な場所だけ再送
3.  **Importance Only**: 重要な物体(ViT)だけ再送
4.  **Edge Baseline**: エッジが強い場所だけ再送
5.  **Proposed Method**: 不確実性と重要性を組み合わせた提案手法 (HPRS)

```bash
# 単発で実行する場合 (-4dB の設定例)
python main_diffcom_retransmission.py \
    --opt configs/diffcom_-4.yaml \
    --run_suite
```

### 🤖 複数のSNRをまとめて実行する場合
`run_comparison.sh` を使用します。このスクリプトは `-8dB` から `0dB` までの実験を順次実行します。

```bash
# 実行権限を付与して実行
chmod +x run_comparison.sh
./run_comparison.sh
```

---

## 🔬 実行方法 2: 特定の条件を詳しく分析する

**「エッジ検出の挙動だけ見たい」「パラメータ $\gamma$ を調整したい」** という場合は `main_diffcom_hqg.py` を使います。

### 例: 提案手法 (Hybrid Priority) を動かす
* `--retrans_basis both`: 不確実性とViTの両方を使用
* `--retrans_gamma 0.3`: 予算の30%を意味的(ViT)に、残りをランダム構造維持に割り当て

```bash
python main_diffcom_hqg.py \
    --opt configs/diffcom_-4.yaml \
    --retrans_mode rate \
    --retrans_value 0.1 \
    --expansion_factor 2.0 \
    --retrans_basis both \
    --retrans_gamma 0.3
```

---

## 📊 実行方法 3: 事後評価 (FID / ID Loss / DISTS)

実験結果（`results_retrans_comparison/` フォルダ内の画像）に対して、後から指標を計算したい場合に使用します。

**使い方:**
各 `calc_*.py` ファイルの下部にある `if __name__ == "__main__":` ブロック内の変数を編集してから実行してください。

```python
# calc_fid.py / calc_id_loss.py / calc_dists.py の編集例

if __name__ == "__main__":
    # 1. データセット名
    DATASET = "ffhq_demo"
    
    # 2. 評価したいSNRリスト
    SNR_LABELS = ["-4", "-2"]
    
    # 3. 実験時のパラメータ (フォルダ特定に使用します)
    RATE = 0.08
    EXP_FACTOR = 3.0
    GAMMA = 0.3
    ...
```

編集後、Pythonコマンドで実行します。

```bash
# FIDの計算
python calc_fid.py

# ID Loss (顔認識精度) の計算
python calc_id_loss.py

# DISTS の計算
python calc_dists.py
```

> **Note:** 計算結果は各実験フォルダ内に `post_process_fid.json` などの名前で保存されます。

---

## 🔥 実行方法 4: Attention Mapの可視化

DINOv3 (ViT) が画像のどこを「重要」と見なしているかを確認するための単体ツールです。

`vit_attention_heatmap.py` 内の `path` 変数に画像のパスを指定して実行します。

```python
if __name__ == "__main__":
    # ここに好きな画像のパスを入れる
    path = "testsets/ffhq_demo/69903.png"
    visualize_attention_heatmap(path)
```

実行:
```bash
python vit_attention_heatmap.py
```
実行すると、`vit_attention_hf.png` という名前でヒートマップ画像が保存されます。

---
## 5. 詳細な実験結果 (Experimental Analysis)

FFHQデータセットを用いた、不確実性の有効性検証およびハイパーパラメータ感度分析の結果です。

### 5.1 不確実性と再構成誤差の相関評価
提案手法の肝となる「不確実性マップ ($U$)」が、実際の「再構成誤差 ($E$)」とどれほど相関しているかを検証しました。
SNRが改善するにつれて相関係数 ($r$) は上昇傾向にあり、不確実性が再送すべき領域を特定するための有効な指標であることが確認されました。

**表: 不確実性と再構成誤差の相関係数 (FFHQ)**
| SNR (dB) | Correlation ($r$) |
| :---: | :---: |
| -8.0 | 0.240 |
| -7.0 | 0.311 |
| -6.0 | 0.358 |
| -5.0 | 0.391 |
| -4.0 | 0.417 |
| -3.0 | 0.423 |
| -2.0 | 0.431 |

### 5.2 ハイパーパラメータの影響評価

HPRSにおける重要なパラメータ（$\gamma, R, \eta$）が画質に与える影響を評価しました。

#### (1) $\gamma$ (Semantic Budget Ratio) の影響
再送予算のうち、ViTによる意味的領域に割り当てる割合 $\gamma$ を変化させた結果です。
($\text{SNR}=-4\text{dB}, R=0.1, \eta=2.0$ 固定)

* **傾向:** $\gamma$ を大きくする（意味的領域を優先する）ほど、**ID Loss（本人同一性の損失）が低下**し、顔の特徴がより保持される傾向が見られました。

| $\gamma$ | PSNR (dB) | LPIPS | DISTS | FID | ID Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 22.19 | 0.2105 | 0.1788 | 109.57 | 0.2258 |
| 0.3 | 22.20 | 0.2118 | 0.1769 | 105.42 | 0.2100 |
| 0.6 | 22.18 | 0.2124 | 0.1767 | 106.60 | 0.2051 |
| 0.9 | 22.15 | 0.2136 | 0.1758 | 109.04 | 0.2004 |
| 1.0 | 22.13 | 0.2144 | 0.1760 | 105.86 | 0.1988 |

#### (2) 再送率 $R$ と探索範囲 $\eta$ のトレードオフ
総データ量が一定になるように $R$ (送る量) と $\eta$ (探す広さ) を調整して比較しました。
($\text{SNR}=-4\text{dB}, \gamma=0.5$ 固定)

* **定義:** データ量一定の制約 $D_{total} = \text{const}$
* **傾向:** 探索範囲 $\eta$ を絞ってでも再送率 $R$ を高めた方（表の下段に行くほど）が、PSNRやFIDなどの全体的な画質指標は向上する傾向にあります。

**設定 A:**

| $R$ (Rate) | $\eta$ (Expansion) | PSNR (dB) | LPIPS | DISTS | FID | ID Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.06 | 8.0 | 21.66 | 0.2300 | 0.1844 | 108.70 | 0.2297 |
| 0.10 | 4.0 | 22.01 | 0.2147 | 0.1799 | 109.42 | 0.1962 |
| 0.15 | 2.0 | 22.50 | 0.1956 | 0.1722 | 102.99 | 0.1667 |
| 0.20 | 1.0 | 23.10 | 0.1790 | 0.1619 | 94.11 | 0.1810 |

**設定 B (追加検証):**

| $R$ (Rate) | $\eta$ (Expansion) | PSNR (dB) | LPIPS | DISTS | FID | ID Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0800 | 3.0 | 21.96 | 0.2187 | 0.1793 | 111.13 | 0.2358 |
| 0.1000 | 2.0 | 22.19 | 0.2116 | 0.1768 | 104.70 | 0.2093 |
| 0.1333 | 1.0 | 22.63 | 0.1973 | 0.1708 | 100.20 | 0.2184 |

## 🧠 アルゴリズム解説 (HPRS)

提案手法 **HPRS (Hybrid-Priority Retransmission Simulation)** は、以下の3ステップで「どこを再送するか」を決めます。

1.  **候補選出 (Candidate Selection)**
    * AIが「自信がない（不確実）」と感じた場所をリストアップします。
    * パラメータ: `expansion_factor` (予算の何倍を候補にするか)

2.  **予算分割 (Budget Splitting)**
    * 限られた再送容量を「意味的に重要な部分」と「全体のバランスを取る部分」に分けます。
    * パラメータ: `gamma` (例: 0.3 なら 30% を重要物体に使う)

3.  **選択実行 (Selection)**
    * **Semantic枠**: 候補の中から、DINOv3 (ViT) が「重要」と判断した物体を選びます。
    * **Structure枠**: 残りの候補からランダムに選び、背景などの多様性を確保します。

---

## 引用
@article{wang2025diffcom,
  title={DiffCom: Channel Received Signal Is a Natural Condition to Guide Diffusion Posterior Sampling},
  author={Wang, Sixian and Dai, Jincheng and Tan, Kailin and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Journal on Selected Areas in Communications},
  volume={43},
  number={7},
  pages={2651--2666},
  year={2025}
}