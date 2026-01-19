# DiffCom 再送実験スイート (Retransmission Experiment Suite)

このリポジトリは、拡散モデルを用いた意味的再送（Semantic Retransmission）の実験スイートです。
効率的な比較検証を行うため、2段階の実験プロセス（v1とv2）をサポートしています。

## 1. バージョンの概要

### v1: フルセット実験 (`main_diffcom_retransmission.py`)
- **出力先:** `results_retrans_comparison`
- **ファイル名接頭辞:** `Retrans_`
- **目的:** "Uncertainty Only" および "Proposed Method"（提案手法）を含む全手法の生成。
- **役割:** 比較対象となる高精度なベースライン（提案手法など）を作成するために使用します。

### v2: 軽量スイート実験 (`main_diffcom_retransmission_v2.py`)
- **出力先:** `results_retrans_comparison_v2`
- **ファイル名接頭辞:** `Retrans_v2_`
- **目的:** 標準的なベースライン（Random, Importance, Edge）を固定パラメータ（Rate=0.1）で高速に回すための構成。
- **特徴:** プロット時に v1 のフォルダから提案手法の結果を自動的に参照し、グラフを統合する機能（ハイブリッドモード）に対応しています。

---

## 2. 推奨ファイル構成

```text
.
├── main_diffcom_retransmission.py      # v1 シミュレーション実行用
├── main_diffcom_retransmission_v2.py   # v2 シミュレーション実行用
│
├── Suite/ (評価用スクリプト格納フォルダ)
│   ├── calc_psnr_updated.py            # PSNR計算 (v1/v2両対応)
│   ├── calc_id_loss_updated.py         # ID Loss計算 (v1/v2両対応)
│   ├── calc_lpips_updated.py           # LPIPS計算 (v1/v2両対応)
│   ├── calc_dists_updated.py           # DISTS計算 (v1/v2両対応)
│   ├── calc_fid_updated.py             # FID計算 (v1/v2両対応)
│   │
│   ├── plot_psnr_suite.py              # PSNRプロット (ハイブリッド対応)
│   ├── plot_id_loss_suite.py           # ID Lossプロット (ハイブリッド対応)
│   ├── plot_lpips_suite.py             # LPIPSプロット (ハイブリッド対応)
│   ├── plot_dists_suite.py             # DISTSプロット (ハイブリッド対応)
│   └── plot_fid_suite.py               # FIDプロット (ハイブリッド対応)
│
├── results_retrans_comparison/         # v1 の出力結果
└── results_retrans_comparison_v2/      # v2 の出力結果
```

---

## 3. シミュレーションの実行方法

### Step 1: 提案手法データの生成 (v1)
オリジナルのスクリプトを実行して、比較対象となる「提案手法 (Proposed)」や「不確実性手法 (Uncertainty)」を生成します。

```bash
python main_diffcom_retransmission.py \
    --opt configs/diffcom_-4.yaml \
    --retrans_mode rate \
    --retrans_value 0.1 \
    --retrans_basis semantic \
    --run_suite
```
*注意: `calc`や`plot`の設定にある `GAMMA` 値（例: 0.9）と一致しているか確認してください。*

### Step 2: 比較用ベースラインの生成 (v2)
v2スクリプトを実行して、一般的なベースライン（Random, Edgeなど）を新しいディレクトリ構造で生成します。

```bash
python main_diffcom_retransmission_v2.py \
    --opt configs/diffcom_-4.yaml \
    --retrans_mode rate \
    --retrans_value 0.1 \
    --retrans_basis semantic \
    --run_suite
```

---

## 4. 評価指標の計算方法 (Calc)

`Suite/calc_*_updated.py` スクリプトを使用して、生成された画像から評価指標を計算し JSON に保存します。

**設定の変更:**
各スクリプトファイルを開き、下部の `Configuration` エリアで `VERSION` を指定してください。

```python
# 対象バージョンの選択
VERSION = "v2"  # "v2" にすると v2フォルダを計算対象にします
```

**実行コマンド例:**
```bash
python Suite/calc_psnr_updated.py
python Suite/calc_id_loss_updated.py
python Suite/calc_lpips_updated.py
python Suite/calc_dists_updated.py
python Suite/calc_fid_updated.py
```

---

## 5. 結果のグラフ化 (Plot / Hybrid Mode)

`Suite/plot_*_suite.py` スクリプトでグラフを作成します。

**重要な機能: ハイブリッドプロット (v2 Mode)**
設定で `VERSION = "v2"` を選択すると、以下の動作を自動的に行います：
1. `results_retrans_comparison_v2` から標準ベースライン（Random, Importance, Edge）の結果を読み込む。
2. 自動的に `results_retrans_comparison` (v1フォルダ) を検索し、「Uncertainty Only」と「Proposed Method」の結果を取得する。
3. これらを**1つのグラフに統合**して出力する。

**設定:**
```python
VERSION = "v2"  # "v1" にすると v1フォルダの中身だけをプロットします
```

**実行コマンド例:**
```
python Suite/plot_psnr_suite.py
python Suite/plot_id_loss_suite.py
# 他の指標も同様
```

---

## 6. 必須要件 (Requirements)

評価指標の計算には以下のライブラリが必要です：

- `torchmetrics` (LPIPS, PSNR, FID用)
- `facenet-pytorch` (ID Loss用)
- `DISTS_pytorch` (DISTS用)

```
pip install torchmetrics[image] facenet-pytorch DISTS-pytorch
```