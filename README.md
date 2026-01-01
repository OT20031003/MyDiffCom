DiffCom: Semantic Retransmission with Uncertainty Guidance

このリポジトリは、論文 **"DiffCom: Channel Received Signal Is a Natural Condition to Guide Diffusion Posterior Sampling"** をベースに、不確実性（Uncertainty）と意味的重要度（Semantic Saliency）に基づく **「意味的再送（Semantic Retransmission）」** メカニズムを実装したものです。

---

## 概要

従来の JSCC（Joint Source-Channel Coding）は、低 SNR 環境下で知覚的な劣化（ボケやアーティファクト）が生じやすいという課題がありました。  
DiffCom は拡散モデルの生成能力を活用し、受信信号を「ガイド」として用いることで、高い忠実度と知覚品質を両立します。

本プロジェクトでは、さらに復元過程の不確実性をリアルタイムで推定し、モデルが **自信のない領域** かつ **人間にとって重要な領域** を特定して部分的に再送（Signal Replacement）を行うことで、効率的な通信品質の向上を実現します。

---

## 主な機能

- **DiffCom Series のサポート**: Standard DiffCom / HiFi-DiffCom / Blind-DiffCom の各モードを搭載。
- **不確実性推定アルゴリズム**:
  - **Perturbation Uncertainty**: 予測画像に微小ノイズを付与し、復元の安定性を分散として算出。
  - **Temporal Uncertainty**: 逆拡散過程の各ステップにおける予測の一貫性を測定。
- **ViT ベースの意味的抽出**: DINOv2（`facebook/dinov2-with-registers-small`）を用いて、画像内の視覚的に重要な領域を特定。
- **2 段階復元プロセス**:
  - **Phase 1**: 初期復元と不確実性マップの生成。
  - **Phase 2**: 不確実性と ViT 重要度を掛け合わせた優先度 \(P = U \times A\) に基づく再送と再復元。

---

## インストール

```
pip install torch torchvision transformers pyyaml scipy tqdm matplotlib

# FID 計算を有効にする場合
pip install torchmetrics
```

## 使用方法

意味的再送のシミュレーションを実行するには、main_diffcom_retransmission.py を使用します。

```
python main_diffcom_retransmission.py \
    --opt ./configs/diffcom.yaml \
    --retrans_mode rate \
    --retrans_value 0.1 \
    --retrans_basis both
```
## 2. 優先度に基づく再送（Priority-based Retransmission）

受信側で計算された **不確実性マップ** \(U\) と、ViT により抽出された **意味的重要度（Attention）マップ** \(A\) を合成し、再送の優先度 \(P\) を決定します。

\[
P = U \odot A
\]

ここで、\(\odot\) は要素ごとの積（Hadamard 積）を表します。

得られた優先度マップ \(P\) に基づき、値の高い領域に対応する **潜在特徴量（Latent Features）** を選択します。  
これらの特徴量に対応するチャネル信号を、再送によって得られた **クリーンな信号** で置き換えた後、拡散モデルへ再入力することで再復元を行います。

この部分的な信号置換（Signal Replacement）により、通信レートを抑えつつも、  
モデルが不確実かつ人間にとって重要な領域の復元品質を重点的に向上させることが可能になります。

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



