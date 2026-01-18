#!/bin/bash

# エラーが発生したら即停止
set -e

# ==============================================================================
# グローバル設定
# ==============================================================================
PYTHON_SCRIPT="main_diffcom_retransmission.py"
TEMPLATE_YAML="configs/diffcom_0.yaml" # テンプレートとして使用する既存のConfig
SNRS=(-6 -4 -2 -8 -5 -3 -7)            # 実行するSNRの範囲

# Configディレクトリの確認
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# ==============================================================================
# 実験実行関数
# 引数:
# 1: SNR
# 2: METHOD_NAME (ログ用)
# 3: MODE (rate / random)
# 4: VALUE (再送率 R)
# 5: EXPANSION (候補係数 eta)
# 6: GAMMA (意味的優先度 gamma)
# 7: BASIS (uncertainty / semantic / edge)
# ==============================================================================
run_experiment() {
    local SNR=$1
    local NAME=$2
    local MODE=$3
    local VAL=$4
    local ETA=$5
    local GAM=$6
    local BASIS=$7

    echo ""
    echo "####################################################################"
    echo "Experiment: $NAME"
    echo "SNR: $SNR | Mode: $MODE | R: $VAL | Eta: $ETA | Gamma: $GAM | Basis: $BASIS"
    echo "####################################################################"

    # Configファイルの生成 (diffcom_SNR.yaml)
    CONFIG_FILE="configs/diffcom_${SNR}.yaml"
    
    if [ ! -f "$TEMPLATE_YAML" ]; then
        echo "Error: Template config '$TEMPLATE_YAML' not found."
        exit 1
    fi

    # テンプレートをコピーしてCSNRを書き換え
    cp "$TEMPLATE_YAML" "$CONFIG_FILE"
    
    # OS判定 (macOSのsed対応)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
    else
        sed -i "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
    fi

    # Pythonスクリプト実行
    python "$PYTHON_SCRIPT" \
        --opt "$CONFIG_FILE" \
        --retrans_mode "$MODE" \
        --retrans_value "$VAL" \
        --expansion_factor "$ETA" \
        --retrans_gamma "$GAM" \
        --retrans_basis "$BASIS" \
        --resume_index 0
}

# ==============================================================================
# メインループ: 各SNRに対して全手法を実行
# ==============================================================================

for SNR in "${SNRS[@]}"; do
    echo "========================================================"
    echo "Starting Batch for SNR = $SNR dB"
    echo "========================================================"

    # --------------------------------------------------------------------------
    # 1. Random Baseline
    # 条件: フィードバック不要 (R=0.2), ランダム選択
    # Params: retrans_mode='random', value=0.2, eta=10.0 (dummy), gamma=0.0
    # --------------------------------------------------------------------------
    run_experiment $SNR "1_Random_Baseline" "random" 0.2 10.0 0.0 "semantic"

    # --------------------------------------------------------------------------
    # 2. Uncertainty Only
    # 条件: フィードバックあり (R=0.1), 不確実性のみ (eta=1.0, gamma=0.0)
    # 解説: eta=1.0で候補=予算となり、gamma=0.0で全予算を「候補(＝高不確実性)」から選ぶため
    #       実質的に「不確実性の高い順」になります。
    # --------------------------------------------------------------------------
    run_experiment $SNR "2_Uncertainty_Only" "rate" 0.1 1.0 0.0 "semantic"

    # --------------------------------------------------------------------------
    # 3. Importance Only
    # 条件: フィードバック不要 (R=0.2), 意味的重要性のみ (eta=10.0, gamma=1.0)
    # 解説: eta=10.0で広範囲を候補とし、gamma=1.0でその中からViTスコア上位のみを選ぶため
    #       純粋なSemantic Importanceベースになります。
    # --------------------------------------------------------------------------
    run_experiment $SNR "3_Importance_Only" "rate" 0.2 10.0 1.0 "semantic"

    # --------------------------------------------------------------------------
    # 4. Edge (Sobel)
    # 条件: フィードバック不要 (R=0.2), エッジ検出ベース
    # Params: basis='edge', value=0.2
    # --------------------------------------------------------------------------
    run_experiment $SNR "4_Edge_Baseline" "rate" 0.2 10.0 0.0 "edge"

    # --------------------------------------------------------------------------
    # 5. Proposed Method
    # 条件: フィードバックあり (R=0.1), 提案手法 (eta=2.0, gamma=0.3)
    # 解説: 不確実性で候補を絞り(eta=2.0)、その中から30%をSemantic、70%をStructureで選択
    # --------------------------------------------------------------------------
    run_experiment $SNR "5_Proposed_Method" "rate" 0.1 2.0 0.3 "semantic"

done

echo ""
echo "All experiments completed successfully."