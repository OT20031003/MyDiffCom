#!/bin/bash

# エラーが発生したら即停止
set -e

# ベース設定
TEMPLATE_YAML="configs/diffcom_0.yaml"
PYTHON_SCRIPT="main_diffcom_retransmission.py"
RETRANS_MODE="rate"
RETRANS_VALUE=0.1
RETRANS_BASIS="semantic"  # semanticで固定

# configsディレクトリが存在しない場合は作成
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# ==============================================================================
# 実験実行用関数
# 引数: 1=SNR, 2=EXPANSION_FACTOR, 3=RETRANS_GAMMA, 4=説明ラベル
# ==============================================================================
run_experiment_step() {
    local SNR=$1
    local EXP_FACTOR=$2
    local GAMMA=$3
    local LABEL=$4

    CONFIG_FILE="configs/diffcom_${SNR}.yaml"
    RESUME_IDX=0

    echo ""
    echo "--------------------------------------------------------"
    echo "Experiment: $LABEL"
    echo "Processing SNR = $SNR"
    echo "Params: Basis=$RETRANS_BASIS, Val=$RETRANS_VALUE, Eta=$EXP_FACTOR, Gamma=$GAMMA"
    echo "--------------------------------------------------------"

    # 設定ファイル生成ロジック (テンプレート利用)
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ ! -f "$TEMPLATE_YAML" ]; then
            echo "Error: Template file '$TEMPLATE_YAML' not found!"
            exit 1
        fi

        echo "Config file '$CONFIG_FILE' not found. Generating from template..."
        cp "$TEMPLATE_YAML" "$CONFIG_FILE"
        
        # CSNRの書き換え (macOS/Linux対応)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        else
            sed -i "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        fi
        echo "Generated '$CONFIG_FILE' with CSNR: $SNR"
    else
        echo "Using existing config file: '$CONFIG_FILE'"
    fi

    # Pythonスクリプトの実行
    python "$PYTHON_SCRIPT" \
        --opt "$CONFIG_FILE" \
        --retrans_mode "$RETRANS_MODE" \
        --retrans_value "$RETRANS_VALUE" \
        --retrans_basis "$RETRANS_BASIS" \
        --expansion_factor "$EXP_FACTOR" \
        --retrans_gamma "$GAMMA" \
        --resume_index "$RESUME_IDX"
}


# ==============================================================================
# 実験1: Unc の挙動を確認するためだけ (eta=1.0, gamma=0.0)
# ==============================================================================
SNRS_EXP1=(-6 -4 -2 -8 -5 -3 -7)
ETA_1=2.0
GAMMA_1=0.5
LABEL_1="Pure Unc Behavior (eta=1.0, gamma=0.0)"

echo "########################################################"
echo "### START Experiment 1: $LABEL_1 ###"
echo "########################################################"

for SNR in "${SNRS_EXP1[@]}"; do
    run_experiment_step $SNR $ETA_1 $GAMMA_1 "$LABEL_1"
done


# ==============================================================================
# 実験2: ViTのみだけ (eta=10.0, gamma=1.0)
# ==============================================================================
SNRS_EXP1=(-6 -4 -2 -8 -5 -3 -7)
ETA_2=3.0
GAMMA_2=0.3
#LABEL_2="ViT Only (eta=10.0, gamma=1.0)"
LABEL_2="Sem (eta=3.0, gamma=0.3)"
echo ""
echo "########################################################"
echo "### START Experiment 2: $LABEL_2 ###"
echo "########################################################"

for SNR in "${SNRS_EXP2[@]}"; do
    run_experiment_step $SNR $ETA_2 $GAMMA_2 "$LABEL_2"
done

echo ""
echo "========================================================"
echo "All experiments completed successfully."
echo "========================================================"