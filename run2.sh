#!/bin/bash

# run2.sh: Table Experiment (Rate vs Eta trade-off)
# SNR = -4 dB, Gamma = 0.5 (Fixed)

# エラーが発生したら即停止
set -e

# ==============================================================================
# 基本設定
# ==============================================================================
TEMPLATE_YAML="configs/diffcom_0.yaml"
PYTHON_SCRIPT="main_diffcom_retransmission.py"

# 実験固定パラメータ
FIXED_SNR=-4
FIXED_GAMMA=0.5
RETRANS_BASIS="semantic"
RESUME_IDX=0  # 全バッチを実行 (必要に応じて変更してください)

# configsディレクトリ確認
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# ==============================================================================
# 実験実行用関数
# 引数: 1=RETRANS_VALUE (Rate), 2=EXPANSION_FACTOR (Eta)
# ==============================================================================
run_experiment_case() {
    local R_VAL=$1
    local ETA_VAL=$2
    
    # 設定ファイル名 (SNRベース)
    CONFIG_FILE="configs/diffcom_${FIXED_SNR}.yaml"

    echo ""
    echo "========================================================"
    echo "Running Case: Rate(R)=$R_VAL, Eta=$ETA_VAL"
    echo "Fixed Params: SNR=${FIXED_SNR}dB, Gamma=${FIXED_GAMMA}, Basis=${RETRANS_BASIS}"
    echo "========================================================"

    # 設定ファイル生成ロジック (テンプレートからコピー & 置換)
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ ! -f "$TEMPLATE_YAML" ]; then
            echo "Error: Template file '$TEMPLATE_YAML' not found!"
            exit 1
        fi

        echo "Config file '$CONFIG_FILE' not found. Generating from template..."
        cp "$TEMPLATE_YAML" "$CONFIG_FILE"
        
        # CSNRの書き換え (OS対応: Mac/Linux)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/CSNR: .*/CSNR: $FIXED_SNR/" "$CONFIG_FILE"
        else
            sed -i "s/CSNR: .*/CSNR: $FIXED_SNR/" "$CONFIG_FILE"
        fi
        echo "Generated '$CONFIG_FILE' with CSNR: $FIXED_SNR"
    else
        echo "Using existing config file: '$CONFIG_FILE'"
    fi

    # Pythonスクリプトの実行
    # 注意: 前回の修正に基づき --retrans_gamma を明示的に渡しています
    python "$PYTHON_SCRIPT" \
        --opt "$CONFIG_FILE" \
        --retrans_mode "rate" \
        --retrans_value "$R_VAL" \
        --expansion_factor "$ETA_VAL" \
        --retrans_gamma "$FIXED_GAMMA" \
        --retrans_basis "$RETRANS_BASIS" \
        --resume_index "$RESUME_IDX"

    echo ">> Case Finished: R=$R_VAL, Eta=$ETA_VAL"
}

# ==============================================================================
# メイン処理: 表の条件を順次実行
# ==============================================================================

echo "########################################################"
echo "### START TABLE EXPERIMENT: Rate vs Eta (SNR=-4, Gamma=0.5) ###"
echo "########################################################"

# Case 1: R=0.06, Eta=8.0
run_experiment_case 0.06 8.0

# Case 2: R=0.10, Eta=4.0
run_experiment_case 0.10 4.0

# Case 3: R=0.15, Eta=2.0
run_experiment_case 0.15 2.0

# Case 4: R=0.20, Eta=1.0
run_experiment_case 0.20 1.0

echo ""
echo "########################################################"
echo "### ALL EXPERIMENTS COMPLETED SUCCESSFULLY ###"
echo "########################################################"