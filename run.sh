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
    
    # ▼▼▼ 修正箇所: GAMMA=0.0のときだけResume Indexを50にする ▼▼▼
    if [ "$GAMMA" == "0.0" ]; then
        RESUME_IDX=0
    else
        RESUME_IDX=0
    fi
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    echo ""
    echo "--------------------------------------------------------"
    echo "Experiment: $LABEL"
    echo "Processing SNR = $SNR"
    echo "Params: Basis=$RETRANS_BASIS, Val=$RETRANS_VALUE, Eta=$EXP_FACTOR, Gamma=$GAMMA"
    echo "Resume Index: $RESUME_IDX"
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
# 1. 優先タスク: 実験2の残り (SNR -3, -7)
# 設定: eta=3.0, gamma=0.3
# ==============================================================================
SNRS_PRIORITY=(-3 -7)
ETA_PRIORITY=3.0
GAMMA_PRIORITY=0.3
LABEL_PRIORITY="Priority Task: Finish Exp 2 (eta=3.0, gamma=0.3)"

echo "########################################################"
echo "### START PRIORITY TASK: $LABEL_PRIORITY ###"
echo "########################################################"

# for SNR in "${SNRS_PRIORITY[@]}"; do
#     run_experiment_step $SNR $ETA_PRIORITY $GAMMA_PRIORITY "$LABEL_PRIORITY"
# done


# ==============================================================================
# 2. 新規実験: Gamma Sweep
# 設定: SNR=-4 (固定), eta=2.0, gamma=(0.0 0.6 0.9 1.0)
# ==============================================================================
FIXED_SNR=-4
FIXED_ETA=2.0
GAMMAS_NEW=(0.0 0.3 0.9 1.0)
LABEL_NEW="New Experiment: Gamma Sweep (SNR=-4, eta=2.0)"

echo ""
echo "########################################################"
echo "### START NEW EXPERIMENT: $LABEL_NEW ###"
echo "########################################################"

for GAMMA in "${GAMMAS_NEW[@]}"; do
    run_experiment_step $FIXED_SNR $FIXED_ETA $GAMMA "$LABEL_NEW"
done

echo ""
echo "========================================================"
echo "All requested experiments completed successfully."
echo "========================================================"