#!/bin/bash

# エラーが発生したら即停止
set -e

# ==============================================================================
# 基本設定
# ==============================================================================
TEMPLATE_YAML="configs/diffcom_0.yaml"   # テンプレート元ファイル（必ず存在する必要があります）
PYTHON_SCRIPT="main_diffcom_retransmission.py"
RETRANS_MODE="rate"       # 提案法の設定
RETRANS_VALUE=0.1         # 再送率
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

    # SNRに応じた設定ファイル名
    CONFIG_FILE="configs/diffcom_${SNR}.yaml"
    
    # ▼▼▼ Resume Indexの設定 (必要に応じて変更してください) ▼▼▼
    # 例: 特定の条件だけ途中から再開したい場合などに分岐を追加
    if [ "$GAMMA" == "0.0" ]; then
        RESUME_IDX=0
    else
        RESUME_IDX=0
    fi
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    echo ""
    echo "--------------------------------------------------------"
    echo "Experiment: $LABEL"
    echo "Target: SNR=$SNR | Gamma=$GAMMA | Eta=$EXP_FACTOR"
    echo "Config: $CONFIG_FILE"
    echo "Basis : $RETRANS_BASIS"
    echo "Resume: $RESUME_IDX"
    echo "--------------------------------------------------------"

    # --------------------------------------------------------
    # 設定ファイル生成ロジック (テンプレート自動コピー & CSNR置換)
    # --------------------------------------------------------
    if [ ! -f "$CONFIG_FILE" ]; then
        # テンプレートが存在するか確認
        if [ ! -f "$TEMPLATE_YAML" ]; then
            echo "Error: Template file '$TEMPLATE_YAML' not found!"
            echo "Please create a base config file at '$TEMPLATE_YAML' first."
            exit 1
        fi

        echo "Config file '$CONFIG_FILE' not found. Generating from template..."
        cp "$TEMPLATE_YAML" "$CONFIG_FILE"
        
        # CSNRの書き換え (macOS/Linux対応のsed分岐)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS (BSD sed) 用
            sed -i '' "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        else
            # Linux (GNU sed) 用
            sed -i "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        fi
        echo "Generated '$CONFIG_FILE' with CSNR set to $SNR."
    else
        echo "Using existing config file: '$CONFIG_FILE'"
    fi

    # --------------------------------------------------------
    # Pythonスクリプトの実行
    # NOTE: --run_suite は指定しない (Single Runモード)
    # --------------------------------------------------------
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
# 実験パラメータ設定: Gamma Sweep
# SNR=-4 (固定), eta=2.0 (固定), Gammaを変化させる
# ==============================================================================

FIXED_SNR=0
FIXED_ETA=2.0
# GAMMAS_NEW=(0.0 0.3 0.9 1.0)
GAMMAS_NEW=(0.6)
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