#!/bin/bash

# run_all.sh
# 1. run.shの残り (Gamma=0.3の途中再開 + 残りのGamma)
# 2. run2.shの全実験 (Rate vs Eta)
# をまとめて実行するスクリプト

# エラーが発生したら即停止
set -e

# ==============================================================================
# 共通設定
# ==============================================================================
TEMPLATE_YAML="configs/diffcom_0.yaml"
PYTHON_SCRIPT="main_diffcom_retransmission.py"
RETRANS_BASIS="semantic"  # 共通してsemanticを使用

# configsディレクトリ確認
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# ==============================================================================
# 汎用実行関数
# 引数: 
#   1: SNR
#   2: Rate (Retrans Value)
#   3: Eta (Expansion Factor)
#   4: Gamma
#   5: Resume Index (開始バッチ番号)
#   6: Description (ログ用)
# ==============================================================================
run_experiment() {
    local SNR=$1
    local R_VAL=$2
    local ETA_VAL=$3
    local GAMMA_VAL=$4
    local START_IDX=$5
    local DESC=$6
    
    CONFIG_FILE="configs/diffcom_${SNR}.yaml"

    echo ""
    echo "========================================================"
    echo "Experiment: $DESC"
    echo "Params: SNR=$SNR, Rate=$R_VAL, Eta=$ETA_VAL, Gamma=$GAMMA_VAL"
    echo ">> Resuming from index: $START_IDX"
    echo "========================================================"

    # 設定ファイル生成 (なければ作成)
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ ! -f "$TEMPLATE_YAML" ]; then
            echo "Error: Template file '$TEMPLATE_YAML' not found!"
            exit 1
        fi
        echo "Generating '$CONFIG_FILE'..."
        cp "$TEMPLATE_YAML" "$CONFIG_FILE"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        else
            sed -i "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        fi
    fi

    # Python実行
    python "$PYTHON_SCRIPT" \
        --opt "$CONFIG_FILE" \
        --retrans_mode "rate" \
        --retrans_value "$R_VAL" \
        --expansion_factor "$ETA_VAL" \
        --retrans_gamma "$GAMMA_VAL" \
        --retrans_basis "$RETRANS_BASIS" \
        --resume_index "$START_IDX"
    
    echo ">> Finished: $DESC"
}

echo "########################################################"
echo "### START INTEGRATED EXPERIMENT RUN ###"
echo "########################################################"

# ==============================================================================
# Part 1: run.sh の残り (Gamma Sweep)
# SNR=-4, Eta=2.0, Rate=0.1 (固定)
# ==============================================================================

# 1-1. 中断箇所の再開 (Gamma=0.3, Index=60から)
run_experiment -4 0.1 2.0 0.3 60 "Resume Gamma Sweep (gamma=0.3)"

# 1-2. 残りのGamma設定 (Gamma=0.9, 1.0)
run_experiment -4 0.1 2.0 0.9 0  "Gamma Sweep (gamma=0.9)"
run_experiment -4 0.1 2.0 1.0 0  "Gamma Sweep (gamma=1.0)"


# ==============================================================================
# Part 2: run2.sh の全実験 (Rate vs Eta Trade-off)
# SNR=-4, Gamma=0.5 (固定), RateとEtaを可変
# ==============================================================================

# 2-1. R=0.06, Eta=8.0
run_experiment -4 0.06 8.0 0.6 0 "Table Exp (R=0.06, Eta=8.0)"

# 2-2. R=0.10, Eta=4.0
run_experiment -4 0.10 4.0 0.6 0 "Table Exp (R=0.10, Eta=4.0)"

# 2-3. R=0.15, Eta=2.0
run_experiment -4 0.15 2.0 0.6 0 "Table Exp (R=0.15, Eta=2.0)"

# 2-4. R=0.20, Eta=1.0
run_experiment -4 0.20 1.0 0.6 0 "Table Exp (R=0.20, Eta=1.0)"


echo ""
echo "########################################################"
echo "### ALL TASKS (run.sh remainder + run2.sh) COMPLETED ###"
echo "########################################################"