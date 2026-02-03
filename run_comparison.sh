#!/bin/bash

# エラーが発生したら即停止
set -e

# ==============================================================================
# グローバル設定
# ==============================================================================
PYTHON_SCRIPT="main_diffcom_gamma_sweep.py"
TEMPLATE_YAML="configs/diffcom_0.yaml" # テンプレートとして使用する既存のConfig

# 指定された実行順序
SNRS=(-6 -4 -2 -8 -5 -3 -7)

# Configディレクトリの確認
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# ==============================================================================
# メインループ: 各SNRに対してスイート実行
# ==============================================================================

for SNR in "${SNRS[@]}"; do
    echo ""
    echo "========================================================"
    echo "Starting Experiment Suite for SNR = $SNR dB"
    echo "========================================================"

    # 1. Configファイルの生成 (diffcom_SNR.yaml)
    CONFIG_FILE="configs/diffcom_${SNR}.yaml"
    
    if [ ! -f "$TEMPLATE_YAML" ]; then
        echo "Error: Template config '$TEMPLATE_YAML' not found."
        exit 1
    fi

    echo "Generating config: $CONFIG_FILE"
    cp "$TEMPLATE_YAML" "$CONFIG_FILE"
    
    # OS判定 (macOSのsed対応)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
    else
        sed -i "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
    fi

    # 2. Pythonスクリプト実行 (--run_suite を使用)
    # 個別のパラメータ(gamma, eta等)はPython内部のEXPERIMENT_SUITEで定義されているため不要です。
    python "$PYTHON_SCRIPT" \
        --opt "$CONFIG_FILE" \
        --run_suite \
        --resume_index 0

done

echo ""
echo "All experiments completed successfully."