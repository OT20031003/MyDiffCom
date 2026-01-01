#!/bin/bash

# エラーが発生したら即停止
set -e

# 実験対象のSNRリスト
SNRS=(-4 -2 0 2 4)

# ベースとなるテンプレート設定ファイル (configsフォルダ内にあると仮定)
TEMPLATE_YAML="configs/diffcom_0.yaml"

# Pythonスクリプト名
PYTHON_SCRIPT="main_diffcom_retransmission.py"

# 再送設定 (必要に応じて変更してください)
RETRANS_MODE="rate"    # 'rate', 'threshold', 'oracle'
RETRANS_VALUE=0.1      # 再送割合 or 閾値
RETRANS_BASIS="both"   # 'uncertainty', 'semantic', 'both'

echo "========================================================"
echo "Start Experiment Batch"
echo "Target SNRs: ${SNRS[*]}"
echo "Base Template: $TEMPLATE_YAML"
echo "========================================================"

# configsディレクトリが存在しない場合は作成（念のため）
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

for SNR in "${SNRS[@]}"; do
    # 設定ファイル名の定義 (パスを configs/ に変更)
    CONFIG_FILE="configs/diffcom_${SNR}.yaml"

    echo ""
    echo "--------------------------------------------------------"
    echo "Processing SNR = $SNR"
    echo "--------------------------------------------------------"

    # 設定ファイルが存在しない場合、テンプレートから自動生成する
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ ! -f "$TEMPLATE_YAML" ]; then
            echo "Error: Template file '$TEMPLATE_YAML' not found!"
            exit 1
        fi

        echo "Config file '$CONFIG_FILE' not found. Generating from template..."
        
        cp "$TEMPLATE_YAML" "$CONFIG_FILE"
        
        # CSNRの値を書き換える (OS分岐)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS (BSD sed)
            sed -i '' "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        else
            # Linux (GNU sed)
            sed -i "s/CSNR: .*/CSNR: $SNR/" "$CONFIG_FILE"
        fi
        
        echo "Generated '$CONFIG_FILE' with CSNR: $SNR"
    else
        echo "Using existing config file: '$CONFIG_FILE'"
    fi

    # Pythonスクリプトの実行
    echo "Running python script..."
    python "$PYTHON_SCRIPT" \
        --opt "$CONFIG_FILE" \
        --retrans_mode "$RETRANS_MODE" \
        --retrans_value "$RETRANS_VALUE" \
        --retrans_basis "$RETRANS_BASIS"

    echo "Finished SNR = $SNR"
done

echo ""
echo "========================================================"
echo "All experiments completed successfully."
echo "========================================================"