#!/bin/bash

# エラーが発生したら即停止
set -e

# 実験対象のSNRリスト
SNRS=(-6 -4 -2 0)

# ベースとなるテンプレート設定ファイル (configsフォルダ内にあると仮定)
TEMPLATE_YAML="configs/diffcom_0.yaml"

# Pythonスクリプト名
PYTHON_SCRIPT="main_diffcom_retransmission.py"

# --- 再送設定 ---
# mode options: 'rate', 'threshold', 'oracle'
RETRANS_MODE="rate"
# ★ここを変更: 複数の値をリスト化しました (0.1を実行後、0.2を実行)
RETRANS_VALUES=(0.1)
# basis options: 'uncertainty', 'semantic', 'both'
RETRANS_BASIS="both"


echo "========================================================"
echo "Start Experiment Batch"
echo "Target SNRs: ${SNRS[*]}"
echo "Target Retrans Values: ${RETRANS_VALUES[*]}"
echo "Base Template: $TEMPLATE_YAML"
echo "Retransmission Basis: $RETRANS_BASIS"
echo "========================================================"

# configsディレクトリが存在しない場合は作成
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# ★外側のループ: Retransmission Value (0.1 -> 0.2)
for R_VAL in "${RETRANS_VALUES[@]}"; do
    echo ""
    echo "########################################################"
    echo "### Processing Retransmission Value = $R_VAL ###"
    echo "########################################################"

    # 内側のループ: SNR
    for SNR in "${SNRS[@]}"; do
        # 設定ファイル名の定義
        CONFIG_FILE="configs/diffcom_${SNR}.yaml"

        echo ""
        echo "--------------------------------------------------------"
        echo "Processing SNR = $SNR (Retrans Value = $R_VAL)"
        echo "--------------------------------------------------------"

        # 設定ファイルが存在しない場合、テンプレートから自動生成する
        if [ ! -f "$CONFIG_FILE" ]; then
            if [ ! -f "$TEMPLATE_YAML" ]; then
                echo "Error: Template file '$TEMPLATE_YAML' not found!"
                exit 1
            fi

            echo "Config file '$CONFIG_FILE' not found. Generating from template..."
            
            cp "$TEMPLATE_YAML" "$CONFIG_FILE"
            
            # CSNRの値を書き換える (OS分岐: macOS vs Linux)
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
        # --retrans_value にループ中の $R_VAL を渡します
        echo "Running python script..."
        python "$PYTHON_SCRIPT" \
            --opt "$CONFIG_FILE" \
            --retrans_mode "$RETRANS_MODE" \
            --retrans_value "$R_VAL" \
            --retrans_basis "$RETRANS_BASIS"

        echo "Finished SNR = $SNR at Retrans Value = $R_VAL"
    done
done

echo ""
echo "========================================================"
echo "All experiments completed successfully."
echo "========================================================"