#!/bin/bash

# エラーが発生したら即停止
set -e

# 実験対象のSNRリスト
SNRS=(-3 -2 -5 -4 -6 -7 -8)

# ベースとなるテンプレート設定ファイル
TEMPLATE_YAML="configs/diffcom_0.yaml"

# Pythonスクリプト名
PYTHON_SCRIPT="main_diffcom_retransmission.py"

# --- 再送設定 (Retransmission Settings) ---
# mode options: 'rate', 'threshold', 'oracle'
RETRANS_MODE="rate"

# 再送率 (リスト化してループ可能)
RETRANS_VALUES=(0.1)

# basis options: 'uncertainty', 'semantic', 'both'
RETRANS_BASIS="both"

# --- HPRS (Hybrid-Priority) パラメータ ---
# 候補領域の拡張係数 (デフォルト: 2.0)
EXPANSION_FACTOR=2.0
# Semantic Priority (ViT) に割り当てる予算の割合 (0.0 ~ 1.0, デフォルト: 0.3)
RETRANS_GAMMA=0.3


echo "========================================================"
echo "Start Experiment Batch"
echo "Target SNRs: ${SNRS[*]}"
echo "Target Retrans Values: ${RETRANS_VALUES[*]}"
echo "Retransmission Basis: $RETRANS_BASIS"
echo "Expansion Factor: $EXPANSION_FACTOR"
echo "Gamma (Semantic Ratio): $RETRANS_GAMMA"
echo "========================================================"

# configsディレクトリが存在しない場合は作成
if [ ! -d "configs" ]; then
    echo "Directory 'configs' not found. Creating it..."
    mkdir -p configs
fi

# 外側のループ: Retransmission Value
for R_VAL in "${RETRANS_VALUES[@]}"; do
    echo ""
    echo "########################################################"
    echo "### Processing Retransmission Value = $R_VAL ###"
    echo "########################################################"

    # 内側のループ: SNR
    for SNR in "${SNRS[@]}"; do
        # 設定ファイル名の定義
        CONFIG_FILE="configs/diffcom_${SNR}.yaml"

        # --- Resume Index の自動設定ロジック ---
        # SNR が -2, -3 の場合は 0 から開始
        # それ以外 (-4, -5, -6, -7, -8) は 50 から開始
        if [[ "$SNR" == "-2" || "$SNR" == "-3" ]]; then
            RESUME_IDX=0
        else
            RESUME_IDX=50
        fi

        echo ""
        echo "--------------------------------------------------------"
        echo "Processing SNR = $SNR (Retrans Value = $R_VAL)"
        echo "Resume Index = $RESUME_IDX"
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

        # Pythonスクリプトの実行 (新しい引数を追加)
        echo "Running python script..."
        python "$PYTHON_SCRIPT" \
            --opt "$CONFIG_FILE" \
            --retrans_mode "$RETRANS_MODE" \
            --retrans_value "$R_VAL" \
            --retrans_basis "$RETRANS_BASIS" \
            --expansion_factor "$EXPANSION_FACTOR" \
            --retrans_gamma "$RETRANS_GAMMA" \
            --resume_index "$RESUME_IDX"

        echo "Finished SNR = $SNR at Retrans Value = $R_VAL"
    done
done

echo ""
echo "========================================================"
echo "All experiments completed successfully."
echo "========================================================"