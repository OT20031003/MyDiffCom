#!/bin/bash

# エラーが発生したら即座に停止する設定
set -e

# スクリプトがあるディレクトリ (Suite) を指定
SCRIPT_DIR="Suite"

echo "================================================="
echo "   Starting Evaluation Suite for Version: v3"
echo "================================================="

# --- 1. 計算フェーズ (Calculation) ---
echo ""
echo "[1/2] Calculating Metrics..."
echo "-------------------------------------------------"

echo "1. Calculating PSNR..."
python ${SCRIPT_DIR}/calc_psnr_suite.py

echo "2. Calculating LPIPS..."
python ${SCRIPT_DIR}/calc_lpips_suite.py

echo "3. Calculating DISTS..."
python ${SCRIPT_DIR}/calc_dists_suite.py

echo "4. Calculating FID..."
python ${SCRIPT_DIR}/calc_fid_suite.py

echo "5. Calculating ID Loss..."
python ${SCRIPT_DIR}/calc_id_loss_suite.py

# --- 2. 描画フェーズ (Plotting) ---
echo ""
echo "[2/2] Plotting Graphs..."
echo "-------------------------------------------------"

echo "1. Plotting PSNR..."
python ${SCRIPT_DIR}/plot_psnr_suite.py

echo "2. Plotting LPIPS..."
python ${SCRIPT_DIR}/plot_lpips_suite.py

echo "3. Plotting DISTS..."
python ${SCRIPT_DIR}/plot_dists_suite.py

echo "4. Plotting FID..."
python ${SCRIPT_DIR}/plot_fid_suite.py

echo "5. Plotting ID Loss..."
python ${SCRIPT_DIR}/plot_id_loss_suite.py

echo ""
echo "================================================="
echo "   All processes finished successfully!"
echo "   Graphs are saved in the './Suite' directory."
echo "================================================="