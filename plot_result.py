import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 設定エリア
# ==========================================

# 1. 探索を開始するルートディレクトリ
# 例: "results_retrans_comparison" や "." (カレント) を指定
ROOT_DIR = "results_retrans_comparison"

# 2. プロットしたいSNRのリスト (None または [] なら全て表示)
TARGET_SNRS = [-4.0, -2.0, 0.0] 

# 3. プロットしたい手法のリスト
TARGET_METHODS = [
    "jscc_init", 
    "phase1_recon", 
    "temporal_smooth_jscc", 
    "temporal_smooth",
    "random_jscc"
]

# 4. 凡例の表示名マッピング
METHOD_LABELS = {
    "jscc_init": "JSCC (Initial)",
    "phase1_recon": "Phase 1 Recon",
    "temporal_smooth_jscc": "temporal_smooth_jscc (JSCC)",
    "temporal_smooth": "temporal_smooth (Smooth)",
    "random_jscc": "Random Baseline",
}

# 5. スタイル設定
STYLE_CONFIG = {
    "jscc_init": {"color": "black", "linestyle": "--", "marker": "x"},
    "phase1_recon": {"color": "blue", "linestyle": "-", "marker": "o"},
    "temporal_smooth_jscc": {"color": "red", "linestyle": "-", "marker": "s", "linewidth": 2},
    "temporal_smooth": {"color": "green", "linestyle": "--", "marker": "^"},
    "random_jscc": {"color": "gray", "linestyle": ":", "marker": "d"},
}

# 6. プロット対象の指標
METRICS = ["psnr", "lpips", "dists", "msssim"]

# ==========================================
# 処理ロジック
# ==========================================

def load_summary_data_recursive():
    # /**/*.json と recursive=True を使うことで、サブフォルダ内をすべて探索します
    search_pattern = os.path.join(ROOT_DIR, "**", "SNR*_Comparison_*.json")
    print(f"Searching in: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    data_store = {}

    for fpath in files:
        # ファイル名からSNRを抽出 (例: SNR-2.0_... -> -2.0)
        filename = os.path.basename(fpath)
        match = re.search(r"SNR(-?\d+\.?\d*)", filename)
        if not match:
            continue
        
        snr = float(match.group(1))

        # SNRフィルタリング
        if TARGET_SNRS and snr not in TARGET_SNRS:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            summary = content.get("summary", {})
            if snr not in data_store:
                data_store[snr] = {}

            for method, metrics in summary.items():
                # 手法フィルタリング
                if TARGET_METHODS and method not in TARGET_METHODS:
                    continue
                
                if isinstance(metrics, dict):
                    data_store[snr][method] = metrics
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_custom_metrics(data_store):
    if not data_store:
        print("指定された条件（SNR/手法/パス）に一致するデータが見つかりませんでした。")
        return

    snr_list = sorted(data_store.keys())
    
    # 手法のリストアップ
    available_methods = set()
    for snr in snr_list:
        available_methods.update(data_store[snr].keys())
    
    plot_methods = TARGET_METHODS if TARGET_METHODS else sorted(list(available_methods))
    plot_methods = [m for m in plot_methods if m in available_methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for method in plot_methods:
            x_vals = []
            y_vals = []
            for snr in snr_list:
                if method in data_store[snr] and metric in data_store[snr][method]:
                    val = data_store[snr][method][metric]
                    if val is not None:
                        x_vals.append(snr)
                        y_vals.append(val)
            
            if x_vals:
                style = STYLE_CONFIG.get(method, {})
                label = METHOD_LABELS.get(method, method)
                ax.plot(x_vals, y_vals, label=label, **style)

        ax.set_title(metric.upper(), fontsize=14, fontweight='bold')
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('recursive_metrics_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # データのロードとプロット
    data = load_summary_data_recursive()
    plot_custom_metrics(data)