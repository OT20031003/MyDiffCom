import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 1. 探索を開始するルートディレクトリ
ROOT_DIR = "results_retrans_comparison"

# 2. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [0, 2, 4]
TARGET_SNRS = None
# 3. プロットしたい手法のリスト (JSONのキーに完全一致させる)
# 提供されたJSONに含まれる新しいキーに対応させました
TARGET_METHODS = [
    #"jscc_init", 
    #"phase1_recon", 
    
    # Temporal (時間的分散) 系
    "temporal_raw_Unc",
    "temporal_raw_Sem",
    
    # Perturbation (摂動分散) 系
    "perturbation_raw_Unc",
    "perturbation_raw_Sem",
    
    # ランダムベースライン
    "random"
]

# 4. 凡例の表示名マッピング
METHOD_LABELS = {
    "jscc_init":            "JSCC (Initial)",
    "phase1_recon":         "Phase 1 Recon",
    
    "temporal_raw_Unc":     "Temporal (Unc)",
    "temporal_raw_Sem":     "Temporal (Sem)",
    
    "perturbation_raw_Unc": "Perturbation (Unc)",
    "perturbation_raw_Sem": "Perturbation (Sem)",
    
    "random":               "Random Baseline",
}

# 5. スタイル設定 (色とマーカーでグループ化)
# 視認性を高めるため、系列ごとに色を統一し、実線/破線やマーカーで区別します
STYLE_CONFIG = {
    "jscc_init":      {"color": "black", "linestyle": ":",  "marker": "x"}, # 初期JSCC: 黒点線
    "phase1_recon":   {"color": "blue",  "linestyle": "-",  "marker": "o"}, # Phase1: 青実線
    
    # Temporal系: 緑色
    "temporal_raw_Unc":     {"color": "green", "linestyle": "-",  "marker": "^"}, # Unc: 実線・三角
    "temporal_raw_Sem":     {"color": "green", "linestyle": "--", "marker": "v"}, # Sem: 破線・逆三角
    
    # Perturbation系: 赤色
    "perturbation_raw_Unc": {"color": "red",   "linestyle": "-",  "marker": "s"}, # Unc: 実線・四角
    "perturbation_raw_Sem": {"color": "red",   "linestyle": "--", "marker": "D"}, # Sem: 破線・ダイヤ
    
    # Random: 灰色
    "random":         {"color": "gray",  "linestyle": "-.", "marker": "d"}, # Random: 一点鎖線
}

# 6. プロット対象の指標 (JSONに含まれるキー)
METRICS = ["psnr", "lpips", "dists", "msssim"]

# ==========================================
# 処理ロジック
# ==========================================

def load_summary_data_recursive():
    # /**/ を使用して、 awgn_02dB などの深い階層にあるファイルを再帰的に探索します
    search_pattern = os.path.join(ROOT_DIR, "**", "SNR*_Comparison_*.json")
    print(f"Searching for files: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    data_store = {}

    for fpath in files:
        filename = os.path.basename(fpath)
        # ファイル名 (SNR2_... や SNR0_...) から数値を抽出
        # SNRの後に続く数字（小数・負数対応）を取得
        match = re.search(r"SNR(-?\d+\.?\d*)", filename)
        if not match:
            continue
        
        snr = float(match.group(1))

        # SNRフィルタリング (指定がある場合)
        if TARGET_SNRS and snr not in TARGET_SNRS:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            summary = content.get("summary", {})
            if snr not in data_store:
                data_store[snr] = {}

            for method in TARGET_METHODS:
                if method in summary:
                    data_store[snr][method] = summary[method]
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_custom_metrics(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パスやファイル名を確認してください。")
        return

    snr_list = sorted(data_store.keys())
    
    # 指標の数に合わせてグラフのレイアウトを調整 (3x2など)
    num_metrics = len(METRICS)
    cols = 2
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for method in TARGET_METHODS:
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
        ax.legend(loc='best', fontsize=9)

    # 余ったグラフ領域を非表示にする
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_name = 'snr_metrics_comparison_updated.png'
    plt.savefig(save_name, dpi=300)
    print(f"グラフを保存しました: {save_name}")
    plt.show()

if __name__ == "__main__":
    data = load_summary_data_recursive()
    plot_custom_metrics(data)