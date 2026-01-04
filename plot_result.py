import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 【追加】 データセットの指定
# ここを "imagenet" や "ffhq_demo" に書き換えてください
DATASET = "imagenet" 
DATASET = "ffhq_demo"

# 1. 探索を開始するルートディレクトリ
# results_retrans_comparison の下の DATASET フォルダをルートとします
BASE_DIR = "results_retrans_comparison"
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# 2. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [-6, -4, -2,0]
#TARGET_SNRS = None

# 【追加】 プロットしたい再送率 (Retrans_rate) のリスト
# None または [] の場合は、見つかった全てのレートについて個別にプロットを作成します
TARGET_RATES = [0.1]
# TARGET_RATES = None

# 3. プロットしたい手法のリスト (JSONのキーに完全一致させる)
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

# 5. スタイル設定
STYLE_CONFIG = {
    "jscc_init":      {"color": "black", "linestyle": ":",  "marker": "x"}, 
    "phase1_recon":   {"color": "blue",  "linestyle": "-",  "marker": "o"}, 
    
    "temporal_raw_Unc":     {"color": "green", "linestyle": "-",  "marker": "^"}, 
    "temporal_raw_Sem":     {"color": "green", "linestyle": "--", "marker": "v"}, 
    
    "perturbation_raw_Unc": {"color": "red",   "linestyle": "-",  "marker": "s"}, 
    "perturbation_raw_Sem": {"color": "red",   "linestyle": "--", "marker": "D"}, 
    
    "random":         {"color": "gray",  "linestyle": "-.", "marker": "d"}, 
}

# 6. プロット対象の指標
METRICS = ["psnr", "lpips", "dists", "msssim"]

# ==========================================
# 処理ロジック
# ==========================================

def load_summary_data_recursive():
    # ROOT_DIR (results_retrans_comparison/DATASET) 以下を探索
    search_pattern = os.path.join(ROOT_DIR, "**", "SNR*_Comparison_*.json")
    print(f"Target Dataset: {DATASET}")
    print(f"Searching for files: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    # 構造: { rate: { snr: { method: metrics } } }
    data_store = {}

    for fpath in files:
        filename = os.path.basename(fpath)
        
        # 1. SNRの抽出
        match_snr = re.search(r"SNR(-?\d+\.?\d*)", filename)
        if not match_snr:
            continue
        snr = float(match_snr.group(1))

        # 2. Retrans_rateの抽出
        match_rate = re.search(r"Retrans_rate_(\d+\.?\d*)", filename)
        if not match_rate:
            continue
        rate = float(match_rate.group(1))

        # フィルタリング
        if TARGET_SNRS and snr not in TARGET_SNRS:
            continue
        if TARGET_RATES and rate not in TARGET_RATES:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            summary = content.get("summary", {})
            
            # 階層構造の初期化
            if rate not in data_store:
                data_store[rate] = {}
            if snr not in data_store[rate]:
                data_store[rate][snr] = {}

            for method in TARGET_METHODS:
                if method in summary:
                    data_store[rate][snr][method] = summary[method]
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_custom_metrics(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パスやファイル名、DATASET設定を確認してください。")
        return

    # 見つかったレートごとにグラフを作成
    rates = sorted(data_store.keys())
    
    for rate in rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        # このレートに対応するデータ (snr -> method -> metrics)
        current_data = data_store[rate]
        snr_list = sorted(current_data.keys())
        
        if not snr_list:
            print(f"Rate {rate} に有効なSNRデータがありません。スキップします。")
            continue

        # レイアウト設定
        num_metrics = len(METRICS)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        # タイトルにレートとデータセットを表示
        fig.suptitle(f"Comparison Metrics ({DATASET} - Rate: {rate})", fontsize=16)

        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            for method in TARGET_METHODS:
                x_vals = []
                y_vals = []
                for snr in snr_list:
                    if method in current_data[snr] and metric in current_data[snr][method]:
                        val = current_data[snr][method][metric]
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

        # 余った領域を非表示
        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 

        # ファイル名にデータセットとレートを含めて保存
        save_name = f'snr_metrics_{DATASET}_rate_{rate}.png'
        plt.savefig(save_name, dpi=300)
        print(f"グラフを保存しました: {save_name}")
        
        plt.close(fig) 

if __name__ == "__main__":
    data = load_summary_data_recursive()
    plot_custom_metrics(data)