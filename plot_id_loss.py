import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 1. データセットとディレクトリ設定
# 例: results_retrans_comparison\ffhq_demo\...
DATASET = "ffhq_demo" 
# DATASET = "imagenet"

BASE_DIR = "results_retrans_comparison"
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# 2. プロット対象のファイル名
TARGET_FILENAME = "post_process_id_loss.json"

# 3. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
# 例: [-6, -4, -2, 0] など
TARGET_SNRS = [-8, -7, -6, -5, -4] 

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# 5. プロットしたいJSON内のキー (手法) のリスト
# JSONのキーに完全一致させる必要があります
TARGET_KEYS = [
    #"1_JSCC_Init",
    #"2_Phase1_Recon",
    "3_P2_Random",
    
    "3_P2_temporal_raw_Unc",
    "3_P2_temporal_raw_Sem",
    
    "3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
]

# 6. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":              "JSCC (Initial)",
    "2_Phase1_Recon":           "Phase 1 Recon",
    "3_P2_Random":              "Random Baseline",
    
    "3_P2_temporal_raw_Unc":    "Temporal (Unc)",
    "3_P2_temporal_raw_Sem":    "Temporal (Sem)",
    
    "3_P2_perturbation_raw_Unc":"Perturbation (Unc)",
    "3_P2_perturbation_raw_Sem":"Perturbation (Sem)",
}

# 7. スタイル設定 (色、線種、マーカー)
STYLE_CONFIG = {
    "1_JSCC_Init":              {"color": "black", "linestyle": ":",  "marker": "x"}, 
    "2_Phase1_Recon":           {"color": "blue",  "linestyle": "-",  "marker": "o"}, 
    "3_P2_Random":              {"color": "gray",  "linestyle": "-.", "marker": "d"}, 

    "3_P2_temporal_raw_Unc":    {"color": "green", "linestyle": "-",  "marker": "^"}, 
    "3_P2_temporal_raw_Sem":    {"color": "green", "linestyle": "--", "marker": "v"}, 
    
    "3_P2_perturbation_raw_Unc":{"color": "red",   "linestyle": "-",  "marker": "s"}, 
    "3_P2_perturbation_raw_Sem":{"color": "red",   "linestyle": "--", "marker": "D"}, 
}

# 8. プロット対象の指標 (JSON内のキー)
METRICS = ["id_loss", "id_similarity"]

# ==========================================
# 処理ロジック
# ==========================================

def load_id_loss_data_recursive():
    """
    ディレクトリを再帰的に探索し、TARGET_FILENAME (post_process_id_loss.json) を読み込む
    パスから SNR (awgn_-6dB) と Retrans_rate (Retrans_rate_0.1) を抽出する
    """
    # ROOT_DIR 以下を探索
    search_pattern = os.path.join(ROOT_DIR, "**", TARGET_FILENAME)
    print(f"Target Dataset: {DATASET}")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Search Pattern: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    # 構造: { rate: { snr: { method: { metric: value } } } }
    data_store = {}

    # 正規表現: 
    # SNR: awgn_-6dB, awgn_10dB などから数値を抽出
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    # Rate: Retrans_rate_0.1 などから数値を抽出
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")

    for fpath in files:
        # ディレクトリパス全体を取得
        dirname = os.path.dirname(fpath)
        
        # 1. SNRの抽出 (ディレクトリ名に含まれると想定)
        match_snr = regex_snr.search(dirname)
        if not match_snr:
            # 念のため親ディレクトリも探すなどのロジックが必要ならここに追加
            # 今回はawgn_-6dBがパスに含まれる前提
            continue
        snr = float(match_snr.group(1))

        # 2. Retrans_rateの抽出
        match_rate = regex_rate.search(dirname)
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
            
            # 階層構造の初期化
            if rate not in data_store:
                data_store[rate] = {}
            if snr not in data_store[rate]:
                data_store[rate][snr] = {}

            # 対象メソッドのデータを格納
            for key_method in TARGET_KEYS:
                if key_method in content:
                    data_store[rate][snr][key_method] = content[key_method]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_id_metrics(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パスやファイル名、DATASET設定を確認してください。")
        return

    # 見つかったレートごとにグラフを作成
    rates = sorted(data_store.keys())
    
    for rate in rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        current_data = data_store[rate]
        snr_list = sorted(current_data.keys())
        
        if not snr_list:
            print(f"Rate {rate} に有効なSNRデータがありません。スキップします。")
            continue
        
        print(f"  SNRs found: {snr_list}")

        # レイアウト設定
        num_metrics = len(METRICS)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        
        # 図のサイズ調整
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f"Identity Preservation Metrics ({DATASET} - Rate: {rate})", fontsize=16)

        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            has_data = False
            
            for method in TARGET_KEYS:
                x_vals = []
                y_vals = []
                
                for snr in snr_list:
                    # methodキーが存在し、かつ metricキーが存在するか確認
                    if method in current_data[snr] and metric in current_data[snr][method]:
                        val = current_data[snr][method][metric]
                        
                        # 数値型のみプロット
                        if isinstance(val, (int, float)):
                            x_vals.append(snr)
                            y_vals.append(val)
                
                if x_vals:
                    has_data = True
                    style = STYLE_CONFIG.get(method, {})
                    label = METHOD_LABELS.get(method, method)
                    ax.plot(x_vals, y_vals, label=label, **style)

            # グラフ装飾
            ax.set_title(metric.replace("_", " ").title(), fontsize=14, fontweight='bold')
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if has_data:
                ax.legend(loc='best', fontsize=9)
            else:
                ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        # 余った領域を非表示
        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90) 

        # 保存
        save_name = f'id_metrics_{DATASET}_rate_{rate}.png'
        plt.savefig(save_name, dpi=300)
        print(f"グラフを保存しました: {save_name}")
        
        plt.close(fig) 

if __name__ == "__main__":
    data = load_id_loss_data_recursive()
    plot_id_metrics(data)