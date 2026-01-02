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

# 2. 対象の再送率 (パスに含まれる文字列 "Retrans_rate_X.X" に一致させる)
#    None の場合は再送率でフィルタリングしません
TARGET_RETRANS_RATE = 0.1

# 3. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
#    例: [-4, -2, 0, 2, 4, 10]
TARGET_SNRS = None 

# 4. プロットしたい手法のリスト (JSONのキーに完全一致させる)
TARGET_METHODS = [
    #"1_JSCC_Init", 
    #"2_Phase1_Recon", 
    
    # Temporal (時間的分散) 系
    "3_P2_temporal_raw_Unc",
    "3_P2_temporal_raw_Sem",
    
    # Perturbation (摂動分散) 系
    "3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
    
    # ランダムベースライン
    "3_P2_Random"
]

# 5. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":               "JSCC (Initial)",
    "2_Phase1_Recon":            "Phase 1 Recon",
    "3_P2_temporal_raw_Unc":     "Temporal (Unc)",
    "3_P2_temporal_raw_Sem":     "Temporal (Sem)",
    "3_P2_perturbation_raw_Unc": "Perturbation (Unc)",
    "3_P2_perturbation_raw_Sem": "Perturbation (Sem)",
    "3_P2_Random":               "Random Baseline",
}

# 6. スタイル設定 (色とマーカーでグループ化)
# FID用コードと同じ配色を使用
STYLE_CONFIG = {
    "1_JSCC_Init":               {"color": "black", "linestyle": ":",  "marker": "x"}, # 初期JSCC
    "2_Phase1_Recon":            {"color": "blue",  "linestyle": "-",  "marker": "o"}, # Phase1
    "3_P2_temporal_raw_Unc":     {"color": "green", "linestyle": "-",  "marker": "^"}, # Temporal Unc
    "3_P2_temporal_raw_Sem":     {"color": "green", "linestyle": "--", "marker": "v"}, # Temporal Sem
    "3_P2_perturbation_raw_Unc": {"color": "red",   "linestyle": "-",  "marker": "s"}, # Perturbation Unc
    "3_P2_perturbation_raw_Sem": {"color": "red",   "linestyle": "--", "marker": "D"}, # Perturbation Sem
    "3_P2_Random":               {"color": "gray",  "linestyle": "-.", "marker": "d"}, # Random
}

# 7. プロットする値のキー (JSON内の構造に基づく)
#    "id_loss" または "id_similarity" を指定可能
METRIC_KEY = "id_loss"
Y_LABEL = "ID Loss (Lower is Better)"

# ==========================================
# 処理ロジック
# ==========================================

def load_id_loss_data_recursive():
    """
    指定ディレクトリ以下の post_process_id_loss.json を探索し、
    パスからSNRを抽出してデータを集計する
    """
    # post_process_id_loss.json を再帰的に探索
    search_pattern = os.path.join(ROOT_DIR, "**", "post_process_id_loss.json")
    print(f"Searching for files: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    data_store = {}
    
    # SNR抽出用正規表現: awgn_-4dB, awgn_00dB, awgn_2.5dB などに対応
    snr_regex = re.compile(r"awgn_(-?\d+\.?\d*)dB")
    
    # 再送率抽出用正規表現: Retrans_rate_0.1 など
    retrans_regex = re.compile(r"Retrans_rate_(\d+\.?\d*)")

    for fpath in files:
        dir_path = os.path.dirname(fpath)
        
        # 1. SNRの抽出
        match_snr = snr_regex.search(dir_path)
        if not match_snr:
            continue
        snr = float(match_snr.group(1))

        # 2. 再送率のフィルタリング
        if TARGET_RETRANS_RATE is not None:
            match_retrans = retrans_regex.search(dir_path)
            if not match_retrans:
                continue 
            retrans_val = float(match_retrans.group(1))
            if abs(retrans_val - TARGET_RETRANS_RATE) > 1e-5:
                continue

        # 3. SNR指定フィルタリング
        if TARGET_SNRS is not None and snr not in TARGET_SNRS:
            continue

        # JSON読み込み
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # データ構造: data_store[snr][method] = metric_value
            if snr not in data_store:
                data_store[snr] = {}

            for method in TARGET_METHODS:
                # post_process_id_loss.json の構造は
                # { "MethodName": { "id_loss": X, "id_similarity": Y, ... }, ... }
                if method in content:
                    method_data = content[method]
                    if isinstance(method_data, dict) and METRIC_KEY in method_data:
                        data_store[snr][method] = method_data[METRIC_KEY]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_id_loss(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パス設定やファイル名を確認してください。")
        return

    snr_list = sorted(data_store.keys())
    print(f"Plotting for SNRs: {snr_list}")
    
    # グラフ作成
    plt.figure(figsize=(10, 7))
    
    for method in TARGET_METHODS:
        x_vals = []
        y_vals = []
        
        for snr in snr_list:
            if method in data_store[snr]:
                val = data_store[snr][method]
                if val is not None:
                    x_vals.append(snr)
                    y_vals.append(val)
        
        if x_vals:
            style = STYLE_CONFIG.get(method, {})
            label = METHOD_LABELS.get(method, method)
            plt.plot(x_vals, y_vals, label=label, **style)

    plt.title(f"{METRIC_KEY} vs SNR (Retrans Rate: {TARGET_RETRANS_RATE})", fontsize=14, fontweight='bold')
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    save_name = f'{METRIC_KEY}_vs_snr_comparison.png'
    plt.savefig(save_name, dpi=300)
    print(f"グラフを保存しました: {save_name}")
    plt.show()

if __name__ == "__main__":
    data = load_id_loss_data_recursive()
    plot_id_loss(data)