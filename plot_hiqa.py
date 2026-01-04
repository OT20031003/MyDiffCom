import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 1. データセットの指定
# ここを "imagenet" や "ffhq_demo" に書き換えてください
DATASET = "ffhq_demo"
DATASET = "imagenet"

# 2. 探索を開始するルートディレクトリ
BASE_DIR = "results_retrans_comparison"
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# 3. 対象の再送率 (パスに含まれる文字列 "Retrans_rate_X.X" に一致させる)
TARGET_RETRANS_RATE = 0.1

# 4. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [-6, -4, -2, 0]
#TARGET_SNRS = None 

# 5. プロットしたい手法のリスト (JSONのキーに完全一致させる)
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

# 6. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":               "JSCC (Initial)",
    "2_Phase1_Recon":            "Phase 1 Recon",
    
    "3_P2_temporal_raw_Unc":     "Temporal (Unc)",
    "3_P2_temporal_raw_Sem":     "Temporal (Sem)",
    
    "3_P2_perturbation_raw_Unc": "Perturbation (Unc)",
    "3_P2_perturbation_raw_Sem": "Perturbation (Sem)",
    
    "3_P2_Random":               "Random Baseline",
}

# 7. スタイル設定
STYLE_CONFIG = {
    "1_JSCC_Init":               {"color": "black", "linestyle": ":",  "marker": "x"}, 
    "2_Phase1_Recon":            {"color": "blue",  "linestyle": "-",  "marker": "o"}, 
    
    "3_P2_temporal_raw_Unc":     {"color": "green", "linestyle": "-",  "marker": "^"}, 
    "3_P2_temporal_raw_Sem":     {"color": "green", "linestyle": "--", "marker": "v"}, 
    
    "3_P2_perturbation_raw_Unc": {"color": "red",   "linestyle": "-",  "marker": "s"}, 
    "3_P2_perturbation_raw_Sem": {"color": "red",   "linestyle": "--", "marker": "D"}, 
    
    "3_P2_Random":               {"color": "gray",  "linestyle": "-.", "marker": "d"}, 
}

# ==========================================
# 処理ロジック
# ==========================================

def load_hiqa_data_recursive():
    """
    指定ディレクトリ以下の post_process_hiqa.json を探索し、データを集計する
    """
    # post_process_hiqa.json を再帰的に探索
    search_pattern = os.path.join(ROOT_DIR, "**", "post_process_hiqa.json")
    print(f"Target Dataset: {DATASET}")
    print(f"Searching for files: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    data_store = {}
    
    # 正規表現
    snr_regex = re.compile(r"awgn_(-?\d+\.?\d*)dB")
    retrans_regex = re.compile(r"Retrans_rate_(\d+\.?\d*)")

    for fpath in files:
        dir_path = os.path.dirname(fpath)
        
        # SNR抽出
        match_snr = snr_regex.search(dir_path)
        if not match_snr: continue
        snr = float(match_snr.group(1))

        # 再送率フィルタリング
        if TARGET_RETRANS_RATE is not None:
            match_retrans = retrans_regex.search(dir_path)
            if not match_retrans: continue
            if abs(float(match_retrans.group(1)) - TARGET_RETRANS_RATE) > 1e-5:
                continue

        # SNRフィルタリング
        if TARGET_SNRS is not None and snr not in TARGET_SNRS:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            if snr not in data_store:
                data_store[snr] = {}

            for method in TARGET_METHODS:
                if method in content:
                    val_obj = content[method]
                    
                    # calc_hiqa.py の出力形式 { "hiqa": score, ... } に対応
                    if isinstance(val_obj, dict) and "hiqa" in val_obj:
                        score = val_obj["hiqa"]
                    elif isinstance(val_obj, (float, int)):
                        score = val_obj
                    else:
                        continue # 形式が不明な場合はスキップ

                    data_store[snr][method] = score
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_hiqa(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。")
        return

    snr_list = sorted(data_store.keys())
    print(f"Plotting for SNRs: {snr_list}")
    
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

    plt.title(f"HyperIQA vs SNR ({DATASET} - Retrans Rate: {TARGET_RETRANS_RATE})", fontsize=14, fontweight='bold')
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("HyperIQA Score (Higher is Better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    save_name = f'hiqa_vs_snr_{DATASET}_rate_{TARGET_RETRANS_RATE}.png'
    plt.savefig(save_name, dpi=300)
    print(f"グラフを保存しました: {save_name}")

if __name__ == "__main__":
    data = load_hiqa_data_recursive()
    plot_hiqa(data)