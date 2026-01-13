import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 1. データセットとディレクトリ設定
DATASET = "imagenet"
BASE_DIR = "results_retrans_comparison"
METHOD_PATH = "diffcom/djscc_2"
ROOT_DIR = os.path.join(BASE_DIR, DATASET, METHOD_PATH)

# 2. プロット対象のファイル名
TARGET_FILENAME = "post_process_dists.json"

# 3. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [-8,-7,-6,-5, -4, -3, -2]
#TARGET_SNRS = []

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# --- 拡張パラメータでのフィルタリング設定 ---
TARGET_EXPS = [5.0]     # 例: [5.0] または None
TARGET_GAMS = [0.7]     # 例: [0.7] または None

# 5. プロットしたいJSON内のキー (手法) のリスト
TARGET_KEYS = [
    # "1_JSCC_Init",
    # "2_Phase1_Recon",
    "3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
    "3_P2_Random"
]

# 6. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":               "JSCC (Initial)",
    "2_Phase1_Recon":            "Phase 1 Recon",
    "3_P2_perturbation_raw_Unc": "Perturbation (Unc)",
    "3_P2_perturbation_raw_Sem": "Perturbation (Sem)",
    "3_P2_Random":               "Random Baseline",
}

# 7. スタイル設定
STYLE_CONFIG = {
    "1_JSCC_Init":               {"color": "black", "linestyle": ":",  "marker": "x"},
    "2_Phase1_Recon":            {"color": "blue",  "linestyle": "-",  "marker": "o"},
    "3_P2_perturbation_raw_Unc": {"color": "red",   "linestyle": "-",  "marker": "s"},
    "3_P2_perturbation_raw_Sem": {"color": "red",   "linestyle": "--", "marker": "D"},
    "3_P2_Random":               {"color": "gray",  "linestyle": "-.", "marker": "d"},
}

# ==========================================
# 処理ロジック
# ==========================================

def load_data_recursive():
    """
    ディレクトリを再帰的に探索しデータを集計する
    """
    search_pattern = os.path.join(ROOT_DIR, "**", TARGET_FILENAME)
    print(f"Target Dataset: {DATASET}")
    print(f"Search Pattern: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    data_store = {}
    
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")
    regex_exp = re.compile(r"_exp(\d+(?:\.\d+)?)")
    regex_gam = re.compile(r"_gam(\d+(?:\.\d+)?)")

    for fpath in files:
        dirname = os.path.dirname(fpath)
        folder_name = os.path.basename(dirname)
        
        # パラメータ抽出
        match_snr = regex_snr.search(dirname)
        match_rate = regex_rate.search(dirname)
        
        if not match_snr or not match_rate:
            continue
            
        snr = float(match_snr.group(1))
        rate = float(match_rate.group(1))
        
        # 拡張パラメータ抽出
        match_exp = regex_exp.search(folder_name)
        current_exp = float(match_exp.group(1)) if match_exp else None
        
        match_gam = regex_gam.search(folder_name)
        current_gam = float(match_gam.group(1)) if match_gam else None

        # フィルタリング
        if TARGET_EXPS and (current_exp is None or current_exp not in TARGET_EXPS):
            continue
        if TARGET_GAMS and (current_gam is None or current_gam not in TARGET_GAMS):
            continue
        if TARGET_SNRS and snr not in TARGET_SNRS:
            continue
        if TARGET_RATES and rate not in TARGET_RATES:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            if rate not in data_store:
                data_store[rate] = {}
            if snr not in data_store[rate]:
                data_store[rate][snr] = {}

            for key_method in TARGET_KEYS:
                if key_method in content:
                    data_store[rate][snr][key_method] = content[key_method]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_dists(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。")
        return

    rates = sorted(data_store.keys())
    
    for rate in rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        current_data = data_store[rate]
        snr_list = sorted(current_data.keys())
        
        if not snr_list:
            continue
        
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        has_data = False
        
        for method in TARGET_KEYS:
            x_vals = []
            y_vals = []
            
            for snr in snr_list:
                if method in current_data[snr]:
                    val = current_data[snr][method]
                    if val is not None:
                        x_vals.append(snr)
                        y_vals.append(val)
            
            if x_vals:
                has_data = True
                style = STYLE_CONFIG.get(method, {})
                label = METHOD_LABELS.get(method, method)
                ax.plot(x_vals, y_vals, label=label, **style)

        # グラフ装飾
        ax.set_title("DISTS vs SNR", fontsize=14, fontweight='bold')
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("DISTS Score (Lower is Better)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if has_data:
            ax.legend(loc='best', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # ファイル名生成
        save_name = f'dists_vs_snr_{DATASET}_rate_{rate}'
        if TARGET_EXPS and len(TARGET_EXPS) == 1:
            save_name += f'_exp{TARGET_EXPS[0]}'
        if TARGET_GAMS and len(TARGET_GAMS) == 1:
            save_name += f'_gam{TARGET_GAMS[0]}'
        save_name += '.png'

        plt.savefig(save_name, dpi=300)
        print(f"Saved: {save_name}")
        plt.close()

if __name__ == "__main__":
    data = load_data_recursive()
    plot_dists(data)