import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 1. データセットとディレクトリ設定
DATASET = "ffhq_demo"
BASE_DIR = "results_retrans_comparison"
METHOD_PATH = "diffcom/djscc_2" # FID計算結果があるサブディレクトリ
ROOT_DIR = os.path.join(BASE_DIR, DATASET, METHOD_PATH)

# 2. プロット対象のファイル名
TARGET_FILENAME = "post_process_fid.json"

# 3. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [-6, -4, -2, 0]
TARGET_SNRS = [-8, -6, -7, -5, -4,-3, -2]

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# --- 拡張パラメータでのフィルタリング設定 ---
# 指定した exp (expansion_factor) や gam (gamma) のファイルのみを抽出します。
# None または [] (空リスト) の場合は、フィルタリングせず全て対象とします。

TARGET_EXPS = [2.0]     # 例: [2.0] または None
TARGET_GAMS = [0.3]     # 例: [0.3, 0.7] または None

# -----------------------------------------------------

# 5. プロットしたいJSON内のキー (手法) のリスト
# calc_fid.py の出力JSONに含まれるキーを指定してください
TARGET_KEYS = [
    # "1_JSCC_Init",
    # "2_Phase1_Recon",
    
    # Perturbation (摂動分散) 系
    "3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
    
    # Temporal (時間的分散) 系
    # "3_P2_temporal_raw_Unc",
    # "3_P2_temporal_raw_Sem",
    
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

# 7. スタイル設定 (色、線種、マーカー)
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

def load_fid_data_recursive():
    """
    ディレクトリを再帰的に探索し、TARGET_FILENAME を読み込む
    パスから SNR, Rate, Exp, Gam を抽出してフィルタリングする
    """
    search_pattern = os.path.join(ROOT_DIR, "**", TARGET_FILENAME)
    print(f"Target Dataset: {DATASET}")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Search Pattern: {search_pattern}")
    
    # フィルタ設定の表示
    if TARGET_EXPS: print(f"Filtering by EXPS: {TARGET_EXPS}")
    if TARGET_GAMS: print(f"Filtering by GAMS: {TARGET_GAMS}")

    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    data_store = {}
    
    # 正規表現
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")
    regex_exp = re.compile(r"_exp(\d+(?:\.\d+)?)")
    regex_gam = re.compile(r"_gam(\d+(?:\.\d+)?)")

    for fpath in files:
        dirname = os.path.dirname(fpath)
        folder_name = os.path.basename(dirname)
        
        # 1. SNRの抽出
        match_snr = regex_snr.search(dirname)
        if not match_snr:
            continue
        snr = float(match_snr.group(1))

        # 2. Retrans_rateの抽出
        match_rate = regex_rate.search(dirname)
        if not match_rate:
            # フォルダ構造によっては上位ディレクトリにある場合も考慮が必要だが、
            # 現状は同じフォルダ名文字列に含まれる想定
            continue
        rate = float(match_rate.group(1))
        
        # 3. 追加パラメータ(exp, gam)の抽出とフィルタリング
        
        # --- Exp (Expansion Factor) ---
        match_exp = regex_exp.search(folder_name)
        current_exp = float(match_exp.group(1)) if match_exp else None
        
        if TARGET_EXPS:
            if current_exp is None or current_exp not in TARGET_EXPS:
                continue
        
        # --- Gam (Gamma) ---
        match_gam = regex_gam.search(folder_name)
        current_gam = float(match_gam.group(1)) if match_gam else None
        
        if TARGET_GAMS:
            if current_gam is None or current_gam not in TARGET_GAMS:
                continue

        # SNR, Rate フィルタリング
        if TARGET_SNRS and snr not in TARGET_SNRS:
            continue
        if TARGET_RATES and rate not in TARGET_RATES:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # データ構造: data_store[rate][snr][method] = score
            if rate not in data_store:
                data_store[rate] = {}
            if snr not in data_store[rate]:
                data_store[rate][snr] = {}

            for key_method in TARGET_KEYS:
                if key_method in content:
                    data_store[rate][snr][key_method] = content[key_method]
            
            # デバッグ用
            # print(f"Loaded: SNR={snr}, Rate={rate}, Exp={current_exp}, Gam={current_gam}")
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_fid(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パスやファイル名、DATASET設定、フィルタ設定を確認してください。")
        return

    rates = sorted(data_store.keys())
    
    for rate in rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        current_data = data_store[rate]
        snr_list = sorted(current_data.keys())
        
        if not snr_list:
            print(f"Rate {rate} に有効なSNRデータがありません。スキップします。")
            continue
        
        print(f"  SNRs found: {snr_list}")

        # 図の生成
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        
        has_data = False
        
        for method in TARGET_KEYS:
            x_vals = []
            y_vals = []
            
            for snr in snr_list:
                if method in current_data[snr]:
                    val = current_data[snr][method]
                    if isinstance(val, (int, float)):
                        x_vals.append(snr)
                        y_vals.append(val)
            
            if x_vals:
                has_data = True
                style = STYLE_CONFIG.get(method, {})
                label = METHOD_LABELS.get(method, method)
                ax.plot(x_vals, y_vals, label=label, **style)

        # グラフ装飾
        # タイトルはシンプルに、もしくは無しにする設定
        ax.set_title("Frechet Inception Distance (FID)", fontsize=14, fontweight='bold')
        
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("FID Score (Lower is Better)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if has_data:
            ax.legend(loc='best', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # ファイル名生成
        save_name = f'fid_vs_snr_{DATASET}_rate_{rate}'
        if TARGET_EXPS and len(TARGET_EXPS) == 1:
            save_name += f'_exp{TARGET_EXPS[0]}'
        if TARGET_GAMS and len(TARGET_GAMS) == 1:
            save_name += f'_gam{TARGET_GAMS[0]}'
        save_name += '.png'

        plt.savefig(save_name, dpi=300)
        print(f"グラフを保存しました: {save_name}")
        
        plt.close()

if __name__ == "__main__":
    data = load_fid_data_recursive()
    plot_fid(data)