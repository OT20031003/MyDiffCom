import os
import glob
import json
import re
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア
# ==========================================

# 1. データセットとディレクトリ設定
DATASET = "ffhq_demo" # 必要に応じて "ffhq_demo" などに変更してください
BASE_DIR = "results_retrans_comparison"
# 検索範囲を広げるため、ROOTはデータセット階層にします
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# 2. プロット対象のファイル名
TARGET_FILENAME = "post_process_lpips.json"

# 3. プロットしたいSNRのリスト
TARGET_SNRS = [-8, -6, -7, -5, -4, -3, -2]

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# ★ 5. 比較したいパラメータ設定のリスト (exp, gamma)
# ここに比較したい組み合わせを定義します。
COMPARISON_CONFIGS = [
    {"exp": 2.0, "gamma": 0.3, "label": "Exp=2.0, Gam=0.3", "linestyle": "-"},
    # 例: 比較対象を追加する場合
    {"exp": 1.0, "gamma": 0.0, "label": "Default", "linestyle": "--"},
]

# 6. プロットしたいJSON内のキー (手法) のリスト
TARGET_KEYS = [
    "3_P2_Random",               # Randomは特別扱い（共通ベースライン）
    
    #"3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
]

# 7. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":               "JSCC (Initial)",
    "2_Phase1_Recon":            "Phase 1 Recon",
    "3_P2_Random":               "Random Baseline",
    
    "3_P2_perturbation_raw_Unc": "Perturb (Unc)",
    "3_P2_perturbation_raw_Sem": "Perturb (Sem)",
}

# 8. スタイル設定
# 線種(linestyle)はCOMPARISON_CONFIGSで上書きされますが、Randomはここの設定が優先されます
STYLE_CONFIG = {
    "1_JSCC_Init":               {"color": "black", "linestyle": ":",  "marker": "x"},
    "2_Phase1_Recon":            {"color": "blue",  "linestyle": "-",  "marker": "o"},
    "3_P2_Random":               {"color": "gray",  "linestyle": "-.", "marker": "d"},
    
    "3_P2_perturbation_raw_Unc": {"color": "red",   "marker": "s"},
    "3_P2_perturbation_raw_Sem": {"color": "green", "marker": "D"},
}

# ==========================================
# 処理ロジック
# ==========================================

def load_data_recursive():
    """
    ディレクトリを再帰的に探索し、データを分類して読み込む
    """
    search_pattern = os.path.join(ROOT_DIR, "**", TARGET_FILENAME)
    print(f"Target Dataset: {DATASET}")
    print(f"Search Pattern: {search_pattern}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    # データ構造
    main_data = {}   # 比較対象用: main_data[config_idx][rate][snr][method]
    random_data = {} # Random用: random_data[rate][snr]
    
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    # パラメータ抽出用正規表現
    regex_params = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)_Comparison_.*_exp(\d+(?:\.\d+)?)_gam(\d+(?:\.\d+)?)_")

    for fpath in files:
        dirname = os.path.dirname(fpath)
        
        # 1. SNRの抽出
        match_snr = regex_snr.search(fpath)
        if not match_snr: continue
        snr = float(match_snr.group(1))

        # 2. Rate, Exp, Gamma の抽出
        match_params = regex_params.search(fpath)
        if not match_params: continue
        
        rate = float(match_params.group(1))
        exp_val = float(match_params.group(2))
        gam_val = float(match_params.group(3))

        # フィルタリング
        if TARGET_SNRS and snr not in TARGET_SNRS: continue
        if TARGET_RATES and rate not in TARGET_RATES: continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # --- Random (共通) の確保 ---
            if "3_P2_Random" in content:
                if rate not in random_data: random_data[rate] = {}
                random_data[rate][snr] = content["3_P2_Random"]

            # --- 比較対象データの確保 ---
            matched_config_idx = -1
            for idx, conf in enumerate(COMPARISON_CONFIGS):
                if abs(conf["exp"] - exp_val) < 1e-5 and abs(conf["gamma"] - gam_val) < 1e-5:
                    matched_config_idx = idx
                    break
            
            if matched_config_idx != -1:
                if matched_config_idx not in main_data:
                    main_data[matched_config_idx] = {}
                if rate not in main_data[matched_config_idx]:
                    main_data[matched_config_idx][rate] = {}
                if snr not in main_data[matched_config_idx][rate]:
                    main_data[matched_config_idx][rate][snr] = {}

                for key_method in TARGET_KEYS:
                    if key_method == "3_P2_Random": continue # Randomは別途確保済み
                    
                    if key_method in content:
                        main_data[matched_config_idx][rate][snr][key_method] = content[key_method]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return main_data, random_data

def plot_lpips(main_data, random_data):
    if not main_data and not random_data:
        print("表示対象のデータが見つかりませんでした。")
        return

    # Rateの集合を取得
    available_rates = set()
    for conf_idx in main_data:
        available_rates.update(main_data[conf_idx].keys())
    available_rates.update(random_data.keys())
    
    sorted_rates = sorted(list(available_rates))
    
    for rate in sorted_rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        has_data = False
        
        # 1. Random (Baseline) のプロット
        if rate in random_data and random_data[rate]:
            r_snrs = sorted(random_data[rate].keys())
            x_rnd = []
            y_rnd = []
            for snr in r_snrs:
                val = random_data[rate][snr]
                if val is not None:
                    x_rnd.append(snr)
                    y_rnd.append(val)
            
            if x_rnd:
                has_data = True
                style = STYLE_CONFIG.get("3_P2_Random", {"color": "gray", "linestyle": "-."})
                label = METHOD_LABELS.get("3_P2_Random", "Random")
                ax.plot(x_rnd, y_rnd, label=label, **style)

        # 2. 比較対象 (Unc, Sem など) のプロット
        for conf_idx, config in enumerate(COMPARISON_CONFIGS):
            if conf_idx not in main_data or rate not in main_data[conf_idx]:
                continue
            
            current_data_group = main_data[conf_idx][rate]
            snr_list = sorted(current_data_group.keys())
            
            for method in TARGET_KEYS:
                if method == "3_P2_Random": continue

                x_vals = []
                y_vals = []
                
                for snr in snr_list:
                    if method in current_data_group[snr]:
                        val = current_data_group[snr][method]
                        if val is not None:
                            x_vals.append(snr)
                            y_vals.append(val)
                
                if x_vals:
                    has_data = True
                    
                    # スタイル決定
                    base_style = STYLE_CONFIG.get(method, {})
                    color = base_style.get("color", "black")
                    marker = base_style.get("marker", "o")
                    
                    linestyle = config.get("linestyle", "-")
                    
                    method_name = METHOD_LABELS.get(method, method)
                    config_label = config.get("label", "")
                    full_label = f"{method_name} [{config_label}]"
                    
                    ax.plot(x_vals, y_vals, 
                            label=full_label, 
                            color=color, 
                            linestyle=linestyle, 
                            marker=marker)

        # グラフ装飾
        ax.set_title("LPIPS vs SNR", fontsize=14, fontweight='bold')
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("LPIPS Score (Lower is Better)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if has_data:
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # ファイル名生成
        save_name = f'lpips_comparison_{DATASET}_rate_{rate}.png'
        plt.savefig(save_name, dpi=300)
        print(f"Saved: {save_name}")
        plt.close()

if __name__ == "__main__":
    m_data, r_data = load_data_recursive()
    plot_lpips(m_data, r_data)