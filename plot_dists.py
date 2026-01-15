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
# 検索範囲を広げるため、ROOTはデータセット階層にします
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# 2. プロット対象のファイル名
TARGET_FILENAME = "post_process_dists.json"

# 3. プロットしたいSNRのリスト
TARGET_SNRS = [-8, -7, -6, -5, -4, -3, -2]

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# ★ 5. 比較したいパラメータ設定のリスト (exp, gamma)
# 【修正箇所】各設定に color と marker を追加し、ラベル表記を統一しました
COMPARISON_CONFIGS = [
    # 設定1: 緑, ダイヤ, 実線
    {
        "exp": 2.0, 
        "gamma": 0.3, 
        "label": r"$\eta=2.0, \gamma=0.3$", 
        "linestyle": "-",
        "color": "green",   # ★追加
        "marker": "D"       # ★追加
    },
    
    # 設定2: 赤, 四角, 破線
    {
        "exp": 1.0, 
        "gamma": 0.0, 
        "label": r"$\eta=1.0, \gamma=0.0$", # "Uncertainty" から統一のため変更
        "linestyle": "--",
        "color": "red",     # ★追加
        "marker": "s"       # ★追加
    },
    # 設定C: 比較2 (紫・三角・一点鎖線)
    {
        "exp": 10.0, 
        "gamma": 1.0, 
        "label": r"$\eta=10.0, \gamma=1.0$", 
        "linestyle": "-.", 
        "color": "purple", 
        "marker": "^"
    }
]

# 6. JSON内のキー分類

# (A) ベースライン (パラメータ比較の対象外で、常に表示する手法)
BASELINE_KEYS = [
    "1_JSCC_Init",
    "2_Phase1_Recon",
    "3_P2_Random"
]

# (B) 比較対象 (パラメータ設定ごとに線を引く手法)
COMPARISON_KEYS = [
    #"3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
]

# 7. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":               "JSCC (Initial)",
    "2_Phase1_Recon":            "Phase 1 Recon (DiffCom)",
    "3_P2_Random":               "Random Baseline",
    
    "3_P2_perturbation_raw_Unc": "Perturb (Unc)",
    "3_P2_perturbation_raw_Sem": "Perturb (Sem)",
}

# 8. スタイル設定
# Baseline手法や、Comparison手法の基本色・マーカー
STYLE_CONFIG = {
    "1_JSCC_Init":               {"color": "black", "linestyle": ":",  "marker": "x"},
    "2_Phase1_Recon":            {"color": "blue",  "linestyle": "-",  "marker": "o"},
    "3_P2_Random":               {"color": "gray",  "linestyle": "-.", "marker": "d"},
    
    # ここでの設定は COMPARISON_CONFIGS に指定がない場合のフォールバックとして使われます
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
    main_data = {}      # 比較対象用: main_data[config_idx][rate][snr][method]
    baseline_data = {}  # ベースライン用(JSCC, Phase1, Random): baseline_data[rate][snr][method]
    
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
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
            
            # --- (A) ベースラインデータの確保 ---
            # どのパラメータフォルダにあっても、見つかれば保存（上書き）
            for b_key in BASELINE_KEYS:
                if b_key in content:
                    if rate not in baseline_data: baseline_data[rate] = {}
                    if snr not in baseline_data[rate]: baseline_data[rate][snr] = {}
                    
                    baseline_data[rate][snr][b_key] = content[b_key]

            # --- (B) 比較対象データの確保 ---
            # CONFIGSと一致するものだけ保存
            matched_config_idx = -1
            for idx, conf in enumerate(COMPARISON_CONFIGS):
                # 浮動小数点の誤差対策
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

                for c_key in COMPARISON_KEYS:
                    if c_key in content:
                        main_data[matched_config_idx][rate][snr][c_key] = content[c_key]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return main_data, baseline_data

def plot_dists(main_data, baseline_data):
    if not main_data and not baseline_data:
        print("表示対象のデータが見つかりませんでした。")
        return

    # Rateの集合を取得
    available_rates = set()
    for conf_idx in main_data:
        available_rates.update(main_data[conf_idx].keys())
    available_rates.update(baseline_data.keys())
    
    sorted_rates = sorted(list(available_rates))
    
    for rate in sorted_rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        has_data = False
        
        # 1. ベースライン (JSCC, Phase1, Random) のプロット
        # これらはパラメータ設定ごとに分割せず、単一の線として描画
        if rate in baseline_data:
            snr_list = sorted(baseline_data[rate].keys())
            
            for method in BASELINE_KEYS:
                x_vals = []
                y_vals = []
                for snr in snr_list:
                    if method in baseline_data[rate][snr]:
                        val = baseline_data[rate][snr][method]
                        if val is not None:
                            x_vals.append(snr)
                            y_vals.append(val)
                
                if x_vals:
                    has_data = True
                    style = STYLE_CONFIG.get(method, {})
                    label = METHOD_LABELS.get(method, method)
                    ax.plot(x_vals, y_vals, label=label, **style)

        # 2. 比較対象 (Perturbation Unc/Sem) のプロット
        # Config (Exp/Gam) ごとにループ
        for conf_idx, config in enumerate(COMPARISON_CONFIGS):
            if conf_idx not in main_data or rate not in main_data[conf_idx]:
                continue
            
            current_data_group = main_data[conf_idx][rate]
            snr_list = sorted(current_data_group.keys())
            
            for method in COMPARISON_KEYS:
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
                    
                    # 【修正箇所】スタイル決定ロジック
                    # 1. まず手法のデフォルトスタイルを取得
                    base_style = STYLE_CONFIG.get(method, {})
                    
                    # 2. Configに指定があればそれを優先、なければデフォルトを使う
                    #    これにより、COMPARISON_CONFIGS で指定した色・マーカーが反映されます
                    color = config.get("color", base_style.get("color", "black"))
                    marker = config.get("marker", base_style.get("marker", "o"))
                    linestyle = config.get("linestyle", base_style.get("linestyle", "-"))
                    
                    method_name = METHOD_LABELS.get(method, method)
                    config_label = config.get("label", "")
                    
                    # 凡例ラベルの構築: MethodName [config]
                    full_label = f"{method_name} [{config_label}]"
                    
                    ax.plot(x_vals, y_vals, 
                            label=full_label, 
                            color=color, 
                            linestyle=linestyle, 
                            marker=marker)

        # グラフ装飾
        ax.set_title("DISTS vs SNR", fontsize=14, fontweight='bold')
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("DISTS Score (Lower is Better)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if has_data:
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # ファイル名生成
        save_name = f'dists_comparison_{DATASET}_rate_{rate}.png'
        plt.savefig(save_name, dpi=300)
        print(f"Saved: {save_name}")
        plt.close()

if __name__ == "__main__":
    m_data, b_data = load_data_recursive()
    plot_dists(m_data, b_data)