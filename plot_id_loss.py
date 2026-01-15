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
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# 2. プロット対象のファイル名
TARGET_FILENAME = "post_process_id_loss.json"

# 3. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [-8, -7, -6, -5, -4, -3, -2] 

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# ★ 5. 比較したいパラメータ設定のリスト (exp, gamma)
# ここに比較したい組み合わせを定義します。
# label: 凡例に表示するパラメータ名, linestyle: この設定の線のスタイル
COMPARISON_CONFIGS = [
    {"exp": 2.0, "gamma": 0.3, "label": "Exp=2.0, Gam=0.3", "linestyle": "-"},  # 実線
    # 必要に応じて他のパラメータを追加してください
    {"exp": 1.0, "gamma": 0.0, "label": "Exp=1.0, Gam=0.0", "linestyle": "--"}
]

# 6. プロットしたいJSON内のキー (手法) のリスト
TARGET_KEYS = [
    "3_P2_Random",               # Randomは特別扱い（共通ベースライン）
    
    #"3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
]

# 7. 凡例の表示名マッピング (手法名)
METHOD_LABELS = {
    "1_JSCC_Init":              "JSCC",
    "2_Phase1_Recon":           "Phase1",
    "3_P2_Random":              "Random Baseline", # 名前変更
    "3_P2_perturbation_raw_Unc":"Perturb(Unc)",
    "3_P2_perturbation_raw_Sem":"Perturb(Sem)",
}

# 8. 手法ごとの基本色・マーカー設定
# 線種(linestyle)は COMPARISON_CONFIGS で上書きされますが、Randomはここで指定したスタイルが使われます
STYLE_CONFIG = {
    "1_JSCC_Init":              {"color": "black", "marker": "x", "linestyle": ":"}, 
    "2_Phase1_Recon":           {"color": "blue",  "marker": "o", "linestyle": "-"}, 
    "3_P2_Random":              {"color": "black", "marker": "d", "linestyle": "-."}, # Random用スタイル

    "3_P2_perturbation_raw_Unc":{"color": "red",   "marker": "s"}, # 赤
    "3_P2_perturbation_raw_Sem":{"color": "green", "marker": "D"}, # 緑
}

# 9. プロット対象の指標
METRICS = ["id_loss"]

# ==========================================
# 処理ロジック
# ==========================================

def load_id_loss_data_recursive():
    """
    ディレクトリを再帰的に探索し、データを読み込む。
    パスから SNR, Rate, Exp, Gamma を抽出する。
    """
    search_pattern = os.path.join(ROOT_DIR, "**", TARGET_FILENAME)
    print(f"Target Dataset: {DATASET}")
    print(f"Root Directory: {ROOT_DIR}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    # データ構造: 
    #   main_data[config_idx][rate][snr][method]  <- 比較対象用
    #   random_data[rate][snr]                    <- Random用 (パラメータ問わず確保)
    main_data = {}
    random_data = {}

    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    # フォルダ名からパラメータを抜く正規表現
    # 例: .../Retrans_rate_0.1_Comparison_both_exp2.0_gam0.3_zeta0.3_seed22/...
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

        # フィルタリング (Rate, SNR)
        if TARGET_SNRS and snr not in TARGET_SNRS: continue
        if TARGET_RATES and rate not in TARGET_RATES: continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # --- Random (共通) の確保 ---
            # 特定のディレクトリ (例: exp2.0, gam0.3) にしかない場合でも、見つかれば保持する
            # ※ もし複数のフォルダにRandomがあっても、上書きして「どれか1つ」を使います
            if "3_P2_Random" in content:
                if rate not in random_data: random_data[rate] = {}
                # まだ登録されていない、あるいは指定のパラメータフォルダ(exp2.0 gam0.3)なら優先して保存などのロジックも可能だが
                # ここでは単純に見つかったものを採用する
                random_data[rate][snr] = content["3_P2_Random"]

            # --- 比較対象データの確保 ---
            # CONFIGSにあるパラメータ設定と一致するか確認
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
                    # Randomは別途確保したのでここではスキップしても良いが、念のため除外
                    if key_method == "3_P2_Random":
                        continue
                    if key_method in content:
                        main_data[matched_config_idx][rate][snr][key_method] = content[key_method]
                        
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return main_data, random_data

def plot_id_metrics(main_data, random_data):
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
        
        num_metrics = len(METRICS)
        cols = 1
        rows = 1
        figsize = (10, 8) if num_metrics == 1 else (14, 6 * ((num_metrics + 1) // 2))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if hasattr(axes, "flatten"): axes = axes.flatten()
        else: axes = [axes]

        for idx, metric in enumerate(METRICS):
            if idx >= len(axes): break
            ax = axes[idx]
            has_data = False
            
            # 1. Random (Baseline) のプロット
            # Randomはパラメータに関係なく1本だけ引く
            if rate in random_data and random_data[rate]:
                r_snrs = sorted(random_data[rate].keys())
                x_rnd = []
                y_rnd = []
                for snr in r_snrs:
                    if metric in random_data[rate][snr]:
                        val = random_data[rate][snr][metric]
                        x_rnd.append(snr)
                        y_rnd.append(val)
                
                if x_rnd:
                    has_data = True
                    style = STYLE_CONFIG.get("3_P2_Random", {"color": "gray", "linestyle": "-."})
                    label = METHOD_LABELS.get("3_P2_Random", "Random")
                    ax.plot(x_rnd, y_rnd, label=label, **style)

            # 2. 比較対象 (Unc, Sem など) のプロット
            # Config (Exp/Gam) ごとにループ
            for conf_idx, config in enumerate(COMPARISON_CONFIGS):
                if conf_idx not in main_data or rate not in main_data[conf_idx]:
                    continue
                
                current_data_group = main_data[conf_idx][rate]
                snr_list = sorted(current_data_group.keys())
                
                # 手法ごとにループ (Random以外)
                for method in TARGET_KEYS:
                    if method == "3_P2_Random": continue # 既にプロット済み

                    x_vals = []
                    y_vals = []
                    
                    for snr in snr_list:
                        if method in current_data_group[snr] and metric in current_data_group[snr][method]:
                            val = current_data_group[snr][method][metric]
                            x_vals.append(snr)
                            y_vals.append(val)
                    
                    if x_vals:
                        has_data = True
                        
                        # スタイル決定
                        base_style = STYLE_CONFIG.get(method, {})
                        color = base_style.get("color", "black")
                        marker = base_style.get("marker", "o")
                        
                        # 線種はConfig依存
                        linestyle = config.get("linestyle", "-")
                        
                        # 凡例: Method [Params]
                        method_name = METHOD_LABELS.get(method, method)
                        config_label = config.get("label", "")
                        full_label = f"{method_name} [{config_label}]"
                        
                        ax.plot(x_vals, y_vals, 
                                label=full_label, 
                                color=color, 
                                linestyle=linestyle, 
                                marker=marker)

            # グラフ装飾
            ax.set_title(metric.replace("_", " ").title(), fontsize=14, fontweight='bold')
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if has_data:
                ax.legend(loc='best', fontsize=9, framealpha=0.9)
            else:
                ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        save_name = f'id_metrics_comparison_{DATASET}_rate_{rate}.png'
        plt.savefig(save_name, dpi=300)
        print(f"グラフを保存しました: {save_name}")
        plt.close(fig)

if __name__ == "__main__":
    m_data, r_data = load_id_loss_data_recursive()
    plot_id_metrics(m_data, r_data)