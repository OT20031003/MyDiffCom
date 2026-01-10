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

# 2. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
# 例: [-7, -4, -1, 2] など。メインスクリプトの出力に合わせて調整してください。
TARGET_SNRS = [-8, -7, -6, -5, -4] 
TARGET_SNRS = []

# 3. プロットしたい再送率 (Retrans_rate) のリスト
# None または [] の場合は、見つかった全てのレートについて個別にプロットを作成します
TARGET_RATES = [0.1]

# --- [追加機能] 拡張パラメータでのフィルタリング設定 ---
# 指定した exp (expansion_factor) や gam (gamma) のファイルのみを抽出します。
# None または [] (空リスト) の場合は、フィルタリングせず全て対象とします。

TARGET_EXPS = [2.0]     # 例: [2.0] または None
TARGET_GAMS = [0.3]     # 例: [0.3, 0.7] または None

# -----------------------------------------------------

# 4. プロットしたい手法のリスト (JSONのキーに完全一致させる)
# main_diffcom_retransmission.py が出力するキーに対応
TARGET_METHODS = [
    #"jscc_init", 
    #"phase1_recon", 
    "random",
    
    # Temporal (時間的分散) 系
    # "temporal_raw_Unc",
    # "temporal_raw_Sem",
    
    # Perturbation (摂動分散) 系
    "perturbation_raw_Unc",
    "perturbation_raw_Sem",
]

# 5. 凡例の表示名マッピング
METHOD_LABELS = {
    "jscc_init":            "JSCC (Initial)",
    "phase1_recon":         "Phase 1 Recon",
    "random":               "Random Baseline",
    
    "temporal_raw_Unc":     "Temporal (Unc)",
    "temporal_raw_Sem":     "Temporal (Sem)",
    
    "perturbation_raw_Unc": "Perturbation (Unc)",
    "perturbation_raw_Sem": "Perturbation (Sem)",
}

# 6. スタイル設定 (色、線種、マーカー)
STYLE_CONFIG = {
    "jscc_init":            {"color": "black", "linestyle": ":",  "marker": "x"}, 
    "phase1_recon":         {"color": "blue",  "linestyle": "-",  "marker": "o"}, 
    "random":               {"color": "gray",  "linestyle": "-.", "marker": "d"}, 

    "temporal_raw_Unc":     {"color": "green", "linestyle": "-",  "marker": "^"}, 
    "temporal_raw_Sem":     {"color": "green", "linestyle": "--", "marker": "v"}, 
    
    "perturbation_raw_Unc": {"color": "red",   "linestyle": "-",  "marker": "s"}, 
    "perturbation_raw_Sem": {"color": "red",   "linestyle": "--", "marker": "D"}, 
}

# 7. プロット対象の指標
# METRICS = ["psnr", "lpips", "dists", "msssim", "fid"]
METRICS = ["psnr", "lpips", "dists",  "fid"]
# ==========================================
# 処理ロジック
# ==========================================

def load_summary_data_recursive():
    """
    ディレクトリを再帰的に探索し、JSONデータを読み込む
    ファイル名形式: SNR-7_Retrans_rate_0.1_Comparison_both_exp2.0_gam0.3_zeta0.3_seed22.json
    """
    # ROOT_DIR 以下を探索
    search_pattern = os.path.join(ROOT_DIR, "**", "SNR*_Retrans_*.json")
    print(f"Target Dataset: {DATASET}")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Search Pattern: {search_pattern}")
    
    # フィルタ設定の表示
    if TARGET_EXPS: print(f"Filtering by EXPS: {TARGET_EXPS}")
    if TARGET_GAMS: print(f"Filtering by GAMS: {TARGET_GAMS}")
    
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files.")

    # 構造: { rate: { snr: { method: metrics } } }
    data_store = {}

    # 正規表現コンパイル
    regex_snr = re.compile(r"SNR(-?\d+(?:\.\d+)?)")
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")
    
    # 追加パラメータ用の正規表現
    # _exp2.0 や _gam0.3 のような形式を想定
    regex_exp = re.compile(r"_exp(\d+(?:\.\d+)?)")
    regex_gam = re.compile(r"_gam(\d+(?:\.\d+)?)")

    for fpath in files:
        filename = os.path.basename(fpath)
        
        # 1. SNRの抽出
        match_snr = regex_snr.search(filename)
        if not match_snr:
            # print(f"Skipping (No SNR found): {filename}")
            continue
        snr = float(match_snr.group(1))

        # 2. Retrans_rateの抽出
        match_rate = regex_rate.search(filename)
        if not match_rate:
            # print(f"Skipping (No Rate found): {filename}")
            continue
        rate = float(match_rate.group(1))

        # 3. 追加パラメータ(exp, gam)の抽出とフィルタリング
        # ファイル名にパラメータが含まれていない場合は、フィルタリング条件が指定されていなければパスさせる
        
        # --- Exp (Expansion Factor) ---
        match_exp = regex_exp.search(filename)
        current_exp = float(match_exp.group(1)) if match_exp else None
        
        # TARGET_EXPSが指定されている場合、一致しなければスキップ
        if TARGET_EXPS:
            # ファイルにexpが書いてない、または値がリストに含まれない場合はスキップ
            if current_exp is None or current_exp not in TARGET_EXPS:
                # print(f"Skipping {filename} (Exp mismatch: {current_exp})")
                continue
        
        # --- Gam (Gamma) ---
        match_gam = regex_gam.search(filename)
        current_gam = float(match_gam.group(1)) if match_gam else None
        
        # TARGET_GAMSが指定されている場合、一致しなければスキップ
        if TARGET_GAMS:
            if current_gam is None or current_gam not in TARGET_GAMS:
                # print(f"Skipping {filename} (Gam mismatch: {current_gam})")
                continue

        # 基本フィルタリング
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

            # 対象メソッドのデータを格納
            for method in TARGET_METHODS:
                if method in summary:
                    data_store[rate][snr][method] = summary[method]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_custom_metrics(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パスやフィルタ設定(SNR, Rate, Exp, Gam)を確認してください。")
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
        
        print(f"  SNRs found: {snr_list}")

        # レイアウト設定
        num_metrics = len(METRICS)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_metrics > 1 else [axes]
        
        # タイトル生成
        title_str = f"Comparison ({DATASET} - Rate: {rate})"
        # フィルタリング条件をタイトルに追記 (単一指定の場合などわかりやすく)
        cond_strs = []
        if TARGET_EXPS and len(TARGET_EXPS) == 1:
            cond_strs.append(f"Exp:{TARGET_EXPS[0]}")
        if TARGET_GAMS and len(TARGET_GAMS) == 1:
            cond_strs.append(f"Gam:{TARGET_GAMS[0]}")
        
        if cond_strs:
            title_str += " [" + ", ".join(cond_strs) + "]"

        fig.suptitle(title_str, fontsize=16)

        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            has_data = False
            
            for method in TARGET_METHODS:
                x_vals = []
                y_vals = []
                for snr in snr_list:
                    if method in current_data[snr] and metric in current_data[snr][method]:
                        val = current_data[snr][method][metric]
                        
                        # 数値以外（エラー文字列など）は除外
                        if isinstance(val, (int, float)):
                            x_vals.append(snr)
                            y_vals.append(val)
                
                if x_vals:
                    has_data = True
                    style = STYLE_CONFIG.get(method, {})
                    label = METHOD_LABELS.get(method, method)
                    ax.plot(x_vals, y_vals, label=label, **style)

            ax.set_title(metric.upper(), fontsize=14, fontweight='bold')
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.6)
            if has_data:
                ax.legend(loc='best', fontsize=9)

        # 余った領域を非表示
        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 

        # ファイル名生成
        save_name = f'snr_metrics_{DATASET}_rate_{rate}'
        if TARGET_EXPS and len(TARGET_EXPS) == 1:
            save_name += f'_exp{TARGET_EXPS[0]}'
        if TARGET_GAMS and len(TARGET_GAMS) == 1:
            save_name += f'_gam{TARGET_GAMS[0]}'
        save_name += '.png'
        
        plt.savefig(save_name, dpi=300)
        print(f"グラフを保存しました: {save_name}")
        
        plt.close(fig) 

if __name__ == "__main__":
    data = load_summary_data_recursive()
    plot_custom_metrics(data)