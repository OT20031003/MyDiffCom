import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# フォント設定 (Font Configuration)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'

# ==========================================
# 設定エリア (Configuration)
# ==========================================

# ★バージョン選択 ("v1", "v2", "v3")
# v1: "results_retrans_comparison" 内の全データをプロット
# v2: "results_retrans_comparison_v2" + v1 (Uncertainty & Proposed)
# v3: "results_retrans_comparison_v3" + v1 (Uncertainty & Proposed)
VERSION = "v2"

# 共通設定
DATASET = "ffhq_demo"
SNR_LABELS = [-7, -6, -5, -4, -3, -2] # プロットしたいSNR範囲

# データのルートディレクトリ定義
ROOT_V1 = os.path.join("results_retrans_comparison_v1", DATASET)
ROOT_V2 = os.path.join("results_retrans_comparison_v2", DATASET)
ROOT_V3 = os.path.join("results_retrans_comparison_v3", DATASET)

# フォルダ名のパターン指定 (正規表現)
# v1/v3は同じ命名規則、v2はprefix付き
PATTERN_V1 = r"Retrans_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"
PATTERN_V2 = r"Retrans_v2_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"
PATTERN_V3 = r"Retrans_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"

# ==========================================
# 手法とデータソースの定義
# ==========================================

# 全ての手法の表示名とスタイル定義
STYLE_CONFIG = {
    # Phase 1 & Common
    # "1_JSCC_Init":         {"label": "Deep JSCC",    "color": "black",  "linestyle": ":",  "marker": "x"},
    "2_Phase1_Recon":      {"label": "DiffCom",     "color": "blue",   "linestyle": "--", "marker": "o"},
    #"1_Random_Baseline":   {"label": "Random Baseline",   "color": "gray",   "linestyle": "-.", "marker": "v"},
    
    # Existing Baselines
    "2_Uncertainty_Only":  {"label": "Uncertainty Only",  "color": "orange", "linestyle": "-",  "marker": "s"},
    "3_Importance_Only":   {"label": "Importance Only",   "color": "purple", "linestyle": "-",  "marker": "^"},
    #"4_Edge_Baseline":     {"label": "Edge Baseline",     "color": "brown",  "linestyle": "-",  "marker": "d"},
    
    # Proposed
    "5_Proposed_Method":   {"label": "Proposed Method",   "color": "red",    "linestyle": "-",  "marker": "*", "linewidth": 2.5},
    
    # New Hybrid Baselines (v3)
    "6_Importance_Random": {"label": "Imp + Random",      "color": "cyan",   "linestyle": "-",  "marker": "o"},
    "7_Edge_Random":       {"label": "Edge + Random",     "color": "lime",   "linestyle": "-",  "marker": "h"},
    "8_Uncertainty_Random":{"label": "Unc + Random",      "color": "magenta","linestyle": "-",  "marker": "D"}, # Added
}

def get_target_methods_and_sources(version):
    """
    バージョンに応じた取得対象メソッドと、その取得先ディレクトリを定義する
    Returns:
        targets: { method_name: source_root_path }
        folder_patterns: { source_root_path: regex_pattern }
    """
    targets = {}
    
    # ルートディレクトリと検索パターンの紐づけ
    patterns = {
        ROOT_V1: PATTERN_V1,
        ROOT_V2: PATTERN_V2,
        ROOT_V3: PATTERN_V3
    }

    if version == "v1":
        # v1: 全て v1 フォルダから取得
        methods = [
            "1_JSCC_Init", "2_Phase1_Recon", "1_Random_Baseline",
            "2_Uncertainty_Only", "3_Importance_Only", 
            "4_Edge_Baseline", "5_Proposed_Method"
        ]
        for m in methods:
            targets[m] = ROOT_V1

    elif version == "v2":
        # v2: v2データ + v1のUncertainty/Proposed
        v2_methods = [
            "1_JSCC_Init", "2_Phase1_Recon", "1_Random_Baseline",
            "3_Importance_Only", "4_Edge_Baseline"
        ]
        for m in v2_methods:
            targets[m] = ROOT_V2
            
        v1_imports = ["2_Uncertainty_Only", "5_Proposed_Method"]
        for m in v1_imports:
            targets[m] = ROOT_V1

    elif version == "v3":
        # v3: v3データ(Hybrid Random等) + v1のUncertainty/Proposed
        
        # v3フォルダから取得するもの
        v3_methods = [
            #"1_JSCC_Init", "2_Phase1_Recon", "1_Random_Baseline", # ベースラインもv3で再計算されている前提
            "6_Importance_Random", "7_Edge_Random", "8_Uncertainty_Random"
        ]
        for m in v3_methods:
            targets[m] = ROOT_V3
        
        # v1フォルダから取得するもの（比較対象として重要）
        v1_imports = ["2_Uncertainty_Only", "5_Proposed_Method"]
        for m in v1_imports:
            targets[m] = ROOT_V1

    return targets, patterns

# ==========================================
# データ読み込み処理
# ==========================================

def load_psnr_data(targets, patterns):
    """
    指定されたターゲット設定に基づいてデータを収集する
    Output: data[method][snr] = value
    """
    data_store = {m: {} for m in targets.keys()}
    
    # 探索対象のルートディレクトリごとにファイルをスキャン
    roots_to_scan = set(targets.values())
    
    file_cache = {} # path -> (snr, json_content)

    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")

    for root in roots_to_scan:
        if not os.path.exists(root):
            print(f"Warning: Directory not found -> {root}")
            continue

        print(f"Scanning directory: {root} ...")
        pattern_str = patterns[root]
        regex_folder = re.compile(pattern_str)
        
        # 再帰的に post_process_psnr.json を探す
        search_path = os.path.join(root, "**", "post_process_psnr.json")
        found_files = glob.glob(search_path, recursive=True)
        
        for fpath in found_files:
            # フォルダ名チェック
            dirname = os.path.basename(os.path.dirname(fpath))
            if not regex_folder.search(dirname):
                continue
                
            # SNRチェック
            match_snr = regex_snr.search(fpath)
            if not match_snr:
                continue
            snr = float(match_snr.group(1))
            
            if snr not in SNR_LABELS:
                continue

            # JSON読み込み
            try:
                with open(fpath, 'r') as f:
                    content = json.load(f)
                    file_cache[fpath] = (snr, content)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    # データ格納
    print("Aggregating data...")
    for method, source_root in targets.items():
        for fpath, (snr, content) in file_cache.items():
            # そのファイルのパスが、指定されたsource_rootから始まっているか確認
            # (正規化して比較することでパス区切りの違いを吸収)
            if os.path.abspath(fpath).startswith(os.path.abspath(source_root)):
                # JSON内のキー名が、指定されたmethod名を含むかチェック
                # NOTE: calc_psnr_suite.py は "3_" などを削除して保存しているため
                # ここでは正確なキーマッチングを行う
                
                # キー名の揺らぎ吸収 (prefix削除後の名前で保存されているため)
                # STYLE_CONFIGのキー (例: "1_Random_Baseline") と JSONキー (例: "1_Random_Baseline")
                if method in content:
                    data_store[method][snr] = content[method]
    
    return data_store

# ==========================================
# プロット処理
# ==========================================

def plot_graph(data_store):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # STYLE_CONFIGの定義順にプロット
    sorted_methods = [k for k in STYLE_CONFIG.keys() if k in data_store]
    
    has_plot = False
    
    for method in sorted_methods:
        snr_dict = data_store[method]
        if not snr_dict:
            # print(f"No data for: {method}")
            continue
            
        sorted_snrs = sorted(snr_dict.keys())
        x_vals = sorted_snrs
        y_vals = [snr_dict[s] for s in sorted_snrs]
        
        style = STYLE_CONFIG[method]
        
        ax.plot(x_vals, y_vals, 
                label=style["label"], 
                color=style["color"], 
                linestyle=style["linestyle"], 
                marker=style["marker"],
                linewidth=style.get("linewidth", 1.5),
                markersize=8)
        has_plot = True

    # グラフ装飾
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(SNR_LABELS)
    
    if has_plot:
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9, shadow=True)
    else:
        ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    
    # ディレクトリ作成
    save_dir = "Suite"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{save_dir}/psnr_comparison_suite_{VERSION}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nGraph saved to: {filename}")
    # plt.show() # 必要に応じてコメントアウト解除

def print_statistics(data_store):
    """
    集計されたPSNRデータをターミナルに表形式で出力する
    """
    print("\n" + "="*80)
    print(f" PSNR Numerical Results (Mode: {VERSION})")
    print("="*80)
    
    # ヘッダー作成 (SNR)
    # Method 名のスペース(25文字) + 各SNR(10文字)
    header = f"{'Method':<25}" + "".join([f"{snr:>10}dB" for snr in SNR_LABELS])
    print(header)
    print("-" * len(header))
    
    # STYLE_CONFIG の順序に従って表示
    sorted_methods = [k for k in STYLE_CONFIG.keys() if k in data_store]
    
    for method in sorted_methods:
        snr_dict = data_store[method]
        
        # データが空の場合はスキップ
        if not snr_dict:
            continue
            
        label_name = STYLE_CONFIG[method]["label"]
        row_str = f"{label_name:<25}"
        
        for snr in SNR_LABELS:
            val = snr_dict.get(snr, None)
            if val is not None:
                # PSNRは小数点以下2～4桁表示が見やすい
                row_str += f"{val:>10.4f}"
            else:
                row_str += f"{'N/A':>10}"
        
        print(row_str)
    
    print("="*80 + "\n")

# ==========================================
# メイン実行
# ==========================================

if __name__ == "__main__":
    print(f"=== Plotting PSNR Suite (Mode: {VERSION}) ===")
    
    # 1. 対象の決定
    targets, patterns = get_target_methods_and_sources(VERSION)
    
    print("\nTarget Methods & Sources:")
    for m, src in targets.items():
        print(f"  - {m:25s} <- {src}")
    print("-" * 50)

    # 2. データ読み込み
    data = load_psnr_data(targets, patterns)
    
    # 3. データの表形式出力
    print_statistics(data)
    
    # 4. プロット
    plot_graph(data)