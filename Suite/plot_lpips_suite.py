import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 設定エリア (Configuration)
# ==========================================

# ★バージョン選択 ("v1" or "v2")
# v1: "results_retrans_comparison" 内の全データをプロット
# v2: "results_retrans_comparison_v2" のデータ + v1 の "Uncertainty" & "Proposed" をプロット
VERSION = "v2"

# 共通設定
DATASET = "ffhq_demo"
SNR_LABELS = [-8, -7, -6, -5, -4, -3, -2] # プロットしたいSNR範囲

# データのルートディレクトリ定義
ROOT_V1 = os.path.join("results_retrans_comparison", DATASET)
ROOT_V2 = os.path.join("results_retrans_comparison_v2", DATASET)

# フォルダ名のパターン指定 (正規表現)
PATTERN_V1 = r"Retrans_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"
PATTERN_V2 = r"Retrans_v2_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"

# 対象メトリクス
TARGET_METRIC_KEY = "lpips" # JSON内のキーはファイル名生成ロジックによりトップレベルにある場合が多いが、関数内で処理

# ==========================================
# 手法とデータソースの定義
# ==========================================

# 全ての手法の表示名とスタイル定義
STYLE_CONFIG = {
    # "1_JSCC_Init":       {"label": "JSCC (Initial)",  "color": "black",  "linestyle": ":",  "marker": "x"},
    # "2_Phase1_Recon":    {"label": "Phase 1 Recon",   "color": "blue",   "linestyle": "--", "marker": "o"},
    "1_Random_Baseline": {"label": "Random Baseline", "color": "gray",   "linestyle": "-.", "marker": "v"},
    "2_Uncertainty_Only":{"label": "Uncertainty Only","color": "orange", "linestyle": "-",  "marker": "s"},
    "3_Importance_Only": {"label": "Importance Only", "color": "purple", "linestyle": "-",  "marker": "^"},
    "4_Edge_Baseline":   {"label": "Edge Baseline",   "color": "brown",  "linestyle": "-",  "marker": "d"},
    "5_Proposed_Method": {"label": "Proposed Method", "color": "red",    "linestyle": "-",  "marker": "*", "linewidth": 2.5},
}

def get_target_methods_and_sources(version):
    """
    バージョンに応じた取得対象メソッドと、その取得先ディレクトリを定義する
    Returns:
        targets: { method_name: source_root_path }
        folder_patterns: { source_root_path: regex_pattern }
    """
    targets = {}
    patterns = {
        ROOT_V1: PATTERN_V1,
        ROOT_V2: PATTERN_V2
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
        # v2: 基本は v2 フォルダだが、特定の手法だけ v1 から取得（混合）
        
        # v2から取得するもの
        v2_methods = [
            "1_JSCC_Init", "2_Phase1_Recon", "1_Random_Baseline",
            "3_Importance_Only", "4_Edge_Baseline"
        ]
        for m in v2_methods:
            targets[m] = ROOT_V2
            
        # v1から取得するもの (Uncertainty, Proposed)
        v1_imports = [
            "2_Uncertainty_Only", "5_Proposed_Method"
        ]
        for m in v1_imports:
            targets[m] = ROOT_V1

    return targets, patterns

# ==========================================
# データ読み込み処理
# ==========================================

def load_lpips_data(targets, patterns):
    """
    指定されたターゲット設定に基づいてデータを収集する
    Output: data[method][snr] = value
    """
    data_store = {m: {} for m in targets.keys()}
    
    # 探索対象のルートディレクトリごとにファイルをスキャン
    roots_to_scan = set(targets.values())
    
    file_cache = {} # path -> json_content

    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")

    for root in roots_to_scan:
        print(f"Scanning directory: {root} ...")
        pattern_str = patterns[root]
        regex_folder = re.compile(pattern_str)
        
        # 再帰的に post_process_lpips.json を探す
        search_path = os.path.join(root, "**", "post_process_lpips.json")
        found_files = glob.glob(search_path, recursive=True)
        
        for fpath in found_files:
            # フォルダ名がパターンにマッチするか確認
            dirname = os.path.basename(os.path.dirname(fpath))
            
            if not regex_folder.search(dirname):
                continue
                
            # SNRを取得
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

    # 収集したキャッシュから必要なデータを抽出して data_store に格納
    print("Aggregating data...")
    for method, source_root in targets.items():
        # source_root に属するファイルキャッシュを走査
        for fpath, (snr, content) in file_cache.items():
            if fpath.startswith(source_root):
                if method in content:
                    val = content[method]
                    # 単純な数値の場合と辞書の場合に対応
                    if isinstance(val, (float, int)):
                         data_store[method][snr] = val
                    elif isinstance(val, dict) and "lpips" in val: # 万が一構造が違う場合
                        data_store[method][snr] = val["lpips"]

    return data_store

# ==========================================
# プロット処理
# ==========================================

def plot_graph(data_store):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # スタイル定義順、あるいはデータが存在する順にプロット
    sorted_methods = [k for k in STYLE_CONFIG.keys() if k in data_store]
    
    has_plot = False
    
    for method in sorted_methods:
        snr_dict = data_store[method]
        if not snr_dict:
            print(f"No data for: {method}")
            continue
            
        # SNRでソートしてリスト化
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
    title_suffix = "(Hybrid)" if VERSION == "v2" else "(Single Run)"
    #ax.set_title(f"LPIPS Comparison {title_suffix} - {DATASET}", fontsize=15, fontweight='bold')
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("LPIPS Score (Lower is Better)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 軸の目盛り設定
    ax.set_xticks(SNR_LABELS)
    
    if has_plot:
        ax.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True)
    else:
        ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    
    filename = f"Suite/lpips_comparison_suite_{VERSION}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nGraph saved to: {filename}")
    plt.show()

# ==========================================
# メイン実行
# ==========================================

if __name__ == "__main__":
    print(f"=== Plotting LPIPS Suite (Mode: {VERSION}) ===")
    
    # 1. 対象の決定
    targets, patterns = get_target_methods_and_sources(VERSION)
    
    print("Target Methods & Sources:")
    for m, src in targets.items():
        print(f"  - {m:20s} from {src}")

    # 2. データ読み込み
    data = load_lpips_data(targets, patterns)
    
    # 3. プロット
    plot_graph(data)