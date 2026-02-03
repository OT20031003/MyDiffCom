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
VERSION = "v2"

# 共通設定
DATASET = "ffhq_demo"
SNR_LABELS = [-7, -6, -5, -4, -3, -2] # プロットしたいSNR範囲

# データのルートディレクトリ定義
ROOT_V1 = os.path.join("results_retrans_comparison_v1", DATASET)
ROOT_V2 = os.path.join("results_retrans_comparison_v2", DATASET)
ROOT_V3 = os.path.join("results_retrans_comparison_v3", DATASET)

# フォルダ名のパターン指定 (正規表現)
PATTERN_V1 = r"Retrans_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"
PATTERN_V2 = r"Retrans_v2_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"
PATTERN_V3 = r"Retrans_rate_0\.1_Comparison_semantic_exp2\.0_gam0\.9_zeta0\.3_seed22"

# 対象メトリクス (JSON内のキー)
TARGET_METRIC_KEY = "id_loss"

# ==========================================
# 手法とデータソースの定義
# ==========================================

STYLE_CONFIG = {
    "2_Phase1_Recon":      {"label": "DiffCom",     "color": "blue",   "linestyle": "--", "marker": "o"},
    "2_Uncertainty_Only":  {"label": "Uncertainty Only",  "color": "orange", "linestyle": "-",  "marker": "s"},
    "3_Importance_Only":   {"label": "Importance Only",   "color": "purple", "linestyle": "-",  "marker": "^"},
    "5_Proposed_Method":   {"label": "Proposed Method",   "color": "red",    "linestyle": "-",  "marker": "*", "linewidth": 2.5},
    "6_Importance_Random": {"label": "Imp + Random",      "color": "cyan",   "linestyle": "-",  "marker": "o"},
    "7_Edge_Random":       {"label": "Edge + Random",     "color": "lime",   "linestyle": "-",  "marker": "h"},
    "8_Uncertainty_Random":{"label": "Unc + Random",      "color": "magenta","linestyle": "-",  "marker": "D"},
}

def get_target_methods_and_sources(version):
    targets = {}
    patterns = {ROOT_V1: PATTERN_V1, ROOT_V2: PATTERN_V2, ROOT_V3: PATTERN_V3}

    if version == "v1":
        methods = ["1_JSCC_Init", "2_Phase1_Recon", "1_Random_Baseline", "2_Uncertainty_Only", "3_Importance_Only", "4_Edge_Baseline", "5_Proposed_Method"]
        for m in methods: targets[m] = ROOT_V1
    elif version == "v2":
        v2_methods = ["1_JSCC_Init", "2_Phase1_Recon", "1_Random_Baseline", "3_Importance_Only", "4_Edge_Baseline"]
        for m in v2_methods: targets[m] = ROOT_V2
        v1_imports = ["2_Uncertainty_Only", "5_Proposed_Method"]
        for m in v1_imports: targets[m] = ROOT_V1
    elif version == "v3":
        v3_methods = ["6_Importance_Random", "7_Edge_Random", "8_Uncertainty_Random"]
        for m in v3_methods: targets[m] = ROOT_V3
        v1_imports = ["2_Uncertainty_Only", "5_Proposed_Method"]
        for m in v1_imports: targets[m] = ROOT_V1
    return targets, patterns

def load_id_loss_data(targets, patterns):
    data_store = {m: {} for m in targets.keys()}
    roots_to_scan = set(targets.values())
    file_cache = {}
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")

    for root in roots_to_scan:
        if not os.path.exists(root): continue
        pattern_str = patterns[root]
        regex_folder = re.compile(pattern_str)
        search_path = os.path.join(root, "**", "post_process_id_loss.json")
        found_files = glob.glob(search_path, recursive=True)
        for fpath in found_files:
            dirname = os.path.basename(os.path.dirname(fpath))
            if not regex_folder.search(dirname): continue
            match_snr = regex_snr.search(fpath)
            if not match_snr: continue
            snr = float(match_snr.group(1))
            if snr not in SNR_LABELS: continue
            try:
                with open(fpath, 'r') as f:
                    content = json.load(f)
                    file_cache[fpath] = (snr, content)
            except Exception: pass

    for method, source_root in targets.items():
        for fpath, (snr, content) in file_cache.items():
            if os.path.abspath(fpath).startswith(os.path.abspath(source_root)):
                if method in content:
                    val = content[method]
                    if isinstance(val, dict) and TARGET_METRIC_KEY in val:
                        data_store[method][snr] = val[TARGET_METRIC_KEY]
                    elif isinstance(val, (float, int)):
                         data_store[method][snr] = val
    return data_store

def plot_graph(data_store):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    sorted_methods = [k for k in STYLE_CONFIG.keys() if k in data_store]
    has_plot = False
    
    for method in sorted_methods:
        snr_dict = data_store[method]
        if not snr_dict: continue
        sorted_snrs = sorted(snr_dict.keys())
        x_vals = sorted_snrs
        y_vals = [snr_dict[s] for s in sorted_snrs]
        style = STYLE_CONFIG[method]
        ax.plot(x_vals, y_vals, label=style["label"], color=style["color"], linestyle=style["linestyle"], 
                marker=style["marker"], linewidth=style.get("linewidth", 1.5), markersize=8)
        has_plot = True

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("ID Loss (Lower is Better)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(SNR_LABELS)
    
    if has_plot:
        # 修正箇所: 論文スタイルの凡例設定
        ax.legend(loc='upper right', 
                  fontsize=11, 
                  frameon=True, 
                  shadow=False, 
                  fancybox=False, 
                  edgecolor='black', 
                  framealpha=1.0)
    else:
        ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    save_dir = "Suite"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    filename = f"{save_dir}/id_loss_comparison_suite_{VERSION}.png"
    plt.savefig(filename, dpi=300)

def print_statistics(data_store):
    print("\n" + "="*80)
    print(f" ID Loss Numerical Results (Mode: {VERSION})")
    print("="*80)
    header = f"{'Method':<25}" + "".join([f"{snr:>10}dB" for snr in SNR_LABELS])
    print(header)
    print("-" * len(header))
    sorted_methods = [k for k in STYLE_CONFIG.keys() if k in data_store]
    for method in sorted_methods:
        snr_dict = data_store[method]
        if not snr_dict: continue
        label_name = STYLE_CONFIG[method]["label"]
        row_str = f"{label_name:<25}"
        for snr in SNR_LABELS:
            val = snr_dict.get(snr, None)
            row_str += f"{val:>10.4f}" if val is not None else f"{'N/A':>10}"
        print(row_str)
    print("="*80 + "\n")

if __name__ == "__main__":
    targets, patterns = get_target_methods_and_sources(VERSION)
    data = load_id_loss_data(targets, patterns)
    print_statistics(data)
    plot_graph(data)