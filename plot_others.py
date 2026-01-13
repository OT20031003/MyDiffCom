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
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# --- [変更点] モデルのサフィックス設定 ---
# calc_others.py で指定したモデルに合わせて変更してください
# 'convnext' または 'swin'
MODEL_SUFFIX = "swin" 

# 2. プロット対象のファイル名 (自動生成)
TARGET_FILENAME = f"semantic_metrics_results_{MODEL_SUFFIX}.json"

# 3. プロットしたいSNRのリスト (None または [] なら見つかったもの全て表示)
TARGET_SNRS = [-8, -7, -6, -5, -4, -3, -2]
# TARGET_SNRS = []

# 4. プロットしたい再送率 (Retrans_rate) のリスト
TARGET_RATES = [0.1]

# --- [追加機能] 拡張パラメータでのフィルタリング設定 ---
# 指定した exp (expansion_factor) や gam (gamma) のファイルのみを抽出します。
# None または [] (空リスト) の場合は、フィルタリングせず全て対象とします。

TARGET_EXPS = [5.0]     # 例: [2.0] または None
TARGET_GAMS = [0.7]     # 例: [0.3, 0.7] または None

# -----------------------------------------------------

# 5. プロットしたいJSON内のキー (手法) のリスト
TARGET_KEYS = [
    #"1_JSCC_Init",
    #"2_Phase1_Recon",
    "3_P2_Random",
    
    # "3_P2_temporal_raw_Unc",
    # "3_P2_temporal_raw_Sem",
    
    "3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
]

# 6. 凡例の表示名マッピング
METHOD_LABELS = {
    "1_JSCC_Init":              "JSCC (Initial)",
    "2_Phase1_Recon":           "Phase 1 Recon",
    "3_P2_Random":              "Random Baseline",
    
    "3_P2_temporal_raw_Unc":    "Temporal (Unc)",
    "3_P2_temporal_raw_Sem":    "Temporal (Sem)",
    
    "3_P2_perturbation_raw_Unc":"Perturbation (Unc)",
    "3_P2_perturbation_raw_Sem":"Perturbation (Sem)",
}

# 7. スタイル設定 (色、線種、マーカー)
STYLE_CONFIG = {
    "1_JSCC_Init":              {"color": "black", "linestyle": ":",  "marker": "x"}, 
    "2_Phase1_Recon":           {"color": "blue",  "linestyle": "-",  "marker": "o"}, 
    "3_P2_Random":              {"color": "gray",  "linestyle": "-.", "marker": "d"}, 

    "3_P2_temporal_raw_Unc":    {"color": "green", "linestyle": "-",  "marker": "^"}, 
    "3_P2_temporal_raw_Sem":    {"color": "green", "linestyle": "--", "marker": "v"}, 
    
    "3_P2_perturbation_raw_Unc":{"color": "red",   "linestyle": "-",  "marker": "s"}, 
    "3_P2_perturbation_raw_Sem":{"color": "red",   "linestyle": "--", "marker": "D"}, 
}

# 8. プロット対象の指標設定 (詳細設定)
# calc_others.py の出力キー ("accuracy", "classifier_confidence", "clip_score") に対応
METRICS_CONFIG = {
    "accuracy": {
        "title": "Classification Consistency (Accuracy)",
        "ylabel": "Accuracy",   # ラベル表記を簡素化
        "ylim": None            # 自動調整に変更 (元: (0.0, 1.05))
    },
    # キー名を resnet_confidence から classifier_confidence に変更
    "classifier_confidence": {
        "title": f"Classifier Confidence ({MODEL_SUFFIX})",
        "ylabel": "Probability",
        "ylim": None            # 自動調整に変更 (元: (0.0, 1.05))
    },
    "clip_score": {
        "title": "CLIP Semantic Score",
        "ylabel": "CLIP Logits",
        "ylim": None            # 元々自動調整
    }
}

# 9. 実際にプロットする指標のリスト (METRICS_CONFIG のキーから選択)
TARGET_METRICS = [
    #"accuracy",
    #"classifier_confidence",
    "clip_score"
]

# ==========================================
# 処理ロジック
# ==========================================

def load_metrics_data_recursive():
    """
    ディレクトリを再帰的に探索し、TARGET_FILENAME を読み込む
    パスから SNR (awgn_-6dB) と Retrans_rate (Retrans_rate_0.1) を抽出する
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
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")
    
    # 追加パラメータ用の正規表現
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

            # JSON構造: { "MethodName": { "accuracy": val, ... }, ... }
            for key_method in TARGET_KEYS:
                if key_method in content:
                    data_store[rate][snr][key_method] = content[key_method]
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return data_store

def plot_other_metrics(data_store):
    if not data_store:
        print("表示対象のデータが見つかりませんでした。パスやファイル名、DATASET設定、フィルタ設定を確認してください。")
        return

    rates = sorted(data_store.keys())
    metrics_list = TARGET_METRICS
    
    for rate in rates:
        print(f"--- Plotting for Retrans Rate: {rate} ---")
        
        current_data = data_store[rate]
        snr_list = sorted(current_data.keys())
        
        if not snr_list:
            print(f"Rate {rate} に有効なSNRデータがありません。スキップします。")
            continue
        
        print(f"  SNRs found: {snr_list}")

        # === レイアウト設定 ===
        num_metrics = len(metrics_list)
        
        if num_metrics == 1:
            cols = 1
            rows = 1
            figsize = (12, 8) 
        else:
            cols = 2
            rows = (num_metrics + cols - 1) // cols
            figsize = (14, 6 * rows)
        
        # 図の生成
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Axesの配列化処理
        if num_metrics == 1:
             axes = [axes] 
        elif hasattr(axes, "flatten"):
            axes = axes.flatten()
        else:
            axes = [axes]
            
        # タイトル生成 (表示はしないがロジックは残す)
        title_str = f"Semantic Metrics ({DATASET} - {MODEL_SUFFIX.upper()} - Rate: {rate})"
        
        cond_strs = []
        if TARGET_EXPS and len(TARGET_EXPS) == 1:
            cond_strs.append(f"Exp:{TARGET_EXPS[0]}")
        if TARGET_GAMS and len(TARGET_GAMS) == 1:
            cond_strs.append(f"Gam:{TARGET_GAMS[0]}")
        
        if cond_strs:
            title_str += " [" + ", ".join(cond_strs) + "]"
        
        # --- 変更点: 全体タイトルを非表示 ---
        # fig.suptitle(title_str, fontsize=16)

        for idx, metric_key in enumerate(metrics_list):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            if metric_key not in METRICS_CONFIG:
                print(f"Warning: Metric '{metric_key}' config not found. Skipping.")
                continue

            config = METRICS_CONFIG[metric_key]
            has_data = False
            
            for method in TARGET_KEYS:
                x_vals = []
                y_vals = []
                
                for snr in snr_list:
                    # データ構造: data_store[rate][snr][method][metric_key]
                    if method in current_data[snr] and metric_key in current_data[snr][method]:
                        val = current_data[snr][method][metric_key]
                        if isinstance(val, (int, float)):
                            x_vals.append(snr)
                            y_vals.append(val)
                
                if x_vals:
                    has_data = True
                    style = STYLE_CONFIG.get(method, {})
                    label = METHOD_LABELS.get(method, method)
                    ax.plot(x_vals, y_vals, label=label, **style)

            # グラフ装飾
            ax.set_title(config["title"], fontsize=14, fontweight='bold')
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(config["ylabel"])
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # ylimがNoneの場合は自動調整 (set_ylimを呼ばない)
            if config["ylim"]:
                ax.set_ylim(config["ylim"])
            
            if has_data:
                ax.legend(loc='best', fontsize=9)
            else:
                ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)

        # 余った領域を非表示
        if num_metrics > 1:
            for i in range(idx + 1, len(axes)):
                axes[i].axis('off')

        plt.tight_layout()
        # --- 変更点: タイトル用の余白調整を削除 ---
        # plt.subplots_adjust(top=0.92)

        # ファイル名生成
        save_name = f'semantic_metrics_{DATASET}_{MODEL_SUFFIX}_rate_{rate}'
        if TARGET_EXPS and len(TARGET_EXPS) == 1:
            save_name += f'_exp{TARGET_EXPS[0]}'
        if TARGET_GAMS and len(TARGET_GAMS) == 1:
            save_name += f'_gam{TARGET_GAMS[0]}'
        save_name += '.png'

        plt.savefig(save_name, dpi=300)
        print(f"グラフを保存しました: {save_name}")
        
        plt.close(fig) 

if __name__ == "__main__":
    data = load_metrics_data_recursive()
    plot_other_metrics(data)