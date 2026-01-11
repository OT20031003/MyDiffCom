import os
import glob
import json
import re
import pandas as pd

# ==========================================
# 設定エリア (User Variables)
# ==========================================

DATASET = "imagenet"
BASE_DIR = "results_retrans_comparison"
ROOT_DIR = os.path.join(BASE_DIR, DATASET)

# フィルタリング設定 (None または [] で全許可)
# ここで実験パラメータを変数として指定します
TARGET_EXPS = [5.0]     
TARGET_GAMS = [0.7]     
TARGET_RATES = [0.1]    

# 相関係数を取得する対象のメソッドキー
# main_diffcom_retransmission.py の出力構造に基づき、不確実性マップの相関 (corr) が含まれるキーを指定
TARGET_KEYS = [
    "perturbation_raw_Unc", 
    "temporal_raw_Unc"
]

# ==========================================

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Directory not found: {ROOT_DIR}")
        print("Please check DATASET and BASE_DIR settings.")
        return

    # ファイル検索
    search_pattern = os.path.join(ROOT_DIR, "**", "SNR*_Retrans_*.json")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Dataset: {DATASET}")
    print(f"Filtering conditions: EXPS={TARGET_EXPS}, GAMS={TARGET_GAMS}, RATES={TARGET_RATES}")
    print(f"Total files found (before filtering): {len(files)}")

    # ファイル名解析用の正規表現
    regex_snr = re.compile(r"SNR(-?\d+(?:\.\d+)?)")
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")
    regex_exp = re.compile(r"_exp(\d+(?:\.\d+)?)")
    regex_gam = re.compile(r"_gam(\d+(?:\.\d+)?)")
    
    records = []

    for fpath in files:
        filename = os.path.basename(fpath)
        
        # パラメータ抽出
        match_snr = regex_snr.search(filename)
        match_rate = regex_rate.search(filename)
        match_exp = regex_exp.search(filename)
        match_gam = regex_gam.search(filename)
        
        if not (match_snr and match_rate):
            continue
            
        snr = float(match_snr.group(1))
        rate = float(match_rate.group(1))
        exp_val = float(match_exp.group(1)) if match_exp else None
        gam_val = float(match_gam.group(1)) if match_gam else None
        
        # フィルタリングロジック
        if TARGET_RATES and rate not in TARGET_RATES: continue
        
        # exp, gam はファイル名に含まれない場合(None)の扱いも考慮
        # TARGETが指定されているのにファイル名に無い場合はスキップするか、
        # "default"として扱うかですが、ここでは厳密に一致するもののみ抽出します。
        if TARGET_EXPS:
            if exp_val is None or exp_val not in TARGET_EXPS: continue
            
        if TARGET_GAMS:
            if gam_val is None or gam_val not in TARGET_GAMS: continue

        # JSON読み込み
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            summary = content.get("summary", {})
            
            # 各ターゲットキーについて相関係数を取得
            for key in TARGET_KEYS:
                # キーが存在し、かつ corr フィールドを持っている場合
                if key in summary and "corr" in summary[key]:
                    corr = summary[key]["corr"]
                    
                    # メソッド名を少し見やすく整形 (例: perturbation_raw_Unc -> perturbation)
                    method_name = key.replace("_raw_Unc", "").replace("_raw_Sem", "")
                    
                    records.append({
                        "Method": method_name,
                        "SNR": snr,
                        "Correlation": corr,
                        "Rate": rate,
                        "Exp": exp_val,
                        "Gam": gam_val
                    })
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not records:
        print("No matching records found based on the criteria.")
        return

    # DataFrame作成
    df = pd.DataFrame(records)
    
    # 整理: Method, Rate, Exp, Gam ごとにグループ化
    unique_groups = df.groupby(["Method", "Rate", "Exp", "Gam"])
    
    for (method, rate, exp, gam), group in unique_groups:
        print("\n" + "="*50)
        print(f"Experimental Condition:")
        print(f"  Method: {method}")
        print(f"  Rate  : {rate}")
        print(f"  Exp   : {exp}")
        print(f"  Gam   : {gam}")
        print("-" * 50)
        
        # SNRで昇順ソート
        sorted_group = group.sort_values("SNR")
        
        # 結果テーブルの表示
        # index=Falseでインデックス番号を非表示に
        print(sorted_group[["SNR", "Correlation"]].to_string(index=False))
        print("="*50)

if __name__ == "__main__":
    main()