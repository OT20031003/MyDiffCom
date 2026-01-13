import os
import glob
import json
import re
import matplotlib.pyplot as plt

# 設定エリア
DATASET = "ffhq_demo"
BASE_DIR = "results_retrans_comparison"
METHOD_PATH = "diffcom/djscc_2"
ROOT_DIR = os.path.join(BASE_DIR, DATASET, METHOD_PATH)

TARGET_FILENAME = "post_process_psnr.json" # 対象ファイル

TARGET_SNRS = []
TARGET_SNRS = [-8, -6, -7, -5, -4]
TARGET_RATES = [0.1]
TARGET_EXPS = [2.0]
TARGET_GAMS = [0.3]

TARGET_KEYS = [
    "3_P2_perturbation_raw_Unc",
    "3_P2_perturbation_raw_Sem",
    "3_P2_Random"
]

METHOD_LABELS = {
    "1_JSCC_Init": "JSCC (Initial)",
    "2_Phase1_Recon": "Phase 1 Recon",
    "3_P2_perturbation_raw_Unc": "Perturbation (Unc)",
    "3_P2_perturbation_raw_Sem": "Perturbation (Sem)",
    "3_P2_Random": "Random Baseline",
}

STYLE_CONFIG = {
    "1_JSCC_Init": {"color": "black", "linestyle": ":", "marker": "x"},
    "2_Phase1_Recon": {"color": "blue", "linestyle": "-", "marker": "o"},
    "3_P2_perturbation_raw_Unc": {"color": "red", "linestyle": "-", "marker": "s"},
    "3_P2_perturbation_raw_Sem": {"color": "red", "linestyle": "--", "marker": "D"},
    "3_P2_Random": {"color": "gray", "linestyle": "-.", "marker": "d"},
}

def load_data_recursive():
    search_pattern = os.path.join(ROOT_DIR, "**", TARGET_FILENAME)
    files = glob.glob(search_pattern, recursive=True)
    data_store = {}
    
    regex_snr = re.compile(r"awgn_(-?\d+(?:\.\d+)?)dB")
    regex_rate = re.compile(r"Retrans_rate_(\d+(?:\.\d+)?)")
    regex_exp = re.compile(r"_exp(\d+(?:\.\d+)?)")
    regex_gam = re.compile(r"_gam(\d+(?:\.\d+)?)")

    for fpath in files:
        dirname = os.path.dirname(fpath)
        folder_name = os.path.basename(dirname)
        
        match_snr = regex_snr.search(dirname)
        match_rate = regex_rate.search(dirname)
        if not match_snr or not match_rate: continue
        
        snr = float(match_snr.group(1))
        rate = float(match_rate.group(1))
        
        match_exp = regex_exp.search(folder_name)
        current_exp = float(match_exp.group(1)) if match_exp else None
        
        match_gam = regex_gam.search(folder_name)
        current_gam = float(match_gam.group(1)) if match_gam else None

        if TARGET_EXPS and (current_exp is None or current_exp not in TARGET_EXPS): continue
        if TARGET_GAMS and (current_gam is None or current_gam not in TARGET_GAMS): continue
        if TARGET_SNRS and snr not in TARGET_SNRS: continue
        if TARGET_RATES and rate not in TARGET_RATES: continue

        try:
            with open(fpath, 'r') as f:
                content = json.load(f)
            if rate not in data_store: data_store[rate] = {}
            if snr not in data_store[rate]: data_store[rate][snr] = {}
            for k in TARGET_KEYS:
                if k in content: data_store[rate][snr][k] = content[k]
        except: pass
    return data_store

def plot_psnr(data_store):
    if not data_store: return
    rates = sorted(data_store.keys())
    for rate in rates:
        current_data = data_store[rate]
        snr_list = sorted(current_data.keys())
        if not snr_list: continue

        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        has_data = False
        
        for method in TARGET_KEYS:
            x, y = [], []
            for snr in snr_list:
                if method in current_data[snr]:
                    x.append(snr)
                    y.append(current_data[snr][method])
            if x:
                has_data = True
                style = STYLE_CONFIG.get(method, {})
                label = METHOD_LABELS.get(method, method)
                ax.plot(x, y, label=label, **style)

        ax.set_title("PSNR vs SNR", fontsize=14, fontweight='bold')
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if has_data: ax.legend(loc='best')
        plt.tight_layout()
        
        save_name = f'psnr_vs_snr_{DATASET}_rate_{rate}'
        if TARGET_EXPS: save_name += f'_exp{TARGET_EXPS[0]}'
        if TARGET_GAMS: save_name += f'_gam{TARGET_GAMS[0]}'
        plt.savefig(save_name + '.png', dpi=300)
        print(f"Saved: {save_name}.png")
        plt.close()

if __name__ == "__main__":
    data = load_data_recursive()
    plot_psnr(data)