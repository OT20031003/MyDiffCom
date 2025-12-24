#!/usr/bin/env python3
"""
detect_ffhq_text.py

FFHQ (or any image folder) の画像を走査し、画像内に文字（テキスト）が存在するかを
Tesseract OCR で判定して、検出結果（index / filename / 検出テキスト / スコア等）を出力します。

使い方例:
  python detect_ffhq_text.py --input_dir /path/to/ffhq/images1024x1024 \
      --output results.csv --min_conf 60 --workers 8 --save_indices indices.txt

出力:
 - CSV (デフォルト): index, filename, word_count, avg_conf, max_conf, concat_text
 - もしくは TXT (1行 = index) / JSON も選べます
"""

import os
import argparse
from PIL import Image
import pytesseract
import concurrent.futures
from tqdm import tqdm
import csv
import json
import sys
import re

def detect_text_in_image(path, tesseract_lang=None, resize_max=None, min_conf=50, min_chars=1):
    """
    画像ファイル path を OCR して、文字の有無を判定する。
    戻り値: dict:
      {
        "filename": ...,
        "index": ... (if filename looks like NNNNN.png it extracts int),
        "words": [ {text, conf, left, top, width, height}, ... ],
        "word_count": int,
        "avg_conf": float,
        "max_conf": int,
        "concat_text": "..."
      }
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return {"filename": os.path.basename(path), "error": f"open_error:{e}"}

    # optional resizing to speed up OCR and avoid huge images
    if resize_max:
        w, h = img.size
        if max(w, h) > resize_max:
            scale = resize_max / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

    tconf_param = "--oem 3 --psm 3"  # default: good general use
    lang_arg = tesseract_lang if tesseract_lang else "eng"
    try:
        # image_to_data returns a TSV with word-level boxes and conf
        ocr_data = pytesseract.image_to_data(img, lang=lang_arg, output_type=pytesseract.Output.DICT, config=tconf_param)
    except Exception as e:
        return {"filename": os.path.basename(path), "error": f"tesseract_error:{e}"}

    words = []
    n_boxes = len(ocr_data.get("text", []))
    for i in range(n_boxes):
        txt = (ocr_data["text"][i] or "").strip()
        conf_raw = ocr_data["conf"][i]
        # pytesseract sometimes returns '-1' for empty; handle safely
        try:
            conf = int(float(conf_raw))
        except:
            try:
                conf = int(conf_raw)
            except:
                conf = -1
        if txt != "" and conf >= min_conf and len(re.sub(r"\s+", "", txt)) >= min_chars:
            words.append({
                "text": txt,
                "conf": conf,
                "left": ocr_data.get("left", [None])[i],
                "top": ocr_data.get("top", [None])[i],
                "width": ocr_data.get("width", [None])[i],
                "height": ocr_data.get("height", [None])[i],
            })

    word_count = len(words)
    avg_conf = sum(w["conf"] for w in words) / word_count if word_count > 0 else 0.0
    max_conf = max((w["conf"] for w in words), default=0)
    concat_text = " ".join(w["text"] for w in words)

    # try to extract numeric index from filename like 00000.png
    fname = os.path.basename(path)
    idx = None
    m = re.match(r"0*([0-9]+)\.", fname)
    if m:
        try:
            idx = int(m.group(1))
        except:
            idx = None

    return {
        "filename": fname,
        "index": idx,
        "words": words,
        "word_count": word_count,
        "avg_conf": avg_conf,
        "max_conf": max_conf,
        "concat_text": concat_text
    }

def worker(args):
    return detect_text_in_image(*args)

def main():
    parser = argparse.ArgumentParser(description="Detect text in FFHQ images using Tesseract OCR.")
    parser.add_argument("--input_dir", default="testsets/ffhq_demo", help="画像フォルダ（FFHQのimages1024x1024等）")
    parser.add_argument("--output", default="results.csv", help="出力ファイル（.csv/.json/.txt のいずれか）")
    parser.add_argument("--min_conf", type=int, default=20, help="文字検出の最小信頼度（0-100）")
    parser.add_argument("--min_chars", type=int, default=1, help="単語の最小文字数（空白を除く）")
    parser.add_argument("--workers", type=int, default=4, help="並列ワーカー数（デフォルト4）")
    parser.add_argument("--resize_max", type=int, default=1024, help="OCR前にリサイズする最大辺（パフォーマンス向上用）。0で無効")
    parser.add_argument("--tesseract_lang", default="eng", help="Tesseract の言語（例: eng, jpn 等）")
    parser.add_argument("--only_indices", action="store_true", help="テキスト検出があった画像の index のみを標準出力に表示")
    parser.add_argument("--save_indices", default=None, help="検出された index を保存するファイル (.txt)")
    args = parser.parse_args()

    input_dir = args.input_dir
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    if not files:
        print("No images found in", input_dir)
        sys.exit(1)

    tasks = []
    for f in files:
        tasks.append((f, args.tesseract_lang, args.resize_max if args.resize_max > 0 else None, args.min_conf, args.min_chars))

    results = []
    # Use ThreadPoolExecutor since pytesseract is I/O/CPU mixed; threads work reasonably
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OCR"):
            try:
                r = fut.result()
            except Exception as e:
                r = {"filename": None, "error": f"worker_exception:{e}"}
            results.append(r)

    # filter those with word_count > 0
    positives = [r for r in results if r.get("word_count", 0) > 0]

    # Save outputs
    out_path = args.output
    _, ext = os.path.splitext(out_path.lower())
    if ext == ".csv" or ext == "":
        # write CSV
        with open(out_path, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["index", "filename", "word_count", "avg_conf", "max_conf", "concat_text"])
            for r in sorted(positives, key=lambda x: (x.get("index") if x.get("index") is not None else 1_000_000, x["filename"])):
                writer.writerow([r.get("index"), r["filename"], r["word_count"], f"{r['avg_conf']:.1f}", r["max_conf"], r["concat_text"]])
        print(f"Saved CSV -> {out_path} (positives: {len(positives)})")
    elif ext == ".json":
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(positives, jf, ensure_ascii=False, indent=2)
        print(f"Saved JSON -> {out_path} (positives: {len(positives)})")
    elif ext == ".txt":
        with open(out_path, "w", encoding="utf-8") as tf:
            for r in sorted(positives, key=lambda x: (x.get("index") if x.get("index") is not None else 1_000_000, x["filename"])):
                idx = r.get("index")
                tf.write(f"{idx if idx is not None else r['filename']}\n")
        print(f"Saved TXT -> {out_path} (positives: {len(positives)})")
    else:
        print("Unsupported output extension; use .csv, .json, or .txt. Defaulting to CSV.")
        # fallback to CSV
        with open("results.csv", "w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["index", "filename", "word_count", "avg_conf", "max_conf", "concat_text"])
            for r in positives:
                writer.writerow([r.get("index"), r["filename"], r["word_count"], f"{r['avg_conf']:.1f}", r["max_conf"], r["concat_text"]])
        print("Saved results.csv")

    # optional: save just indices file
    if args.save_indices:
        with open(args.save_indices, "w", encoding="utf-8") as si:
            for r in sorted(positives, key=lambda x: (x.get("index") if x.get("index") is not None else 1_000_000, x["filename"])):
                idx = r.get("index")
                si.write(f"{idx if idx is not None else r['filename']}\n")
        print(f"Saved indices -> {args.save_indices}")

    if args.only_indices:
        for r in sorted(positives, key=lambda x: (x.get("index") if x.get("index") is not None else 1_000_000, x["filename"])):
            print(r.get("index") if r.get("index") is not None else r["filename"])


if __name__ == "__main__":
    main()
