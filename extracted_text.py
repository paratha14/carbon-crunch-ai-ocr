from preprocessing import preprocess
from ocr import extract_ocr_result
import re
import json
import os

TOTAL_KEYWORDS     = ["grand total", "total amount", "amount due", "total", "amt due", "balance due"]
SKIP_ITEM_KEYWORDS = ["total", "subtotal", "sub total", "tax", "cash", "change", "discount",
                      "saving", "balance", "vat", "gst", "tip", "tender", "open", "store",
                      "thank", "www", "items", "grocery non", "see back", "scan with",
                      "voided", "you saved", "was ", "id #", "tc#", "st#"]
PRICE_PATTERN  = r'^\$?(\d{1,3}(?:,\d{3})*\.\d{2})$'
INLINE_PRICE   = r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})\s*[A-Z]?\s*$'
DATE_PATTERNS  = [
    r'\b(\d{2}/\d{2}/\d{4})\b',
    r'\b(\d{2}-\d{2}-\d{4})\b',
    r'\b(\d{4}-\d{2}-\d{2})\b',
    r'\b(\d{2}/\d{2}/\d{2})\b',
    r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
]
TIME_PATTERN = r'\b(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2}\s?[APap][Mm])\b'

STORE_KEYWORDS = ["walmart", "target", "trader joe", "costco", "kroger", "whole foods",
                  "aldi", "safeway", "publix", "cvs", "walgreens", "amazon", "mcdonald"]


def is_price_only(text: str) -> bool:
    return bool(re.match(PRICE_PATTERN, text.strip()))


def is_skip_line(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in SKIP_ITEM_KEYWORDS)


def has_known_store(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in STORE_KEYWORDS)

def extract_store_name(ocr_result):
    blocks = ocr_result["blocks"]

    for block in blocks:
        text = block["text"].strip()
        conf = block["confidence"]
        if has_known_store(text):
            
            text = re.sub(r'^[A-Z]\s+', '', text).strip()
            
            for kw in STORE_KEYWORDS:
                if kw in text.lower():
                
                    match = re.search(kw, text, re.IGNORECASE)
                    if match and len(text) > 30:
                        text = text[match.start():match.start()+len(kw)].title()
                    break
            print(f"Extracted store name: '{text}' with confidence {conf:.2f}")
            return {"value": text, "confidence": round(conf, 4)}

   
    for block in blocks[:5]:
        text = block["text"].strip()
        conf = block["confidence"]
        if len(text) < 3 or re.fullmatch(r'[\d\s\W]+', text) or is_skip_line(text):
            continue
        text = re.sub(r'^[A-Z]\s+', '', text).strip()
        return {"value": text, "confidence": round(conf, 4)}

    return {"value": None, "confidence": 0.0}

def extract_date_time(ocr_result):
    result = {"date": {"value": None, "confidence": 0.0},
              "time": {"value": None, "confidence": 0.0}}

    for block in ocr_result["blocks"]:
        text = block["text"]
        conf = block["confidence"]

        if result["date"]["value"] is None:
            for pattern in DATE_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["date"] = {"value": match.group(1), "confidence": round(conf, 4)}
                    time_match = re.search(TIME_PATTERN, text, re.IGNORECASE)
                    if time_match:
                        result["time"] = {"value": time_match.group(1), "confidence": round(conf, 4)}
                    break

        if result["time"]["value"] is None:
            time_match = re.search(TIME_PATTERN, text, re.IGNORECASE)
            if time_match and "OPEN" not in text.upper():
                result["time"] = {"value": time_match.group(1), "confidence": round(conf, 4)}

    return result

def extract_items(ocr_result):
    blocks = ocr_result["blocks"]
    items  = []
    used   = set()

    for i, block in enumerate(blocks):
        text = block["text"].strip()
        conf = block["confidence"]

        if i in used or is_skip_line(text):
            continue

        # Layout A — price only block, look back up to 2 blocks for name
        if is_price_only(text):
            if i == 0:
                continue
            name, name_conf = None, 0.0

            for lookback in [1, 2]:
                if i - lookback < 0:
                    break
                candidate      = blocks[i - lookback]["text"].strip()
                candidate_conf = blocks[i - lookback]["confidence"]

                if re.fullmatch(r'[\d\s]+', candidate):  # skip barcode blocks
                    continue
                if len(candidate) < 2:                    # skip single letters
                    continue
                if is_skip_line(candidate):
                    break

                name      = re.sub(r'\s+\d{6,}', '', candidate).strip()
                name_conf = candidate_conf
                used.add(i - lookback)
                break

            if not name or len(name) < 2:
                continue

            price_match = re.match(PRICE_PATTERN, text)
            items.append({
                "name"       : name,
                "price"      : price_match.group(1),
                "confidence" : round((conf + name_conf) / 2, 4)
            })
            used.add(i)
            continue

        # Layout B — name + barcode + price all inline
        inline_match = re.search(INLINE_PRICE, text)
        if inline_match:
            price = inline_match.group(1)
            name  = text[:inline_match.start()].strip()
            name  = re.sub(r'\s+\d{8,}\s*[A-Z]?\s*$', '', name).strip()
            name  = re.sub(r'\s+[A-Z]$', '', name).strip()
            if len(name) < 2 or is_skip_line(name):
                continue
            items.append({
                "name"       : name,
                "price"      : price,
                "confidence" : round(conf, 4)
            })
            used.add(i)

    return items

def extract_total(ocr_result):
    blocks = ocr_result["blocks"]

    for i, block in enumerate(blocks):
        text = block["text"]
        conf = block["confidence"]

        if not any(kw in text.lower() for kw in TOTAL_KEYWORDS):
            continue
        
        if "sub" in text.lower():
            continue

        match = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})', text)
        if match:
            print(f"Found total: {match.group(1)}")
            return {"value": match.group(1), "confidence": round(conf, 4)}

        if i + 1 < len(blocks):
            next_match = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})', blocks[i+1]["text"])
            if next_match:
                avg_conf = (conf + blocks[i+1]["confidence"]) / 2
                print(f"Found total: {next_match.group(1)}")
                return {"value": next_match.group(1), "confidence": round(avg_conf, 4)}

    return {"value": None, "confidence": 0.0}

def extract_all(ocr_result):
    store     = extract_store_name(ocr_result)
    date_time = extract_date_time(ocr_result)
    items     = extract_items(ocr_result)
    total     = extract_total(ocr_result)

    missing = [f for f, v in [("store_name", store), ("date", date_time["date"]),
                               ("total_amount", total)] if v["value"] is None]
    low_conf = [f for f, v in [("store_name", store), ("date", date_time["date"]),
                                ("total_amount", total)] if v["value"] and v["confidence"] < 0.7]

    return {
        "store_name"   : store,
        "date"         : date_time["date"],
        "time"         : date_time["time"],
        "items"        : items,
        "total_amount" : total,
        "flags"        : {"missing_fields": missing, "low_confidence_fields": low_conf}
    }


def save_json(data, image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    base     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_receipt_image>")
        sys.exit(1)

    image_path   = sys.argv[1]
    preprocessed = preprocess(image_path)
    ocr_result   = extract_ocr_result(preprocessed)
    extracted    = extract_all(ocr_result)

    print(json.dumps(extracted, indent=2))
    save_json(extracted, image_path)