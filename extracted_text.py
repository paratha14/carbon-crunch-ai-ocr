from preprocessing import preprocess
from ocr import extract_ocr_result
import re
import json
import os

TOTAL_KEYWORDS     = ["grand total", "total amount", "amount due", "total", "amt due", "balance due"]
SKIP_ITEM_KEYWORDS = ["total", "subtotal", "sub total", "tax", "cash", "change", "discount",
                      "saving", "balance", "vat", "gst", "tip", "tender", "open", "store",
                      "thank", "www", "items", "grocery non"]
PRICE_PATTERN = r'^\$?(\d{1,3}(?:,\d{3})*\.\d{2})$'
DATE_PATTERNS = [
    r'\b(\d{2}/\d{2}/\d{4})\b',
    r'\b(\d{2}-\d{2}-\d{4})\b',
    r'\b(\d{4}-\d{2}-\d{2})\b',
    r'\b(\d{2}/\d{2}/\d{2})\b',
    r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
]
TIME_PATTERN = r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\b'


def is_price_only(text: str) -> bool:
    """Returns True if the block is purely a price value."""
    return bool(re.match(PRICE_PATTERN, text.strip()))


def is_skip_line(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in SKIP_ITEM_KEYWORDS)


def extract_store_name(ocr_result):
    for block in ocr_result["blocks"][:5]:
        text = block["text"].strip()
        conf = block["confidence"]

        if len(text) < 3:
            continue
        if re.fullmatch(r'[\d\s\W]+', text):
            continue

        text = re.sub(r'^[A-Z]\s+', '', text).strip()

        if conf < 0.7:
            print(f"Warning: Low confidence ({conf:.2f}) for store name: '{text}'")
        else:
            print(f"Extracted store name: '{text}' with confidence {conf:.2f}")

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
                    # Check same line for transaction time
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
    used   = set()  # track used block indices to avoid duplicates

    for i, block in enumerate(blocks):
        text = block["text"].strip()
        conf = block["confidence"]

        if i in used:
            continue

        # Case 1: price-only block — pair with previous block as name
        if is_price_only(text):
            if i == 0:
                continue
            prev = blocks[i - 1]
            name = prev["text"].strip()

            if is_skip_line(name) or is_skip_line(text):
                continue
            if len(name) < 2 or is_price_only(name):
                continue

            price_match = re.match(PRICE_PATTERN, text)
            avg_conf    = (conf + prev["confidence"]) / 2

            items.append({
                "name"       : name,
                "price"      : price_match.group(1),
                "confidence" : round(avg_conf, 4)
            })
            used.add(i)
            used.add(i - 1)
            continue

        # Case 2: name + price on same block (e.g. "2@0.49")
        inline_match = re.search(r'\$?\s*(\d{1,3}\.\d{2})', text)
        if inline_match:
            if is_skip_line(text):
                continue
            name = text[:inline_match.start()].strip(" -:@")
            if len(name) < 2:
                continue
            items.append({
                "name"       : name,
                "price"      : inline_match.group(1),
                "confidence" : round(conf, 4)
            })
            used.add(i)

    return items


def extract_total_amount(ocr_result):
    blocks = ocr_result["blocks"]

    for i, block in enumerate(blocks):
        text = block["text"]
        conf = block["confidence"]

        if not any(kw in text.lower() for kw in TOTAL_KEYWORDS):
            continue

      
        match = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})', text)
        if match:
            print(f"Found total: {match.group(1)}")
            return {"value": match.group(1), "confidence": round(conf, 4)}

        if i + 1 < len(blocks):
            next_match = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})', blocks[i + 1]["text"])
            if next_match:
                avg_conf = (conf + blocks[i + 1]["confidence"]) / 2
                print(f"Found total: {next_match.group(1)}")
                return {"value": next_match.group(1), "confidence": round(avg_conf, 4)}

    return {"value": None, "confidence": 0.0}


def extract_all(ocr_result):
    store     = extract_store_name(ocr_result)
    date_time = extract_date_time(ocr_result)
    items     = extract_items(ocr_result)
    total     = extract_total_amount(ocr_result)

    return {
        "store_name"   : store,
        "date"         : date_time["date"],
        "time"         : date_time["time"],
        "items"        : items,
        "total_amount" : total,
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