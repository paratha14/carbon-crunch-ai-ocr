import numpy as np
import cv2
from paddleocr import PaddleOCR
from PIL import Image
from typing import Union
import logging
import os

logging.getLogger("ppocr").setLevel(logging.ERROR)


# INITIALIZE OCR ENGINE

def get_ocr_engine() -> PaddleOCR:
    """
    Initialize and return a PaddleOCR instance.

    Settings explained:
    - use_angle_cls=True  : detects and corrects rotated text (90°, 180°, 270°)
    - lang='en'           : English language model
    - use_gpu=False       : CPU mode 
    - show_log=False      : suppress console spam
    - det_db_thresh=0.3   : lower = detects more text boxes (good for faint text)
    - rec_batch_num=6     : how many text boxes to recognize in one batch
    """
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        text_det_thresh=0.3,
        text_det_box_thresh=0.4,
        text_recognition_batch_size=6,
        enable_mkldnn=False
    )
    return ocr


# Single global instance
_ocr_engine = None

def get_engine() -> PaddleOCR:
    """Returns a singleton OCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR] Loading PaddleOCR model (first time only)...")
        _ocr_engine = get_ocr_engine()
        print("[OCR] Model loaded successfully.")
    return _ocr_engine

def parse_paddlex_dict(result_dict: dict) -> list:
    """
    Parse the new PaddleX-based output format.
    The result is a single dict containing:
      - 'rec_texts'  : list of extracted strings
      - 'rec_scores' : list of confidence floats
      - 'rec_polys'  : list of bounding box arrays
    """
    texts  = result_dict.get("rec_texts", [])
    scores = result_dict.get("rec_scores", [])
    polys  = result_dict.get("rec_polys", result_dict.get("dt_polys", []))
 
    blocks = []
    for idx, (text, score) in enumerate(zip(texts, scores)):
        text = str(text).strip()
        if not text:
            continue
        if score < 0.1:
            continue
 
        # Get bounding box — convert numpy array to plain list
        bbox = None
        if idx < len(polys):
            try:
                bbox = polys[idx].tolist()
            except Exception:
                bbox = []
 
        blocks.append({
            "text"       : text,
            "confidence" : round(float(score), 4),
            "bbox"       : bbox,
            "line_index" : idx,
        })
 
    return blocks
 

def parse_ocr_line(line) -> tuple:
    """
    Safely parse a single OCR result line regardless of PaddleOCR version. 
    Returns: (bbox, text, confidence)
    """
    try:
        if len(line) == 2:
            # Format A: [bbox, (text, conf)]
            bbox, text_conf = line
            if isinstance(text_conf, (list, tuple)):
                text, confidence = text_conf
            else:
                text = str(text_conf)
                confidence = 0.5
        elif len(line) == 3:
            # Format B: [bbox, text, confidence]
            bbox, text, confidence = line
        else:
            print(f"[OCR WARN] Unexpected line format with {len(line)} elements: {line}")
            return None, None, None
 
        return bbox, str(text).strip(), float(confidence)
 
    except Exception as e:
        print(f"[OCR WARN] Could not parse line: {line} — {e}")
        return None, None, None

# CORE OCR FUNCTION

def run_ocr(image_input: Union[str, np.ndarray, Image.Image]) -> list:
    """
    Run PaddleOCR on an image and return structured text blocks.
 
    Returns:
        List of dicts:
        {
            "text"       : str,
            "confidence" : float (0.0-1.0),
            "bbox"       : list of 4 corner points,
            "line_index" : int
        }
    """
    engine = get_engine()
 
    # --- Normalize input ---
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image: {image_input}")
 
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
 
    elif isinstance(image_input, np.ndarray):
        img = image_input
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")
 
    # --- Run OCR ---
    try:
        results = engine.predict(img)
    except Exception as e:
        print(f"[OCR ERROR] OCR failed: {e}")
        return []
 
    # --- Handle empty results ---
    if results is None:
        print("[OCR] No results returned.")
        return []
 
    # Auto-detect format
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        return parse_paddlex_dict(results[0])

    if isinstance(results, dict):
        return parse_paddlex_dict(results)

    raw_lines = results[0] if isinstance(results[0], list) else results
    if raw_lines is None:
        print("[OCR] No text detected in image.")
        return []

    text_blocks = []
    for idx, line in enumerate(raw_lines):
        bbox, text, confidence = parse_ocr_line(line)
        if text is None or not text:
            continue
        if confidence is not None and confidence < 0.1:
            continue
        text_blocks.append({
            "text"       : text,
            "confidence" : round(confidence, 4),
            "bbox"       : bbox,
            "line_index" : idx,
        })
    return text_blocks

# GET PLAIN TEXT (top-to-bottom order)

def get_sorted_text_blocks(image_input: Union[str, np.ndarray, Image.Image]) -> list[dict]:
    """
    Run OCR and return text blocks sorted top-to-bottom by vertical position.
    This is important — PaddleOCR doesn't always return lines in reading order.
    """
    blocks = run_ocr(image_input)

    # Sort by the top-left Y coordinate of bounding box
    blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][0][1] if b["bbox"] else 0)

    # Re-index after sorting
    for i, block in enumerate(blocks_sorted):
        block["line_index"] = i

    return blocks_sorted


def get_full_text(image_input: Union[str, np.ndarray, Image.Image]) -> str:
    """
    Returns all extracted text as a single plain string (top to bottom).
    Useful for regex-based field extraction.
    """
    blocks = get_sorted_text_blocks(image_input)
    return "\n".join(b["text"] for b in blocks)


def get_average_confidence(blocks: list[dict]) -> float:
    """
    Returns average OCR confidence across all detected text blocks.
    Useful as a document-level quality signal.
    """
    if not blocks:
        return 0.0
    return round(sum(b["confidence"] for b in blocks) / len(blocks), 4)


# 4. FULL OCR RESULT OBJECT

def extract_ocr_result(image_input: Union[str, np.ndarray, Image.Image]) -> dict:
    """
    Master function — returns everything from OCR in one structured object.

    Returns:
    {
        "blocks"            : list of text blocks (sorted top to bottom),
        "full_text"         : full plain text string,
        "avg_confidence"    : average confidence across all blocks,
        "total_blocks"      : number of text blocks detected,
        "low_conf_blocks"   : blocks with confidence < 0.7 (flagged),
    }
    """
    blocks = get_sorted_text_blocks(image_input)
    full_text = "\n".join(b["text"] for b in blocks)
    avg_conf = get_average_confidence(blocks)
    low_conf = [b for b in blocks if b["confidence"] < 0.7]

    return {
        "blocks"          : blocks,
        "full_text"       : full_text,
        "avg_confidence"  : avg_conf,
        "total_blocks"    : len(blocks),
        "low_conf_blocks" : low_conf,
    }


def print_ocr_result(ocr_result: dict) -> None:
    """Print OCR results in a readable format for debugging."""
    print("\n" + "="*60)
    print("OCR RESULTS")
    print("="*60)
    print(f"Total blocks detected : {ocr_result['total_blocks']}")
    print(f"Average confidence    : {ocr_result['avg_confidence']:.2%}")
    print(f"Low confidence blocks : {len(ocr_result['low_conf_blocks'])}")
    print("-"*60)
    print("EXTRACTED TEXT (top to bottom):")
    print("-"*60)
    for block in ocr_result["blocks"]:
        flag = " LOW CONF" if block["confidence"] < 0.7 else ""
        print(f"  [{block['confidence']:.2f}] {block['text']}{flag}")
    print("="*60 + "\n")

# QUICK TEST


if __name__ == "__main__":
    import sys
    from preprocessing import preprocess

    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <path_to_receipt_image>")
        print("Example: python ocr_engine.py images/receipt1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"\nPreprocessing image: {image_path}")
    preprocessed = preprocess(image_path)

    print("Running PaddleOCR...")
    result = extract_ocr_result(preprocessed)

    print_ocr_result(result)

    print("\n[FULL TEXT OUTPUT]")
    print(result["full_text"])