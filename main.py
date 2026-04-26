import os
import sys
import cv2
import json
from preprocessing import preprocess
from ocr import extract_ocr_result
from extracted_text import extract_all, save_json
from summary import generate_summary

IMAGES_DIR = "images"
OUTPUT_DIR = "outputs"
SUPPORTED  = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def process_image(image_path: str) -> dict | None:
    """
    Full pipeline for a single image.
    Returns extracted data dict or None if failed.
    """
    print(f"\n{'='*55}")
    print(f"Processing: {image_path}")
    print(f"{'='*55}")

    try:
        # Step 1 — Preprocess
        preprocessed = preprocess(image_path)

        # Step 2 — OCR
        ocr_result = extract_ocr_result(preprocessed)

        if ocr_result["total_blocks"] == 0:
            print("[WARN] No text detected — skipping.")
            return None

        # Step 3 — Extract fields
        extracted = extract_all(ocr_result)

        # Step 4 — Save JSON
        save_json(extracted, image_path, OUTPUT_DIR)

        return extracted

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None


def run_pipeline(images_dir: str = IMAGES_DIR):
    """
    Runs pipeline on all images in the given directory.
    """
    if not os.path.exists(images_dir):
        print(f"[ERROR] Images folder '{images_dir}' not found.")
        sys.exit(1)

    # Collect all supported image files
    image_files = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(SUPPORTED)
    ]

    if not image_files:
        print(f"[ERROR] No images found in '{images_dir}'.")
        sys.exit(1)

    print(f"\nFound {len(image_files)} image(s) to process.")

    results   = []
    failed    = []

    for image_path in image_files:
        result = process_image(image_path)
        if result:
            results.append(result)
        else:
            failed.append(image_path)

    # Summary
    print(f"\n{'='*55}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*55}")
    print(f"Processed : {len(results)} / {len(image_files)}")
    if failed:
        print(f"Failed    : {[os.path.basename(f) for f in failed]}")

    # Generate financial summary
    print("\nGenerating financial summary...")
    generate_summary(OUTPUT_DIR)


if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        path = sys.argv[1]
        if os.path.isfile(path):
            process_image(path)
            generate_summary(OUTPUT_DIR)
        elif os.path.isdir(path):
            run_pipeline(path)
        else:
            print(f"[ERROR] '{path}' is not a valid file or directory.")
    else:
        run_pipeline(IMAGES_DIR)