# Carbon Crunch — AI OCR Pipeline Documentation

## Approach

The pipeline processes receipt images in four sequential stages:

**1. Image Preprocessing** — Each image is loaded, resized to a max width of 1200px, denoised using Non-Local Means filtering, deskewed via Hough Line Transform, contrast-enhanced with CLAHE, and binarized using adaptive thresholding. A quality check compares the white pixel ratio of the final output — if preprocessing degraded the image (ratio > 0.97 or < 0.50), the pipeline falls back to grayscale to avoid feeding garbage to the OCR engine.

**2. OCR** — PaddleOCR (PaddleX-based) is used as the sole OCR engine. It returns `rec_texts`, `rec_scores`, and `rec_polys` per image. Blocks are sorted top-to-bottom by bounding box Y coordinate to ensure reading order. A singleton engine instance is used across all images to avoid repeated model loading.

**3. Field Extraction** — A regex and heuristic-based extractor pulls four fields from OCR blocks:
- **Store name** — scans all blocks for known store keywords first, falls back to first valid top-5 block
- **Date + Time** — tries six regex patterns covering common date formats; time is extracted from the same line as the date when possible, skipping store hour lines
- **Items** — handles two layouts: (A) price in a separate block below the name, with lookback up to 2 blocks skipping barcode strings; (B) name + barcode + price inline, stripped via regex
- **Total** — keyword search for "total" variants, skipping subtotal, checking same line then next block

**4. Confidence Scoring** — Each field gets a composite score: `OCR confidence × pattern match score × heuristic score`. Fields below 0.70 are flagged. Missing fields are tracked separately.

---

## Tools Used

| Tool | Purpose |
|---|---|
| PaddleOCR (PaddleX) | Text detection + recognition + confidence scores |
| OpenCV | Image preprocessing pipeline |
| NumPy | Array operations |
| Python `re` | Regex-based field extraction |
| JSON (stdlib) | Structured output |
| Pandas | Financial summary aggregation |

---

## Challenges Faced

**PaddleOCR version format change** — The newer PaddleX-based version returns a dict with `rec_texts`, `rec_scores`, `rec_polys` instead of the classic list-of-lines format. Required building a format auto-detector.

**Preprocessing hurting some images** — The Walmart receipt on an orange background was actively degraded by binarization. Solved with a white pixel ratio quality check that falls back to grayscale when preprocessing fails.

**Split blocks per line** — PaddleOCR often splits a single receipt line into multiple blocks (name, barcode, price separately). Solved by pairing price-only blocks with their preceding name block using a lookback strategy.

**Receipts with no standard layout** — Store name location varies (top, middle, after address). Solved by scanning the entire receipt for known store keywords before falling back to positional heuristics.

**OCR misreads on noisy images** — Characters like `5→6`, `B→8`, `LB→L8` occur on low quality images. These are OCR-level errors outside the scope of post-processing fixes without a fine-tuned model.

---

## Improvements With More Time

- **Fine-tune PaddleOCR** on a receipt-specific dataset to improve accuracy on noisy, skewed, or handwritten receipts
- **Perspective correction** using four-point transform for receipts photographed at an angle
- **NLP-based store name extraction** using NER instead of a fixed keyword list
- **Line grouping** — cluster OCR blocks by Y coordinate proximity to reconstruct full receipt lines before extraction
- **Multi-currency support** — extend price regex to handle £, €, ₹ etc.
- **Barcode/QR detection** to extract transaction IDs reliably