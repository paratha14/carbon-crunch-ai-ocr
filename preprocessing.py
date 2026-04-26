import cv2
import numpy as np
from PIL import Image
import os


# LOAD IMAGE

def load_image(image_path: str) -> np.ndarray:
    """Load image from path and convert to numpy array (BGR)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


# RESIZE (optional, keeps aspect ratio)

def resize_image(img: np.ndarray, max_width: int = 1200) -> np.ndarray:
    """
    Resize image if too large, maintaining aspect ratio.
    Larger images = better OCR accuracy up to a point.
    """
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


# DENOISE

def denoise(img: np.ndarray) -> np.ndarray:
    """
    Remove noise using Non-Local Means Denoising.
    Works better than Gaussian blur for preserving text edges.
    """
    # Converting to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10,              # filter strength
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised


# DESKEW

def get_skew_angle(gray_img: np.ndarray) -> float:
    """
    Detect skew angle of the image using Hough Line Transform.
    Returns angle in degrees.
    """
    # Edge detection
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # skip vertical lines
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (text lines)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def deskew(gray_img: np.ndarray) -> np.ndarray:
    """
    Correct skew/rotation in the image.
    Skew under 0.5 degrees is ignored (not worth rotating).
    """
    angle = get_skew_angle(gray_img)

    if abs(angle) < 0.5:
        return gray_img  

    h, w = gray_img.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

   
    rotated = cv2.warpAffine(
        gray_img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    return rotated


# CONTRAST ENHANCEMENT (CLAHE)

def enhance_contrast(gray_img: np.ndarray) -> np.ndarray:
    """
    Applying CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,        
        tileGridSize=(8, 8)   
    )
    enhanced = clahe.apply(gray_img)
    return enhanced


# BINARIZATION (Thresholding)

def binarize(gray_img: np.ndarray) -> np.ndarray:
    """
    Convert grayscale to clean black & white using adaptive thresholding.
    Adaptive is better than Otsu for receipts with uneven backgrounds.
    """
    binary = cv2.adaptiveThreshold(
        gray_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,    
        C=10             
    )
    return binary


# MORPHOLOGICAL CLEANUP

def morphological_cleanup(binary_img: np.ndarray) -> np.ndarray:
    """
    Remove tiny noise blobs and fill small gaps in characters.
    Helps OCR read broken or speckled text.
    """
    kernel = np.ones((1, 1), np.uint8)

    # Remove small noise
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Fill small gaps
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# FULL PIPELINE

def preprocess(image_path: str, save_debug: bool = False, debug_dir: str = "debug") -> np.ndarray:
    img = load_image(image_path)
    img = resize_image(img, max_width=1200)
    gray = denoise(img)
    deskewed = deskew(gray)
    enhanced = enhance_contrast(deskewed)
    binary = binarize(enhanced)
    final = morphological_cleanup(binary)

    # Quality check — if preprocessing made it worse, return original
    # Compare average confidence proxy: if final has very few non-white pixels
    # it means binarization destroyed the image
    white_ratio = np.sum(final == 255) / final.size
    if white_ratio > 0.97 or white_ratio < 0.5:
        print("[PREPROCESS] Quality check failed — using grayscale fallback")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return final


def preprocess_to_pil(image_path: str) -> Image.Image:
    """
    Convenience wrapper — returns a PIL Image instead of numpy array.
    PaddleOCR accepts both, but PIL is useful for quick inspection.
    """
    processed = preprocess(image_path)
    return Image.fromarray(processed)

'''
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_receipt_image>")
        print("Example: python preprocessing.py")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Processing: {path}")

    result = preprocess(path, save_debug=True, debug_dir="debug")

    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
   

    # Show result
    cv2.imshow("Preprocessed Receipt", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''