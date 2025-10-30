# detect_cells.py
"""
Core image processing pipeline for cell detection using morphological opening/closing.
Functions are small and testable.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

def load_image(path_or_bytes) -> np.ndarray:
    """Return BGR image as numpy array.
    path_or_bytes: filesystem path or bytes-like (if bytes, np.frombuffer used)"""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not load image.")
    return img

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB to grayscale (uint8)."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img_gray: np.ndarray, method='gaussian', ksize=5) -> np.ndarray:
    """Denoise grayscale image."""
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    if method == 'median':
        return cv2.medianBlur(img_gray, k)
    else:
        return cv2.GaussianBlur(img_gray, (k, k), 0)

def enhance_contrast(img_gray: np.ndarray, clipLimit=2.0, tileGridSize=(8,8)) -> np.ndarray:
    """CLAHE contrast enhancement (helps uneven illumination)."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

def threshold_image(img_gray: np.ndarray, method='otsu', manual_thresh: Optional[int]=None,
                    adaptive='gaussian', block_size=35, C=5) -> np.ndarray:
    """Return binary (0/255) image."""
    if method == 'manual' and manual_thresh is not None:
        _, binary = cv2.threshold(img_gray, int(manual_thresh), 255, cv2.THRESH_BINARY)
        return binary
    if method == 'adaptive':
        bs = block_size
        if bs % 2 == 0:
            bs += 1
        return cv2.adaptiveThreshold(img_gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptive=='gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, bs, C)
    # Otsu (global)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def morph_open_close(binary_img: np.ndarray, kernel_shape=(3,3), opening_iter=1, closing_iter=1) -> np.ndarray:
    """Apply opening then closing and return cleaned binary image (0/255)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, k, iterations=int(opening_iter))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=int(closing_iter))
    return closed

def separate_touching_cells(binary_img: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """Optional watershed-based separation. Returns labeled image (int32).
    Input binary_img expected to be 0/255 with objects in 255."""
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    _, dist_thresh = cv2.threshold((dist/np.max(dist)*255).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts = cv2.connectedComponents(dist_thresh)[1]
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.erode(binary_img, kernel, iterations=1)
    unknown = cv2.subtract(sure_bg, dist_thresh)
    ret, markers = cv2.connectedComponents(dist_thresh)
    markers = markers + 1
    markers[unknown==255] = 0
    color = cv2.cvtColor(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    markers = cv2.watershed(color, markers.astype('int32'))
    return markers

def find_and_filter_contours(binary_img: np.ndarray, min_area=50, max_area: Optional[int]=None) -> Tuple[np.ndarray, int]:
    """
    Find contours, filter by area, draw them on an output image, 
    and return (output_image, cell_count).
    """
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cell_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        if max_area is not None and area > float(max_area):
            continue
        cv2.drawContours(output, [c], -1, (0,255,0), 2)
        cell_count += 1

    return output, cell_count

def draw_detections(original_img: np.ndarray, detections: List[Dict], draw_style='contour') -> np.ndarray:
    """Draw contours, centroids and ids on a copy of original BGR image."""
    out = original_img.copy()
    for obj in detections:
        c = obj['contour']
        cv2.drawContours(out, [c], -1, (0,255,0), 2)
        cx, cy = obj['centroid']
        cv2.circle(out, (cx, cy), 3, (0,0,255), -1)
        cv2.putText(out, str(obj['id']), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    return out
