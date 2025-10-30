# tests/test_detect_cells.py
import numpy as np
import cv2
from detect_cells import to_grayscale, denoise, threshold_image, morph_open_close, find_and_filter_contours
import pytest

def make_circle_img():
    img = np.zeros((200,200), dtype=np.uint8)
    cv2.circle(img, (50,50), 20, 255, -1)
    cv2.circle(img, (150,150), 15, 255, -1)
    return img

def test_threshold_and_contours():
    img = make_circle_img()
    den = denoise(img, method='gaussian', ksize=3)
    binary = threshold_image(den, method='manual', manual_thresh=127)
    cleaned = morph_open_close(binary, kernel_shape=(3,3), opening_iter=1, closing_iter=1)
    objs = find_and_filter_contours(cleaned, min_area=50)
    assert len(objs) == 2

if __name__ == "__main__":
    pytest.main([__file__])
