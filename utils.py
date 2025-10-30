# utils.py
import os
import cv2
import pandas as pd
from typing import List, Dict

def save_image(path: str, img) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, img)

def detections_to_dataframe(detections: List[Dict]) -> pd.DataFrame:
    rows = []
    for d in detections:
        x,y,w,h = d['bbox']
        cx,cy = d['centroid']
        rows.append({
            'id': d['id'],
            'centroid_x': cx,
            'centroid_y': cy,
            'area_px': d['area'],
            'bbox_x': x,
            'bbox_y': y,
            'bbox_w': w,
            'bbox_h': h,
            'circularity': d['circularity'],
            'equivalent_diameter': d['equivalent_diameter']
        })
    return pd.DataFrame(rows)

def export_csv(detections: List[Dict], out_path: str) -> None:
    df = detections_to_dataframe(detections)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False)
