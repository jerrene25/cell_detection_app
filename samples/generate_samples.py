# samples/generate_samples.py
import numpy as np
import cv2
import os
import random

def generate_image(path, size=(512,512), n_circles=50, min_r=5, max_r=25, noise_level=20):
    img = np.zeros(size, dtype=np.uint8)
    h,w = size
    for i in range(n_circles):
        r = random.randint(min_r, max_r)
        x = random.randint(r, w-r-1)
        y = random.randint(r, h-r-1)
        cv2.circle(img, (x,y), r, (255), -1)
    # add gaussian blur and noise to mimic microscope
    img = cv2.GaussianBlur(img, (5,5), 0)
    noise = (np.random.randn(*img.shape) * noise_level).astype(np.uint8)
    noisy = cv2.add(img, noise)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, noisy)

if __name__=="__main__":
    os.makedirs("samples", exist_ok=True)
    generate_image("samples/sample1.png", size=(1024,1024), n_circles=120)
    print("Generated samples/sample1.png")
