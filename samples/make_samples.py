"""Tạo ảnh mẫu từ bộ dữ liệu công khai của scikit-image (CC0 / public domain)."""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data

out = Path(__file__).parent
Image.fromarray(data.camera()).save(out / "cameraman.png")            # bầu trời trơn + cỏ
Image.fromarray(data.astronaut()).convert("L").save(out / "astronaut.png")
Image.fromarray(data.coffee()).convert("L").save(out / "coffee.png")

# Ảnh tổng hợp: nửa trên "bầu trời" (gradient trơn), nửa dưới "cỏ" (texture thật)
grass = np.asarray(Image.fromarray(data.grass()).resize((512, 256)), dtype=np.float64)
yy = np.linspace(0, 1, 256)[:, None]
sky = 200 - 60 * yy + 0 * np.zeros((256, 512))
img = np.vstack([sky, grass * 0.8 + 20])
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out / "troi_va_co.png")
print("ok")
