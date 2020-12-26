import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.img_connect import get_concat_h_multi_blank

out_dir = 'out/bilateral/ds/'


def bi(img, i=1, d=15):
    if i is 0:
        return img
    for _ in range(i):
        img = cv2.bilateralFilter(img, d, 20, 20)
    return img


# origin img
origin_f_name = 'glasses.jpg'
img = cv2.imread(f'data/{origin_f_name}', cv2.IMREAD_COLOR)

ds = [5, 15, 25, 50, 70]
for d in ds:
    cv2.imwrite(out_dir + f'{origin_f_name}_bi{d}.jpg', bi(img, d=d))
    print('d', d)

# PIL 画像結合
imgs = []
for d in ds:
    imgs += [Image.open(out_dir + f'{origin_f_name}_bi{d}.jpg')]
print(str(ds))
get_concat_h_multi_blank(imgs).save(out_dir + f'compare{str(ds)}.jpg')
