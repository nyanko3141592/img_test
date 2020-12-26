import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.img_connect import get_concat_h_multi_blank

out_dir = 'out/bilateral/times/'
def bi(img, i=1):
    if i == 0:
        return img
    for _ in range(i):
        img = cv2.bilateralFilter(img, 15, 20, 20)
    return img


# origin img
origin_f_name = 'glasses.jpg'
img = cv2.imread(f'data/{origin_f_name}', cv2.IMREAD_COLOR)

bi_times = [0, 3, 10, 30, 50]
for times in bi_times:
    cv2.imwrite(out_dir + f'{origin_f_name}_bi{times}.jpg', bi(img, times))

# PIL 画像結合
imgs = []
for times in bi_times:
    imgs += [Image.open(out_dir + f'{origin_f_name}_bi{times}.jpg')]
print(str(bi_times))
get_concat_h_multi_blank(imgs).save(out_dir + f'compare{str(bi_times)}.jpg')
