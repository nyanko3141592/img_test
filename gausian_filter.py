import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussianFilter(img, k, s):
    w, h, c = img.shape
    size = k // 2

    # ０パディング処理
    _img = np.zeros((w + 2 * size, h + 2 * size, c), dtype=np.uint8)
    _img[size:size + w, size:size + h] = img.copy().astype(np.uint8)
    dst = _img.copy()

    # フィルタ作成
    ker = np.zeros((k, k), dtype=np.float)
    for x in range(-1 * size, k - size):
        for y in range(-1 * size, k - size):
            ker[x + size, y + size] = (1 / (2 * np.pi * (s ** 2))) * np.exp(-1 * (x ** 2 + y ** 2) / (2 * (s ** 2)))

    ker /= ker.sum()

    # フィルタリング処理
    for x in range(w):
        for y in range(h):
            for z in range(c):
                dst[x + size, y + size, z] = np.sum(ker * _img[x:x + k, y:y + k, z])

    dst = dst[size:size + w, size:size + h].astype(np.uint8)

    return dst


# 画像読込
img = cv2.imread('data/IMG_4216.png', cv2.IMREAD_COLOR)

# ガウシアンフィルタ
# 第2引数：フィルタサイズ、第3引数：標準偏差(σ)
dst = gaussianFilter(img, 21, 2)

# 画像保存
# cv2.imwrite('result.jpg', img)
# 画像表示
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
