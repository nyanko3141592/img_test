import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/IMG_4216.png', cv2.IMREAD_COLOR)

kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
