import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/IMG_4216.png', cv2.IMREAD_COLOR)

dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()
