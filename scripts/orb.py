import numpy as np
import cv2
from matplotlib import pyplot as plt


def crop(img, size):
    # Get center of image
    x = int(img.shape[1] / 2)
    y = int(img.shape[0] / 2)

    return img[y: y + size, x: x + size].copy()


img1 = cv2.imread('2.jpg', 1)
img2 = cv2.imread('b.jpg', 1)

orb = cv2.ORB_create(nfeatures=500)
orb2 = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb2.detectAndCompute(img2, None)

n = 1000 - len(des1)
if n > 0:
    t = [[0] * 32] * n
    t = np.array(t)
    t = t.astype(np.uint8)
    des1 = np.append(des1, t, axis=0)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2)
plt.imshow(img3)
plt.show()
