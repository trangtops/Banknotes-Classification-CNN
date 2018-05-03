import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('3.jpg')

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# orb = cv.ORB_create(nfeatures=500)
orb = cv.ORB_create(nfeatures=1000)

kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)


# n = 1500 - len(kp)
#
# if n > 0:
#     t = [[0] * 32] * n
#     t = np.array(t)
#     t = t.astype(np.uint8)
#     des = np.append(des, t, axis=0)

print(len(kp))
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()
