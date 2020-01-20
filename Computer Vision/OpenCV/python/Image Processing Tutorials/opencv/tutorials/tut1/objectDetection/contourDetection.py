import cv2
import numpy as np
import matplotlib.pyplot as plt


objects = cv2.imread("../../../gallery/objects.jpg", cv2.IMREAD_COLOR)
grayObjs = cv2.cvtColor(objects, cv2.COLOR_BGR2GRAY)

mask = cv2.adaptiveThreshold(
    grayObjs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 125, 1)
img, hierarchy = cv2.findContours(grayObjs, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(img)
# compelete_contours: the number of found contours in the image
# hierarchy: the point which draws the contours
# internal & external shapes found in the image

# contMask = np.zeros_like(mask)
# for i in range(len(compContours)):
#     if hierarchy[0][i][3] == -1:     # external contour
#         cv2.drawContours(contMask, compContours, i, (255, 0, 0), -1)

cv2.imshow("img", img)
# cv2.imshow("FG", contMask)
# cv2.imshow("BKG", imgBKG)
cv2.waitKey(0)
cv2.destroyAllWindows()
