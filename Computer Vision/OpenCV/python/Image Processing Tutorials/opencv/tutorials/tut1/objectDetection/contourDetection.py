import cv2
import numpy as np
import matplotlib.pyplot as plt


pennyImg = cv2.imread("../../../gallery/penny.jpg", cv2.IMREAD_COLOR)
pennyImg = cv2.resize(pennyImg, None, fx=0.5, fy=0.5)
grayPenny = cv2.cvtColor(pennyImg, cv2.COLOR_BGR2GRAY)
grayPenny = cv2.medianBlur(grayPenny, 9)
_, mask = cv2.threshold(grayPenny, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(pennyImg, contours, -1, (255, 0, 0), 3)


cv2.imshow("mask", pennyImg)
cv2.imshow("ranged-frame", grayPenny)
cv2.waitKey(0)
cv2.destroyAllWindows()
