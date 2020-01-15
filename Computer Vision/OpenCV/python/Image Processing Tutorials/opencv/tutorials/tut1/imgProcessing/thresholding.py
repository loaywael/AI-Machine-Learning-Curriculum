import cv2
import numpy as np


hourseImg = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_GRAYSCALE)
tomatoImg = cv2.imread("../../../gallery/tomato.jpg", cv2.IMREAD_GRAYSCALE)
tomatoImg = cv2.resize(tomatoImg, (1000, 667))
logoImg = cv2.imread("../../../gallery/logo.png", cv2.IMREAD_GRAYSCALE)
outImg = hourseImg.copy() * 0

# cv2.addWeighted(hourseImg, 0.4, tomatoImg, 0.7, 0, outImg)
# cv2.imshow("logo", outImg)
# cv2.waitKey(0)
# cv2.add(hourseImg, tomatoImg, outImg)
# cv2.imshow("logo", outImg)
# cv2.waitKey(0)
# outImg = hourseImg + tomatoImg
# cv2.imshow("logo", outImg)
# cv2.waitKey(0)

rows, cols = logoImg.shape
roi = logoImg[:rows, :cols]
# _, threshholdLogo = cv2.threshold(logoImg, 240, 255, cv2.THRESH_BINARY)
_, invThreshholdLogo = cv2.threshold(logoImg, 240, 255, cv2.THRESH_BINARY_INV)

print(roi.shape)

cv2.imshow("logo", logBkg)
cv2.waitKey(0)
# cv2.imshow("threshold", threshholdLogo)
# cv2.waitKey(0)
cv2.imshow("inv-threshold", invThreshholdLogo)
cv2.waitKey(0)
cv2.destroyAllWindows()
