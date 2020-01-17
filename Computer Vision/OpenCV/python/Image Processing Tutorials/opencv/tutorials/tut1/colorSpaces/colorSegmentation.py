import cv2
import numpy as np


def creatTrackBar(name):
    cv2.namedWindow(name)
    cv2.createTrackbar("lowerHue", name, 170, 179, lambda x: None)
    cv2.createTrackbar("lowerSat", name, 60, 255, lambda x: None)
    cv2.createTrackbar("lowerVal", name, 115, 255, lambda x: None)
    cv2.createTrackbar("upperHue", name, 179, 179, lambda x: None)
    cv2.createTrackbar("upperSat", name, 92, 255, lambda x: None)
    cv2.createTrackbar("upperVal", name, 206, 255, lambda x: None)


def getTrackBarVals(name):

    lH = cv2.getTrackbarPos("lowerHue", name)
    lS = cv2.getTrackbarPos("lowerSat", name)
    lV = cv2.getTrackbarPos("lowerVal", name)
    uH = cv2.getTrackbarPos("upperHue", name)
    uS = cv2.getTrackbarPos("upperSat", name)
    uV = cv2.getTrackbarPos("upperVal", name)

    return np.array((lH, lS, lV)), np.array((uH, uS, uV))


cap = cv2.VideoCapture(0)
creatTrackBar("color-range TrackBar")

while True:
    _, frame = cap.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowObjColor, highObjColor = getTrackBarVals("color-range TrackBar")
    colorRange = cv2.inRange(hsvFrame, lowObjColor, highObjColor)
    gauBlr = cv2.GaussianBlur(colorRange, (9, 9), 1)
    medBlr = cv2.medianBlur(colorRange, 15)
    detectedObj = cv2.bitwise_or(gauBlr, medBlr)
    colorRange = cv2.morphologyEx(detectedObj, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))
    colorRange = cv2.bitwise_and(frame, frame,  mask=detectedObj)
    cv2.imshow("frame", colorRange)
    cv2.imshow("ranged-frame", detectedObj)
    if cv2.waitKey(1) & 0xFF == ord('q') or not cap.isOpened():
        break


cap.release()
cv2.destroyAllWindows()
