import cv2
import numpy as np
import matplotlib.pyplot as plt


bgr = cv2.imread("../../../gallery/hourse.jpeg", cv2.IMREAD_COLOR)  # bluish img


fontSettings = {
    "text": "hourse in Blue channel",
    "org": (200, 500),
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale": 1,
    "color": (0, 255, 0),
    "thickness": 1,
    "lineType": cv2.LINE_AA
}

cv2.putText(bgrN, **fontSettings)
cv2.imshow("blue-channel", bgrN)
cv2.waitKey(0)
cv2.destroyAllWindows()
