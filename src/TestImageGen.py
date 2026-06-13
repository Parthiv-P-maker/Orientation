import cv2
import numpy as np

img = np.ones((500, 500, 3), dtype=np.uint8) * 255

rect = ((250, 250), (200, 80), 30)

box = cv2.boxPoints(rect)
box = np.int32(box)

cv2.drawContours(img, [box], 0, (0, 0, 0), -1)

cv2.imwrite("images/test_rect_30deg.png", img)

cv2.imshow("Generated", img)
cv2.waitKey(0)
cv2.destroyAllWindows()