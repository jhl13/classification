import cv2
import numpy as np

image = np.array(cv2.imread("/home/luo13/Pictures/webwxgetmsgimg.jpeg"), dtype=np.float32)
print (image)
cv2.imshow("image", image)
cv2.waitKey(0)

image = image[:, ::-1, :]
cv2.imshow("image", image)
cv2.waitKey(0)

image = image[::-1, :, :]
cv2.imshow("image", image)
cv2.waitKey(0)