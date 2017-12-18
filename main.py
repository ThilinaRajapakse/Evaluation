import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample.jpg')
# img = cv2.resize(img, None, fx=0.2, fy=0.2)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(imgray,9,75,75)
blurred = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

# kernel = np.ones((3, 3), np.uint8)
# blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=3)


# plt.imshow(blurred, cmap='gray')
# plt.show()

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

# cv2.imshow("Original", img)
# cv2.imshow("Edges", np.hstack([wide, tight, auto]))
# cv2.waitKey(0)

edges = cv2.Canny(imgray,100,200)

image, contours, hierarchy = cv2.findContours(blurred,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

# plt.imshow(img, cmap='gray')
# plt.show()
boxes = []

for cont in contours:
    if cv2.contourArea(cont) > 500:
        rect = cv2.minAreaRect(cont)
        points = cv2.boxPoints(rect)
        points = np.int0(points)

        if 0.7 < (points[0][1] - points[1][1]) / (points[2][0] - points[0][0]) < 0.9:
            boxes.append(points)

options = []
for box in boxes:
    if box[0][0] > 1000:
        options.append(box)

boxes = options
img = cv2.drawContours(img, boxes, -1, (0,255,0), 3)

print(len(boxes))
print(boxes[0].shape)

coords = np.concatenate(boxes)
xmax, ymax = np.max(coords, axis=0)
xmin, ymin = np.min(coords, axis=0)

img = img[ymin:ymax, xmin:xmax]

print(xmax, xmin, ymax, ymin)

plt.imshow(img, cmap='gray')
plt.show()


