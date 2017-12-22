import cv2
import numpy as np
from matplotlib import pyplot as plt
from sort_contours import sort_contours

img = cv2.imread('3.jpg')
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
contours, bounding_boxes = sort_contours(contours)

# img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

# plt.imshow(img, cmap='gray')
# plt.show()
boxes = []
contour_boxes = []
contours = list(contours)
contours.sort(key=cv2.contourArea)

for cont in contours:
    if 5000 > cv2.contourArea(cont) > 700:
        rect = cv2.minAreaRect(cont)
        points = cv2.boxPoints(rect)
        points = np.int0(points)

        if 0.7 < (points[0][1] - points[1][1]) / (points[2][0] - points[0][0]) < 0.9:
            if points[0][0] > 1000:
                if points[0][0] == points[1][0] and points[3][0] == points[2][0] and points[0][1] == points[3][1] \
                        and points[1][1] == points[2][1]:
                    boxes.append(points)
                    contour_boxes.append(cont)

# options = []
# for box in boxes:
#     if box[0][0] > 1000:
#         if box[0][0] == box[1][0] and box[3][0] == box[2][0] and box[0][1] == box[3][1] and box[1][1] == box[2][1]:
#             options.append(box)
#
# boxes = options
# img = cv2.drawContours(img, boxes, -1, (0,255,0), 1)

coords = np.concatenate(boxes)
xmax, ymax = np.max(coords, axis=0)
xmin, ymin = np.min(coords, axis=0)

# img = img[ymin:ymax, xmin:xmax]

print(len(boxes))
print(len(contour_boxes))
# areas = []
#
# for cont in contour_boxes:
#     areas.append(cv2.contourArea(cont))
#
# areas.sort()
# print(areas)
# plt.imshow(img, cmap='gray')
# plt.show()

# contour_boxes.sort(key=cv2.contourArea)
# rect = cv2.minAreaRect(contour_boxes[-1])
# points = cv2.boxPoints(rect)
# points = np.int0(points)


def check_if_inside(points, boxes):
    for box in boxes:
        if box[0][1] > points[0][1] > box[1][1] and box[3][0] > points[0][0] > box[0][0]\
                and box[0][1] > points[2][1] > box[1][1] and box[3][0] > points[2][0] > box[0][0]:
            return True
    return False


boxes_unique = []
contours_unique = []

# contour_boxes, bnd = sort_contours(contour_boxes)

for cont in contour_boxes:
    rect = cv2.minAreaRect(cont)
    points = cv2.boxPoints(rect)
    box = np.int0(points)
    if not check_if_inside(box, boxes):
        boxes_unique.append(box)
        contours_unique.append(cont)

print(len(boxes_unique))
print(len(contours_unique))
# for cont in contour_boxes:
#     print(cv2.contourArea(cont))

heights_of_boxex = [(box[0][1] - box[1][1]) for box in boxes_unique]
print(np.mean(heights_of_boxex))

img = cv2.drawContours(img, boxes_unique, -1, (0,255,0), 2)
img = img[ymin:ymax, xmin:xmax]
plt.imshow(img, cmap='gray')
plt.show()




